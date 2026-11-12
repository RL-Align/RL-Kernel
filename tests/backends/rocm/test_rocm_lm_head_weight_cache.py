# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import rl_engine.integrations.engines.rollout.vllm.runtime as vllm_runtime
from rl_engine.backends.rocm.gemm.det_gemm import (
    RocmDetGemmOp,
    det_gemm_linear_prepared,
    prepare_det_gemm_linear_weight,
)


class _CpuRocmTensor(torch.Tensor):
    """CPU tensor that reaches the allocation-free ROCm helper validations."""

    @staticmethod
    def __new__(cls, value: torch.Tensor) -> _CpuRocmTensor:
        return torch.Tensor._make_subclass(cls, value, value.requires_grad)

    @property
    def is_cuda(self) -> bool:
        return True


def _cpu_rocm_tensor(value: torch.Tensor) -> _CpuRocmTensor:
    return _CpuRocmTensor(value)


def _prepare_on_cpu(
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    transposed = weight.detach().transpose(0, 1).contiguous()
    if out is None:
        return transposed
    with torch.no_grad():
        out.copy_(transposed)
    return out


class _FakeLmHead(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, *, strict: bool = True) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        if strict:
            setattr(
                self,
                vllm_runtime._STRICT_PROJECTION_MARKER,
                "lm_head",
            )


def _linear_method_class():
    class FakeUnquantizedEmbeddingMethod:
        def __init__(self) -> None:
            self.original_apply_calls = []
            self.process_calls = []

        def process_weights_after_loading(self, layer):
            self.process_calls.append(layer)
            return "processed"

        def apply(self, layer, x, bias=None):
            self.original_apply_calls.append((layer, x, bias))
            return x

    return FakeUnquantizedEmbeddingMethod


class _FakeDetGemm:
    def __init__(self) -> None:
        self.prepared_calls = []
        self.linear_calls = []

    def linear_prepared(self, x: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        self.prepared_calls.append((x, weight_t))
        return x.float().matmul(weight_t.float())

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        self.linear_calls.append((x, weight))
        return x.float().matmul(weight.float().transpose(0, 1))


def test_prepare_det_gemm_linear_weight_creates_and_refreshes_stable_storage():
    source = _cpu_rocm_tensor(torch.arange(12, dtype=torch.bfloat16).reshape(3, 4))

    prepared = prepare_det_gemm_linear_weight(source)
    original_ptr = prepared.data_ptr()
    assert prepared.shape == (4, 3)
    assert prepared.is_contiguous()
    assert not prepared.requires_grad
    assert torch.equal(prepared, source.as_subclass(torch.Tensor).transpose(0, 1))

    with torch.no_grad():
        source.add_(16)
    refreshed = prepare_det_gemm_linear_weight(source, out=prepared)

    assert refreshed is prepared
    assert refreshed.data_ptr() == original_ptr
    assert torch.equal(refreshed, source.as_subclass(torch.Tensor).transpose(0, 1))


@pytest.mark.parametrize(
    ("source", "error", "message"),
    (
        (torch.ones(4, dtype=torch.bfloat16), ValueError, "must be 2-D"),
        (torch.ones(2, 3, dtype=torch.float16), TypeError, "must be BF16"),
        (
            torch.ones(2, 3, dtype=torch.bfloat16),
            RuntimeError,
            "must be on ROCm",
        ),
        (
            _cpu_rocm_tensor(torch.arange(12, dtype=torch.bfloat16).reshape(4, 3).transpose(0, 1)),
            ValueError,
            "source deterministic linear weights must be contiguous",
        ),
    ),
)
def test_prepare_det_gemm_linear_weight_rejects_invalid_sources(
    source: torch.Tensor,
    error: type[Exception],
    message: str,
):
    with pytest.raises(error, match=message):
        prepare_det_gemm_linear_weight(source)


@pytest.mark.parametrize(
    ("out", "error", "message"),
    (
        (
            torch.empty(3, 4, dtype=torch.bfloat16),
            ValueError,
            "must have shape",
        ),
        (
            torch.empty(4, 3, dtype=torch.float16),
            TypeError,
            "dtype must match",
        ),
        (
            torch.empty(3, 4, dtype=torch.bfloat16).transpose(0, 1),
            ValueError,
            "must be contiguous",
        ),
        (
            torch.empty(4, 3, dtype=torch.bfloat16, requires_grad=True),
            ValueError,
            "must not require gradients",
        ),
    ),
)
def test_prepare_det_gemm_linear_weight_rejects_invalid_refresh_buffers(
    out: torch.Tensor,
    error: type[Exception],
    message: str,
):
    source = _cpu_rocm_tensor(torch.ones(3, 4, dtype=torch.bfloat16))

    with pytest.raises(error, match=message):
        prepare_det_gemm_linear_weight(source, out=out)


def test_prepare_det_gemm_linear_weight_rejects_source_storage_overlap():
    storage = torch.arange(18, dtype=torch.bfloat16)
    source = _cpu_rocm_tensor(storage[:9].reshape(3, 3))
    overlapping_out = storage[1:10].reshape(3, 3)

    with pytest.raises(ValueError, match="must not alias its source"):
        prepare_det_gemm_linear_weight(source, out=overlapping_out)


@pytest.mark.parametrize("alias", ("activation", "weight"))
def test_det_gemm_linear_prepared_rejects_output_alias(alias):
    # A square projection makes both aliases otherwise-valid output buffers.
    activation = torch.ones(3, 3, dtype=torch.bfloat16)
    weight_t = torch.ones(3, 3, dtype=torch.bfloat16)
    out = activation if alias == "activation" else weight_t

    with pytest.raises(ValueError, match="output must not alias its inputs"):
        det_gemm_linear_prepared(activation, weight_t, out=out)


def test_vllm_rocm_lm_head_post_load_refreshes_cache_in_place(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    method_cls = _linear_method_class()
    det_gemm = _FakeDetGemm()
    prepare_calls = []

    def prepare(weight, *, out=None):
        prepare_calls.append((weight, out))
        return _prepare_on_cpu(weight, out=out)

    vllm_runtime._patch_strict_lm_head_linear(
        linear_method_cls=method_cls,
        det_gemm=det_gemm,
        prepare_weight=prepare,
    )
    method = method_cls()
    layer = _FakeLmHead(torch.arange(20, dtype=torch.bfloat16).reshape(5, 4))

    assert method.process_weights_after_loading(layer) == "processed"
    state = getattr(layer, vllm_runtime._STRICT_LM_HEAD_CACHE_STATE)
    cached = getattr(layer, vllm_runtime._STRICT_LM_HEAD_CACHE_BUFFER)
    cache_ptr = cached.data_ptr()
    assert method.process_calls == [layer]
    assert prepare_calls == [(layer.weight, None)]
    assert state.generation == 1
    assert state.valid
    assert state.weight_t is cached
    assert cached.data_ptr() == cache_ptr
    assert vllm_runtime._validated_lm_head_weight_cache(layer) is cached
    assert torch.equal(cached, layer.weight.detach().transpose(0, 1))
    assert vllm_runtime._STRICT_LM_HEAD_CACHE_BUFFER in layer._non_persistent_buffers_set

    x = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4)
    bias = torch.arange(5, dtype=torch.float32)
    output = method.apply(layer, x, bias)
    expected = x.reshape(-1, 4).float().matmul(cached.float())
    expected = expected.reshape(2, 3, 5) + bias
    torch.testing.assert_close(output, expected)
    assert len(det_gemm.prepared_calls) == 1
    called_x, called_weight_t = det_gemm.prepared_calls[0]
    assert torch.equal(called_x, x.reshape(-1, 4))
    assert called_weight_t is cached
    assert det_gemm.linear_calls == []

    replacement = torch.arange(20, 40, dtype=torch.bfloat16).reshape(5, 4)
    layer.weight.data.copy_(replacement)
    vllm_runtime._invalidate_lm_head_weight_cache(layer.weight)
    with pytest.raises(RuntimeError, match="invalid during weight update"):
        method.apply(layer, x)

    assert method.process_weights_after_loading(layer) == "processed"
    refreshed_state = getattr(layer, vllm_runtime._STRICT_LM_HEAD_CACHE_STATE)
    assert refreshed_state is state
    assert refreshed_state.generation == 2
    assert refreshed_state.valid
    assert refreshed_state.weight_t is cached
    assert cached.data_ptr() == cache_ptr
    assert prepare_calls[-1] == (layer.weight, cached)
    assert torch.equal(cached, replacement.transpose(0, 1))


def test_vllm_layerwise_reload_defers_refresh_until_stable_weight_is_restored(
    monkeypatch,
):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    method_cls = _linear_method_class()
    det_gemm = _FakeDetGemm()
    prepare_calls = []

    def prepare(weight, *, out=None):
        prepare_calls.append((weight, out))
        return _prepare_on_cpu(weight, out=out)

    vllm_runtime._patch_strict_lm_head_linear(
        linear_method_cls=method_cls,
        det_gemm=det_gemm,
        prepare_weight=prepare,
    )
    method = method_cls()
    layer = _FakeLmHead(torch.zeros(5, 4, dtype=torch.bfloat16))
    method.process_weights_after_loading(layer)
    state = getattr(layer, vllm_runtime._STRICT_LM_HEAD_CACHE_STATE)
    stable_weight = layer.weight
    stable_cache = state.weight_t
    stable_cache_ptr = stable_cache.data_ptr()

    # vLLM checkpoint-format layerwise reload removes the stable tensors,
    # materializes temporary Parameters, runs post-load processing, then copies
    # back and restores the original objects.
    delattr(layer, "weight")
    delattr(layer, vllm_runtime._STRICT_LM_HEAD_CACHE_BUFFER)
    replacement = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4)
    temporary_weight = torch.nn.Parameter(replacement.clone(), requires_grad=False)
    layer.register_parameter("weight", temporary_weight)

    method.process_weights_after_loading(layer)
    assert not state.valid
    assert state.refresh_pending
    assert len(prepare_calls) == 1

    stable_weight.data.copy_(temporary_weight)
    delattr(layer, "weight")
    layer.register_parameter("weight", stable_weight)
    # vLLM 43914dd74 restores saved buffers with register_buffer's default
    # persistence.  The cache hook must repair that metadata before state_dict
    # serialization, and forward validation must preserve it thereafter.
    layer.register_buffer(vllm_runtime._STRICT_LM_HEAD_CACHE_BUFFER, stable_cache)
    assert vllm_runtime._STRICT_LM_HEAD_CACHE_BUFFER not in layer._non_persistent_buffers_set
    assert vllm_runtime._STRICT_LM_HEAD_CACHE_BUFFER not in layer.state_dict()
    assert vllm_runtime._STRICT_LM_HEAD_CACHE_BUFFER in layer._non_persistent_buffers_set

    x = torch.arange(4, dtype=torch.bfloat16).reshape(1, 4)
    output = method.apply(layer, x)
    expected = x.float().matmul(replacement.transpose(0, 1).float())
    torch.testing.assert_close(output, expected)
    assert state.valid
    assert not state.refresh_pending
    assert state.generation == 2
    assert state.weight_t.data_ptr() == stable_cache_ptr
    assert prepare_calls[-1] == (stable_weight, stable_cache)
    assert torch.equal(stable_cache, replacement.transpose(0, 1))


def test_vllm_rocm_non_lm_head_uses_original_apply(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    method_cls = _linear_method_class()
    det_gemm = _FakeDetGemm()
    vllm_runtime._patch_strict_lm_head_linear(
        linear_method_cls=method_cls,
        det_gemm=det_gemm,
        prepare_weight=_prepare_on_cpu,
    )
    method = method_cls()
    layer = _FakeLmHead(torch.ones(3, 2, dtype=torch.bfloat16), strict=False)
    x = torch.ones(4, 2, dtype=torch.bfloat16)
    bias = torch.ones(2, dtype=torch.bfloat16)

    assert method.process_weights_after_loading(layer) == "processed"
    assert not hasattr(layer, vllm_runtime._STRICT_LM_HEAD_CACHE_STATE)
    assert method.apply(layer, x, bias) is x
    assert method.original_apply_calls == [(layer, x, bias)]
    assert det_gemm.prepared_calls == []
    assert det_gemm.linear_calls == []


def test_vllm_rocm_lm_head_fails_closed_for_missing_or_mutated_cache(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    method_cls = _linear_method_class()
    det_gemm = _FakeDetGemm()
    vllm_runtime._patch_strict_lm_head_linear(
        linear_method_cls=method_cls,
        det_gemm=det_gemm,
        prepare_weight=_prepare_on_cpu,
    )
    method = method_cls()
    layer = _FakeLmHead(torch.ones(3, 2, dtype=torch.bfloat16))
    x = torch.ones(1, 2, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="was not prepared after model loading"):
        method.apply(layer, x)
    assert det_gemm.prepared_calls == []

    method.process_weights_after_loading(layer)
    with torch.no_grad():
        layer.weight.add_(1)
    with pytest.raises(RuntimeError, match="changed without a cache refresh"):
        method.apply(layer, x)
    state = getattr(layer, vllm_runtime._STRICT_LM_HEAD_CACHE_STATE)
    assert not state.valid
    assert det_gemm.prepared_calls == []


def test_vllm_rocm_lm_head_fails_closed_for_mutated_or_replaced_cache(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")
    method_cls = _linear_method_class()
    det_gemm = _FakeDetGemm()
    vllm_runtime._patch_strict_lm_head_linear(
        linear_method_cls=method_cls,
        det_gemm=det_gemm,
        prepare_weight=_prepare_on_cpu,
    )
    method = method_cls()
    layer = _FakeLmHead(torch.ones(3, 2, dtype=torch.bfloat16))
    method.process_weights_after_loading(layer)
    state = getattr(layer, vllm_runtime._STRICT_LM_HEAD_CACHE_STATE)

    with torch.no_grad():
        state.weight_t.add_(1)
    with pytest.raises(RuntimeError, match="cache bytes changed after refresh"):
        method.apply(layer, torch.ones(1, 2, dtype=torch.bfloat16))

    state.valid = True
    replacement = state.weight_t.clone()
    setattr(layer, vllm_runtime._STRICT_LM_HEAD_CACHE_BUFFER, replacement)
    with pytest.raises(RuntimeError, match="cache buffer was replaced"):
        method.apply(layer, torch.ones(1, 2, dtype=torch.bfloat16))


def test_vllm_rocm_lm_head_failed_lazy_refresh_cannot_reuse_old_cache():
    layer = _FakeLmHead(torch.ones(3, 2, dtype=torch.bfloat16))
    state = vllm_runtime._refresh_lm_head_weight_cache(layer, _prepare_on_cpu)
    vllm_runtime._invalidate_lm_head_weight_cache(layer.weight)
    vllm_runtime._mark_lm_head_weight_cache_refreshable(layer.weight)

    def fail_prepare(_weight, *, out=None):
        del out
        raise RuntimeError("injected refresh failure")

    with pytest.raises(RuntimeError, match="injected refresh failure"):
        vllm_runtime._validated_lm_head_weight_cache(layer, fail_prepare)
    assert not state.valid
    assert not state.refresh_pending
    with pytest.raises(RuntimeError, match="invalid during weight update"):
        vllm_runtime._validated_lm_head_weight_cache(layer, _prepare_on_cpu)


def test_vllm_rocm_lm_head_cache_survives_level_two_buffer_restore():
    layer = _FakeLmHead(torch.arange(6, dtype=torch.bfloat16).reshape(3, 2))
    state = vllm_runtime._refresh_lm_head_weight_cache(layer, _prepare_on_cpu)
    saved_buffers = {name: buffer.cpu().clone() for name, buffer in layer.named_buffers()}

    # vLLM's sleep allocator remaps storage, then wake_up restores with
    # ``buffer.data.copy_``; neither operation advances the Tensor's counter.
    state.weight_t.data.zero_()
    for name, buffer in layer.named_buffers():
        buffer.data.copy_(saved_buffers[name].data)

    assert torch.equal(state.weight_t, layer.weight.detach().transpose(0, 1))
    assert vllm_runtime._validated_lm_head_weight_cache(layer) is state.weight_t


@pytest.mark.parametrize("real_vocab,padded_vocab", [(6, 8), (151936, 152064), (151936, 152576)])
def test_qwen_padding_preserves_canonical_vocab(monkeypatch, real_vocab, padded_vocab):
    monkeypatch.setenv("RL_KERNEL_VLLM_REAL_VOCAB_SIZE", str(real_vocab))
    monkeypatch.setenv("RL_KERNEL_VLLM_PADDED_VOCAB_SIZE", str(padded_vocab))

    class Embedding:
        def weight_loader(self, param, loaded_weight):
            pass

    class Head(Embedding):
        def __init__(self, num_embeddings, *, org_num_embeddings, padding_size):
            self.org_vocab_size = org_num_embeddings
            self.num_embeddings_padded = (
                (num_embeddings + padding_size - 1) // padding_size * padding_size
            )

    module_name = "vllm.model_executor.layers.vocab_parallel_embedding"
    module = ModuleType(module_name)
    module.ParallelLMHead = Head
    module.VocabParallelEmbedding = Embedding
    monkeypatch.setitem(sys.modules, module_name, module)
    vllm_runtime._patch_qwen_lm_head_padding()
    assert Head(real_vocab).num_embeddings_padded == padded_vocab


def test_qwen_weight_loader_invalidates_cache_before_data_write(
    monkeypatch,
):
    monkeypatch.setenv("RL_KERNEL_VLLM_REAL_VOCAB_SIZE", "6")
    monkeypatch.setenv("RL_KERNEL_VLLM_PADDED_VOCAB_SIZE", "8")

    class FakeVocabParallelEmbedding:
        def weight_loader(self, param, loaded_weight):
            self.original_loader_calls += 1
            param.data.copy_(loaded_weight)

    class FakeParallelLMHead(FakeVocabParallelEmbedding, torch.nn.Module):
        def __init__(
            self,
            num_embeddings,
            embedding_dim=3,
            *,
            org_num_embeddings=None,
            padding_size=None,
        ):
            super().__init__()
            del padding_size
            self.original_loader_calls = 0
            self.org_vocab_size = int(org_num_embeddings or num_embeddings)
            self.num_embeddings_padded = int(num_embeddings)
            self.shard_indices = SimpleNamespace(
                org_vocab_start_index=0,
                org_vocab_end_index=int(num_embeddings),
            )
            self.weight = torch.nn.Parameter(
                torch.zeros(
                    int(num_embeddings),
                    int(embedding_dim),
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            self.weight.output_dim = 0
            self.weight.packed_dim = None

        def tie_weights(self, embed_tokens):
            self.weight = embed_tokens.weight
            return self

    module_name = "vllm.model_executor.layers.vocab_parallel_embedding"
    fake_module = ModuleType(module_name)
    fake_module.ParallelLMHead = FakeParallelLMHead
    fake_module.VocabParallelEmbedding = FakeVocabParallelEmbedding
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    vllm_runtime._patch_qwen_lm_head_padding()
    layer = FakeParallelLMHead(6, 3)
    state = vllm_runtime._refresh_lm_head_weight_cache(layer, _prepare_on_cpu)
    cache_ptr = state.weight_t.data_ptr()
    loaded = torch.arange(18, dtype=torch.bfloat16).reshape(6, 3)

    layer.weight_loader(layer.weight, loaded)

    assert not state.valid
    assert layer.original_loader_calls == 0
    assert torch.equal(layer.weight[:6], loaded)
    assert torch.count_nonzero(layer.weight[6:]) == 0
    refreshed_weight = vllm_runtime._validated_lm_head_weight_cache(layer, _prepare_on_cpu)
    assert refreshed_weight is state.weight_t
    assert state.generation == 2
    assert state.weight_t.data_ptr() == cache_ptr
    assert torch.equal(state.weight_t, layer.weight.detach().transpose(0, 1))

    embedding = torch.nn.Embedding(8, 3, dtype=torch.bfloat16)
    assert layer.tie_weights(embedding) is layer
    with pytest.raises(RuntimeError, match="does not support tied embeddings"):
        vllm_runtime._refresh_lm_head_weight_cache(layer, _prepare_on_cpu)
    with pytest.raises(RuntimeError, match="does not support tied embeddings"):
        vllm_runtime._validated_lm_head_weight_cache(layer, _prepare_on_cpu)


def test_qwen_original_weight_loader_refreshes_only_after_success(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_VLLM_REAL_VOCAB_SIZE", "6")
    monkeypatch.setenv("RL_KERNEL_VLLM_PADDED_VOCAB_SIZE", "8")

    class FakeVocabParallelEmbedding:
        def weight_loader(self, param, loaded_weight):
            self.original_loader_calls += 1
            if self.raise_during_load:
                raise RuntimeError("injected loader failure")
            param.data.copy_(loaded_weight)

    class FakeParallelLMHead(FakeVocabParallelEmbedding, torch.nn.Module):
        def __init__(self, num_embeddings, embedding_dim=3, **_kwargs):
            super().__init__()
            self.original_loader_calls = 0
            self.raise_during_load = False
            self.org_vocab_size = int(num_embeddings)
            self.num_embeddings_padded = int(num_embeddings)
            self.weight = torch.nn.Parameter(
                torch.zeros(num_embeddings, embedding_dim, dtype=torch.bfloat16),
                requires_grad=False,
            )
            self.weight.output_dim = 0
            # A packed destination selects vLLM's original-loader branch.
            self.weight.packed_dim = 0

    module_name = "vllm.model_executor.layers.vocab_parallel_embedding"
    fake_module = ModuleType(module_name)
    fake_module.ParallelLMHead = FakeParallelLMHead
    fake_module.VocabParallelEmbedding = FakeVocabParallelEmbedding
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    vllm_runtime._patch_qwen_lm_head_padding()
    layer = FakeParallelLMHead(6, 3)
    state = vllm_runtime._refresh_lm_head_weight_cache(layer, _prepare_on_cpu)
    loaded = torch.arange(24, dtype=torch.bfloat16).reshape(8, 3)

    layer.weight_loader(layer.weight, loaded)
    assert layer.original_loader_calls == 1
    assert not state.valid
    assert state.refresh_pending
    refreshed = vllm_runtime._validated_lm_head_weight_cache(layer, _prepare_on_cpu)
    assert torch.equal(refreshed, loaded.transpose(0, 1))

    layer.raise_during_load = True
    with pytest.raises(RuntimeError, match="injected loader failure"):
        layer.weight_loader(layer.weight, loaded + 1)
    assert not state.valid
    assert not state.refresh_pending
    with pytest.raises(RuntimeError, match="invalid during weight update"):
        vllm_runtime._validated_lm_head_weight_cache(layer, _prepare_on_cpu)


def test_qwen_compute_logits_rejects_directly_shared_embedding(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test-rocm")

    class FakeQwenForCausalLM:
        def __init__(self):
            shared = torch.nn.Embedding(8, 3, dtype=torch.bfloat16)
            self.model = SimpleNamespace(embed_tokens=shared)
            # vLLM 43914dd74 assigns this directly for Qwen tied weights; it
            # does not call ParallelLMHead.tie_weights.
            self.lm_head = shared

        def compute_logits(self, hidden_states):
            return hidden_states

    distributed_module = ModuleType("vllm.distributed")
    distributed_module.get_pp_group = lambda: SimpleNamespace(is_last_rank=True)
    distributed_module.get_tp_group = lambda: SimpleNamespace(world_size=1)
    qwen2_module = ModuleType("vllm.model_executor.models.qwen2")
    qwen2_module.Qwen2ForCausalLM = FakeQwenForCausalLM
    qwen3_module = ModuleType("vllm.model_executor.models.qwen3")
    qwen3_module.Qwen3ForCausalLM = FakeQwenForCausalLM
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen2", qwen2_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.qwen3", qwen3_module)
    integration = SimpleNamespace(record_installed_hook=lambda *_args: None)

    vllm_runtime._patch_qwen_compute_logits(integration)
    model = FakeQwenForCausalLM()
    with pytest.raises(RuntimeError, match="does not support tied embeddings"):
        model.compute_logits(torch.ones(1, 3, dtype=torch.bfloat16))
    assert getattr(model.lm_head, vllm_runtime._STRICT_LM_HEAD_TIED)


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="requires ROCm",
)
def test_rocm_layerwise_reload_uses_new_weight_with_raw_byte_identity():
    method_cls = _linear_method_class()
    operator = RocmDetGemmOp()
    vllm_runtime._patch_strict_lm_head_linear(
        linear_method_cls=method_cls,
        det_gemm=operator,
        prepare_weight=prepare_det_gemm_linear_weight,
    )
    generator = torch.Generator(device="cpu").manual_seed(20260906)
    initial = torch.randn(1024, 256, dtype=torch.bfloat16, generator=generator)
    replacement = torch.randn(1024, 256, dtype=torch.bfloat16, generator=generator)
    x = torch.randn(3, 256, dtype=torch.bfloat16, generator=generator).cuda()
    layer = _FakeLmHead(initial.cuda())
    method = method_cls()

    method.process_weights_after_loading(layer)
    state = getattr(layer, vllm_runtime._STRICT_LM_HEAD_CACHE_STATE)
    stable_weight = layer.weight
    stable_cache = state.weight_t
    stable_cache_ptr = stable_cache.data_ptr()
    with torch.inference_mode():
        initial_output = method.apply(layer, x)
        initial_reference = operator.linear(x, stable_weight)
    torch.cuda.synchronize()
    assert torch.equal(
        initial_output.view(torch.uint8),
        initial_reference.view(torch.uint8),
    )

    delattr(layer, "weight")
    delattr(layer, vllm_runtime._STRICT_LM_HEAD_CACHE_BUFFER)
    temporary_weight = torch.nn.Parameter(replacement.cuda(), requires_grad=False)
    layer.register_parameter("weight", temporary_weight)
    method.process_weights_after_loading(layer)
    stable_weight.data.copy_(temporary_weight)
    delattr(layer, "weight")
    layer.register_parameter("weight", stable_weight)
    layer.register_buffer(vllm_runtime._STRICT_LM_HEAD_CACHE_BUFFER, stable_cache)

    with torch.inference_mode():
        updated_output = method.apply(layer, x)
        updated_reference = operator.linear(x, stable_weight)
    torch.cuda.synchronize()
    assert stable_cache.data_ptr() == stable_cache_ptr
    assert torch.equal(
        updated_output.view(torch.uint8),
        updated_reference.view(torch.uint8),
    )
    assert not torch.equal(
        initial_output.view(torch.uint8),
        updated_output.view(torch.uint8),
    )
