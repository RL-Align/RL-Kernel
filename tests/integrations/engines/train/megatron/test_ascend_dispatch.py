# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from types import SimpleNamespace

import pytest
import torch

import rl_engine.integrations.common.operators as common_operators
import rl_engine.integrations.engines.train.megatron.operators as megatron_operators
from rl_engine.integrations.engines.train.megatron.operators import (
    MegatronAttentionOperator,
    MegatronFFNOperator,
)
import rl_engine.integrations.engines.train.megatron.runtime as megatron_runtime
from rl_engine.integrations.engines.train.megatron.runtime import (
    _CanonicalColumnProjection,
    _patch_strict_attention_projections,
)


def _parallel_state(tp_world: int = 1, cp_world: int = 1):
    return SimpleNamespace(
        get_context_parallel_world_size=lambda: cp_world,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: tp_world,
        get_tensor_model_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: None,
    )


def test_projection_op_selects_ascend_when_npu_is_available(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(common_operators, "_npu_available", lambda: True)
    monkeypatch.setattr(
        "rl_engine.backends.ascend.gemm.det_gemm.DetGemmAscendOp",
        lambda: sentinel,
        raising=False,
    )

    import rl_engine.backends.ascend.gemm.det_gemm as det_gemm

    monkeypatch.setattr(det_gemm, "DetGemmAscendOp", lambda: sentinel)
    assert common_operators._strict_attention_projection_op() is sentinel


def test_npu_projection_calls_linear_without_cuda_column_autograd(monkeypatch):
    calls = []

    class Gemm:
        def linear(self, activations, weight):
            calls.append((tuple(activations.shape), tuple(weight.shape)))
            return activations @ weight.t()

    def rejected(*args, **kwargs):
        del args, kwargs
        raise AssertionError("NPU projection must not use the CUDA column autograd")

    monkeypatch.setattr(megatron_runtime, "_tensor_on_npu", lambda tensor: True)
    monkeypatch.setattr(_CanonicalColumnProjection, "apply", staticmethod(rejected))

    class ColumnLinear:
        def __init__(self):
            self.allreduce_dgrad = True

        def _forward_impl(self, input, weight, *args, **kwargs):
            del args, kwargs
            return input

    class RowLinear:
        def _forward_impl(self, input, weight, *args, **kwargs):
            del args, kwargs
            return input

    class SelfAttention:
        def __init__(self):
            self.linear_qkv = ColumnLinear()
            self.linear_proj = RowLinear()

    _patch_strict_attention_projections(
        self_attention_cls=SelfAttention,
        column_linear_cls=ColumnLinear,
        row_linear_cls=RowLinear,
        det_gemm=Gemm(),
    )
    attention = SelfAttention()
    value = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    output = attention.linear_qkv._forward_impl(value, weight, bias=None)

    assert calls == [((1, 2), (3, 2))]
    assert torch.equal(output, torch.tensor([[1.0, 2.0, 3.0]]))


def test_npu_attention_calls_ascend_kernel(monkeypatch):
    seen = {}

    class Attention:
        def forward_with_lse(self, q, k, v, *, causal, scale):
            seen["shape"] = tuple(q.shape)
            seen["causal"] = causal
            seen["scale"] = scale
            return q, None

    monkeypatch.setattr(megatron_operators, "_is_npu_tensor", lambda tensor: True)
    monkeypatch.setattr(
        megatron_operators,
        "_megatron_parallel_state",
        lambda: _parallel_state(),
    )
    operator = MegatronAttentionOperator(handle=SimpleNamespace())
    operator._ascend_attention = Attention()
    query = torch.zeros(3, 1, 2, 4)
    module = SimpleNamespace(softmax_scale=0.25)

    output = operator(module, query, query, query, attention_mask=None)

    assert seen == {"shape": (1, 2, 3, 4), "causal": True, "scale": 0.25}
    assert output.shape == (3, 1, 8)


def test_npu_attention_rejects_tensor_parallel(monkeypatch):
    monkeypatch.setattr(megatron_operators, "_is_npu_tensor", lambda tensor: True)
    monkeypatch.setattr(
        megatron_operators,
        "_megatron_parallel_state",
        lambda: _parallel_state(tp_world=2),
    )
    operator = MegatronAttentionOperator(handle=SimpleNamespace())
    operator._ascend_attention = SimpleNamespace()
    query = torch.zeros(2, 1, 2, 4)

    with pytest.raises(RuntimeError, match="TP=1"):
        operator(SimpleNamespace(), query, query, query, attention_mask=None)


def test_npu_ffn_uses_separate_ascend_linears(monkeypatch):
    calls = []

    class Gemm:
        def linear(self, activations, weight):
            calls.append(tuple(weight.shape))
            return activations @ weight.t()

    class SwiGLU:
        def __call__(self, gate, up):
            return gate + up

    monkeypatch.setattr(megatron_operators, "_is_npu_tensor", lambda tensor: True)
    monkeypatch.setattr(
        megatron_operators,
        "_megatron_parallel_state",
        lambda: _parallel_state(),
    )
    operator = MegatronFFNOperator(handle=SimpleNamespace(provenance={}))
    operator._ascend_gemm = Gemm()
    operator._ascend_swiglu = SwiGLU()
    hidden = torch.ones(2, 1, 4)
    fused = torch.ones(10, 4)
    down = torch.ones(4, 5)
    module = SimpleNamespace(
        config=SimpleNamespace(
            add_bias_linear=False,
            gated_linear_unit=True,
            sequence_parallel=False,
        ),
        linear_fc1=SimpleNamespace(weight=fused),
        linear_fc2=SimpleNamespace(weight=down),
    )

    output, bias = operator(module, hidden)

    assert calls == [(5, 4), (5, 4), (4, 5)]
    assert bias is None
    assert output.shape == (2, 1, 4)
    assert operator.provenance["execution"]["actual_backend"] == (
        "rlkernel.ascend.det_gemm_swiglu"
    )


def _install_fake_megatron(monkeypatch, tp_world: int = 1):
    import sys
    from types import ModuleType

    def package(name):
        module = ModuleType(name)
        module.__package__ = name
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        parent_name, _, child = name.rpartition(".")
        if parent_name and parent_name in sys.modules:
            setattr(sys.modules[parent_name], child, module)
        return module

    package("megatron")
    package("megatron.core")
    package("megatron.core.models")
    package("megatron.core.models.gpt")
    package("megatron.core.tensor_parallel")

    class ParallelState:
        @staticmethod
        def model_parallel_is_initialized():
            return True

        @staticmethod
        def get_tensor_model_parallel_world_size():
            return tp_world

    parallel_state = ModuleType("megatron.core.parallel_state")
    parallel_state.model_parallel_is_initialized = ParallelState.model_parallel_is_initialized
    parallel_state.get_tensor_model_parallel_world_size = (
        ParallelState.get_tensor_model_parallel_world_size
    )
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    sys.modules["megatron.core"].parallel_state = parallel_state

    class ColumnParallelLinear:
        def __init__(self):
            self.weight = None
            self.bias = None
            self.sequence_parallel = False

        def forward(self, input_, weight=None, runtime_gather_output=None):
            del weight, runtime_gather_output
            return ("original", tuple(input_.shape))

    layers = ModuleType("megatron.core.tensor_parallel.layers")
    layers.ColumnParallelLinear = ColumnParallelLinear
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel.layers", layers)
    sys.modules["megatron.core.tensor_parallel"].layers = layers

    class GPTModel:
        def __init__(self):
            self.output_layer = ColumnParallelLinear()

    gpt_model = ModuleType("megatron.core.models.gpt.gpt_model")
    gpt_model.GPTModel = GPTModel
    monkeypatch.setitem(sys.modules, "megatron.core.models.gpt.gpt_model", gpt_model)
    sys.modules["megatron.core.models.gpt"].gpt_model = gpt_model

    def vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):
        del target, label_smoothing
        return torch.zeros(vocab_parallel_logits.shape[:-1])

    cross_entropy = ModuleType("megatron.core.tensor_parallel.cross_entropy")
    cross_entropy.vocab_parallel_cross_entropy = vocab_parallel_cross_entropy
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel.cross_entropy", cross_entropy)
    tensor_parallel = sys.modules["megatron.core.tensor_parallel"]
    tensor_parallel.cross_entropy = cross_entropy
    tensor_parallel.vocab_parallel_cross_entropy = vocab_parallel_cross_entropy
    return GPTModel, ColumnParallelLinear, tensor_parallel


def test_npu_logp_uses_existing_ascend_ops(monkeypatch):
    gemm_calls = []
    logp_calls = []

    class Gemm:
        def linear(self, activations, weight):
            gemm_calls.append((tuple(activations.shape), tuple(weight.shape)))
            return torch.ones(activations.shape[0], weight.shape[0])

    class Logp:
        def __call__(self, logits, target):
            logp_calls.append((tuple(logits.shape), tuple(target.shape)))
            return torch.full(logits.shape[:-1], -1.5)

    import rl_engine.backends.ascend.gemm.det_gemm as det_gemm
    import rl_engine.backends.ascend.logprob.logp as logp_mod

    monkeypatch.setattr(det_gemm, "DetGemmAscendOp", lambda: Gemm())
    monkeypatch.setattr(logp_mod, "FusedLogpAscendOp", lambda: Logp())
    monkeypatch.setattr(megatron_runtime, "_is_npu_tensor", lambda tensor: True)
    GPTModel, ColumnParallelLinear, tensor_parallel = _install_fake_megatron(monkeypatch)

    megatron_runtime._patch_npu_logp()
    model = GPTModel()
    hidden = torch.zeros(2, 3, 4)
    weight = torch.zeros(6, 4)
    logits, bias = model.output_layer.forward(hidden, weight)
    other = ColumnParallelLinear()
    original = other.forward(hidden, weight)
    target = torch.zeros(2, 3, dtype=torch.long)
    loss = tensor_parallel.vocab_parallel_cross_entropy(logits, target)

    assert getattr(model.output_layer, "_rlk_npu_output_layer", False) is True
    assert gemm_calls == [((6, 4), (6, 4))]
    assert bias is None
    assert logits.shape == (2, 3, 6)
    assert original == ("original", (2, 3, 4))
    assert logp_calls == [((2, 3, 6), (2, 3))]
    assert torch.equal(loss, torch.full((2, 3), 1.5))


def test_npu_logp_keeps_megatron_cross_entropy_off_npu(monkeypatch):
    import rl_engine.backends.ascend.gemm.det_gemm as det_gemm
    import rl_engine.backends.ascend.logprob.logp as logp_mod

    monkeypatch.setattr(det_gemm, "DetGemmAscendOp", lambda: SimpleNamespace())
    monkeypatch.setattr(logp_mod, "FusedLogpAscendOp", lambda: SimpleNamespace())
    monkeypatch.setattr(megatron_runtime, "_is_npu_tensor", lambda tensor: False)
    _gpt, _column, tensor_parallel = _install_fake_megatron(monkeypatch)

    megatron_runtime._patch_npu_logp()
    logits = torch.zeros(2, 3, 6)
    target = torch.zeros(2, 3, dtype=torch.long)
    loss = tensor_parallel.vocab_parallel_cross_entropy(logits, target)

    assert torch.equal(loss, torch.zeros(2, 3))


def test_npu_logp_rejects_tensor_parallel(monkeypatch):
    import rl_engine.backends.ascend.gemm.det_gemm as det_gemm
    import rl_engine.backends.ascend.logprob.logp as logp_mod

    monkeypatch.setattr(det_gemm, "DetGemmAscendOp", lambda: SimpleNamespace())
    monkeypatch.setattr(logp_mod, "FusedLogpAscendOp", lambda: SimpleNamespace())
    _install_fake_megatron(monkeypatch, tp_world=2)

    with pytest.raises(RuntimeError, match="TP=1"):
        megatron_runtime._patch_npu_logp()
