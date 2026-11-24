# SPDX-License-Identifier: Apache-2.0
"""DSV4 cache lifecycle on CPU; exact original-kernel comparisons on ROCm."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.shared.triton.embedding import lm_head
from rl_engine.backends.shared.triton.logprob.logp import TritonLogpOp
from rl_engine.integrations.common.mhc_worker import P1Worker
from rl_engine.integrations.engines.rollout.vllm import outputs, runtime


class _Wrapper(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.lm_head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.lm_head.weight = torch.nn.Parameter(weight)
        self.config = SimpleNamespace(vocab_size=weight.shape[0])
        self.logits_processor = SimpleNamespace(scale=1.0, soft_cap=None)
        self.fail_load = False
        self.load_calls = 0

    def load_weights(self, weights):
        self.load_calls += 1
        for name, value in weights:
            assert name == "lm_head.weight"
            # Match vLLM's .data loader: the Parameter version does not change.
            self.lm_head.weight.data.copy_(value)
        if self.fail_load:
            raise RuntimeError("injected loader failure")
        return {"lm_head.weight"}


def _worker():
    return P1Worker("verl", "rollout", backend="reference")


@pytest.fixture
def cpu_projection(monkeypatch):
    """Real tensors/worker/cache with a CPU GEMM stand-in, not GPU acceptance."""
    prepares = []
    gemms = []

    def prepare(weight, *, out=None):
        prepares.append(out)
        with torch.inference_mode(False), torch.no_grad():
            if out is None:
                out = torch.empty(weight.shape[1], weight.shape[0], dtype=weight.dtype)
            out.copy_(weight.t())
        return out

    def gemm(a, b, *, output_dtype=None):
        gemms.append(b)
        result = a.float() @ b.float()
        return result.to(output_dtype or a.dtype)

    monkeypatch.setattr(outputs, "prepare_det_gemm_linear_weight", prepare)
    monkeypatch.setattr(lm_head, "_triton_gemm", gemm)
    return prepares, gemms


def _bits(tensor):
    return tensor.contiguous().view(torch.uint8)


def test_repeated_calls_reuse_transpose_and_preserve_original_fp32_results(cpu_projection):
    prepares, gemms = cpu_projection
    wrapper = _Wrapper(torch.arange(40, dtype=torch.bfloat16).reshape(10, 4) / 16)
    outputs.bind_output_head(wrapper, _worker())
    cached = getattr(wrapper.lm_head, runtime._STRICT_LM_HEAD_CACHE_BUFFER)
    hidden = torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4) / 16

    with torch.inference_mode():
        actual = [wrapper.compute_logits(hidden), wrapper.compute_logits(hidden)]
        assert len(prepares) == 1
        assert all(rhs is cached for rhs in gemms)
        expected = lm_head.TritonLMHeadOp().forward_fp32(hidden, wrapper.lm_head.weight)
    assert all(torch.equal(_bits(value), _bits(expected)) for value in actual)
    assert actual[0].dtype == torch.float32
    assert runtime._STRICT_LM_HEAD_CACHE_BUFFER not in wrapper.lm_head.state_dict()


@pytest.mark.parametrize("inference_weight", [False, True])
def test_data_reload_refreshes_same_buffer_and_new_policy_output(cpu_projection, inference_weight):
    prepares, _ = cpu_projection
    with torch.inference_mode(inference_weight):
        wrapper = _Wrapper(torch.zeros(10, 4, dtype=torch.bfloat16))
    worker = _worker()
    outputs.bind_output_head(wrapper, worker)
    cached = getattr(wrapper.lm_head, runtime._STRICT_LM_HEAD_CACHE_BUFFER)
    pointer = cached.data_ptr()
    replacement = torch.arange(40, dtype=torch.bfloat16).reshape(10, 4) / 16
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)
    with torch.inference_mode():
        before = wrapper.compute_logits(hidden)
        with worker.updating(1, lambda: None):
            loaded = wrapper.load_weights(iter([("lm_head.weight", replacement)]))
        after = wrapper.compute_logits(hidden)
        expected = lm_head.TritonLMHeadOp().forward_fp32(hidden, wrapper.lm_head.weight)
    assert loaded == {"lm_head.weight"} and wrapper.load_calls == 1
    assert worker.version == 1 and cached.data_ptr() == pointer
    assert len(prepares) == 2 and prepares[1] is cached
    assert torch.equal(_bits(after), _bits(expected))
    assert not torch.equal(_bits(before), _bits(after))


def test_failed_load_cannot_serve_old_cache_and_complete_reload_recovers(cpu_projection):
    wrapper = _Wrapper(torch.zeros(10, 4, dtype=torch.bfloat16))
    outputs.bind_output_head(wrapper, _worker())
    hidden = torch.ones(1, 4, dtype=torch.bfloat16)
    wrapper.fail_load = True
    with pytest.raises(RuntimeError, match="injected loader failure"):
        wrapper.load_weights([("lm_head.weight", torch.ones_like(wrapper.lm_head.weight))])
    with torch.no_grad(), pytest.raises(RuntimeError, match="invalid during weight update"):
        wrapper.compute_logits(hidden)
    wrapper.fail_load = False
    wrapper.load_weights([("lm_head.weight", torch.zeros_like(wrapper.lm_head.weight))])
    with torch.no_grad():
        assert torch.count_nonzero(wrapper.compute_logits(hidden)) == 0


@pytest.mark.parametrize("target", ["parameter", "cache"])
def test_unannounced_parameter_or_cache_mutation_is_rejected(cpu_projection, target):
    wrapper = _Wrapper(torch.ones(10, 4, dtype=torch.bfloat16))
    outputs.bind_output_head(wrapper, _worker())
    with torch.no_grad():
        if target == "parameter":
            wrapper.lm_head.weight.add_(1)
            message = "changed without a cache refresh"
        else:
            getattr(wrapper.lm_head, runtime._STRICT_LM_HEAD_CACHE_BUFFER).add_(1)
            message = "cache bytes changed"
        with pytest.raises(RuntimeError, match=message):
            wrapper.compute_logits(torch.ones(1, 4, dtype=torch.bfloat16))


def test_grad_enabled_call_preserves_original_backward(cpu_projection):
    prepares, _ = cpu_projection
    wrapper = _Wrapper(torch.ones(10, 4, dtype=torch.bfloat16))
    outputs.bind_output_head(wrapper, _worker())
    hidden = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)
    wrapper.compute_logits(hidden).sum().backward()
    assert len(prepares) == 1
    assert torch.equal(hidden.grad, torch.full_like(hidden, 10))
    assert torch.equal(wrapper.lm_head.weight.grad, torch.full_like(wrapper.lm_head.weight, 2))


def test_prepared_api_rejects_gradient_requests_and_invalid_layout(cpu_projection):
    op = lm_head.TritonLMHeadOp()
    hidden = torch.ones(2, 4, dtype=torch.bfloat16, requires_grad=True)
    prepared = torch.ones(4, 10, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="inference-only"):
        op.forward_fp32_prepared(hidden, prepared)
    with torch.no_grad(), pytest.raises(ValueError, match="contiguous"):
        op.forward_fp32_prepared(hidden, torch.ones(10, 4, dtype=torch.bfloat16).t())


@pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a real ROCm GPU"
)
@pytest.mark.parametrize("vocab,hidden_size", [(257, 128), (129280, 4096)])
def test_rocm_original_logits_logprobs_and_reloaded_policy_are_bitwise_equal(vocab, hidden_size):
    generator = torch.Generator().manual_seed(1234)
    weight = torch.randn(vocab, hidden_size, dtype=torch.bfloat16, generator=generator).cuda()
    wrapper = _Wrapper(weight)
    worker = _worker()
    with torch.inference_mode():
        outputs.bind_output_head(wrapper, worker)
        cached = getattr(wrapper.lm_head, runtime._STRICT_LM_HEAD_CACHE_BUFFER)
        pointer = cached.data_ptr()
        rows = torch.randn(17, hidden_size, dtype=torch.bfloat16, generator=generator).cuda()
        actual = wrapper.compute_logits(rows)
        expected = lm_head.TritonLMHeadOp().forward_fp32(rows, wrapper.lm_head.weight)
        assert torch.equal(_bits(actual), _bits(expected))
        reordered = wrapper.compute_logits(rows.flip(0)).flip(0)
        chunked = torch.cat([wrapper.compute_logits(part) for part in rows.split(4)])
        assert torch.equal(_bits(actual), _bits(reordered))
        assert torch.equal(_bits(actual), _bits(chunked))
        targets = torch.arange(rows.shape[0], device=rows.device, dtype=torch.int64) % vocab
        logp = TritonLogpOp()
        assert torch.equal(_bits(logp(actual, targets)), _bits(logp(expected, targets)))
        replacement = weight + 0.125
        with worker.updating(1, torch.cuda.synchronize):
            wrapper.load_weights([("lm_head.weight", replacement)])
        refreshed = wrapper.compute_logits(rows)
        reference = lm_head.TritonLMHeadOp().forward_fp32(rows, wrapper.lm_head.weight)
        assert cached.data_ptr() == pointer
        assert torch.equal(_bits(refreshed), _bits(reference))
        assert torch.equal(_bits(logp(refreshed, targets)), _bits(logp(reference, targets)))
        assert not torch.equal(_bits(refreshed), _bits(actual))
