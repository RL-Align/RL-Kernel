# SPDX-License-Identifier: Apache-2.0
"""Run unchanged on CUDA and ROCm; CPU environments skip hardware validation."""

import pytest
import torch

pytest.importorskip("triton")
from rl_engine.backends.shared.triton import autotuning  # noqa: E402
from rl_engine.platforms.autotuner import PersistentAutotuner  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm required")


@pytest.fixture
def tuning(monkeypatch, tmp_path):
    monkeypatch.setenv("RL_KERNEL_MODE", "auto")
    monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "1")
    monkeypatch.setattr(autotuning, "_TUNER", PersistentAutotuner(tmp_path))
    return tmp_path


def equal_bits(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _case(name, dtype):
    def tensor(shape):
        return torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)

    if name in ("silu", "swiglu"):
        from rl_engine.backends.shared.triton.activation.swiglu import TritonSiLUOp, TritonSwiGLUOp

        x = tensor((3, 1030))[:, ::2]
        return (
            (TritonSiLUOp(), (x,)) if name == "silu" else (TritonSwiGLUOp(), (x, tensor(x.shape)))
        )
    if name == "rope":
        from rl_engine.backends.shared.triton.rope.rope import TritonRoPEOp

        return TritonRoPEOp(), (tensor((2, 5, 128)), torch.arange(5, device="cuda"))
    if name == "embedding":
        from rl_engine.backends.shared.triton.embedding.embedding import TritonEmbeddingOp

        return TritonEmbeddingOp(), (torch.tensor([1, 3, 1, 0], device="cuda"), tensor((7, 65)))
    if name == "rmsnorm":
        from rl_engine.backends.shared.triton.norm.rmsnorm import RMSNormTritonOp

        return RMSNormTritonOp(), (tensor((3, 129)), tensor((129,)))
    if name == "ratio_kl":
        from rl_engine.backends.shared.triton.loss.ratio_kl import TritonRatioKLOp

        return TritonRatioKLOp(), (
            tensor((2, 3, 133)),
            tensor((2, 3, 133)).detach(),
            torch.zeros((2, 3), device="cuda", dtype=torch.long),
            torch.tensor([[1, 0, 1], [1, 1, 0]], device="cuda"),
            torch.randn((2, 3), device="cuda"),
        )
    if name == "logp":
        from rl_engine.backends.shared.triton.tuned import TunedLogpOp

        return TunedLogpOp(), (tensor((3, 1031)), torch.tensor([0, -100, 1030], device="cuda"))
    if name == "gemm":
        from rl_engine.backends.shared.triton.tuned import TunedGemmOp

        op = TunedGemmOp()
        return (op if dtype == torch.bfloat16 else op.forward_fp32), (
            tensor((5, 65)),
            tensor((65, 7)),
        )
    if name == "attention":
        from rl_engine.backends.shared.triton.tuned import TunedAttentionOp

        op = TunedAttentionOp(require_bitwise_libm=False)
        return op, (tensor((1, 2, 3, 128)), tensor((1, 1, 5, 128)), tensor((1, 1, 5, 128)))
    raise AssertionError(name)


@pytest.mark.parametrize(
    "name",
    ["silu", "swiglu", "rope", "embedding", "rmsnorm", "ratio_kl", "logp", "gemm", "attention"],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_forward_backward_and_hot_cache(tuning, monkeypatch, name, dtype):
    if (name == "gemm" and dtype == torch.float16) or (
        name == "attention" and dtype == torch.float32
    ):
        pytest.skip("Existing operator does not support this dtype")
    op, inputs = _case(name, dtype)
    variables = tuple(x for x in inputs if x.requires_grad)
    if name == "gemm" and dtype == torch.float32:
        variables = ()  # Existing FP32-input GEMM supports forward only.
    monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "0")
    baseline = op(*inputs)
    baseline = baseline if isinstance(baseline, tuple) else (baseline,)
    grad_outputs = tuple(torch.randn_like(x) for x in baseline)
    expected_grads = torch.autograd.grad(baseline, variables, grad_outputs) if variables else ()
    monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "1")
    for repetition in range(2):
        if repetition:

            def unexpected_selection(*args, **kwargs):
                pytest.fail("Hot launches must bypass selection and disk access")

            monkeypatch.setattr(autotuning._TUNER, "select", unexpected_selection)
        actual = op(*inputs)
        actual = actual if isinstance(actual, tuple) else (actual,)
        # Backward must retain the policy chosen by forward.
        monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "0")
        grads = torch.autograd.grad(actual, variables, grad_outputs) if variables else ()
        monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "1")
        for left, right in zip(baseline + expected_grads, actual + grads, strict=True):
            equal_bits(left, right)
    assert autotuning._TUNER._memory


def test_grpo_normalization(tuning, monkeypatch):
    from rl_engine.backends.shared.triton.loss.grpo_loss import TritonGRPOLossOp

    op = TritonGRPOLossOp()
    rewards = torch.randn(12, device="cuda")
    monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "0")
    baseline = op.group_advantages(rewards, samples_per_prompt=3)
    monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "1")
    equal_bits(baseline, op.group_advantages(rewards, samples_per_prompt=3))


def test_strict_core_never_tunes(tuning):
    from rl_engine.backends.shared.triton.logprob.batch_invariant_logp import (
        TritonBatchInvariantLogpOp,
    )

    x = torch.randn((3, 17), device="cuda", requires_grad=True)
    out = TritonBatchInvariantLogpOp()(x, torch.zeros(3, device="cuda", dtype=torch.long))
    out.sum().backward()
    assert not autotuning._TUNER._memory


def test_graph_capture_cold_and_warm(tuning):
    from rl_engine.backends.shared.triton.activation.swiglu import TritonSiLUOp

    op = TritonSiLUOp()
    x = torch.randn(123, device="cuda")
    # Compile and initialize CUDA without populating the tuning cache.
    with autotuning.tuning_scope(autotuning.TuneState(False)):
        op(x)
    torch.cuda.synchronize()
    for warm in (False, True):
        if warm:
            op(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = op(x)
        graph.replay()
        equal_bits(y, op(x))


def test_special_values_and_empty(tuning, monkeypatch):
    from rl_engine.backends.shared.triton.activation.swiglu import TritonSiLUOp

    op = TritonSiLUOp()
    x = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-38], device="cuda")
    monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "0")
    expected = op(x)
    monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "1")
    equal_bits(expected, op(x))
    assert op(torch.empty((0, 13), device="cuda")).shape == (0, 13)


@pytest.mark.parametrize("mode", ["strict", "audit", "off"])
def test_modes_disable_even_explicit_backends(tuning, monkeypatch, mode):
    from rl_engine.backends.shared.triton.tuned import TunedLogpOp

    monkeypatch.setenv("RL_KERNEL_MODE", mode)
    op = TunedLogpOp()
    x = torch.randn((2, 33), device="cuda", requires_grad=True)
    op(x, torch.zeros(2, device="cuda", dtype=torch.long)).sum().backward()
    assert not autotuning._TUNER._memory
    assert not op.autotune_provenance["enabled"]


def test_fixed_gemm_attention_and_indirect_lm_head(tuning):
    from rl_engine.backends.shared.triton.attention.deterministic_attn import (
        TritonDeterministicAttentionOp,
    )
    from rl_engine.backends.shared.triton.embedding.lm_head import TritonLMHeadOp
    from rl_engine.backends.shared.triton.gemm.det_gemm import TritonDetGemmOp

    a = torch.randn((2, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn((32, 8), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    TritonDetGemmOp()(a, b).sum().backward()
    TritonLMHeadOp()(a, b.t().contiguous()).sum().backward()
    q = torch.randn((1, 1, 2, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    TritonDeterministicAttentionOp(require_bitwise_libm=False)(q, q, q).sum().backward()
    assert not autotuning._TUNER._memory


@pytest.mark.parametrize("name", ["rmsnorm", "ratio_kl", "logp"])
def test_cached_configuration_on_new_values(tuning, monkeypatch, name):
    op, inputs = _case(name, torch.float32)
    op(*inputs)
    for _ in range(3):
        inputs = tuple(
            torch.randn_like(x).requires_grad_(x.requires_grad) if x.is_floating_point() else x
            for x in inputs
        )
        monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "0")
        expected = op(*inputs)
        monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "1")
        actual = op(*inputs)
        if not isinstance(expected, tuple):
            expected, actual = (expected,), (actual,)
        for left, right in zip(expected, actual, strict=True):
            equal_bits(left, right)


def test_fp32_input_gemm_backward_retains_existing_error(tuning, monkeypatch):
    from rl_engine.backends.shared.triton.tuned import TunedGemmOp

    a = torch.randn((2, 8), device="cuda", requires_grad=True)
    b = torch.randn((8, 3), device="cuda", requires_grad=True)
    for enabled in ("0", "1"):
        monkeypatch.setenv("RL_KERNEL_AUTOTUNE", enabled)
        with pytest.raises(TypeError, match="BF16 inputs"):
            TunedGemmOp().forward_fp32(a, b).sum().backward()


def test_graph_profiler_resets_read_write_buffers(tuning):
    from rl_engine.backends.shared.triton.attention.deterministic_attn import (
        _softmax_backward_kernel,
    )
    from rl_engine.platforms.autotuner import TuneConfig

    ds = torch.arange(256, device="cuda", dtype=torch.float32).reshape(1, 256)
    original = ds.clone()
    probabilities = torch.full_like(ds, 1 / 256)
    reference = ds.clone()
    raw = _softmax_backward_kernel.kernel[(1,)]
    raw(reference, probabilities, 256, LANES=256, num_warps=4)
    observed = []

    def run(config, buffers):
        observed[:] = buffers
        return raw(*buffers, LANES=256, num_warps=4, **config.to_dict())

    timings = autotuning._profile_candidates(
        (TuneConfig(),), (ds, probabilities, 256), (0,), (0,), run
    )
    assert timings[TuneConfig()] > 0
    equal_bits(ds, original)
    # After many graph replays, scratch must still represent exactly one update.
    equal_bits(observed[0], reference)


def test_profiling_preserves_callers_stream(tuning):
    from rl_engine.backends.shared.triton.activation.swiglu import TritonSiLUOp

    op = TritonSiLUOp()
    x = torch.randn(577, device="cuda")
    caller = torch.cuda.Stream()
    caller.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(caller):
        actual = op(x)
        assert torch.cuda.current_stream() == caller
        with autotuning.tuning_scope(autotuning.TuneState(False)):
            expected = op(x)
        equal_bits(actual, expected)
    torch.cuda.current_stream().wait_stream(caller)


@pytest.mark.parametrize(
    "name",
    ["silu", "swiglu", "rope", "embedding", "rmsnorm", "ratio_kl", "gemm", "attention", "logp"],
)
def test_hot_new_values_and_gradients(tuning, monkeypatch, name):
    from benchmarks.tuning.autotuning_cases import make_case

    for seed in (0, 1, 2026):
        torch.manual_seed(seed)
        case = make_case(name, torch.bfloat16, width=128)
        monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "0")
        baseline = case.forward()
        grad = tuple(torch.randn_like(x) for x in baseline)
        expected = torch.autograd.grad(baseline, case.variables, grad)
        monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "1")
        if seed != 0:
            monkeypatch.setattr(
                autotuning._TUNER, "select", lambda *a, **kw: pytest.fail("Hot selection")
            )
        actual = case.forward()
        monkeypatch.setenv("RL_KERNEL_MODE", "strict")
        result = torch.autograd.grad(actual, case.variables, grad)
        monkeypatch.setenv("RL_KERNEL_MODE", "auto")
        for a, b in zip(baseline + expected, actual + result, strict=True):
            equal_bits(a, b)


@pytest.mark.parametrize("name", ["silu", "rmsnorm", "gemm", "attention", "logp"])
@pytest.mark.parametrize("warm", [False, True])
def test_forward_backward_graph_no_disk_or_profile(tuning, monkeypatch, name, warm):
    from benchmarks.tuning.autotuning_cases import make_case

    caller = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(caller)
    with torch.cuda.stream(stream):
        torch.manual_seed(2026)
        case = make_case(name, torch.bfloat16, width=128)

        def run():
            values = case.forward()
            gradients = torch.autograd.grad(
                values, case.variables, tuple(torch.ones_like(x) for x in values)
            )
            return values + gradients

        monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "0")
        expected = run()  # Compile both fixed forward and backward before capture.
        monkeypatch.setenv("RL_KERNEL_AUTOTUNE", "1")
        if warm:
            run()
        monkeypatch.setattr(
            autotuning._TUNER, "_disk_lock", lambda *a, **kw: pytest.fail("Capture disk access")
        )
        monkeypatch.setattr(
            autotuning, "_profile_candidates", lambda *a, **kw: pytest.fail("Capture profiling")
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = run()
        for _ in range(3):
            graph.replay()
            for a, b in zip(actual, expected, strict=True):
                equal_bits(a, b)
    caller.wait_stream(stream)


@pytest.mark.parametrize("name", ["gemm", "attention"])
def test_unsupported_core_dtype_preserves_fixed_error(tuning, name):
    from rl_engine.backends.shared.triton.tuned import TunedAttentionOp, TunedGemmOp

    if name == "gemm":
        from rl_engine.backends.shared.triton.gemm.det_gemm import TritonDetGemmOp

        fixed, tuned = TritonDetGemmOp(), TunedGemmOp()
        inputs = (
            torch.ones((2, 8), device="cuda", dtype=torch.float16),
            torch.ones((8, 3), device="cuda", dtype=torch.float16),
        )
        error_type = AssertionError
    else:
        from rl_engine.backends.shared.triton.attention.deterministic_attn import (
            TritonDeterministicAttentionOp,
        )

        fixed = TritonDeterministicAttentionOp(require_bitwise_libm=False)
        tuned = TunedAttentionOp()
        inputs = (torch.ones((1, 1, 2, 128), device="cuda", dtype=torch.float32),) * 3
        error_type = ValueError
    with pytest.raises(error_type) as baseline:
        fixed(*inputs)
    with pytest.raises(error_type) as actual:
        tuned(*inputs)
    assert str(actual.value) == str(baseline.value)
    assert not autotuning._TUNER._memory
