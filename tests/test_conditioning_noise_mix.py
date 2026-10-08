# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 noise mixing: independent CPU reference, exact CUDA parity and invariance."""

import argparse
import importlib

import pytest
import torch

import envs
from rl_engine.kernels.gtest import run_operator_suite
from rl_engine.kernels.gtest.operator_specs import make_candidate, make_operator_case
from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.cuda.conditioning_noise_mix import ConditioningNoiseMixCudaOp
from rl_engine.kernels.ops.pytorch.conditioning_noise_mix import NativeConditioningNoiseMixOp
from rl_engine.kernels.registry import KernelRegistry, OpBackend

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_HAS_CUDA = (
    torch.cuda.is_available()
    and torch.version.hip is None
    and _EXT_AVAILABLE
    and hasattr(_C, "conditioning_noise_mix_forward")
)
requires_cuda_kernel = pytest.mark.skipif(
    not _HAS_CUDA, reason="native conditioning_noise_mix CUDA build required"
)


def _make_tensors(shape, dtype=torch.float32):
    generator = torch.Generator().manual_seed(420)
    return [torch.randn(shape, generator=generator).to(dtype) for _ in range(3)]


def _assert_bitwise_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("time", [0.0, 0.37, 1.0])
def test_reference_endpoints_and_input_gradients(dtype, time):
    sample, noise, grad = _make_tensors((2, 3, 17), dtype)
    sample.requires_grad_()
    noise.requires_grad_()
    out = NativeConditioningNoiseMixOp()(sample, time, noise)
    dx, dn = torch.autograd.grad(out, (sample, noise), grad)
    t = torch.tensor(time, dtype=dtype)
    _assert_bitwise_equal(dx, grad * t)
    _assert_bitwise_equal(dn, grad * (1 - t))
    if time == 0.0:
        _assert_bitwise_equal(out, noise)
    elif time == 1.0:
        _assert_bitwise_equal(out, sample)


def test_reference_fp32_against_float64_and_finite_differences():
    sample, noise, grad = _make_tensors((2, 3, 5))
    sample.requires_grad_()
    noise.requires_grad_()
    time = torch.tensor([0.13, 0.79])
    op = NativeConditioningNoiseMixOp()
    out = op(sample, time, noise)
    t64 = time.double()[:, None, None]
    expected = t64 * sample.double() + (1 - t64) * noise.double()
    torch.testing.assert_close(out.double(), expected, rtol=2e-6, atol=2e-7)
    derivatives = torch.autograd.grad(out, (sample, noise), grad)
    for which, original in enumerate((sample, noise)):
        plus, minus = original.detach().clone(), original.detach().clone()
        plus[1, 1, 2] += 0.001
        minus[1, 1, 2] -= 0.001
        args_plus, args_minus = [sample, noise], [sample, noise]
        args_plus[which], args_minus[which] = plus, minus
        numerical = (
            (op(args_plus[0], time, args_plus[1]) - op(args_minus[0], time, args_minus[1])) * grad
        ).sum() / 0.002
        torch.testing.assert_close(numerical, derivatives[which][1, 1, 2], atol=1e-4, rtol=2e-4)


@pytest.mark.parametrize("time_shape", [(), (1,), (2,), (2, 1), (2, 1, 1)])
def test_scalar_and_per_sample_timestep(time_shape):
    x, n, _ = _make_tensors((2, 3, 5))
    t = torch.full(time_shape, 0.5)
    _assert_bitwise_equal(NativeConditioningNoiseMixOp()(x, t, n), x * 0.5 + n * 0.5)


@pytest.mark.parametrize("method", ["forward", "forward_fp32"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda_kernel)])
@pytest.mark.parametrize(
    "case, error, match",
    [
        ("shape", ValueError, "shape"),
        ("dtype", TypeError, "dtype"),
        ("double", TypeError, "dtype"),
        ("empty", ValueError, "non-empty"),
        ("scalar_sample", ValueError, "batch dimension"),
        ("strided", ValueError, "contiguous"),
        ("time_shape", ValueError, "scalar"),
        ("time_dtype", TypeError, "timestep"),
        ("time_grad", ValueError, "metadata"),
        ("time_strided", ValueError, "contiguous"),
        ("time_integer", TypeError, "timestep"),
        ("device", ValueError, "device"),
    ],
)
def test_invalid_inputs(case, error, match, device, method):
    x, n, _ = [value.to(device) for value in _make_tensors((2, 3, 5))]
    t = torch.tensor([0.2, 0.8], device=device)
    if case == "shape":
        n = n[:, :, :4].contiguous()
    elif case == "dtype":
        n = n.half()
    elif case == "double":
        x, n = x.double(), n.double()
    elif case == "empty":
        x, n = x[:0], n[:0]
    elif case == "scalar_sample":
        x, n = x[0, 0, 0], n[0, 0, 0]
    elif case == "strided":
        x, n = x.transpose(1, 2), n.transpose(1, 2)
    elif case == "time_shape":
        t = torch.ones(2, 3, device=device)
    elif case == "time_dtype":
        t = t.long()
    elif case == "time_grad":
        t.requires_grad_()
    elif case == "time_strided":
        t = torch.arange(4, dtype=torch.float32, device=device)[::2]
    elif case == "time_integer":
        t = 1
    elif case == "device":
        n = torch.empty_like(x, device="meta")
    op = KernelRegistry().get_op("conditioning_noise_mix", device=device)
    with pytest.raises(error, match=match):
        getattr(op, method)(x, t, n)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("time_kind", ["float", "scalar_tensor", "per_sample_tensor"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda_kernel)])
def test_fp32_preserves_timestep_precision(dtype, time_kind, device):
    sample, noise, grad = [value.to(device) for value in _make_tensors((2, 7), dtype)]
    sample.requires_grad_()
    noise.requires_grad_()
    if time_kind == "float":
        timestep = 0.37
    elif time_kind == "scalar_tensor":
        timestep = torch.tensor(0.37, dtype=torch.float32, device=device)
    else:
        timestep = torch.tensor([0.13, 0.79], dtype=torch.float32, device=device)
    time = torch.as_tensor(timestep, dtype=torch.float32, device=device).reshape(-1, 1)
    expected = time * sample.float() + (1.0 - time) * noise.float()
    expected_grads = torch.autograd.grad(expected, (sample, noise), grad.float())

    op = KernelRegistry().get_op("conditioning_noise_mix", device=device)
    actual = op.forward_fp32(sample, timestep, noise)
    actual_grads = torch.autograd.grad(actual, (sample, noise), grad.float())
    _assert_bitwise_equal(actual, expected)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        _assert_bitwise_equal(got, want)


def test_registry_cpu_reference():
    registry = KernelRegistry()
    op = registry.get_op("conditioning_noise_mix", device="cpu")
    assert isinstance(op, NativeConditioningNoiseMixOp)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("input_mode", ["random", "constant"])
@pytest.mark.parametrize(
    "device, backend",
    [("cpu", "pytorch"), pytest.param("cuda", "cuda", marks=requires_cuda_kernel)],
)
def test_gtest_forward_backward(dtype, input_mode, device, backend):
    args = argparse.Namespace(
        op="conditioning_noise_mix",
        candidate=backend,
        arch_key=device,
        batch=2,
        seq=3,
        normalized_dim=7,
        seed=420,
        input_mode=input_mode,
    )
    case = make_operator_case(args, dtype, torch.device(device))
    assert case.inputs["sample"].shape == (2, 3, 7)
    candidate = make_candidate(args)
    report = run_operator_suite(
        args.op, candidates=[candidate], cases=[case], check_grad=True, grad_mode="random"
    )
    assert report.passed, report.to_dict()


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", [(3, 257), (2, 32, 17), (2, 24, 1, 48, 80)])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda_kernel)])
def test_forward_and_gradients_against_fp32_reference(dtype, shape, device):
    sample, noise, grad = _make_tensors(shape, dtype)
    timestep = torch.linspace(0.13, 0.79, shape[0], dtype=dtype)
    # Promote already-quantized inputs before making the reference leaves so
    # autograd returns FP32 gradients, rather than casting them back to dtype.
    sample_fp32 = sample.float().detach().requires_grad_()
    noise_fp32 = noise.float().detach().requires_grad_()
    time_fp32 = timestep.float().reshape((-1,) + (1,) * (len(shape) - 1))
    # Independent scale_noise arithmetic from Diffusers f53d552; no shared
    # timestep preparation or candidate implementation is used by this gold.
    expected = time_fp32 * sample_fp32 + (1.0 - time_fp32) * noise_fp32
    expected_grads = torch.autograd.grad(expected, (sample_fp32, noise_fp32), grad.float())
    assert expected.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in expected_grads)

    op = KernelRegistry().get_op("conditioning_noise_mix", device=device)
    actual_sample = sample.to(device).requires_grad_()
    actual_noise = noise.to(device).requires_grad_()
    actual = op(sample=actual_sample, timestep=timestep.to(device), noise=actual_noise)
    actual_grads = torch.autograd.grad(actual, (actual_sample, actual_noise), grad.to(device))

    contract = load_contract()
    for judgment, actuals, references in (
        ("forward_accuracy", (actual,), (expected,)),
        ("gradient_accuracy", actual_grads, expected_grads),
    ):
        tolerance = resolve_tolerance(
            contract, judgment=judgment, op_class="elementwise", dtype=dtype
        )
        for got, want in zip(actuals, references, strict=True):
            torch.testing.assert_close(
                got.detach().cpu().float(),
                want.detach(),
                atol=tolerance.atol,
                rtol=tolerance.rtol,
            )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA CUDA GPU",
)
@pytest.mark.skipif(
    not envs.env_flag(envs.RL_KERNEL_REQUIRE_EXT),
    reason="native extension enforcement is opt-in via RL_KERNEL_REQUIRE_EXT=1",
)
def test_required_cuda_extension_has_operator_symbols():
    assert _EXT_AVAILABLE and all(
        hasattr(_C, name)
        for name in ("conditioning_noise_mix_forward", "conditioning_noise_mix_backward")
    ), "rl_engine._C must provide both conditioning_noise_mix kernels"


def test_cuda_registry_fails_closed(monkeypatch):
    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_get_or_create_backend", lambda backend: None)
    assert registry._priority_map["cuda"]["conditioning_noise_mix"] == [
        OpBackend.CUDA_CONDITIONING_NOISE_MIX
    ]
    with pytest.raises(RuntimeError, match="No functional backend"):
        registry.get_op("conditioning_noise_mix", device="cuda")
    assert registry._priority_map["rocm"]["conditioning_noise_mix"] == []
    assert registry._priority_map["musa"]["conditioning_noise_mix"] == []
    assert registry._priority_map["npu"]["conditioning_noise_mix"] == []


def test_missing_extension_fails_closed(monkeypatch):
    module = importlib.import_module("rl_engine.kernels.ops.cuda.conditioning_noise_mix")
    monkeypatch.setattr(module, "_EXT_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="compiled CUDA"):
        module.ConditioningNoiseMixCudaOp()


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", [(1, 1), (3, 257), (2, 24, 1, 48, 80)])
@pytest.mark.parametrize("per_sample", [False, True])
def test_cuda_matches_eager_cpu_forward_backward(dtype, shape, per_sample):
    x, n, g = _make_tensors(shape, dtype)
    t = torch.linspace(0.13, 0.79, shape[0], dtype=dtype) if per_sample else torch.tensor(0.37)
    cpu_x, cpu_n = x.requires_grad_(), n.requires_grad_()
    expected = NativeConditioningNoiseMixOp()(cpu_x, t, cpu_n)
    expected_grads = torch.autograd.grad(expected, (cpu_x, cpu_n), g)
    gpu_x, gpu_n = x.detach().cuda().requires_grad_(), n.detach().cuda().requires_grad_()
    op = ConditioningNoiseMixCudaOp()
    actual = op(gpu_x, t.cuda(), gpu_n)
    actual_grads = torch.autograd.grad(actual, (gpu_x, gpu_n), g.cuda())
    _assert_bitwise_equal(actual.cpu(), expected.detach())
    for got, want in zip(actual_grads, expected_grads):
        _assert_bitwise_equal(got.cpu(), want)


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", _DTYPES)
def test_forward_backward_invariance(dtype):
    op = ConditioningNoiseMixCudaOp()
    x, n, g = [v.cuda() for v in _make_tensors((1, 257), dtype)]
    t = torch.tensor([0.37], dtype=dtype, device="cuda")

    def run(a, b, time, grad):
        a, b = a.clone().requires_grad_(), b.clone().requires_grad_()
        y = op(a, time, b)
        return (y.detach(), *torch.autograd.grad(y, (a, b), grad))

    single = run(x, n, t, g)
    for batch, position, width in [(3, 0, 257), (4, 2, 263), (7, 6, 513)]:
        a, b, grad = [v.cuda() for v in _make_tensors((batch, width), dtype)]
        a[position, :257], b[position, :257], grad[position, :257] = x[0], n[0], g[0]
        times = torch.linspace(0, 1, batch, device="cuda", dtype=dtype)
        times[position] = t[0]
        first = run(a, b, times, grad)
        for _ in range(5):
            for got, want in zip(run(a, b, times, grad), first, strict=True):
                _assert_bitwise_equal(got, want)
        for got, want in zip(first, single):
            _assert_bitwise_equal(got[position, :257], want[0])
        permuted = run(a.flip(0), b.flip(0), times.flip(0), grad.flip(0))
        for got, want in zip(permuted, first, strict=True):
            _assert_bitwise_equal(got.flip(0), want)
        split = batch // 2
        chunks = [
            run(a[start:end], b[start:end], times[start:end], grad[start:end])
            for start, end in ((0, split), (split, batch))
        ]
        for index, want in enumerate(first):
            _assert_bitwise_equal(torch.cat([chunk[index] for chunk in chunks]), want)
        other = (position + 1) % batch
        a[other].fill_(99)
        b[other].fill_(-42)
        again = run(a, b, times, grad)
        for got, want in zip(again, first):
            _assert_bitwise_equal(got[position], want[position])


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", _DTYPES)
def test_low_precision_rounding_boundaries(dtype):
    x, n, _ = _make_tensors((2, 8193), dtype)
    t = torch.tensor([0.37, 0.79], dtype=dtype)
    expected = NativeConditioningNoiseMixOp()(x, t, n)
    actual = ConditioningNoiseMixCudaOp()(x.cuda(), t.cuda(), n.cuda()).cpu()
    _assert_bitwise_equal(actual, expected)
    if dtype != torch.float32:
        fused = (t.float()[:, None] * x.float() + (1 - t.float()[:, None]) * n.float()).to(dtype)
        assert not torch.equal(expected, fused), "fixture must distinguish intermediate rounding"


@requires_cuda_kernel
def test_current_stream_and_noncontiguous_upstream_gradient():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x, n = torch.ones(2, 257, device="cuda"), torch.full((2, 257), 3.0, device="cuda")
        x.requires_grad_()
        n.requires_grad_()
        out = ConditioningNoiseMixCudaOp()(x, 0.25, n)
        grad = torch.ones(257, 2, device="cuda").t()
        assert not grad.is_contiguous()
        dx, dn = torch.autograd.grad(out, (x, n), grad_outputs=grad)
    stream.synchronize()
    _assert_bitwise_equal(out.cpu(), torch.full((2, 257), 2.5))
    _assert_bitwise_equal(dx.cpu(), torch.full((2, 257), 0.25))
    _assert_bitwise_equal(dn.cpu(), torch.full((2, 257), 0.75))


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", _DTYPES)
def test_native_binding_matches_public_argument_order(dtype):
    sample, noise, _ = [value.cuda() for value in _make_tensors((2, 257), dtype)]
    timestep = torch.tensor([0.25, 0.75], dtype=dtype, device="cuda")
    expected = NativeConditioningNoiseMixOp()(sample, timestep, noise)
    _assert_bitwise_equal(_C.conditioning_noise_mix_forward(sample, timestep, noise), expected)


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("requires_grad", [(True, False), (False, True), (True, True)])
def test_gradients_follow_public_argument_order(dtype, requires_grad):
    sample, noise, grad = [value.cuda() for value in _make_tensors((2, 257), dtype)]
    sample.requires_grad_(requires_grad[0])
    noise.requires_grad_(requires_grad[1])
    timestep = torch.tensor([0.25, 0.75], dtype=dtype, device="cuda")
    leaves = tuple(value for value in (sample, noise) if value.requires_grad)
    expected = NativeConditioningNoiseMixOp()(sample, timestep, noise)
    expected_grads = torch.autograd.grad(expected, leaves, grad)
    actual = ConditioningNoiseMixCudaOp()(sample, timestep, noise)
    actual_grads = torch.autograd.grad(actual, leaves, grad)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        _assert_bitwise_equal(got, want)


@requires_cuda_kernel
def test_native_binding_validates_before_launch():
    x = torch.ones(2, 3, device="cuda")
    time = torch.ones(2, device="cuda")
    with pytest.raises(RuntimeError, match="dtype"):
        _C.conditioning_noise_mix_forward(x, time, x.half())
    with pytest.raises(RuntimeError, match="batch"):
        _C.conditioning_noise_mix_backward(x, torch.ones(3, device="cuda"))
    with pytest.raises(RuntimeError, match="contiguous"):
        _C.conditioning_noise_mix_forward(x.t(), time, x.t())


@requires_cuda_kernel
def test_768p_style_video_latents():
    # 24 latent channels, 17 latent frames, 768x1280 at the VAE's 16x spatial reduction.
    shape = (1, 24, 17, 48, 80)
    x, n, g = _make_tensors(shape)
    expected = NativeConditioningNoiseMixOp()(x, 0.37, n)
    gx, gn = x.cuda().requires_grad_(), n.cuda().requires_grad_()
    actual = ConditioningNoiseMixCudaOp()(gx, 0.37, gn)
    dx, dn = torch.autograd.grad(actual, (gx, gn), g.cuda())
    _assert_bitwise_equal(actual.cpu(), expected)
    _assert_bitwise_equal(dx.cpu(), g * torch.tensor(0.37))
    _assert_bitwise_equal(dn.cpu(), g * (1 - torch.tensor(0.37)))


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cuda_endpoints_subnormals_and_aliased_inputs(dtype):
    tiny = torch.finfo(dtype).tiny
    data = torch.tensor([[0.0, -0.0, tiny, -tiny, tiny / 2, 1.0, -1.0]], dtype=dtype)
    for time in (0.0, 0.37, 1.0):
        cpu = data.clone().requires_grad_()
        gpu = data.cuda().requires_grad_()
        expected = NativeConditioningNoiseMixOp()(cpu, time, cpu)
        actual = ConditioningNoiseMixCudaOp()(gpu, time, gpu)
        _assert_bitwise_equal(actual.cpu(), expected)
        # Exercise subnormal backward products as well as forward inputs.
        cpu_grad = torch.autograd.grad(expected, cpu, data)[0]
        gpu_grad = torch.autograd.grad(actual, gpu, data.cuda())[0]
        _assert_bitwise_equal(gpu_grad.cpu(), cpu_grad)


@requires_cuda_kernel
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("negative_view", ["sample", "noise", "timestep", "grad"])
def test_negative_views_match_reference(dtype, negative_view):
    x, n, grad = [v.cuda() for v in _make_tensors((2, 257), dtype)]
    time = torch.tensor([0.25, 0.75], dtype=dtype, device="cuda")
    values = {"sample": x, "noise": n, "timestep": time, "grad": grad}
    # Preserve valid positive timesteps while representing them as a lazy negation.
    original = -time if negative_view == "timestep" else values[negative_view]
    values[negative_view] = torch._neg_view(original)
    assert values[negative_view].is_neg() and values[negative_view].is_contiguous()
    x, n = values["sample"].requires_grad_(), values["noise"].requires_grad_()
    time, grad = values["timestep"], values["grad"]
    cx, cn = x.detach().cpu().requires_grad_(), n.detach().cpu().requires_grad_()
    expected = NativeConditioningNoiseMixOp()(cx, time.cpu(), cn)
    expected_grads = torch.autograd.grad(expected, (cx, cn), grad.cpu())
    actual = ConditioningNoiseMixCudaOp()(x, time, n)
    actual_grads = torch.autograd.grad(actual, (x, n), grad)
    _assert_bitwise_equal(actual.cpu(), expected.detach())
    for got, want in zip(actual_grads, expected_grads):
        _assert_bitwise_equal(got.cpu(), want)


@requires_cuda_kernel
def test_native_binding_rejects_unresolved_negative_views():
    # Public complex-tensor operations can produce contiguous floating negative views.
    x = torch.tensor([1 + 2j], device="cuda").conj().imag
    assert x.is_neg() and x.is_contiguous()
    time = torch.tensor([0.25], device="cuda")
    with pytest.raises(RuntimeError, match="negative view bit"):
        _C.conditioning_noise_mix_forward(x, time, x.resolve_neg())
    with pytest.raises(RuntimeError, match="negative view bit"):
        _C.conditioning_noise_mix_backward(x, time)
