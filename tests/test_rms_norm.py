# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.cuda.norm.rmsnorm import RMSNormCudaOp, rmsnorm_cuda
from rl_engine.kernels.ops.pytorch.norm.rms_norm import NativeRMSNormOp
from rl_engine.kernels.ops.triton.rmsnorm_triton import (
    _TRITON_AVAILABLE,
    RMSNormTritonOp,
    rmsnorm_triton,
)
from rl_engine.platforms.device import device_ctx

try:
    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE

    _HAS_CUDA_RMSNORM = _EXT_AVAILABLE and all(
        hasattr(_C, name)
        for name in ("rmsnorm_forward", "rmsnorm_backward_dx", "rmsnorm_backward_dw")
    )
except ImportError:  # pragma: no cover - import can fail when the extension is not built.
    _HAS_CUDA_RMSNORM = False

# Qwen3-8B normalized dims this op must cover.
_HIDDEN = 4096  # input / post-attention norm
_HEAD_DIM = 128  # QK-Norm (per-head RMSNorm on Q and K)
_EPS = 1e-6


# Shared helpers
def _rand(shape, *, seed, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=dtype)


def _manual_rms_norm(x, weight, *, eps=_EPS):
    """Independent hand-written fp32 reference (NOT the op under test)."""
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    return x_f * torch.rsqrt(var + eps) * weight.float()


def _dtype_tolerance(dtype):
    if dtype is torch.float32:
        return 2e-5, 2e-5
    if dtype is torch.float16:
        return 3e-3, 3e-3
    if dtype is torch.bfloat16:
        return 2e-2, 2e-2
    raise ValueError(f"unsupported dtype: {dtype}")


def _run_forward_backward(fn, x, weight, dy):
    x_req = x.detach().clone().contiguous().requires_grad_(True)
    w_req = weight.detach().clone().contiguous().requires_grad_(True)
    y = fn(x_req, w_req)
    y.backward(dy.detach().clone().contiguous())
    return y.detach(), x_req.grad.detach(), w_req.grad.detach()


def _native_rstd(x, eps=_EPS):
    return torch.rsqrt(x.float().pow(2).mean(dim=-1) + eps)


def _native_dw(x, dy, rstd, mask):
    contrib = dy.float() * x.float() * rstd.float().unsqueeze(-1)
    return contrib.masked_fill(~mask[:, None], 0.0).sum(dim=0)


def _build_padded_layout(x_real, dy_real, total_rows, real_positions):
    dtype = x_real.dtype
    device = x_real.device
    real_rows, hidden = x_real.shape

    assert len(real_positions) == real_rows
    assert total_rows >= real_rows

    x_pad = torch.randn((total_rows, hidden), device=device, dtype=torch.float32).to(dtype)
    dy_pad = torch.randn((total_rows, hidden), device=device, dtype=torch.float32).to(dtype)
    mask = torch.zeros((total_rows,), device=device, dtype=torch.bool)

    for src_t, dst_t in enumerate(real_positions):
        x_pad[dst_t] = x_real[src_t]
        dy_pad[dst_t] = dy_real[src_t]
        mask[dst_t] = True

    return x_pad, dy_pad, mask


def _run_cuda_dw(x, dy, weight, mask, eps=_EPS):
    x_req = x.detach().clone().contiguous().requires_grad_(True)
    w_req = weight.detach().clone().contiguous().requires_grad_(True)
    y = rmsnorm_cuda(x_req, w_req, eps=eps, mask=mask)
    y.backward(dy.detach().clone().contiguous())
    return w_req.grad.detach()


requires_cuda_rmsnorm = pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAS_CUDA_RMSNORM),
    reason="CUDA RMSNorm extension is not available",
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


# 1. Primary correctness check vs PyTorch's own F.rms_norm. This is a *truly*
# independent implementation -- it may reduce in a different float order than
# our op, so a shared formula bug (eps placement, wrong reduction dim) cannot
# hide here. Tolerance-based (assert_close), NOT torch.equal, precisely because
# the reduction order is allowed to differ.
@pytest.mark.parametrize("N", [_HIDDEN, _HEAD_DIM])
def test_forward_fp32_matches_torch_reference(N):
    op = NativeRMSNormOp()
    x, w = _rand((2, 16, N), seed=0), _rand((N,), seed=1)
    ref = F.rms_norm(x.float(), (N,), weight=w.float(), eps=_EPS)
    torch.testing.assert_close(op.forward_fp32(x, w), ref, rtol=1e-6, atol=1e-6)


# 1b. Secondary sanity check vs a hand-written fp32 formula in the same float
# order -> bitwise equal. Pins the exact reference semantics; the F.rms_norm
# test above is the independent guard against a formula bug.
@pytest.mark.parametrize("N", [_HIDDEN, _HEAD_DIM])
def test_forward_fp32_matches_manual_reference(N):
    op = NativeRMSNormOp()
    x, w = _rand((2, 16, N), seed=0), _rand((N,), seed=1)
    assert torch.equal(op.forward_fp32(x, w), _manual_rms_norm(x, w))


# 2. Axis A -- batch invariance, bitwise (the WS1 "aligned" property)
@pytest.mark.parametrize("N", [_HIDDEN, _HEAD_DIM])
def test_batch_invariance_slice(N):
    """A row's output must not depend on how many rows share the batch."""
    op = NativeRMSNormOp()
    w, x = _rand((N,), seed=1), _rand((8, 32, N), seed=2)
    full = op.forward_fp32(x, w)  # compute on full batch...
    assert torch.equal(op.forward_fp32(x[:1], w), full[:1])  # ...then slice
    assert torch.equal(op.forward_fp32(x[3:5], w), full[3:5])


def test_batch_invariance_with_padding():
    """Padding extra rows must not perturb the real rows (bitwise)."""
    op = NativeRMSNormOp()
    w = _rand((_HIDDEN,), seed=1)
    x = _rand((4, _HIDDEN), seed=3)
    padded = torch.cat([x, _rand((6, _HIDDEN), seed=99)], dim=0)
    assert torch.equal(op.forward_fp32(padded, w)[:4], op.forward_fp32(x, w))


# 3. dtype behavior -- forward follows input, forward_fp32 forces fp32
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtype_paths(dtype):
    op = NativeRMSNormOp()
    x = _rand((2, 16, _HIDDEN), seed=4).to(dtype)
    w = _rand((_HIDDEN,), seed=5).to(dtype)
    assert op.forward(x, w).dtype == dtype
    assert op.forward_fp32(x, w).dtype == torch.float32


# 4. Axis B -- low-precision forward stays within tolerance of fp32 reference
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [(torch.bfloat16, 2e-2, 1.6e-2), (torch.float16, 1e-3, 1e-3)],
)
def test_low_precision_within_tolerance(dtype, atol, rtol):
    op = NativeRMSNormOp()
    x, w = _rand((4, 64, _HIDDEN), seed=6), _rand((_HIDDEN,), seed=7)
    ref = op.forward_fp32(x, w)
    got = op.forward(x.to(dtype), w.to(dtype)).float()
    assert torch.allclose(got, ref, atol=atol, rtol=rtol)


# 5. eps lives INSIDE the sqrt: zero input -> finite (zero) output
def test_eps_inside_sqrt():
    op = NativeRMSNormOp()
    out = op.forward_fp32(torch.zeros(1, _HIDDEN), torch.ones(_HIDDEN))
    assert torch.isfinite(out).all() and torch.equal(out, torch.zeros(1, _HIDDEN))


# 6. Plain weight scaling, NOT the (1 + weight) variant
def test_weight_scaling_no_plus_one():
    op = NativeRMSNormOp()
    x = _rand((2, _HEAD_DIM), seed=8)
    base = op.forward_fp32(x, torch.ones(_HEAD_DIM))
    doubled = op.forward_fp32(x, torch.full((_HEAD_DIM,), 2.0))
    assert torch.allclose(doubled, 2.0 * base, atol=1e-5)


# 7. Shape guard fires
def test_bad_weight_shape_raises():
    op = NativeRMSNormOp()
    x = _rand((2, _HIDDEN), seed=9)
    with pytest.raises(ValueError):
        op.forward_fp32(x, _rand((_HEAD_DIM,), seed=10))  # 128 != 4096
    with pytest.raises(ValueError):
        op.forward_fp32(x, _rand((1, _HIDDEN), seed=10))  # not 1-D


# 8. Purity -- inputs not mutated in-place
def test_inputs_not_mutated():
    op = NativeRMSNormOp()
    x, w = _rand((2, _HIDDEN), seed=11), _rand((_HIDDEN,), seed=12)
    xc, wc = x.clone(), w.clone()
    op.forward(x, w)
    op.forward_fp32(x, w)
    assert torch.equal(x, xc) and torch.equal(w, wc)


# 9. Gradient flows (fp32 autograd = backward golden source)
def test_gradient_flows():
    op = NativeRMSNormOp()
    x = _rand((2, _HIDDEN), seed=13).requires_grad_(True)
    w = _rand((_HIDDEN,), seed=14).requires_grad_(True)
    op.forward_fp32(x, w).sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


# 9b. Axis A for gradients -- backward must be batch-invariant too (needed for
# #153). Slicing the batch must yield bitwise-identical input gradients to the
# full-batch backward. Compute on the full batch, then compare against a
# batch-of-1 recompute fed the matching slice of the upstream gradient.
def test_backward_batch_invariance_slice():
    op = NativeRMSNormOp()

    w_full = _rand((_HIDDEN,), seed=1).requires_grad_(True)
    x_full = _rand((8, 32, _HIDDEN), seed=2).requires_grad_(True)
    out_full = op.forward_fp32(x_full, w_full)
    dy_full = _rand(out_full.shape, seed=3)
    out_full.backward(dy_full)
    grad_x_full_sliced = x_full.grad[:1].clone()

    w_slice = _rand((_HIDDEN,), seed=1).requires_grad_(True)
    x_slice = _rand((8, 32, _HIDDEN), seed=2)[:1].detach().requires_grad_(True)
    out_slice = op.forward_fp32(x_slice, w_slice)
    out_slice.backward(dy_full[:1])  # matching slice of the upstream gradient

    assert torch.equal(x_slice.grad, grad_x_full_sliced)


# 10. Registry dispatch resolves to the native op
def test_registry_dispatches_rms_norm():
    from rl_engine.kernels.registry import kernel_registry

    op = kernel_registry.get_op("rms_norm")
    if device_ctx.is_musa:
        if _TRITON_AVAILABLE:
            assert isinstance(op, RMSNormTritonOp)
            assert hasattr(op, "forward")
        else:
            assert isinstance(op, NativeRMSNormOp)
            assert hasattr(op, "forward") and hasattr(op, "forward_fp32")
    elif torch.cuda.is_available() and _HAS_CUDA_RMSNORM:
        assert isinstance(op, RMSNormCudaOp)
        assert hasattr(op, "forward")
    else:
        assert isinstance(op, NativeRMSNormOp)
        assert hasattr(op, "forward") and hasattr(op, "forward_fp32")


@requires_cuda
@pytest.mark.parametrize("impl", ["triton", "cuda"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("rows, hidden", [(1, 128), (8, 768), (32, 2048), (128, _HIDDEN)])
def test_cuda_triton_rms_norm_matches_native_forward_and_backward(impl, dtype, rows, hidden):
    if impl == "cuda" and not _HAS_CUDA_RMSNORM:
        pytest.skip("CUDA RMSNorm extension is not available")

    torch.manual_seed(0)
    native = NativeRMSNormOp()
    x_cpu = torch.randn(rows, hidden, device="cpu", dtype=torch.float32)
    w_cpu = torch.randn(hidden, device="cpu", dtype=torch.float32)
    dy_cpu = torch.randn(rows, hidden, device="cpu", dtype=torch.float32)

    x_ref = x_cpu.to(dtype).float().detach().requires_grad_(True)
    w_ref = w_cpu.to(dtype).float().detach().requires_grad_(True)
    dy_ref = dy_cpu.to(dtype).float()
    y_ref = native.forward_fp32(x_ref, w_ref, eps=_EPS)
    y_ref.backward(dy_ref)

    x_gpu = x_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    w_gpu = w_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    dy_gpu = dy_cpu.to(device="cuda", dtype=dtype)
    fn = rmsnorm_cuda if impl == "cuda" else rmsnorm_triton
    y_gpu = fn(x_gpu, w_gpu, eps=_EPS)
    y_gpu.backward(dy_gpu)

    assert w_gpu.grad.dtype == w_gpu.dtype

    atol, rtol = _dtype_tolerance(dtype)
    dw_scale = max(1.0, rows**0.5 / 4.0)
    torch.testing.assert_close(y_gpu.detach().cpu().float(), y_ref.detach(), atol=atol, rtol=rtol)
    torch.testing.assert_close(x_gpu.grad.detach().cpu().float(), x_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        w_gpu.grad.detach().cpu().float(),
        w_ref.grad,
        atol=atol * dw_scale,
        rtol=rtol * dw_scale,
    )


@requires_cuda
@pytest.mark.parametrize("impl", ["triton", "cuda"])
def test_cuda_triton_rms_norm_deterministic_repeat(impl):
    if impl == "cuda" and not _HAS_CUDA_RMSNORM:
        pytest.skip("CUDA RMSNorm extension is not available")

    torch.manual_seed(0)
    fn = rmsnorm_cuda if impl == "cuda" else rmsnorm_triton
    x = torch.randn(128, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(_HIDDEN, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(128, _HIDDEN, device="cuda", dtype=torch.bfloat16)

    y0, dx0, dw0 = _run_forward_backward(fn, x, weight, dy)
    torch.cuda.synchronize()
    for _ in range(10):
        y, dx, dw = _run_forward_backward(fn, x, weight, dy)
        torch.cuda.synchronize()
        assert torch.equal(y0, y)
        assert torch.equal(dx0, dx)
        assert torch.equal(dw0, dw)


@requires_cuda
def test_triton_rms_norm_long_context_dw_reduction():
    torch.manual_seed(2)
    rows, hidden = 1025, _HEAD_DIM
    dtype = torch.bfloat16
    native = NativeRMSNormOp()
    x_cpu = torch.randn(rows, hidden, device="cpu", dtype=torch.float32)
    w_cpu = torch.randn(hidden, device="cpu", dtype=torch.float32)
    dy_cpu = torch.randn(rows, hidden, device="cpu", dtype=torch.float32)

    x_ref = x_cpu.to(dtype).float().detach().requires_grad_(True)
    w_ref = w_cpu.to(dtype).float().detach().requires_grad_(True)
    y_ref = native.forward_fp32(x_ref, w_ref, eps=_EPS)
    y_ref.backward(dy_cpu.to(dtype).float())

    x_gpu = x_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    w_gpu = w_cpu.to(device="cuda", dtype=dtype).detach().requires_grad_(True)
    dy_gpu = dy_cpu.to(device="cuda", dtype=dtype)
    y_gpu = rmsnorm_triton(x_gpu, w_gpu, eps=_EPS)
    y_gpu.backward(dy_gpu)

    assert w_gpu.grad.dtype == w_gpu.dtype
    torch.testing.assert_close(y_gpu.detach().cpu().float(), y_ref.detach(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        w_gpu.grad.detach().cpu().float(),
        w_ref.grad,
        atol=8e-2,
        rtol=8e-2,
    )


@requires_cuda
@pytest.mark.parametrize("impl", ["triton", "cuda"])
@pytest.mark.parametrize("hidden", [63, 64, 65, 127, 128, 129, 255, 256, 257, _HIDDEN])
def test_cuda_triton_rms_norm_forward_dx_layout_invariance(impl, hidden):
    if impl == "cuda" and not _HAS_CUDA_RMSNORM:
        pytest.skip("CUDA RMSNorm extension is not available")

    torch.manual_seed(1)
    fn = rmsnorm_cuda if impl == "cuda" else rmsnorm_triton
    dtype = torch.bfloat16
    target_x = torch.randn(1, hidden, device="cuda", dtype=dtype)
    target_dy = torch.randn(1, hidden, device="cuda", dtype=dtype)
    weight = torch.randn(hidden, device="cuda", dtype=dtype)

    y_single, dx_single, _ = _run_forward_backward(fn, target_x, weight, target_dy)
    for total_rows, row_id in [(16, 0), (16, 7), (64, 63)]:
        x = torch.randn(total_rows, hidden, device="cuda", dtype=dtype)
        dy = torch.randn(total_rows, hidden, device="cuda", dtype=dtype)
        x[row_id : row_id + 1] = target_x
        dy[row_id : row_id + 1] = target_dy
        y, dx, _ = _run_forward_backward(fn, x, weight, dy)
        assert torch.equal(y_single[0], y[row_id])
        assert torch.equal(dx_single[0], dx[row_id])

    valid_rows = 4
    positions = [1, 5, 9, 14]
    valid_x = torch.randn(valid_rows, hidden, device="cuda", dtype=dtype)
    valid_dy = torch.randn(valid_rows, hidden, device="cuda", dtype=dtype)
    x_a = torch.randn(16, hidden, device="cuda", dtype=dtype)
    dy_a = torch.randn(16, hidden, device="cuda", dtype=dtype)
    x_a[:valid_rows] = valid_x
    dy_a[:valid_rows] = valid_dy
    y_a, dx_a, _ = _run_forward_backward(fn, x_a, weight, dy_a)

    x_b = torch.randn(16, hidden, device="cuda", dtype=dtype)
    dy_b = torch.randn(16, hidden, device="cuda", dtype=dtype)
    for idx, pos in enumerate(positions):
        x_b[pos] = valid_x[idx]
        dy_b[pos] = valid_dy[idx]
    y_b, dx_b, _ = _run_forward_backward(fn, x_b, weight, dy_b)

    for idx, pos in enumerate(positions):
        assert torch.equal(y_a[idx], y_b[pos])
        assert torch.equal(dx_a[idx], dx_b[pos])


@requires_cuda_rmsnorm
def test_cuda_rms_norm_masked_dw_layout_invariance():
    torch.manual_seed(0)
    rows, hidden = 128, _HIDDEN
    dtype = torch.bfloat16
    x_real = torch.randn((rows, hidden), device="cuda", dtype=torch.float32).to(dtype)
    dy_real = torch.randn((rows, hidden), device="cuda", dtype=torch.float32).to(dtype)
    weight = torch.randn((hidden,), device="cuda", dtype=torch.float32).to(dtype)

    x1 = x_real.clone()
    dy1 = dy_real.clone()
    mask1 = torch.ones((rows,), device="cuda", dtype=torch.bool)
    x2, dy2, mask2 = _build_padded_layout(
        x_real=x_real,
        dy_real=dy_real,
        total_rows=2 * rows,
        real_positions=[2 * i + 1 for i in range(rows)],
    )
    x3, dy3, mask3 = _build_padded_layout(
        x_real=x_real,
        dy_real=dy_real,
        total_rows=2 * rows + 1,
        real_positions=[2 * i for i in range(rows)],
    )

    dw1 = _run_cuda_dw(x1, dy1, weight, mask1)
    dw2 = _run_cuda_dw(x2, dy2, weight, mask2)
    dw3 = _run_cuda_dw(x3, dy3, weight, mask3)

    ref_dw1 = _native_dw(x1, dy1, _native_rstd(x1), mask1)
    ref_dw2 = _native_dw(x2, dy2, _native_rstd(x2), mask2)
    ref_dw3 = _native_dw(x3, dy3, _native_rstd(x3), mask3)

    atol, rtol = _dtype_tolerance(dtype)
    torch.testing.assert_close(dw1.float(), dw2.float(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(dw2.float(), dw3.float(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(dw1.float(), ref_dw1.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(dw2.float(), ref_dw2.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(dw3.float(), ref_dw3.float(), atol=atol, rtol=rtol)
