# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.pytorch.norm.rms_norm import NativeRMSNormOp

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
    assert isinstance(op, NativeRMSNormOp)
    assert hasattr(op, "forward") and hasattr(op, "forward_fp32")
