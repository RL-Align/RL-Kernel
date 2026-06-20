# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

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


# 1. Correctness vs an independent fp32 formula (both normalized dims)
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


# 10. Registry dispatch resolves to the native op
def test_registry_dispatches_rms_norm():
    from rl_engine.kernels.registry import kernel_registry

    op = kernel_registry.get_op("rms_norm")
    assert isinstance(op, NativeRMSNormOp)
    assert hasattr(op, "forward") and hasattr(op, "forward_fp32")
