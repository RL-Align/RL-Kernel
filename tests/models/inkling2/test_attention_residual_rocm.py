# SPDX-License-Identifier: Apache-2.0
"""INK-17 ROCm attention O-projection residual tests (MI300X)."""

import pytest
import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.models.inkling2.attention_residual import (
    attention_o_residual,
    attention_o_residual_rocm,
)

pytestmark = pytest.mark.skipif(
    torch.version.hip is None
    or not torch.cuda.is_available()
    or not _EXT_AVAILABLE
    or not hasattr(_C, "inkling2_attention_o_residual_forward"),
    reason="ROCm device and Inkling 2 attention O residual extension required",
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _make(rows, inp, out, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape, s=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * s).to(dtype)

    return r(rows, inp), r(rows, out), r(out, inp, s=inp**-0.5), r(out)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 5, 5), (3, 70, 67), (65, 129, 130), (0, 8, 8)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_forward_matches_reference(dtype, shape, use_bias):
    attn, res, w, b = _make(*shape, dtype)
    b = b if use_bias else None
    out = attention_o_residual_rocm(attn, res, w, b)
    ref = attention_o_residual(attn, res, w, b)
    assert out.shape == ref.shape and out.dtype == dtype
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", DTYPES)
def test_real_inkling_shape_forward_and_determinism(dtype):
    # hidden 2304 = Q24 * D128 attention output projected back to 2304
    attn, res, w, b = _make(40, 3072, 2304, dtype, seed=2)
    a = attention_o_residual_rocm(attn, res, w, b)
    c = attention_o_residual_rocm(attn, res, w, b)
    assert torch.equal(a, c)
    ref = attention_o_residual(attn, res, w, b)
    tol = 1e-3 if dtype == torch.float32 else 5e-2
    torch.testing.assert_close(a.float(), ref.float(), atol=tol, rtol=tol)


def test_residual_added_after_projection_in_fp32():
    # residual is added to the FP32 accumulator before the single bf16 cast
    attn = torch.zeros(1, 4, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(4, 4, device="cuda", dtype=torch.bfloat16)
    bias = torch.full((4,), 2**-9, device="cuda", dtype=torch.bfloat16)
    res = torch.full((1, 4), 1.0, device="cuda", dtype=torch.bfloat16)
    out = attention_o_residual_rocm(attn, res, w, bias)
    expected = (bias.float() + res.float()).to(torch.bfloat16)
    assert torch.equal(out, expected.reshape(1, 4))


@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_invariance_bitwise(dtype):
    attn, res, w, b = _make(75, 130, 131, dtype, seed=6)
    base = attention_o_residual_rocm(attn, res, w, b)
    assert torch.equal(attention_o_residual_rocm(attn[:9], res[:9], w, b), base[:9])
    assert torch.equal(attention_o_residual_rocm(attn[40:41], res[40:41], w, b), base[40:41])
    other = attn.clone()
    other[30:] = torch.randn_like(other[30:])
    assert torch.equal(attention_o_residual_rocm(other, res, w, b)[:30], base[:30])
    perm = torch.randperm(75, device="cuda")
    assert torch.equal(attention_o_residual_rocm(attn[perm], res[perm], w, b), base[perm])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(5, 70, 67), (33, 129, 65)])
def test_backward_matches_reference_and_is_deterministic(dtype, shape):
    leaves = [x.requires_grad_(True) for x in _make(*shape[:1], shape[1], shape[2], dtype, seed=8)]
    out = attention_o_residual_rocm(*leaves)
    dy = torch.randn_like(out)
    out.backward(dy)
    grads = [x.grad.clone() for x in leaves]
    for x in leaves:
        x.grad = None
    attention_o_residual_rocm(*leaves).backward(dy)
    for g, x in zip(grads, leaves, strict=True):
        assert torch.equal(g, x.grad)
    assert torch.equal(grads[1], dy)  # d_residual is dy

    ref_leaves = [x.detach().clone().requires_grad_(True) for x in leaves]
    attention_o_residual(*ref_leaves).backward(dy)
    tol = 1e-3 if dtype == torch.float32 else 5e-2
    for g, r in zip(grads, ref_leaves, strict=True):
        torch.testing.assert_close(g.float(), r.grad.float(), atol=tol, rtol=tol)


def test_rejects_cpu_tensors():
    attn, res, w, _ = _make(2, 8, 8, torch.float32)
    with pytest.raises(ValueError):
        attention_o_residual_rocm(attn.cpu(), res.cpu(), w.cpu())
