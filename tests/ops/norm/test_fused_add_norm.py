# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Muse-Glimmer [F1] norm references against independent FP64 definitions.

The references under test are written from the #558 fingerprint. These tests are
written from the definition a second time, in FP64, without importing the
production helpers, so a misreading of the contract cannot cancel out on both
sides. Each guard test then checks that the FP64 comparison actually rejects the
mistake it is supposed to catch, because a tolerance check that cannot fail is
worth nothing.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.reference.norm.fused_add_norm import (
    HIDDEN_SIZE,
    POST_NORM_EPS,
    RMS_NORM_EPS,
    NativeFinalNormOp,
    NativeFusedAddNormOp,
    final_norm_backward,
    fused_add_norm_backward,
)
from rl_engine.reference.norm.rms_norm import shape_invariant_meansq

_FP64_TOL = {"atol": 1e-6, "rtol": 1e-6}


def _rand(shape, *, seed: int, dtype=torch.float64) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float64).to(dtype)


def _ref_meansq(x: torch.Tensor) -> torch.Tensor:
    """Mean of squares, written out rather than reusing the production helper."""
    return (x * x).sum(dim=-1, keepdim=True) / x.shape[-1]


def _ref_centered(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``centered_rmsnorm_one_plus_weight``, FP64, from the definition."""
    return x / torch.sqrt(_ref_meansq(x) + eps) * (1.0 + weight)


def _ref_scaled(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``rmsnorm_times_weight``, FP64, from the definition."""
    return x / torch.sqrt(_ref_meansq(x) + eps) * weight


SHAPES = [(1, HIDDEN_SIZE), (8, HIDDEN_SIZE), (2, 3, HIDDEN_SIZE)]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_fused_add_norm_matches_fp64_definition(shape):
    """forward_fp32 reproduces post-norm, add and pre-norm computed in FP64."""
    hidden = _rand(shape, seed=20)
    residual = _rand(shape, seed=21)
    post_w = _rand((shape[-1],), seed=22)
    pre_w = _rand((shape[-1],), seed=23)

    expected_residual = residual + _ref_centered(hidden, post_w, POST_NORM_EPS)
    expected_pre = _ref_centered(expected_residual, pre_w, RMS_NORM_EPS)

    got_residual, got_pre = NativeFusedAddNormOp().forward_fp32(
        hidden.float(), residual.float(), post_w.float(), pre_w.float()
    )
    torch.testing.assert_close(got_residual.double(), expected_residual, **_FP64_TOL)
    torch.testing.assert_close(got_pre.double(), expected_pre, **_FP64_TOL)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_final_norm_matches_fp64_definition(shape):
    x = _rand(shape, seed=24)
    weight = _rand((shape[-1],), seed=25)
    expected = _ref_scaled(x, weight, RMS_NORM_EPS)
    got = NativeFinalNormOp().forward_fp32(x.float(), weight.float())
    torch.testing.assert_close(got.double(), expected, **_FP64_TOL)


def test_fp64_check_rejects_a_missing_one_plus_offset():
    """Guard: the decoder norm is (1+w), and dropping the 1 must be caught.

    This is the mistake the fingerprint exists to prevent. Folding the offset into
    a BF16 weight, or forgetting it, is silent at the API level.
    """
    shape = (4, HIDDEN_SIZE)
    x = _rand(shape, seed=26)
    weight = _rand((shape[-1],), seed=27)
    correct = _ref_centered(x, weight, RMS_NORM_EPS)
    wrong = _ref_scaled(x, weight, RMS_NORM_EPS)  # w instead of (1 + w)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(wrong, correct, **_FP64_TOL)


def test_fp64_check_rejects_swapped_eps():
    """Guard: post-norm eps is 1e-8 and pre-norm eps is 1e-5, not interchangeable.

    Two orders of magnitude apart, but both tiny next to a unit-variance row, so a
    loose tolerance would let the swap through. This pins the scale at which the
    difference is still visible: rows scaled down so the eps term dominates.
    """
    shape = (4, HIDDEN_SIZE)
    x = _rand(shape, seed=28) * 1e-3
    weight = _rand((shape[-1],), seed=29)
    with_post = _ref_centered(x, weight, POST_NORM_EPS)
    with_pre = _ref_centered(x, weight, RMS_NORM_EPS)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(with_pre, with_post, **_FP64_TOL)


def test_forward_keeps_the_narrow_cast_before_the_residual_add():
    """The post-norm result is cast before the add, and that is observable.

    ``weight_applied_before_cast`` in the fingerprint puts the cast at the end of
    the norm, so in BF16 the value entering the add is already rounded. If a kernel
    kept the post-norm in FP32 and added there, it would disagree with eager HF.
    This test records that ``forward`` and ``forward_fp32`` are therefore not the
    same computation widened, which is why both exist.
    """
    shape = (8, HIDDEN_SIZE)
    hidden = _rand(shape, seed=30, dtype=torch.bfloat16)
    residual = _rand(shape, seed=31, dtype=torch.bfloat16)
    post_w = _rand((shape[-1],), seed=32, dtype=torch.bfloat16)
    pre_w = _rand((shape[-1],), seed=33, dtype=torch.bfloat16)

    op = NativeFusedAddNormOp()
    narrow_residual, _ = op.forward(hidden, residual, post_w, pre_w)
    wide_residual, _ = op.forward_fp32(hidden, residual, post_w, pre_w)
    assert not torch.equal(narrow_residual.float(), wide_residual)


@pytest.mark.parametrize(
    "mutate, exc, match",
    [
        (lambda h, r, pw, vw: (h[:, :-1], r, pw, vw), ValueError, "share shape"),
        (lambda h, r, pw, vw: (h, r.to(torch.float16), pw, vw), ValueError, "share dtype"),
        (lambda h, r, pw, vw: (h, r, pw[:-1], vw), ValueError, "post_weight must be 1-D"),
        (lambda h, r, pw, vw: (h, r, pw, vw[:-1]), ValueError, "pre_weight must be 1-D"),
        (
            lambda h, r, pw, vw: (h.to(torch.float64), r.to(torch.float64), pw, vw),
            TypeError,
            "fp16, bf16, or fp32",
        ),
    ],
)
def test_fused_add_norm_fails_closed(mutate, exc, match):
    """Shape, dtype and weight-width mismatches raise rather than broadcast."""
    shape = (4, HIDDEN_SIZE)
    args = (
        _rand(shape, seed=34, dtype=torch.float32),
        _rand(shape, seed=35, dtype=torch.float32),
        _rand((shape[-1],), seed=36, dtype=torch.float32),
        _rand((shape[-1],), seed=37, dtype=torch.float32),
    )
    with pytest.raises(exc, match=match):
        NativeFusedAddNormOp().forward(*mutate(*args))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_outputs_keep_the_input_dtype(dtype):
    shape = (4, HIDDEN_SIZE)
    args = (
        _rand(shape, seed=38, dtype=dtype),
        _rand(shape, seed=39, dtype=dtype),
        _rand((shape[-1],), seed=40, dtype=dtype),
        _rand((shape[-1],), seed=41, dtype=dtype),
    )
    residual_out, pre_normed = NativeFusedAddNormOp().forward(*args)
    assert residual_out.dtype == dtype
    assert pre_normed.dtype == dtype
    assert NativeFinalNormOp().forward(args[0], args[2]).dtype == dtype


# --------------------------------------------------------------------------- #
# Backward
# --------------------------------------------------------------------------- #
# The closed-form backward is what the CUDA and Triton kernels will implement, so
# it has to be checked against something that did not come from the same algebra.
# Two independent checks here: autograd on the reference forward, and autograd on
# an FP64 forward written from the definition.


def _ref_fused_forward_fp64(hidden, residual, post_w, pre_w):
    """FP64 forward from the definition, mirroring forward_fp32's structure."""
    post = _ref_centered(hidden, post_w, POST_NORM_EPS)
    residual_out = residual + post
    return residual_out, _ref_centered(residual_out, pre_w, RMS_NORM_EPS)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_closed_form_backward_matches_autograd(shape):
    """The hand-derived gradients agree with autograd on the same forward.

    Both output cotangents are random and non-zero. An all-ones cotangent would
    hide the rstd term, and a zero cotangent on residual_out would hide that the
    residual output is live on two paths at once.
    """
    hidden = _rand(shape, seed=50, dtype=torch.float32).requires_grad_(True)
    residual = _rand(shape, seed=51, dtype=torch.float32).requires_grad_(True)
    post_w = _rand((shape[-1],), seed=52, dtype=torch.float32).requires_grad_(True)
    pre_w = _rand((shape[-1],), seed=53, dtype=torch.float32).requires_grad_(True)
    g_residual = _rand(shape, seed=54, dtype=torch.float32)
    g_pre = _rand(shape, seed=55, dtype=torch.float32)

    residual_out, pre_normed = NativeFusedAddNormOp().forward_fp32(hidden, residual, post_w, pre_w)
    torch.autograd.backward((residual_out, pre_normed), grad_tensors=(g_residual, g_pre))

    d_hidden, d_residual, d_post_rows, d_pre_rows = fused_add_norm_backward(
        g_residual, g_pre, hidden.detach(), residual.detach(), post_w.detach(), pre_w.detach()
    )
    lead = tuple(range(d_post_rows.dim() - 1))
    torch.testing.assert_close(d_hidden, hidden.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(d_residual, residual.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(d_post_rows.sum(dim=lead), post_w.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(d_pre_rows.sum(dim=lead), pre_w.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_closed_form_backward_matches_fp64_definition(shape):
    """Same gradients against autograd on an independent FP64 forward."""
    hidden = _rand(shape, seed=56)
    residual = _rand(shape, seed=57)
    post_w = _rand((shape[-1],), seed=58)
    pre_w = _rand((shape[-1],), seed=59)
    g_residual = _rand(shape, seed=60)
    g_pre = _rand(shape, seed=61)

    leaves = [t.clone().requires_grad_(True) for t in (hidden, residual, post_w, pre_w)]
    out_residual, out_pre = _ref_fused_forward_fp64(*leaves)
    torch.autograd.backward((out_residual, out_pre), grad_tensors=(g_residual, g_pre))

    d_hidden, d_residual, d_post_rows, d_pre_rows = fused_add_norm_backward(
        g_residual.float(),
        g_pre.float(),
        hidden.float(),
        residual.float(),
        post_w.float(),
        pre_w.float(),
    )
    lead = tuple(range(d_post_rows.dim() - 1))
    torch.testing.assert_close(d_hidden.double(), leaves[0].grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(d_residual.double(), leaves[1].grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        d_post_rows.sum(dim=lead).double(), leaves[2].grad, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        d_pre_rows.sum(dim=lead).double(), leaves[3].grad, atol=1e-4, rtol=1e-4
    )


def test_backward_needs_both_cotangents():
    """Guard: dropping the downstream residual cotangent changes the answer.

    residual_out feeds both the next residual add and the pre-norm. A kernel that
    only propagated the pre-norm path would pass any test that left g_residual at
    zero, so this pins that the two contributions really do both land.
    """
    shape = (4, HIDDEN_SIZE)
    args = (
        _rand(shape, seed=62, dtype=torch.float32),
        _rand(shape, seed=63, dtype=torch.float32),
        _rand((shape[-1],), seed=64, dtype=torch.float32),
        _rand((shape[-1],), seed=65, dtype=torch.float32),
    )
    g_pre = _rand(shape, seed=66, dtype=torch.float32)
    with_downstream = fused_add_norm_backward(
        _rand(shape, seed=67, dtype=torch.float32), g_pre, *args
    )
    without = fused_add_norm_backward(torch.zeros(shape), g_pre, *args)
    assert not torch.equal(with_downstream[0], without[0])
    assert not torch.equal(with_downstream[1], without[1])


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_final_norm_backward_matches_fp64_definition(shape):
    x = _rand(shape, seed=68)
    weight = _rand((shape[-1],), seed=69)
    cotangent = _rand(shape, seed=70)

    x_leaf = x.clone().requires_grad_(True)
    w_leaf = weight.clone().requires_grad_(True)
    _ref_scaled(x_leaf, w_leaf, RMS_NORM_EPS).backward(cotangent)

    dx, dw_rows = final_norm_backward(cotangent.float(), x.float(), weight.float())
    lead = tuple(range(dw_rows.dim() - 1))
    torch.testing.assert_close(dx.double(), x_leaf.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(dw_rows.sum(dim=lead).double(), w_leaf.grad, atol=1e-4, rtol=1e-4)


def test_backward_row_reduction_is_batch_invariant():
    """A row's gradient bytes do not change with batch size or position.

    Axis A applies to the backward too. The reduction inside the closed form is
    the chunked row sum for exactly this reason; a plain sum would drift here while
    the forward still looked invariant.
    """
    width = HIDDEN_SIZE
    big = 16
    hidden = _rand((big, width), seed=71, dtype=torch.float32)
    residual = _rand((big, width), seed=72, dtype=torch.float32)
    post_w = _rand((width,), seed=73, dtype=torch.float32)
    pre_w = _rand((width,), seed=74, dtype=torch.float32)
    g_residual = _rand((big, width), seed=75, dtype=torch.float32)
    g_pre = _rand((big, width), seed=76, dtype=torch.float32)

    full = fused_add_norm_backward(g_residual, g_pre, hidden, residual, post_w, pre_w)
    for row in (0, 1, big // 2, big - 1):
        sl = slice(row, row + 1)
        single = fused_add_norm_backward(
            g_residual[sl], g_pre[sl], hidden[sl], residual[sl], post_w, pre_w
        )
        for whole, part, name in zip(
            full, single, ("d_hidden", "d_residual", "d_post", "d_pre"), strict=True
        ):
            assert torch.equal(whole[sl], part), f"{name} drifted at row {row}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_matches_an_eager_composition(dtype):
    """``forward`` narrows in the same two places eager does.

    Eager computes ``residual + post_norm(x)`` in the model dtype and hands that
    rounded value to the next norm, so there are two narrowing points and both
    change the answer. Composed here by hand, step by step, and required to match
    bitwise. Run in the narrow dtypes because in FP32 both narrowings vanish and
    the test would pass either way.

    This shares the production statistic deliberately. What is under test is where
    the casts land, not the reduction; swapping in a plain ``sum`` here would
    change ``post`` by a ULP on one element in 53k and the comparison would start
    failing for a reason that has nothing to do with casts. The statistic and the
    formula are checked independently by the FP64 tests above.
    """
    shape = (8, HIDDEN_SIZE)
    hidden = _rand(shape, seed=80, dtype=dtype)
    residual = _rand(shape, seed=81, dtype=dtype)
    post_w = _rand((HIDDEN_SIZE,), seed=82, dtype=dtype)
    pre_w = _rand((HIDDEN_SIZE,), seed=83, dtype=dtype)

    def centered(x, w, eps):
        x_f = x.float()
        rstd = torch.rsqrt(shape_invariant_meansq(x_f) + eps).unsqueeze(-1)
        return ((x_f * rstd) * (1.0 + w.float())).to(dtype)

    expected_residual = residual + centered(hidden, post_w, POST_NORM_EPS)
    expected_pre = centered(expected_residual, pre_w, RMS_NORM_EPS)

    got_residual, got_pre = NativeFusedAddNormOp().forward(hidden, residual, post_w, pre_w)
    assert torch.equal(got_residual, expected_residual)
    assert torch.equal(got_pre, expected_pre)


@pytest.mark.parametrize("rows", [8, 512, 2048])
def test_closed_form_backward_accumulates_the_residual_gradient_like_eager(rows):
    """The residual output's two incoming gradients are summed in the model dtype.

    Eager autograd rounds what arrives at a bf16 tensor to bf16 and adds in bf16.
    Keeping that sum in FP32 is one ULP per element, and the post-norm weight
    gradient sums it over rows: measured against the reference's own autograd, the
    FP32 variant sits 2.6 to 2.9 times outside the bf16 contract row at 512 and 2048
    rows, a single rounding of the FP32 sum does not help, and the eager
    accumulation brings it inside. This pins the closed form to autograd on
    ``forward`` at bf16 at the row counts where the difference is visible, and
    asserts the FP32-accumulated variant really does fail there, so the rounding
    cannot be removed as a simplification.
    """
    dtype = torch.bfloat16
    shape = (rows, HIDDEN_SIZE)
    hidden = _rand(shape, seed=90, dtype=dtype)
    residual = _rand(shape, seed=91, dtype=dtype)
    post_w = _rand((HIDDEN_SIZE,), seed=92, dtype=dtype)
    pre_w = _rand((HIDDEN_SIZE,), seed=93, dtype=dtype)
    g_res = _rand(shape, seed=94, dtype=dtype)
    g_pre = _rand(shape, seed=95, dtype=dtype)

    leaves = [t.clone().requires_grad_(True) for t in (hidden, residual, post_w, pre_w)]
    out_res, out_pre = NativeFusedAddNormOp().forward(*leaves)
    torch.autograd.backward((out_res, out_pre), grad_tensors=(g_res, g_pre))
    want = [t.grad.float() for t in leaves]

    atol, rtol = 1e-1, 2e-2  # the contract's bf16 gradient_accuracy reduction row
    lead = (0,)

    eager = fused_add_norm_backward(g_res, g_pre, hidden, residual, post_w, pre_w)
    got = [eager[0], eager[1], eager[2].sum(dim=lead), eager[3].sum(dim=lead)]
    for g, w, name in zip(
        got, want, ("d_hidden", "d_residual", "d_post_w", "d_pre_w"), strict=True
    ):
        torch.testing.assert_close(
            g.to(dtype).float(), w, atol=atol, rtol=rtol, msg=lambda m, n=name: f"{n}: {m}"
        )

    if rows >= 512:
        fp32 = fused_add_norm_backward(
            g_res, g_pre, hidden, residual, post_w, pre_w, grad_dtype=torch.float32
        )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                fp32[2].sum(dim=lead).to(dtype).float(), want[2], atol=atol, rtol=rtol
            )
