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

# The chunked reductions in rms_norm.py exist to keep the statistic bitwise equal
# across batch layouts on devices whose reduction kernel is shape-dependent. On CPU
# plain torch.mean/torch.sum already are invariant, so a CPU-only invariance test
# cannot fail for the mutation it names: replacing shape_invariant_meansq with
# x.pow(2).mean(-1) leaves all 119 tests in this operator's three files green.
# Every invariance test below therefore runs on CUDA as well, where it does fail.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _rand(shape, *, seed: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float64).to(device=device, dtype=dtype)


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


def test_production_pre_norm_is_not_the_missing_offset_form():
    """Guard: the decoder norm is (1+w), and dropping the 1 must be caught.

    This is the mistake the fingerprint exists to prevent, and it is silent at the
    API level. The comparison is deliberately the wrong way round: the production
    output is held *against the wrong FP64 form* and required to disagree. An
    earlier version of this test compared ``_ref_centered`` with ``_ref_scaled``,
    both defined in this file, so it asserted that two test-local expressions
    differ -- arithmetic, not a property of the code under test. It passed with
    ``_centered_norm`` replaced by a function returning zeros.
    """
    shape = (4, HIDDEN_SIZE)
    # FP32 is what the reference accepts, so the FP64 side starts from the same
    # values rather than from the unrounded draw.
    hidden = _rand(shape, seed=26).float()
    residual = _rand(shape, seed=120).float()
    post_w = _rand((shape[-1],), seed=27).float()
    pre_w = _rand((shape[-1],), seed=121).float()

    _, got_pre = NativeFusedAddNormOp().forward_fp32(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
    )
    post = _ref_centered(hidden.double(), post_w.double(), POST_NORM_EPS)
    residual_out = residual.double() + post
    correct = _ref_centered(residual_out, pre_w.double(), RMS_NORM_EPS)
    missing_offset = _ref_scaled(residual_out, pre_w.double(), RMS_NORM_EPS)  # w, not (1 + w)

    torch.testing.assert_close(got_pre.double(), correct, **_FP64_TOL)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got_pre.double(), missing_offset, **_FP64_TOL)


def test_production_eps_are_not_interchangeable():
    """Guard: post-norm eps is 1e-8 and pre-norm eps is 1e-5, not interchangeable.

    Two orders of magnitude apart but both tiny next to a unit-variance row, so the
    rows are scaled down until the eps term dominates, which is the only scale where
    any comparison can see the difference. As above, the production output is the
    left-hand side: it must match the FP64 form built with the eps it was given and
    must not match the form built with them exchanged. The previous version compared
    two test-local expressions at two eps values and never called the reference.
    """
    shape = (4, HIDDEN_SIZE)
    scale = 1e-3
    hidden = (_rand(shape, seed=28) * scale).float()
    residual = (_rand(shape, seed=122) * scale).float()
    post_w = _rand((shape[-1],), seed=29).float()
    pre_w = _rand((shape[-1],), seed=123).float()

    op = NativeFusedAddNormOp()
    got_res, got_pre = op.forward_fp32(
        hidden, residual, post_w, pre_w, post_eps=POST_NORM_EPS, pre_eps=RMS_NORM_EPS
    )

    def compose(post_eps, pre_eps):
        post = _ref_centered(hidden.double(), post_w.double(), post_eps)
        res = residual.double() + post
        return res, _ref_centered(res, pre_w.double(), pre_eps)

    want_res, want_pre = compose(POST_NORM_EPS, RMS_NORM_EPS)
    swapped_res, swapped_pre = compose(RMS_NORM_EPS, POST_NORM_EPS)

    torch.testing.assert_close(got_res.double(), want_res, **_FP64_TOL)
    torch.testing.assert_close(got_pre.double(), want_pre, **_FP64_TOL)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got_res.double(), swapped_res, **_FP64_TOL)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got_pre.double(), swapped_pre, **_FP64_TOL)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_differs_from_the_composition_without_the_pre_add_cast(dtype):
    """Negative control for the first narrowing point.

    ``test_forward_matches_an_eager_composition`` above already pins, bitwise, that
    ``forward`` *is* the narrowed composition. The half that was missing is that it
    is not the un-narrowed one, and the test that was supposed to cover it only
    asserted that ``forward`` and ``forward_fp32`` differ. Any narrowing anywhere
    makes those two differ, including the final output cast, so it passed with this
    cast removed and it passed with ``forward`` rewritten as
    ``forward_fp32(...).to(dtype)`` -- the exact mistake its own docstring named.
    Both measured before this was changed.

    The composition is built with the production statistic for the same reason the
    test above gives: what is under test is where the cast lands, not the
    reduction. The only difference between the two expressions below is the
    ``.to(dtype)`` on ``post``, and it moves 13.8k of 53k elements at BF16 and
    13.7k at FP16, so the control is not vacuous at either dtype.
    """
    shape = (8, HIDDEN_SIZE)
    hidden = _rand(shape, seed=30, dtype=dtype)
    residual = _rand(shape, seed=31, dtype=dtype)
    post_w = _rand((HIDDEN_SIZE,), seed=32, dtype=dtype)

    x_f = hidden.float()
    rstd = torch.rsqrt(shape_invariant_meansq(x_f) + POST_NORM_EPS).unsqueeze(-1)
    post = (x_f * rstd) * (1.0 + post_w.float())

    with_cast = residual + post.to(dtype)
    without_cast = (residual.float() + post).to(dtype)
    assert not torch.equal(with_cast, without_cast), "the control cannot fail at this dtype"

    got_residual, _ = NativeFusedAddNormOp().forward(hidden, residual, post_w, post_w)
    assert torch.equal(got_residual, with_cast)
    assert not torch.equal(got_residual, without_cast), (
        "forward matches the composition with the pre-add cast removed, so the cast "
        "is not where the fingerprint puts it"
    )



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


@pytest.mark.parametrize("device", DEVICES)
def test_backward_row_reduction_is_batch_invariant(device):
    """A row's gradient bytes do not change with batch size or position.

    Axis A applies to the backward too. The reduction inside the closed form is
    the chunked row sum for exactly this reason. Measured: with
    ``shape_invariant_rowsum`` and ``shape_invariant_meansq`` replaced by plain
    ``sum``/``mean``, this test passes on CPU and fails on CUDA on eleven
    (row, gradient) pairs, which is why ``device`` is a parameter rather than a
    default. All four gradients are compared because the reference returns the
    weight gradients unreduced, so they are still per-row here.
    """
    width = HIDDEN_SIZE
    big = 16
    kw = {"dtype": torch.float32, "device": device}
    hidden = _rand((big, width), seed=71, **kw)
    residual = _rand((big, width), seed=72, **kw)
    post_w = _rand((width,), seed=73, **kw)
    pre_w = _rand((width,), seed=74, **kw)
    g_residual = _rand((big, width), seed=75, **kw)
    g_pre = _rand((big, width), seed=76, **kw)

    full = fused_add_norm_backward(g_residual, g_pre, hidden, residual, post_w, pre_w)
    for count in (1, 2, 7, 15):
        for start in (0, 1, big - count):
            sl = slice(start, start + count)
            part = fused_add_norm_backward(
                g_residual[sl], g_pre[sl], hidden[sl], residual[sl], post_w, pre_w
            )
            for whole, piece, name in zip(
                full, part, ("d_hidden", "d_residual", "d_post", "d_pre"), strict=True
            ):
                where = f"rows [{start}, {start + count}) on {device}"
                assert torch.equal(whole[sl], piece), f"{name} drifted at {where}"


@pytest.mark.parametrize("device", DEVICES)
def test_reference_forward_is_batch_invariant(device):
    """The gold itself has to be batch invariant, not only the kernels.

    The reference is what both GPU backends are measured against and is the
    registered backend for cpu, musa and npu, so if its statistic moves with the
    batch layout then every comparison against it moves too. The kernels had their
    own invariance tests from the start; this one was missing.

    The second half is the layout the chunked reduction was written for:
    ``rms_norm.shape_invariant_meansq`` names ``[1,7,H]`` against ``[1,20,H]`` as
    the case where torch picks a different reduction kernel. Both outputs are
    compared, so the pre-norm's statistic over the narrowed sum is covered too.
    """
    width = HIDDEN_SIZE
    big = 16
    kw = {"dtype": torch.float32, "device": device}
    hidden = _rand((big, width), seed=80, **kw)
    residual = _rand((big, width), seed=81, **kw)
    post_w = _rand((width,), seed=82, **kw)
    pre_w = _rand((width,), seed=83, **kw)

    op = NativeFusedAddNormOp()
    full = op.forward_fp32(hidden, residual, post_w, pre_w)
    for count in (1, 2, 7, 15):
        for start in (0, 1, big - count):
            sl = slice(start, start + count)
            part = op.forward_fp32(hidden[sl], residual[sl], post_w, pre_w)
            for whole, piece, name in zip(full, part, ("residual_out", "pre_normed"), strict=True):
                where = f"rows [{start}, {start + count}) on {device}"
                assert torch.equal(whole[sl], piece), f"{name} drifted at {where}"

    # [1,7,H] against [1,20,H]: the same seven rows, two leading layouts.
    wide = _rand((1, 20, width), seed=84, **kw)
    wide_res = _rand((1, 20, width), seed=85, **kw)
    narrow_out = op.forward_fp32(wide[:, :7], wide_res[:, :7], post_w, pre_w)
    wide_out = op.forward_fp32(wide, wide_res, post_w, pre_w)
    for whole, piece, name in zip(wide_out, narrow_out, ("residual_out", "pre_normed"), strict=True):
        assert torch.equal(whole[:, :7], piece), f"{name} moved between [1,7,H] and [1,20,H]"


@pytest.mark.parametrize("device", DEVICES)
def test_final_norm_is_batch_invariant(device):
    """Same requirement for the final norm, which shares the chunked statistic."""
    width = HIDDEN_SIZE
    kw = {"dtype": torch.float32, "device": device}
    x = _rand((1, 20, width), seed=86, **kw)
    weight = _rand((width,), seed=87, **kw)

    op = NativeFinalNormOp()
    assert torch.equal(op.forward_fp32(x[:, :7], weight), op.forward_fp32(x, weight)[:, :7])
    assert torch.equal(op.forward_fp32(x[:, 3:5], weight), op.forward_fp32(x, weight)[:, 3:5])


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


@pytest.mark.parametrize("device", DEVICES)
def test_reference_uses_the_eps_it_is_given(device):
    """Both eps have to reach the arithmetic, not just the signature.

    Measured: with ``_run`` overwriting its ``post_eps``/``pre_eps`` arguments
    with the module constants, the whole suite stayed green -- every reference test
    used the defaults, so nothing distinguished a plumbed parameter from a
    hardcoded one. The Triton file has ``test_eps_values_reach_the_kernel`` for the
    same failure mode in the kernel; this is the reference's version of it.

    The rows are scaled down so the eps term dominates, which is the only regime
    where a comparison can see eps at all, and the two values are chosen far from
    the defaults so a partial plumbing (one of the two) also shows up.
    """
    width = HIDDEN_SIZE
    scale = 1e-3
    kw = {"dtype": torch.float32, "device": device}
    hidden = (_rand((4, width), seed=90, **kw) * scale).float()
    residual = (_rand((4, width), seed=91, **kw) * scale).float()
    post_w = _rand((width,), seed=92, **kw)
    pre_w = _rand((width,), seed=93, **kw)

    odd_post, odd_pre = 3e-4, 7e-2
    op = NativeFusedAddNormOp()
    got_res, got_pre = op.forward_fp32(
        hidden, residual, post_w, pre_w, post_eps=odd_post, pre_eps=odd_pre
    )

    def compose(post_eps, pre_eps):
        post = _ref_centered(hidden.double(), post_w.double(), post_eps)
        res = residual.double() + post
        return res, _ref_centered(res, pre_w.double(), pre_eps)

    want_res, want_pre = compose(odd_post, odd_pre)
    torch.testing.assert_close(got_res.double(), want_res, **_FP64_TOL)
    torch.testing.assert_close(got_pre.double(), want_pre, **_FP64_TOL)

    # And must not match the defaults, which is what a hardcoded eps would give.
    default_res, default_pre = compose(POST_NORM_EPS, RMS_NORM_EPS)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got_res.double(), default_res, **_FP64_TOL)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got_pre.double(), default_pre, **_FP64_TOL)


@pytest.mark.parametrize("device", DEVICES)
def test_final_norm_uses_the_eps_it_is_given(device):
    """Same for the final norm, which takes a single eps."""
    width = HIDDEN_SIZE
    kw = {"dtype": torch.float32, "device": device}
    x = (_rand((4, width), seed=94, **kw) * 1e-3).float()
    weight = _rand((width,), seed=95, **kw)

    odd = 5e-2
    got = NativeFinalNormOp().forward_fp32(x, weight, eps=odd)
    torch.testing.assert_close(
        got.double(), _ref_scaled(x.double(), weight.double(), odd), **_FP64_TOL
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            got.double(), _ref_scaled(x.double(), weight.double(), RMS_NORM_EPS), **_FP64_TOL
        )


def test_final_norm_uses_pow_neg_half_and_decoder_norms_use_rsqrt(monkeypatch):
    """The fingerprint separates the two spellings, so the code has to as well.

    No numerical test can tell them apart here: ``x ** -0.5`` and
    ``torch.rsqrt(x)`` are bitwise equal over 42M FP32 inputs spanning twelve
    decades, on both CPU and this GPU, which I measured before writing this. The
    distinction exists because the fingerprint freezes it and because a device
    whose ``pow`` and ``rsqrt`` differ would otherwise change the model silently.

    So this pins it behaviourally instead: ``torch.rsqrt`` is replaced with a
    counter. The decoder norms must go through it and the final norm must not.
    Measured: with ``_scaled_norm``'s ``** -0.5`` replaced by ``torch.rsqrt``, the
    suite was entirely green, and this test is now the one that fails.
    """
    calls = []
    real_rsqrt = torch.rsqrt
    monkeypatch.setattr(torch, "rsqrt", lambda x: (calls.append(1), real_rsqrt(x))[1])

    width = HIDDEN_SIZE
    hidden = _rand((4, width), seed=96).float()
    residual = _rand((4, width), seed=97).float()
    post_w = _rand((width,), seed=98).float()
    pre_w = _rand((width,), seed=99).float()

    NativeFinalNormOp().forward_fp32(hidden, post_w)
    assert not calls, "the final norm is pow_neg_half in the fingerprint, not rsqrt"

    NativeFusedAddNormOp().forward_fp32(hidden, residual, post_w, pre_w)
    assert calls, "the decoder norms are rsqrt in the fingerprint"
