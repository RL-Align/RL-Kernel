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
)

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
