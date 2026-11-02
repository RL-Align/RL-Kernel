# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton [F1] kernels: post-norm, residual add, pre-norm, and the final norm.

Mirrors ``rl_engine.reference.norm.fused_add_norm``. The frozen semantics and the
two narrowing points are documented there; this file implements them and should
not be read as the contract.

One program per row, one block per row, so the reduction order is fixed by
``BLOCK_H`` and never depends on the batch layout. That is what makes Axis-A batch
invariance hold here.

A note on the reduction, because it limits what this file can claim. RFC #501 asks
for CUDA == Triton bitwise. For an elementwise operator that is reachable, and
``activation/swiglu.py`` does it. For a row reduction it is not reachable by
writing the same source in both languages: the fold is exactly reproducible when
fed precomputed squares, but once the square is in-kernel the optimiser reshapes
the reduction chain, and an inline-asm barrier that removes every ``fma`` from the
PTX still leaves the result differing from a hand-written tree on a few rows in
sixty-four. The repo's own position matches: ``tests/ops/norm/test_rms_norm.py``
compares backends with a tolerance ``NOT torch.equal, precisely because the
reduction order is allowed to differ``, and the vendor-exact sequences in
``attention/deterministic_attn.py`` needed hand-written assembly. So the tests for
this operator hold CUDA and Triton to the tolerance contract and to identical
batch invariance, and the bitwise question is raised in the PR rather than
answered here.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.autograd.vjp_fp32 import reduce_rows_fp32
from rl_engine.reference.norm.fused_add_norm import POST_NORM_EPS, RMS_NORM_EPS

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only builds
    _TRITON_AVAILABLE = False

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_BLOCK = 1 << 16


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_add_norm_fwd_kernel(
        HIDDEN,
        RESIDUAL,
        POST_W,
        PRE_W,
        RES_OUT,
        PRE_OUT,
        RSTD_POST,
        RSTD_PRE,
        H: tl.constexpr,
        POST_EPS: tl.constexpr,
        PRE_EPS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        mask = offs < H
        base = row * H

        hidden = tl.load(HIDDEN + base + offs, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(RESIDUAL + base + offs, mask=mask, other=0.0).to(tl.float32)
        post_w = tl.load(POST_W + offs, mask=mask, other=0.0).to(tl.float32)
        pre_w = tl.load(PRE_W + offs, mask=mask, other=0.0).to(tl.float32)

        # Masked lanes load as 0.0 and contribute nothing to the sum of squares.
        rstd_post = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / H + POST_EPS)
        post = hidden * rstd_post * (1.0 + post_w)

        # First narrowing: the post-norm's cast is its last step.
        narrow = RES_OUT.dtype.element_ty
        res_out = residual + post.to(narrow).to(tl.float32)
        # Second narrowing: the pre-norm reads the rounded sum, as eager does.
        res_narrow = res_out.to(narrow)
        res_f = res_narrow.to(tl.float32)

        rstd_pre = tl.rsqrt(tl.sum(res_f * res_f, axis=0) / H + PRE_EPS)
        pre = res_f * rstd_pre * (1.0 + pre_w)

        tl.store(RES_OUT + base + offs, res_narrow, mask=mask)
        tl.store(PRE_OUT + base + offs, pre.to(PRE_OUT.dtype.element_ty), mask=mask)
        tl.store(RSTD_POST + row, rstd_post)
        tl.store(RSTD_PRE + row, rstd_pre)

    @triton.jit
    def _fused_add_norm_bwd_kernel(
        G_RES,
        G_PRE,
        HIDDEN,
        RES_OUT,
        POST_W,
        PRE_W,
        RSTD_POST,
        RSTD_PRE,
        D_HIDDEN,
        D_RESIDUAL,
        PARTIAL_POST_W,
        PARTIAL_PRE_W,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_H)
        mask = offs < H
        base = row * H

        g_res = tl.load(G_RES + base + offs, mask=mask, other=0.0).to(tl.float32)
        g_pre = tl.load(G_PRE + base + offs, mask=mask, other=0.0).to(tl.float32)
        hidden = tl.load(HIDDEN + base + offs, mask=mask, other=0.0).to(tl.float32)
        res_out = tl.load(RES_OUT + base + offs, mask=mask, other=0.0).to(tl.float32)
        post_gain = 1.0 + tl.load(POST_W + offs, mask=mask, other=0.0).to(tl.float32)
        pre_gain = 1.0 + tl.load(PRE_W + offs, mask=mask, other=0.0).to(tl.float32)
        rstd_post = tl.load(RSTD_POST + row).to(tl.float32)
        rstd_pre = tl.load(RSTD_PRE + row).to(tl.float32)

        # Pre-norm backward, then fold in the cotangent the residual output gets
        # from downstream. It is live on both paths; dropping either is silent.
        # Accumulate the two the way eager does: the residual output is a
        # model-dtype tensor, so autograd rounds the arriving gradient to that
        # dtype and adds in that dtype. Keeping this in FP32 is one ULP per
        # element and puts the post-norm weight gradient 2.6-2.9x outside the
        # contract row at 512 and 2048 rows; see the reference's docstring.
        narrow = G_RES.dtype.element_ty
        scaled_pre = g_pre * pre_gain
        dot_pre = tl.sum(scaled_pre * res_out, axis=0) / H
        d_from_pre = rstd_pre * scaled_pre - res_out * rstd_pre * rstd_pre * rstd_pre * dot_pre
        d_res_out = (g_res + d_from_pre.to(narrow).to(tl.float32)).to(narrow).to(tl.float32)

        scaled_post = d_res_out * post_gain
        dot_post = tl.sum(scaled_post * hidden, axis=0) / H
        d_hidden = rstd_post * scaled_post - hidden * rstd_post * rstd_post * rstd_post * dot_post

        tl.store(D_HIDDEN + base + offs, d_hidden, mask=mask)
        tl.store(D_RESIDUAL + base + offs, d_res_out, mask=mask)
        tl.store(PARTIAL_PRE_W + base + offs, g_pre * res_out * rstd_pre, mask=mask)
        tl.store(PARTIAL_POST_W + base + offs, d_res_out * hidden * rstd_post, mask=mask)


def _require_triton() -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for the fused add-norm kernels")


def _validate(hidden: Tensor, residual: Tensor, post_w: Tensor, pre_w: Tensor) -> int:
    for name, t in (("hidden", hidden), ("residual", residual)):
        if t.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {t.dtype}.")
        if not t.is_cuda:
            raise RuntimeError(f"{name} must be a GPU tensor, got device '{t.device}'.")
    if hidden.shape != residual.shape:
        raise ValueError(
            f"hidden and residual must share shape, got {tuple(hidden.shape)} "
            f"and {tuple(residual.shape)}."
        )
    if hidden.dtype != residual.dtype:
        raise ValueError(
            f"hidden and residual must share dtype, got {hidden.dtype} and {residual.dtype}."
        )
    width = hidden.shape[-1]
    for name, w in (("post_weight", post_w), ("pre_weight", pre_w)):
        if w.dim() != 1 or w.shape[0] != width:
            raise ValueError(
                f"{name} must be 1-D of size x.shape[-1]={width}, "
                f"got tuple({name}.shape)={tuple(w.shape)}"
            )
    block = triton.next_power_of_2(width)
    if block > _MAX_BLOCK:
        raise ValueError(
            f"hidden width {width} needs BLOCK_H {block}, above the {_MAX_BLOCK} "
            "single-block limit; a row must fit one program for the reduction order "
            "to stay independent of the batch layout."
        )
    return block


def fused_add_norm_triton_forward(
    hidden: Tensor,
    residual: Tensor,
    post_weight: Tensor,
    pre_weight: Tensor,
    *,
    post_eps: float,
    pre_eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns ``(residual_out, pre_normed, rstd_post, rstd_pre)``.

    The two statistics come back so the backward does not recompute them and
    cannot drift from what the forward actually used.
    """
    _require_triton()
    block = _validate(hidden, residual, post_weight, pre_weight)
    width = hidden.shape[-1]

    hidden_2d = hidden.contiguous().view(-1, width)
    residual_2d = residual.contiguous().view(-1, width)
    rows = hidden_2d.shape[0]

    res_out = torch.empty_like(hidden_2d)
    pre_out = torch.empty_like(hidden_2d)
    rstd_post = torch.empty(rows, device=hidden.device, dtype=torch.float32)
    rstd_pre = torch.empty(rows, device=hidden.device, dtype=torch.float32)

    _fused_add_norm_fwd_kernel[(rows,)](
        hidden_2d,
        residual_2d,
        post_weight.contiguous(),
        pre_weight.contiguous(),
        res_out,
        pre_out,
        rstd_post,
        rstd_pre,
        H=width,
        POST_EPS=float(post_eps),
        PRE_EPS=float(pre_eps),
        BLOCK_H=block,
    )
    return (
        res_out.view_as(hidden),
        pre_out.view_as(hidden),
        rstd_post,
        rstd_pre,
    )


def fused_add_norm_triton_backward(
    grad_residual_out: Tensor,
    grad_pre_normed: Tensor,
    hidden: Tensor,
    residual_out: Tensor,
    post_weight: Tensor,
    pre_weight: Tensor,
    rstd_post: Tensor,
    rstd_pre: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns ``(d_hidden, d_residual, partial_post_w, partial_pre_w)`` in FP32.

    The two weight entries are per-row contributions, matching the convention in
    ``norm/rmsnorm.py``; the caller folds them with ``reduce_rows_fp32`` so the
    declared reduction order over rows is preserved.
    """
    _require_triton()
    width = hidden.shape[-1]
    block = triton.next_power_of_2(width)

    hidden_2d = hidden.contiguous().view(-1, width)
    res_out_2d = residual_out.contiguous().view(-1, width)
    g_res_2d = grad_residual_out.contiguous().view(-1, width)
    g_pre_2d = grad_pre_normed.contiguous().view(-1, width)
    rows = hidden_2d.shape[0]

    d_hidden = torch.empty(rows, width, device=hidden.device, dtype=torch.float32)
    d_residual = torch.empty_like(d_hidden)
    partial_post = torch.empty_like(d_hidden)
    partial_pre = torch.empty_like(d_hidden)

    _fused_add_norm_bwd_kernel[(rows,)](
        g_res_2d,
        g_pre_2d,
        hidden_2d,
        res_out_2d,
        post_weight.contiguous(),
        pre_weight.contiguous(),
        rstd_post,
        rstd_pre,
        d_hidden,
        d_residual,
        partial_post,
        partial_pre,
        H=width,
        BLOCK_H=block,
    )
    return d_hidden, d_residual, partial_post, partial_pre


class _FusedAddNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, residual, post_weight, pre_weight, post_eps, pre_eps):
        res_out, pre_out, rstd_post, rstd_pre = fused_add_norm_triton_forward(
            hidden,
            residual,
            post_weight,
            pre_weight,
            post_eps=post_eps,
            pre_eps=pre_eps,
        )
        ctx.save_for_backward(hidden, res_out, post_weight, pre_weight, rstd_post, rstd_pre)
        return res_out, pre_out

    @staticmethod
    def backward(ctx, grad_residual_out, grad_pre_normed):
        hidden, res_out, post_w, pre_w, rstd_post, rstd_pre = ctx.saved_tensors
        d_hidden, d_residual, partial_post, partial_pre = fused_add_norm_triton_backward(
            grad_residual_out,
            grad_pre_normed,
            hidden,
            res_out,
            post_w,
            pre_w,
            rstd_post,
            rstd_pre,
        )
        record_backward(
            "fused_post_norm_residual_pre_norm",
            kernel_id=(
                "rl_engine.backends.shared.triton.norm.fused_add_norm"
                "._fused_add_norm_bwd_kernel"
                "+rl_engine.ops.autograd.vjp_fp32.reduce_rows_fp32"
            ),
            impl="triton_fused_add_norm_declared_fp32_rowfold_dw",
            family="triton",
        )
        width = hidden.shape[-1]
        d_post_w = reduce_rows_fp32(partial_post.view(-1, width)).to(post_w.dtype)
        d_pre_w = reduce_rows_fp32(partial_pre.view(-1, width)).to(pre_w.dtype)
        return (
            d_hidden.view_as(hidden).to(hidden.dtype),
            d_residual.view_as(hidden).to(hidden.dtype),
            d_post_w,
            d_pre_w,
            None,
            None,
        )


class TritonFusedAddNormOp:
    """Triton [F1] wrapper shaped like the other operator-harness ops."""

    backward_impl = "triton_fused_add_norm_declared_fp32_rowfold_dw"

    def __call__(
        self,
        hidden,
        residual,
        post_weight,
        pre_weight,
        *,
        post_eps=POST_NORM_EPS,
        pre_eps=RMS_NORM_EPS,
    ):
        return self.forward(
            hidden, residual, post_weight, pre_weight, post_eps=post_eps, pre_eps=pre_eps
        )

    def forward(
        self,
        hidden,
        residual,
        post_weight,
        pre_weight,
        *,
        post_eps=POST_NORM_EPS,
        pre_eps=RMS_NORM_EPS,
    ):
        return _FusedAddNormTriton.apply(
            hidden, residual, post_weight, pre_weight, post_eps, pre_eps
        )
