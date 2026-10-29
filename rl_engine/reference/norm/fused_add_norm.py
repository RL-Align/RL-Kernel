# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Muse-Glimmer-30B [F1] norm references: post-norm, residual add, pre-norm.

RFC #501 work item ``fused_post_norm_residual_pre_norm``. Every rule below comes
from the ``frozen_semantics`` block of the #558 architecture fingerprint rather
than from the HF config, because the config does not state most of it.

A decoder layer runs the order the fingerprint pins as ``norm_residual_order``::

    input_layernorm -> self_attn -> post_attention_layernorm -> residual_add
    -> pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm -> residual_add

so the operator fused here is the tail of each half-layer: take a sublayer output,
post-norm it, add it to the residual stream, and pre-norm the result for whatever
comes next. Two outputs, because both are live: the updated residual continues the
chain and the pre-normed tensor feeds the next sublayer.

``decoder_norm`` (all four per-layer norms)
    form ``centered_rmsnorm_one_plus_weight``: ``x * rstd * (1 + w)``. The ``1 +``
    is a kernel parameter, not something a caller folds into the weight, because
    the weights are BF16 and folding rounds the offset away.
    compute dtype FP32, ``rsqrt_form`` ``rsqrt``, ``weight_applied_before_cast``
    true, so the multiply happens in FP32 and the cast is the last step.
    eps is per site: ``rms_norm_eps`` for ``input_layernorm`` and
    ``pre_feedforward_layernorm``, ``post_norm_eps`` for the two post-norms.

``final_norm``
    form ``rmsnorm_times_weight``: ``x * rstd * w``, no offset, and
    ``rsqrt_form`` ``pow_neg_half`` rather than ``rsqrt``. Kept as a separate entry
    point for exactly that reason. The two spellings are bitwise equal for FP32 on
    SM 8.9, measured over 4096 rows at hidden 6656 for both eps values, but the
    fingerprint distinguishes them so this code does too.

Both forms share ``shape_invariant_meansq`` for the reduction. That is what makes
Axis-A batch invariance hold: the statistic is summed in fixed 32-wide chunks so no
intermediate shape depends on the batch layout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rl_engine.reference.norm.rms_norm import shape_invariant_meansq, shape_invariant_rowsum

# Muse-Glimmer-30B text config; see the #558 manifest.
HIDDEN_SIZE = 6656
RMS_NORM_EPS = 1e-5
POST_NORM_EPS = 1e-8

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_row(x: Tensor, name: str) -> None:
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {x.dtype}.")


def _validate_weight(weight: Tensor, hidden: int, name: str) -> None:
    if weight.dim() != 1 or weight.shape[0] != hidden:
        raise ValueError(
            f"{name} must be 1-D of size x.shape[-1]={hidden}, "
            f"got tuple({name}.shape)={tuple(weight.shape)}"
        )


def _centered_norm(x_f: Tensor, weight: Tensor, eps: float) -> Tensor:
    """``x * rstd * (1 + w)`` in FP32. Decoder-norm form, ``rsqrt`` spelling."""
    rstd = torch.rsqrt(shape_invariant_meansq(x_f) + float(eps)).unsqueeze(-1)
    return (x_f * rstd) * (1.0 + weight.float())


def _scaled_norm(x_f: Tensor, weight: Tensor, eps: float) -> Tensor:
    """``x * rstd * w`` in FP32. Final-norm form, ``pow_neg_half`` spelling."""
    rstd = ((shape_invariant_meansq(x_f) + float(eps)) ** -0.5).unsqueeze(-1)
    return (x_f * rstd) * weight.float()


class NativeFusedAddNormOp(nn.Module):
    """Post-norm, residual add, pre-norm, as one operator.

    Returns ``(residual_out, pre_normed)``. ``residual_out`` is the updated
    residual stream and ``pre_normed`` is what the next sublayer consumes.

    Two narrowing points, both matching eager HF rather than being incidental.
    ``weight_applied_before_cast`` puts the post-norm's cast at its last step, so
    the value entering the add is already narrow. The sum is then narrowed again
    before the pre-norm reads it, because eager adds in the model dtype and the
    next norm sees that rounded value. ``forward_fp32`` has neither narrowing,
    which is why it is a different computation rather than a wider one.
    """

    def forward(
        self,
        hidden: Tensor,
        residual: Tensor,
        post_weight: Tensor,
        pre_weight: Tensor,
        *,
        post_eps: float = POST_NORM_EPS,
        pre_eps: float = RMS_NORM_EPS,
    ) -> tuple[Tensor, Tensor]:
        return self._run(
            hidden,
            residual,
            post_weight,
            pre_weight,
            post_eps=post_eps,
            pre_eps=pre_eps,
            output_dtype=hidden.dtype,
        )

    def forward_fp32(
        self,
        hidden: Tensor,
        residual: Tensor,
        post_weight: Tensor,
        pre_weight: Tensor,
        *,
        post_eps: float = POST_NORM_EPS,
        pre_eps: float = RMS_NORM_EPS,
    ) -> tuple[Tensor, Tensor]:
        """Ground truth: same arithmetic, FP32 out, and the add stays in FP32.

        Not simply ``forward`` with a wider output. The narrow cast between the
        post-norm and the add is part of ``forward``'s contract, so removing it
        changes the result; this entry point exists to measure how much.
        """
        return self._run(
            hidden,
            residual,
            post_weight,
            pre_weight,
            post_eps=post_eps,
            pre_eps=pre_eps,
            output_dtype=torch.float32,
        )

    def _run(
        self,
        hidden: Tensor,
        residual: Tensor,
        post_weight: Tensor,
        pre_weight: Tensor,
        *,
        post_eps: float,
        pre_eps: float,
        output_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        _validate_row(hidden, "hidden")
        _validate_row(residual, "residual")
        if hidden.shape != residual.shape:
            raise ValueError(
                f"hidden and residual must share shape, got {tuple(hidden.shape)} "
                f"and {tuple(residual.shape)}."
            )
        if hidden.dtype != residual.dtype:
            raise ValueError(
                f"hidden and residual must share dtype, got {hidden.dtype} "
                f"and {residual.dtype}."
            )
        width = hidden.shape[-1]
        _validate_weight(post_weight, width, "post_weight")
        _validate_weight(pre_weight, width, "pre_weight")

        post = _centered_norm(hidden.float(), post_weight, post_eps)
        # Both casts are load-bearing, see the class docstring. The pre-norm reads
        # the narrowed residual, not the FP32 sum: eager adds in the model dtype
        # and the next norm sees that rounded value, so feeding the unrounded sum
        # forward would quietly disagree with HF on the BF16 path.
        residual_out = (residual.float() + post.to(output_dtype).float()).to(output_dtype)
        pre_normed = _centered_norm(residual_out.float(), pre_weight, pre_eps)
        return residual_out, pre_normed.to(output_dtype)


class NativeFinalNormOp(nn.Module):
    """The model's last norm: ``rmsnorm_times_weight`` with ``pow_neg_half``."""

    def forward(self, x: Tensor, weight: Tensor, *, eps: float = RMS_NORM_EPS) -> Tensor:
        return self._run(x, weight, eps=eps, output_dtype=x.dtype)

    def forward_fp32(self, x: Tensor, weight: Tensor, *, eps: float = RMS_NORM_EPS) -> Tensor:
        return self._run(x, weight, eps=eps, output_dtype=torch.float32)

    def _run(self, x: Tensor, weight: Tensor, *, eps: float, output_dtype: torch.dtype) -> Tensor:
        _validate_row(x, "x")
        _validate_weight(weight, x.shape[-1], "weight")
        return _scaled_norm(x.float(), weight, eps).to(output_dtype)


# --------------------------------------------------------------------------- #
# Closed-form backward
# --------------------------------------------------------------------------- #
# Autograd already differentiates the references above. These functions exist
# because the CUDA and Triton kernels cannot, and they have to agree with autograd
# before that arithmetic gets written twice in two languages. The tests compare
# them against autograd on the reference and against FP64.
#
# For y = x * rstd * g with rstd = (meansq(x) + eps) ** -0.5 and g fixed per column:
#
#     dx_j = rstd * dy_j * g_j - x_j * rstd**3 * c,   c = rowsum(dy * g * x) / n
#     dw_j = sum over rows of dy_j * x_j * rstd
#
# g is (1 + w) for the decoder norms and w for the final norm, and dw is the same
# either way because d(1 + w)/dw is 1. The row reduction in c uses
# shape_invariant_rowsum for the same reason the forward uses the chunked mean: a
# plain sum makes the gradient depend on the batch layout.


def _centered_backward(
    dy_f: Tensor, x_f: Tensor, gain: Tensor, rstd: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(dx, dweight_rows)`` for one norm, all FP32.

    ``gain`` is the full multiplier the forward applied, so ``1 + w`` for a decoder
    norm and ``w`` for the final norm. ``rstd`` is the forward's statistic, passed
    in rather than recomputed so the backward cannot drift from the forward.
    """
    hidden = x_f.shape[-1]
    scaled = dy_f * gain
    c = (shape_invariant_rowsum(scaled * x_f) / float(hidden)).unsqueeze(-1)
    dx = rstd * scaled - x_f * rstd.pow(3) * c
    dweight_rows = dy_f * x_f * rstd
    return dx, dweight_rows


def fused_add_norm_backward(
    grad_residual_out: Tensor,
    grad_pre_normed: Tensor,
    hidden: Tensor,
    residual: Tensor,
    post_weight: Tensor,
    pre_weight: Tensor,
    *,
    post_eps: float = POST_NORM_EPS,
    pre_eps: float = RMS_NORM_EPS,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Gradients of ``NativeFusedAddNormOp.forward_fp32``.

    Returns ``(d_hidden, d_residual, d_post_weight_rows, d_pre_weight_rows)``. The
    two weight entries are per-row FP32 contributions, not sums, matching
    ``parameter_vjp_contributions_fp32`` elsewhere in the repo; the caller folds
    them with ``reduce_rows_fp32``.

    ``residual_out`` is both an output and the pre-norm's input, so the gradient
    reaching it is the sum of what comes back from downstream and what comes back
    through the pre-norm. Getting that wrong is invisible when the downstream
    cotangent is zero, which is why the tests drive both outputs.
    """
    hidden_f = hidden.float()
    residual_f = residual.float()
    post_gain = 1.0 + post_weight.float()
    pre_gain = 1.0 + pre_weight.float()

    # Recompute the forward to recover both statistics and the intermediate.
    rstd_post = torch.rsqrt(shape_invariant_meansq(hidden_f) + float(post_eps)).unsqueeze(-1)
    residual_out = residual_f + (hidden_f * rstd_post) * post_gain
    rstd_pre = torch.rsqrt(shape_invariant_meansq(residual_out) + float(pre_eps)).unsqueeze(-1)

    d_from_pre, d_pre_weight_rows = _centered_backward(
        grad_pre_normed.float(), residual_out, pre_gain, rstd_pre
    )
    # The residual output is live on two paths; add them before going further back.
    d_residual_out = grad_residual_out.float() + d_from_pre

    d_hidden, d_post_weight_rows = _centered_backward(
        d_residual_out, hidden_f, post_gain, rstd_post
    )
    return d_hidden, d_residual_out, d_post_weight_rows, d_pre_weight_rows


def final_norm_backward(
    grad_output: Tensor, x: Tensor, weight: Tensor, *, eps: float = RMS_NORM_EPS
) -> tuple[Tensor, Tensor]:
    """Gradients of ``NativeFinalNormOp.forward_fp32``: ``(dx, dweight_rows)``.

    Separate from the decoder-norm path because the gain is ``w`` rather than
    ``1 + w`` and the statistic is spelled ``pow_neg_half``.
    """
    x_f = x.float()
    rstd = ((shape_invariant_meansq(x_f) + float(eps)) ** -0.5).unsqueeze(-1)
    return _centered_backward(grad_output.float(), x_f, weight.float(), rstd)
