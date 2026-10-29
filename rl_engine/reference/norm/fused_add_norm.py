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

from rl_engine.reference.norm.rms_norm import shape_invariant_meansq

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

    The residual add happens in the input dtype, not FP32. That follows from
    ``weight_applied_before_cast``: the post-norm's cast is its last step, so the
    value reaching the add is already narrow, which is also what eager HF does.
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
        # The cast is load-bearing, see the class docstring.
        residual_out = residual.float() + post.to(output_dtype).float()
        pre_normed = _centered_norm(residual_out, pre_weight, pre_eps)
        return residual_out.to(output_dtype), pre_normed.to(output_dtype)


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
