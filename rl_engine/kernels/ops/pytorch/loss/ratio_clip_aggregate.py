# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional, Tuple

import torch


def validate_ratio_clip_inputs(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    penalty_terms: Optional[torch.Tensor],
    clip_low: float,
    clip_high: float,
) -> bool:
    """Validate the shared contract and return whether advantages are per-token."""
    if ratio.ndim not in (1, 2):
        raise ValueError(f"ratio must be 1D or 2D; got shape {tuple(ratio.shape)}.")
    if ratio.numel() == 0:
        raise ValueError("ratio must contain at least one element.")
    if not ratio.is_floating_point() or not advantages.is_floating_point():
        raise TypeError("ratio and advantages must be floating-point tensors.")
    if mask.shape != ratio.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match ratio shape {tuple(ratio.shape)}."
        )
    if mask.is_floating_point() or mask.is_complex():
        raise TypeError("mask must have boolean or integer dtype.")
    if ratio.device != advantages.device or ratio.device != mask.device:
        raise ValueError("ratio, advantages, and mask must be on the same device.")
    if advantages.requires_grad:
        raise ValueError("advantages must be detached RL targets (requires_grad=False).")

    per_token_advantages = advantages.shape == ratio.shape
    per_sequence_advantages = ratio.ndim == 2 and advantages.shape == ratio.shape[:-1]
    if not (per_token_advantages or per_sequence_advantages):
        raise ValueError(
            "advantages must be per-token with ratio.shape or per-sequence "
            f"with ratio.shape[:-1]; got {tuple(advantages.shape)} for "
            f"ratio shape {tuple(ratio.shape)}."
        )

    if penalty_terms is not None:
        if not penalty_terms.is_floating_point():
            raise TypeError("penalty_terms must be a floating-point tensor.")
        if penalty_terms.shape != ratio.shape:
            raise ValueError(
                f"penalty_terms shape {tuple(penalty_terms.shape)} must match "
                f"ratio shape {tuple(ratio.shape)}."
            )
        if penalty_terms.device != ratio.device:
            raise ValueError("penalty_terms must be on the same device as ratio.")

    if not 0.0 <= float(clip_low) < 1.0:
        raise ValueError(f"clip_low must satisfy 0 <= clip_low < 1; got {clip_low}.")
    if float(clip_high) < 0.0:
        raise ValueError(f"clip_high must be non-negative; got {clip_high}.")
    return per_token_advantages


class NativeRatioClipAggregateOp:
    """PyTorch reference for the ratio-clip-aggregate policy loss primitive."""

    def __call__(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        *,
        clip_low: float = 0.2,
        clip_high: float = 0.2,
        penalty_terms: Optional[torch.Tensor] = None,
        penalty_coef: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(
            ratio,
            advantages,
            mask,
            clip_low=clip_low,
            clip_high=clip_high,
            penalty_terms=penalty_terms,
            penalty_coef=penalty_coef,
        )

    def forward(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        *,
        clip_low: float = 0.2,
        clip_high: float = 0.2,
        penalty_terms: Optional[torch.Tensor] = None,
        penalty_coef: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        per_token_advantages = validate_ratio_clip_inputs(
            ratio, advantages, mask, penalty_terms, clip_low, clip_high
        )
        bool_mask = mask.to(torch.bool)
        if per_token_advantages:
            token_advantages = advantages
        else:
            token_advantages = advantages[:, None]
        ratio_fp32 = ratio.float()
        advantages_fp32 = token_advantages.float()
        clipped_ratio = ratio_fp32.clamp(1.0 - clip_low, 1.0 + clip_high)
        policy_terms = -torch.minimum(ratio_fp32 * advantages_fp32, clipped_ratio * advantages_fp32)

        count = bool_mask.sum().to(torch.float32).clamp_min(1.0)
        policy_loss = policy_terms.masked_fill(~bool_mask, 0.0).sum() / count
        if penalty_terms is None:
            mean_penalty = policy_loss.new_zeros(())
        else:
            mean_penalty = penalty_terms.masked_fill(~bool_mask, 0.0).float().sum() / count
        clip_fraction = (
            (ratio_fp32 < 1.0 - clip_low) | (ratio_fp32 > 1.0 + clip_high)
        ).masked_fill(~bool_mask, False).float().sum() / count
        total = policy_loss + float(penalty_coef) * mean_penalty
        return total, policy_loss, mean_penalty, clip_fraction
