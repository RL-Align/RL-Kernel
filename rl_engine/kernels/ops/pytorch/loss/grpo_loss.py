# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from rl_engine.kernels.ops.pytorch.loss.ratio_kl import NativeRatioKLOp


class NativeGRPOLossOp:
    """Pure PyTorch native fallback for the fused GRPO loss.

    Consumes logits directly: the per-token ``policy_ratio`` / ``kl_penalty`` come
    from the fused ratio/KL op, and the group-normalized advantages + clipped
    surrogate + reference-KL reduction are applied on top.
    """

    def __init__(self) -> None:
        self._ratio_kl = NativeRatioKLOp()

    def __call__(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_ids: torch.Tensor,
        old_logps: torch.Tensor,
        rewards: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        clip_eps: float = 0.2,
        beta: float = 0.0,
        samples_per_prompt: Optional[int] = None,
        group_boundaries: Optional[Sequence[int] | torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(
            policy_logits,
            ref_logits,
            action_ids,
            old_logps,
            rewards,
            completion_mask,
            clip_eps=clip_eps,
            beta=beta,
            samples_per_prompt=samples_per_prompt,
            group_boundaries=group_boundaries,
            eps=eps,
        )

    def group_advantages(
        self,
        rewards: torch.Tensor,
        *,
        samples_per_prompt: Optional[int] = None,
        group_boundaries: Optional[Sequence[int] | torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Normalize raw per-sequence rewards within each generation group."""
        flat_rewards = rewards.reshape(-1).float()
        num_sequences = flat_rewards.numel()
        group_id = self._resolve_group_ids(
            num_sequences,
            device=flat_rewards.device,
            samples_per_prompt=samples_per_prompt,
            group_boundaries=group_boundaries,
        )

        num_groups = int(group_id.max().item()) + 1 if num_sequences else 0
        counts = flat_rewards.new_zeros(num_groups).index_add_(
            0, group_id, torch.ones_like(flat_rewards)
        )
        sums = flat_rewards.new_zeros(num_groups).index_add_(0, group_id, flat_rewards)
        sq_sums = flat_rewards.new_zeros(num_groups).index_add_(
            0, group_id, flat_rewards * flat_rewards
        )

        means = sums / counts
        variance = (sq_sums / counts) - means * means
        stds = variance.clamp_min(eps**2).sqrt()

        return (flat_rewards - means[group_id]) / stds[group_id]

    @staticmethod
    def expand_advantages(
        sample_advantages: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast per-sequence advantages to per-token, zeroing masked tokens."""
        bool_mask = completion_mask.bool()
        expanded = sample_advantages.reshape(-1, 1).expand_as(bool_mask).clone()
        return expanded.masked_fill(~bool_mask, 0.0)

    def apply(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_ids: torch.Tensor,
        old_logps: torch.Tensor,
        sample_advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        clip_eps: float = 0.2,
        beta: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ``(loss, policy_loss, kl)`` from logits + per-sequence advantages.

        ``sample_advantages`` is per-sequence and is broadcast to per-token here.
        The ratio and KL come from the fused ratio/KL op; gradients flow into
        ``policy_logits``.
        """
        ratio, kl_terms = self._ratio_kl(
            policy_logits, ref_logits, action_ids, completion_mask, old_logps
        )
        bool_mask = completion_mask.bool()
        adv = self.expand_advantages(sample_advantages, completion_mask).float()
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        policy_loss_terms = -torch.minimum(unclipped, clipped)

        policy_loss = self._masked_mean(policy_loss_terms, bool_mask)
        kl = self._masked_mean(kl_terms, bool_mask)
        return policy_loss + beta * kl, policy_loss, kl

    def forward(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        action_ids: torch.Tensor,
        old_logps: torch.Tensor,
        rewards: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        clip_eps: float = 0.2,
        beta: float = 0.0,
        samples_per_prompt: Optional[int] = None,
        group_boundaries: Optional[Sequence[int] | torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_advantages = self.group_advantages(
            rewards,
            samples_per_prompt=samples_per_prompt,
            group_boundaries=group_boundaries,
            eps=eps,
        )
        return self.apply(
            policy_logits,
            ref_logits,
            action_ids,
            old_logps,
            sample_advantages,
            completion_mask,
            clip_eps=clip_eps,
            beta=beta,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _masked_mean(
        values: torch.Tensor, bool_mask: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        masked = values.masked_fill(~bool_mask, 0.0)
        denom = bool_mask.sum().to(values.dtype).clamp_min(eps)
        return masked.sum() / denom

    @staticmethod
    def _resolve_group_ids(
        num_sequences: int,
        *,
        device: torch.device,
        samples_per_prompt: Optional[int],
        group_boundaries: Optional[Sequence[int] | torch.Tensor],
    ) -> torch.Tensor:
        provided = [spec is not None for spec in (samples_per_prompt, group_boundaries)]
        if sum(provided) != 1:
            raise ValueError("Provide exactly one of samples_per_prompt or group_boundaries.")

        if samples_per_prompt is not None:
            if samples_per_prompt < 2:
                raise ValueError("samples_per_prompt must be at least 2 for group normalization.")
            if num_sequences % samples_per_prompt != 0:
                raise ValueError(
                    f"num_sequences ({num_sequences}) must be divisible by "
                    f"samples_per_prompt ({samples_per_prompt})."
                )
            return torch.arange(num_sequences, device=device) // samples_per_prompt

        boundaries = torch.as_tensor(group_boundaries, device=device, dtype=torch.long)
        if boundaries.ndim != 1 or boundaries.numel() < 2:
            raise ValueError("group_boundaries must be a 1D tensor of length num_groups + 1.")
        sizes = boundaries[1:] - boundaries[:-1]

        if int(sizes.sum().item()) != num_sequences:
            raise ValueError(
                f"group sizes sum to {int(sizes.sum().item())} but there are "
                f"{num_sequences} sequences."
            )
        if bool((sizes < 1).any().item()):
            raise ValueError("each group must contain at least one sequence.")

        group_index = torch.arange(sizes.numel(), device=device)
        return torch.repeat_interleave(group_index, sizes)
