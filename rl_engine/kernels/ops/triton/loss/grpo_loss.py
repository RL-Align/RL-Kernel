# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.triton.loss.ratio_kl import TritonRatioKLOp


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


_MAX_GROUP_SIZE = 1024


def _check_group_block_limit(max_group: int) -> None:
    if max_group > _MAX_GROUP_SIZE:
        raise ValueError(
            f"max group size {max_group} exceeds the Triton GRPO kernel limit of "
            f"{_MAX_GROUP_SIZE}. Reduce samples_per_prompt / group sizes, or use a "
            "tiled reduction kernel for larger groups."
        )


@triton.jit
def _group_norm_kernel(
    rewards_ptr,
    bounds_ptr,  # int32[num_groups + 1], CSR-style group offsets
    adv_ptr,  # float32[N], per-sequence advantages (output)
    eps,
    GROUP_BLOCK: tl.constexpr,
):
    g = tl.program_id(0)
    start = tl.load(bounds_ptr + g)
    end = tl.load(bounds_ptr + g + 1)
    size = end - start

    offs = tl.arange(0, GROUP_BLOCK)
    keep = offs < size
    rewards = tl.load(rewards_ptr + start + offs, mask=keep, other=0.0).to(tl.float32)

    count = (end - start).to(tl.float32)
    mean = tl.sum(rewards, axis=0) / count
    # Population variance (unbiased=False): E[x^2] - E[x]^2. Masked lanes are 0.
    sq_mean = tl.sum(rewards * rewards, axis=0) / count
    std = tl.sqrt(tl.maximum(sq_mean - mean * mean, 0.0))
    std = tl.maximum(std, eps)

    adv = (rewards - mean) / std
    tl.store(adv_ptr + start + offs, adv, mask=keep)


class TritonGRPOLossOp:
    """Triton fused GRPO loss op.

    The per-token ``policy_ratio`` / ``kl_penalty`` are produced by the fused
    ``ratio_kl`` Triton kernel (logits -> ratio/KL via online softmax, with an
    analytic backward into ``policy_logits``); reward normalization runs in the
    ``_group_norm_kernel``; the clipped surrogate + reference-KL reduction are a
    thin autograd-friendly PyTorch layer on top. ``forward`` takes raw rewards;
    ``apply`` takes the per-sequence advantage vector directly.
    """

    def __init__(self) -> None:
        self._ratio_kl = TritonRatioKLOp()

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

    @staticmethod
    def _build_bounds(
        num_sequences: int,
        device: torch.device,
        samples_per_prompt: Optional[int],
        group_boundaries: Optional[Sequence[int] | torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Return CSR-style group offsets (int32) and the max group size."""
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
            _check_group_block_limit(samples_per_prompt)
            bounds = torch.arange(
                0, num_sequences + 1, samples_per_prompt, device=device, dtype=torch.int32
            )
            return bounds, samples_per_prompt

        bounds = torch.as_tensor(group_boundaries, device=device, dtype=torch.int32)
        if bounds.ndim != 1 or bounds.numel() < 2:
            raise ValueError("group_boundaries must be a 1D tensor of length num_groups + 1.")
        sizes = bounds[1:] - bounds[:-1]
        if int(bounds[0].item()) != 0 or int(bounds[-1].item()) != num_sequences:
            raise ValueError("group_boundaries must start at 0 and end at num_sequences.")
        if bool((sizes < 1).any().item()):
            raise ValueError("each group must contain at least one sequence.")
        max_group = int(sizes.max().item())
        _check_group_block_limit(max_group)
        return bounds, max_group

    def group_advantages(
        self,
        rewards: torch.Tensor,
        *,
        samples_per_prompt: Optional[int] = None,
        group_boundaries: Optional[Sequence[int] | torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Per-sequence reward normalization, computed by the Triton group kernel."""
        if not rewards.is_cuda:
            raise RuntimeError("TritonGRPOLossOp requires CUDA tensors.")
        flat = rewards.reshape(-1).to(torch.float32)
        n = flat.numel()
        bounds, max_group = self._build_bounds(n, flat.device, samples_per_prompt, group_boundaries)
        num_groups = bounds.numel() - 1
        adv = torch.empty(n, device=flat.device, dtype=torch.float32)

        # TODO: for larger groups, implement a tiled reduction version of the
        # kernel that can handle >1024 sequences per group.
        _group_norm_kernel[(num_groups,)](
            flat,
            bounds,
            adv,
            float(eps),
            GROUP_BLOCK=_next_pow2(max_group),
        )
        return adv

    @staticmethod
    def expand_advantages(
        sample_advantages: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast per-sequence advantages to per-token, zeroing masked tokens."""
        bool_mask = completion_mask.bool()
        expanded = sample_advantages.reshape(-1, 1).expand_as(bool_mask).clone()
        return expanded.masked_fill(~bool_mask, 0.0)

    @staticmethod
    def _masked_mean(
        values: torch.Tensor, bool_mask: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        masked = values.masked_fill(~bool_mask, 0.0)
        denom = bool_mask.sum().to(values.dtype).clamp_min(eps)
        return masked.sum() / denom

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
        """Evaluate the loss from logits + per-sequence advantages."""
        if not policy_logits.is_cuda:
            raise RuntimeError("TritonGRPOLossOp requires CUDA tensors.")
        if completion_mask.ndim != 2:
            raise ValueError("completion_mask must be 2D [num_sequences, completion_len].")

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
