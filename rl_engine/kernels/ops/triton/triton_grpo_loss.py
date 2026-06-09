# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

_BLOCK = 1024


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


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


@triton.jit
def _grpo_fwd_kernel(
    cur_ptr,
    old_ptr,
    ref_ptr,
    adv_seq_ptr,  # float32[B], per-sequence advantages
    mask_ptr,
    partials_ptr,  # float32[2]: (policy_sum, kl_sum)
    n_elements,
    T,  # completion length (tokens per sequence)
    clip_eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    bound = offs < n_elements
    seq_id = offs // T

    cur = tl.load(cur_ptr + offs, mask=bound, other=0.0).to(tl.float32)
    old = tl.load(old_ptr + offs, mask=bound, other=0.0).to(tl.float32)
    ref = tl.load(ref_ptr + offs, mask=bound, other=0.0).to(tl.float32)
    adv = tl.load(adv_seq_ptr + seq_id, mask=bound, other=0.0).to(tl.float32)
    active = tl.load(mask_ptr + offs, mask=bound, other=0).to(tl.float32)
    keep = bound & (active != 0.0)

    ratio = tl.exp(cur - old)
    lo = 1.0 - clip_eps
    hi = 1.0 + clip_eps
    unclipped = ratio * adv
    clipped = tl.minimum(tl.maximum(ratio, lo), hi) * adv
    policy_term = -tl.minimum(unclipped, clipped)

    diff = ref - cur
    kl_term = tl.exp(diff) - diff - 1.0

    policy_term = tl.where(keep, policy_term, 0.0)
    kl_term = tl.where(keep, kl_term, 0.0)

    tl.atomic_add(partials_ptr + 0, tl.sum(policy_term, axis=0))
    tl.atomic_add(partials_ptr + 1, tl.sum(kl_term, axis=0))


@triton.jit
def _grpo_bwd_kernel(
    cur_ptr,
    old_ptr,
    ref_ptr,
    adv_seq_ptr,
    mask_ptr,
    grad_cur_ptr,
    scale,  # grad_output / num_active
    beta,
    clip_eps,
    n_elements,
    T,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    bound = offs < n_elements
    seq_id = offs // T

    cur = tl.load(cur_ptr + offs, mask=bound, other=0.0).to(tl.float32)
    old = tl.load(old_ptr + offs, mask=bound, other=0.0).to(tl.float32)
    ref = tl.load(ref_ptr + offs, mask=bound, other=0.0).to(tl.float32)
    adv = tl.load(adv_seq_ptr + seq_id, mask=bound, other=0.0).to(tl.float32)
    active = tl.load(mask_ptr + offs, mask=bound, other=0).to(tl.float32)
    keep = bound & (active != 0.0)

    ratio = tl.exp(cur - old)
    lo = 1.0 - clip_eps
    hi = 1.0 + clip_eps
    unclipped = ratio * adv
    clipped = tl.minimum(tl.maximum(ratio, lo), hi) * adv

    # d(ratio)/d(cur) = ratio. The surrogate selects the smaller branch; the
    # clamped branch has zero gradient outside (lo, hi).
    in_range = (ratio > lo) & (ratio < hi)
    d_clipped = tl.where(in_range, ratio * adv, 0.0)
    sel_unclipped = unclipped <= clipped
    deriv_sel = tl.where(sel_unclipped, ratio * adv, d_clipped)
    d_policy = -deriv_sel

    # d(kl)/d(cur) for kl = exp(ref - cur) - (ref - cur) - 1.
    d_kl = 1.0 - tl.exp(ref - cur)

    grad = scale * (d_policy + beta * d_kl)
    grad = tl.where(keep, grad, 0.0)
    tl.store(grad_cur_ptr + offs, grad, mask=bound)


class _GRPOLossFunction(torch.autograd.Function):
    """Autograd wrapper around the token-parallel Triton kernels."""

    @staticmethod
    def forward(
        ctx, current_logps, old_logps, ref_logps, adv_seq, mask, completion_len, clip_eps, beta
    ):
        cur = current_logps.contiguous()
        old = old_logps.contiguous().to(cur.dtype)
        ref = ref_logps.contiguous().to(cur.dtype)
        adv = adv_seq.contiguous().to(torch.float32)
        mask_f = mask.contiguous().to(torch.float32)

        n = cur.numel()
        num_active = mask_f.sum().clamp_min(1e-8)
        partials = torch.zeros(2, device=cur.device, dtype=torch.float32)

        grid = (triton.cdiv(n, _BLOCK),)
        _grpo_fwd_kernel[grid](
            cur.reshape(-1),
            old.reshape(-1),
            ref.reshape(-1),
            adv,
            mask_f.reshape(-1),
            partials,
            n,
            int(completion_len),
            float(clip_eps),
            BLOCK=_BLOCK,
        )

        policy_loss = partials[0] / num_active
        kl = partials[1] / num_active
        loss = policy_loss + beta * kl

        ctx.save_for_backward(cur, old, ref, adv, mask_f)
        ctx.clip_eps = float(clip_eps)
        ctx.beta = float(beta)
        ctx.completion_len = int(completion_len)
        ctx.num_active = num_active
        ctx.mark_non_differentiable(policy_loss, kl)
        return loss, policy_loss, kl

    @staticmethod
    def backward(ctx, grad_loss, grad_policy, grad_kl):
        cur, old, ref, adv, mask_f = ctx.saved_tensors
        n = cur.numel()
        grad_cur = torch.empty_like(cur, dtype=torch.float32)
        scale = float((grad_loss / ctx.num_active).item())

        grid = (triton.cdiv(n, _BLOCK),)
        _grpo_bwd_kernel[grid](
            cur.reshape(-1),
            old.reshape(-1),
            ref.reshape(-1),
            adv,
            mask_f.reshape(-1),
            grad_cur.reshape(-1),
            scale,
            ctx.beta,
            ctx.clip_eps,
            n,
            ctx.completion_len,
            BLOCK=_BLOCK,
        )

        grad_cur = grad_cur.to(cur.dtype)
        # Inputs: current, old, ref, adv_seq, mask, completion_len, clip_eps, beta.
        return grad_cur, None, None, None, None, None, None, None


class TritonGRPOLossOp:
    """Triton fused GRPO loss op.

    ``forward`` is the drop-in equivalent of ``NativeGRPOLossOp.forward`` (raw
    rewards in, scalar loss out). ``apply`` takes the per-sequence advantage
    vector directly (the fused representation), not a per-token tensor.
    """

    def __call__(
        self,
        current_logps: torch.Tensor,
        old_logps: torch.Tensor,
        ref_logps: torch.Tensor,
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
            current_logps,
            old_logps,
            ref_logps,
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
        return bounds, int(sizes.max().item())

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
        _group_norm_kernel[(num_groups,)](
            flat,
            bounds,
            adv,
            float(eps),
            GROUP_BLOCK=_next_pow2(max_group),
        )
        return adv

    def apply(
        self,
        current_logps: torch.Tensor,
        old_logps: torch.Tensor,
        ref_logps: torch.Tensor,
        sample_advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        clip_eps: float = 0.2,
        beta: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the loss from per-sequence advantages (gathered per token)."""
        if not current_logps.is_cuda:
            raise RuntimeError("TritonGRPOLossOp requires CUDA tensors.")
        if completion_mask.ndim != 2:
            raise ValueError("completion_mask must be 2D [num_sequences, completion_len].")
        completion_len = completion_mask.shape[1]
        return _GRPOLossFunction.apply(
            current_logps,
            old_logps,
            ref_logps,
            sample_advantages,
            completion_mask,
            completion_len,
            clip_eps,
            beta,
        )

    def forward(
        self,
        current_logps: torch.Tensor,
        old_logps: torch.Tensor,
        ref_logps: torch.Tensor,
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
            current_logps,
            old_logps,
            ref_logps,
            sample_advantages,
            completion_mask,
            clip_eps=clip_eps,
            beta=beta,
        )
