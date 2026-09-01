# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fused ratio clipping, masking, and deterministic loss aggregation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.pytorch.loss.ratio_clip_aggregate import validate_ratio_clip_inputs

_BLOCK = 256
_NUM_WARPS = 4
_SINGLE_PASS_TILE = 8192
_SINGLE_PASS_MAX = 65536
_SINGLE_PASS_NUM_WARPS = 8
_MAX_PARTIALS = 65536


@triton.jit
def _ratio_clip_single_pass_kernel(
    ratio_ptr,
    advantage_ptr,
    mask_ptr,
    penalty_ptr,
    total_ptr,
    policy_ptr,
    mean_penalty_ptr,
    clip_fraction_ptr,
    active_count_ptr,
    n_elements,
    tokens_per_sequence,
    clip_low,
    clip_high,
    penalty_coef,
    HAS_PENALTY: tl.constexpr,
    ADV_PER_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    lower = 1.0 - clip_low
    upper = 1.0 + clip_high
    lane_offsets = tl.arange(0, BLOCK)
    policy_sum = 0.0
    penalty_sum = 0.0
    clipped_sum = 0.0
    active_count = 0.0

    for tile_id in range(NUM_TILES):
        offsets = tile_id * BLOCK + lane_offsets
        in_bounds = offsets < n_elements
        active = in_bounds & (tl.load(mask_ptr + offsets, mask=in_bounds, other=0) != 0)

        ratio = tl.load(ratio_ptr + offsets, mask=in_bounds, other=1.0).to(tl.float32)
        if ADV_PER_TOKEN:
            advantage = tl.load(advantage_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)
        else:
            sequence_ids = offsets // tokens_per_sequence
            advantage = tl.load(advantage_ptr + sequence_ids, mask=in_bounds, other=0.0).to(
                tl.float32
            )

        clipped_ratio = tl.minimum(tl.maximum(ratio, lower), upper)
        policy_term = -tl.minimum(ratio * advantage, clipped_ratio * advantage)
        policy_sum += tl.sum(tl.where(active, policy_term, 0.0), axis=0)

        if HAS_PENALTY:
            penalty = tl.load(penalty_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)
            penalty_sum += tl.sum(tl.where(active, penalty, 0.0), axis=0)

        clipped_sum += tl.sum(
            (((ratio < lower) | (ratio > upper)) & active).to(tl.float32),
            axis=0,
        )
        active_count += tl.sum(active.to(tl.float32), axis=0)

    denominator = tl.maximum(active_count, 1.0)
    policy_loss = policy_sum / denominator
    mean_penalty = penalty_sum / denominator

    tl.store(total_ptr, policy_loss + penalty_coef * mean_penalty)
    tl.store(policy_ptr, policy_loss)
    tl.store(mean_penalty_ptr, mean_penalty)
    tl.store(clip_fraction_ptr, clipped_sum / denominator)
    tl.store(active_count_ptr, active_count)


@triton.jit
def _ratio_clip_partial_kernel(
    ratio_ptr,
    advantage_ptr,
    mask_ptr,
    penalty_ptr,
    partials_ptr,
    n_elements,
    partial_count,
    tokens_per_sequence,
    clip_low,
    clip_high,
    HAS_PENALTY: tl.constexpr,
    ADV_PER_TOKEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < n_elements
    active = in_bounds & (tl.load(mask_ptr + offsets, mask=in_bounds, other=0) != 0)

    ratio = tl.load(ratio_ptr + offsets, mask=in_bounds, other=1.0).to(tl.float32)
    if ADV_PER_TOKEN:
        advantage = tl.load(advantage_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)
    else:
        sequence_ids = offsets // tokens_per_sequence
        advantage = tl.load(advantage_ptr + sequence_ids, mask=in_bounds, other=0.0).to(tl.float32)

    lower = 1.0 - clip_low
    upper = 1.0 + clip_high
    clipped_ratio = tl.minimum(tl.maximum(ratio, lower), upper)
    policy_term = -tl.minimum(ratio * advantage, clipped_ratio * advantage)
    policy_term = tl.where(active, policy_term, 0.0)

    if HAS_PENALTY:
        penalty = tl.load(penalty_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)
        penalty = tl.where(active, penalty, 0.0)
    else:
        penalty = tl.zeros((BLOCK,), dtype=tl.float32)

    clipped = ((ratio < lower) | (ratio > upper)) & active
    count = active.to(tl.float32)
    tl.store(partials_ptr + block_id, tl.sum(policy_term, axis=0))
    tl.store(partials_ptr + partial_count + block_id, tl.sum(penalty, axis=0))
    tl.store(
        partials_ptr + 2 * partial_count + block_id,
        tl.sum(clipped.to(tl.float32), axis=0),
    )
    tl.store(partials_ptr + 3 * partial_count + block_id, tl.sum(count, axis=0))


@triton.jit
def _ratio_clip_finalize_kernel(
    partials_ptr,
    total_ptr,
    policy_ptr,
    mean_penalty_ptr,
    clip_fraction_ptr,
    active_count_ptr,
    partial_count,
    penalty_coef,
    REDUCE_BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, REDUCE_BLOCK)
    keep = offsets < partial_count
    policy_sum = tl.sum(tl.load(partials_ptr + offsets, mask=keep, other=0.0), axis=0)
    penalty_sum = tl.sum(
        tl.load(partials_ptr + partial_count + offsets, mask=keep, other=0.0),
        axis=0,
    )
    clipped_sum = tl.sum(
        tl.load(partials_ptr + 2 * partial_count + offsets, mask=keep, other=0.0),
        axis=0,
    )
    active_count = tl.sum(
        tl.load(partials_ptr + 3 * partial_count + offsets, mask=keep, other=0.0),
        axis=0,
    )
    denominator = tl.maximum(active_count, 1.0)
    policy_loss = policy_sum / denominator
    mean_penalty = penalty_sum / denominator

    tl.store(total_ptr, policy_loss + penalty_coef * mean_penalty)
    tl.store(policy_ptr, policy_loss)
    tl.store(mean_penalty_ptr, mean_penalty)
    tl.store(clip_fraction_ptr, clipped_sum / denominator)
    tl.store(active_count_ptr, active_count)


@triton.jit
def _ratio_clip_backward_kernel(
    ratio_ptr,
    advantage_ptr,
    mask_ptr,
    active_count_ptr,
    grad_total_ptr,
    grad_policy_ptr,
    grad_mean_penalty_ptr,
    grad_ratio_ptr,
    grad_penalty_ptr,
    n_elements,
    tokens_per_sequence,
    clip_low,
    clip_high,
    penalty_coef,
    HAS_PENALTY: tl.constexpr,
    ADV_PER_TOKEN: tl.constexpr,
    HAS_GRAD_TOTAL: tl.constexpr,
    HAS_GRAD_POLICY: tl.constexpr,
    HAS_GRAD_MEAN_PENALTY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    in_bounds = offsets < n_elements
    active = in_bounds & (tl.load(mask_ptr + offsets, mask=in_bounds, other=0) != 0)

    ratio = tl.load(ratio_ptr + offsets, mask=in_bounds, other=1.0).to(tl.float32)
    if ADV_PER_TOKEN:
        advantage = tl.load(advantage_ptr + offsets, mask=in_bounds, other=0.0).to(tl.float32)
    else:
        sequence_ids = offsets // tokens_per_sequence
        advantage = tl.load(advantage_ptr + sequence_ids, mask=in_bounds, other=0.0).to(tl.float32)

    lower = 1.0 - clip_low
    upper = 1.0 + clip_high
    clipped_ratio = tl.minimum(tl.maximum(ratio, lower), upper)
    unclipped_term = ratio * advantage
    clipped_term = clipped_ratio * advantage
    use_unclipped = unclipped_term <= clipped_term
    clipped_derivative = tl.where((ratio >= lower) & (ratio <= upper), advantage, 0.0)
    policy_derivative = -tl.where(use_unclipped, advantage, clipped_derivative)

    active_count = tl.load(active_count_ptr)
    denominator = tl.maximum(active_count, 1.0)
    if HAS_GRAD_TOTAL:
        grad_total = tl.load(grad_total_ptr)
    else:
        grad_total = 0.0
    if HAS_GRAD_POLICY:
        grad_policy_output = tl.load(grad_policy_ptr)
    else:
        grad_policy_output = 0.0
    policy_scale = (grad_total + grad_policy_output) / denominator
    grad_ratio = tl.where(active, policy_scale * policy_derivative, 0.0)
    tl.store(grad_ratio_ptr + offsets, grad_ratio, mask=in_bounds)

    if HAS_PENALTY:
        if HAS_GRAD_MEAN_PENALTY:
            grad_mean_penalty = tl.load(grad_mean_penalty_ptr)
        else:
            grad_mean_penalty = 0.0
        penalty_scale = (grad_total * penalty_coef + grad_mean_penalty) / denominator
        grad_penalty = tl.where(active, penalty_scale, 0.0)
        tl.store(grad_penalty_ptr + offsets, grad_penalty, mask=in_bounds)


class _RatioClipAggregateFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ratio,
        advantages,
        mask,
        penalty_terms,
        clip_low,
        clip_high,
        penalty_coef,
    ):
        if not ratio.is_cuda:
            raise RuntimeError("TritonRatioClipAggregateOp requires CUDA/ROCm tensors.")
        per_token_advantages = validate_ratio_clip_inputs(
            ratio, advantages, mask, penalty_terms, clip_low, clip_high
        )

        ratio_flat = ratio.contiguous().view(-1)
        advantages_flat = advantages.contiguous().view(-1)
        mask_flat = mask.contiguous().view(-1)
        has_penalty = penalty_terms is not None
        penalty_flat = penalty_terms.contiguous().view(-1) if has_penalty else ratio_flat
        n_elements = ratio_flat.numel()
        if n_elements == 0:
            raise ValueError("ratio must contain at least one element.")

        needs_backward = ratio.requires_grad or (
            penalty_terms is not None and penalty_terms.requires_grad
        )
        if needs_backward:
            # Independent storages prevent an in-place update to one public
            # scalar from invalidating a sibling saved for backward.
            total = torch.empty((), device=ratio.device, dtype=torch.float32)
            policy = torch.empty((), device=ratio.device, dtype=torch.float32)
            mean_penalty = torch.empty((), device=ratio.device, dtype=torch.float32)
            clip_fraction = torch.empty((), device=ratio.device, dtype=torch.float32)
            active_count = torch.empty((), device=ratio.device, dtype=torch.float32)
        else:
            # Avoid five allocator calls on the latency-sensitive inference path.
            packed_outputs = torch.empty(5, device=ratio.device, dtype=torch.float32)
            total, policy, mean_penalty, clip_fraction, active_count = packed_outputs.unbind()
        tokens_per_sequence = ratio.shape[-1] if ratio.ndim == 2 else n_elements

        if n_elements <= _SINGLE_PASS_MAX:
            single_block = min(_SINGLE_PASS_TILE, triton.next_power_of_2(n_elements))
            num_tiles = triton.cdiv(n_elements, single_block)
            _ratio_clip_single_pass_kernel[(1,)](
                ratio_flat,
                advantages_flat,
                mask_flat,
                penalty_flat,
                total,
                policy,
                mean_penalty,
                clip_fraction,
                active_count,
                n_elements,
                tokens_per_sequence,
                float(clip_low),
                float(clip_high),
                float(penalty_coef),
                HAS_PENALTY=has_penalty,
                ADV_PER_TOKEN=per_token_advantages,
                BLOCK=single_block,
                NUM_TILES=num_tiles,
                num_warps=_SINGLE_PASS_NUM_WARPS,
            )
        else:
            partial_count = triton.cdiv(n_elements, _BLOCK)
            if partial_count > _MAX_PARTIALS:
                raise ValueError(
                    f"ratio has {n_elements} elements, exceeding the deterministic "
                    f"reduction limit of {_MAX_PARTIALS * _BLOCK}."
                )
            partials = torch.empty((4, partial_count), device=ratio.device, dtype=torch.float32)
            _ratio_clip_partial_kernel[(partial_count,)](
                ratio_flat,
                advantages_flat,
                mask_flat,
                penalty_flat,
                partials,
                n_elements,
                partial_count,
                tokens_per_sequence,
                float(clip_low),
                float(clip_high),
                HAS_PENALTY=has_penalty,
                ADV_PER_TOKEN=per_token_advantages,
                BLOCK=_BLOCK,
                num_warps=_NUM_WARPS,
            )
            reduce_block = triton.next_power_of_2(partial_count)
            _ratio_clip_finalize_kernel[(1,)](
                partials,
                total,
                policy,
                mean_penalty,
                clip_fraction,
                active_count,
                partial_count,
                float(penalty_coef),
                REDUCE_BLOCK=reduce_block,
                num_warps=min(8, max(1, reduce_block // 128)),
            )

        ctx.save_for_backward(ratio_flat, advantages_flat, mask_flat, active_count)
        ctx.has_penalty = has_penalty
        ctx.per_token_advantages = per_token_advantages
        ctx.tokens_per_sequence = tokens_per_sequence
        ctx.clip_low = float(clip_low)
        ctx.clip_high = float(clip_high)
        ctx.penalty_coef = float(penalty_coef)
        ctx.ratio_shape = tuple(ratio.shape)
        ctx.ratio_dtype = ratio.dtype
        ctx.penalty_shape = tuple(penalty_terms.shape) if penalty_terms is not None else None
        ctx.penalty_dtype = penalty_terms.dtype if penalty_terms is not None else None

        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(clip_fraction)
        return total, policy, mean_penalty, clip_fraction

    @staticmethod
    def backward(ctx, grad_total, grad_policy, grad_mean_penalty, grad_clip_fraction):
        ratio, advantages, mask, active_count = ctx.saved_tensors
        grad_ratio = torch.empty_like(ratio)
        empty_placeholder = ratio.new_empty(0)
        grad_penalty = (
            torch.empty(ctx.penalty_shape, device=ratio.device, dtype=ctx.penalty_dtype).view(-1)
            if ctx.has_penalty
            else empty_placeholder
        )

        has_grad_total = grad_total is not None
        has_grad_policy = grad_policy is not None
        has_grad_mean_penalty = grad_mean_penalty is not None
        grad_total = empty_placeholder if grad_total is None else grad_total.contiguous()
        grad_policy = empty_placeholder if grad_policy is None else grad_policy.contiguous()
        grad_mean_penalty = (
            empty_placeholder if grad_mean_penalty is None else grad_mean_penalty.contiguous()
        )
        grid = (triton.cdiv(ratio.numel(), _BLOCK),)
        _ratio_clip_backward_kernel[grid](
            ratio,
            advantages,
            mask,
            active_count,
            grad_total,
            grad_policy,
            grad_mean_penalty,
            grad_ratio,
            grad_penalty,
            ratio.numel(),
            ctx.tokens_per_sequence,
            ctx.clip_low,
            ctx.clip_high,
            ctx.penalty_coef,
            HAS_PENALTY=ctx.has_penalty,
            ADV_PER_TOKEN=ctx.per_token_advantages,
            HAS_GRAD_TOTAL=has_grad_total,
            HAS_GRAD_POLICY=has_grad_policy,
            HAS_GRAD_MEAN_PENALTY=has_grad_mean_penalty,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )

        grad_ratio = grad_ratio.view(ctx.ratio_shape).to(ctx.ratio_dtype)
        if ctx.has_penalty:
            grad_penalty = grad_penalty.view(ctx.penalty_shape).to(ctx.penalty_dtype)
        else:
            grad_penalty = None
        return grad_ratio, None, None, grad_penalty, None, None, None


class TritonRatioClipAggregateOp:
    """Triton ratio-clip-aggregate operator for PPO/GRPO policy losses."""

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
        return _RatioClipAggregateFunction.apply(
            ratio,
            advantages,
            mask,
            penalty_terms,
            clip_low,
            clip_high,
            penalty_coef,
        )
