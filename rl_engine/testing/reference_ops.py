# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _bool_mask(mask: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    return mask.to(device=device, dtype=torch.bool)


def vocab_shard_ranges(vocab_size: int, tp_size: int) -> list[tuple[int, int]]:
    """Return contiguous vocab shard ranges with uneven tails distributed first."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than zero")
    if tp_size <= 0:
        raise ValueError("tp_size must be greater than zero")
    if tp_size > vocab_size:
        raise ValueError("tp_size must be less than or equal to vocab_size")

    base = vocab_size // tp_size
    remainder = vocab_size % tp_size
    ranges: list[tuple[int, int]] = []
    start = 0
    for rank in range(tp_size):
        shard_size = base + (1 if rank < remainder else 0)
        end = start + shard_size
        ranges.append((start, end))
        start = end
    return ranges


def shard_logits_by_vocab(logits: torch.Tensor, tp_size: int) -> list[torch.Tensor]:
    """Split full logits into contiguous vocab shards for simulated TP tests."""

    ranges = vocab_shard_ranges(int(logits.size(-1)), tp_size)
    return [logits[..., start:end] for start, end in ranges]


def _resolve_vocab_start_indices(
    logit_shards: Sequence[torch.Tensor],
    vocab_start_indices: Sequence[int] | None,
) -> list[int]:
    if not logit_shards:
        raise ValueError("logit_shards must contain at least one shard")
    if vocab_start_indices is None:
        starts: list[int] = []
        cursor = 0
        for shard in logit_shards:
            starts.append(cursor)
            cursor += int(shard.size(-1))
        return starts

    starts = [int(start) for start in vocab_start_indices]
    if len(starts) != len(logit_shards):
        raise ValueError("vocab_start_indices length must match logit_shards")
    if any(start < 0 for start in starts):
        raise ValueError("vocab_start_indices must be non-negative")
    ranges = [
        (start, start + int(shard.size(-1)))
        for start, shard in zip(starts, logit_shards, strict=True)
    ]
    for (prev_start, prev_end), (next_start, _next_end) in zip(
        ranges,
        ranges[1:],
        strict=False,
    ):
        if next_start < prev_end or next_start < prev_start:
            raise ValueError("vocab_start_indices must define non-overlapping sorted shards")
    return starts


def _validate_logit_shards(
    logit_shards: Sequence[torch.Tensor],
    token_ids: torch.Tensor,
) -> tuple[torch.device, torch.Size]:
    if not logit_shards:
        raise ValueError("logit_shards must contain at least one shard")

    first = logit_shards[0]
    if first.ndim < 1:
        raise ValueError("each logit shard must have at least one dimension")
    if first.size(-1) <= 0:
        raise ValueError("logit shards must have non-empty vocab dimensions")

    device = first.device
    leading_shape = first.shape[:-1]
    if leading_shape != token_ids.shape:
        raise ValueError(
            f"logit shard leading shape {tuple(leading_shape)} must match "
            f"token_ids shape {tuple(token_ids.shape)}"
        )

    for shard in logit_shards[1:]:
        if shard.device != device:
            raise ValueError("all logit shards must be on the same device")
        if shard.shape[:-1] != leading_shape:
            raise ValueError("all logit shards must have the same leading shape")
        if shard.size(-1) <= 0:
            raise ValueError("logit shards must have non-empty vocab dimensions")

    return device, leading_shape


def owner_ranks_for_token_ids(
    token_ids: torch.Tensor,
    shard_ranges: Sequence[tuple[int, int]],
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map global token ids to owning TP rank, using -1 for inactive or uncovered ids."""

    owners = torch.full(token_ids.shape, -1, device=token_ids.device, dtype=torch.long)
    active = torch.ones_like(token_ids, dtype=torch.bool)
    if mask is not None:
        if mask.shape != token_ids.shape:
            raise ValueError("mask shape must match token_ids shape")
        active = _bool_mask(mask, device=token_ids.device)

    for rank, (start, end) in enumerate(shard_ranges):
        owns = active & (token_ids >= int(start)) & (token_ids < int(end))
        owners = torch.where(owns, torch.full_like(owners, rank), owners)
    return owners


def selected_logprobs_reference(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference selected-token logprobs for RL kernel validation."""

    if temperature <= 0.0:
        raise ValueError("temperature must be greater than zero")
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError(
            f"logits leading shape {tuple(logits.shape[:-1])} must match "
            f"token_ids shape {tuple(token_ids.shape)}"
        )
    if mask is not None and mask.shape != token_ids.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} must match token_ids shape")

    gather_token_ids = token_ids.long()
    active_mask = None
    if mask is not None:
        active_mask = _bool_mask(mask, device=token_ids.device)
        gather_token_ids = gather_token_ids.masked_fill(~active_mask, 0)

    scaled_logits = logits.float() / float(temperature)
    log_probs = torch.log_softmax(scaled_logits, dim=-1)
    selected = torch.gather(log_probs, dim=-1, index=gather_token_ids.unsqueeze(-1)).squeeze(-1)

    if active_mask is not None:
        selected = selected.masked_fill(~active_mask.to(device=selected.device), 0.0)

    return selected.to(dtype=output_dtype)


def selected_logprobs_tp_reference(
    logit_shards: Sequence[torch.Tensor],
    token_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    vocab_start_indices: Sequence[int] | None = None,
    temperature: float = 1.0,
    output_dtype: torch.dtype = torch.float32,
    reduction_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """TP-invariant selected logprobs from vocab-sharded logits.

    The denominator is a global online-softmax-style reduction: all-rank max
    first, then all-rank exp-sum in ``reduction_dtype``. This is the semantic
    reference for matching FSDP(TP=1) against TP>1 rollout or scoring paths.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be greater than zero")
    if mask is not None and mask.shape != token_ids.shape:
        raise ValueError("mask shape must match token_ids shape")

    device, _leading_shape = _validate_logit_shards(logit_shards, token_ids)
    starts = _resolve_vocab_start_indices(logit_shards, vocab_start_indices)
    token_ids_device = token_ids.to(device=device, dtype=torch.long)
    active_mask = None
    if mask is not None:
        active_mask = _bool_mask(mask, device=device)

    scaled_shards = [
        shard.to(device=device, dtype=reduction_dtype) / float(temperature)
        for shard in logit_shards
    ]
    local_maxes = [shard.amax(dim=-1) for shard in scaled_shards]
    global_max = torch.stack(local_maxes, dim=0).amax(dim=0)
    global_sum = torch.zeros_like(global_max, dtype=reduction_dtype, device=device)
    for shard in scaled_shards:
        global_sum = global_sum + torch.exp(shard - global_max.unsqueeze(-1)).sum(dim=-1)
    global_lse = global_max + torch.log(global_sum)

    selected_logits = torch.zeros_like(global_lse, dtype=reduction_dtype, device=device)
    covered = torch.zeros_like(token_ids_device, dtype=torch.bool, device=device)
    if active_mask is None:
        token_active = torch.ones_like(token_ids_device, dtype=torch.bool, device=device)
    else:
        token_active = active_mask

    for start, shard in zip(starts, scaled_shards, strict=True):
        end = start + int(shard.size(-1))
        owns = token_active & (token_ids_device >= start) & (token_ids_device < end)
        safe_local_ids = (token_ids_device - start).clamp(min=0, max=int(shard.size(-1)) - 1)
        gathered = torch.gather(shard, dim=-1, index=safe_local_ids.unsqueeze(-1)).squeeze(-1)
        selected_logits = torch.where(owns, gathered, selected_logits)
        covered = covered | owns

    if bool((token_active & ~covered).any().item()):
        first_bad = (token_active & ~covered).nonzero(as_tuple=False)[0]
        token_id = int(token_ids_device[tuple(first_bad.tolist())].item())
        raise ValueError(f"active token id {token_id} is not covered by any vocab shard")

    selected = selected_logits - global_lse
    if active_mask is not None:
        selected = selected.masked_fill(~active_mask, 0.0)

    return selected.to(dtype=output_dtype)


def _require_initialized_distributed():
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed process group is not initialized")
    return torch.distributed


def selected_logprobs_distributed_tp_reference(
    local_logits: torch.Tensor,
    token_ids: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    vocab_start_index: int,
    group: Any | None = None,
    temperature: float = 1.0,
    output_dtype: torch.dtype = torch.float32,
    reduction_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Distributed TP selected logprobs using real all-reduce collectives.

    Each rank provides one contiguous vocab shard. The returned tensor is the
    same on every rank and matches ``selected_logprobs_tp_reference`` for the
    same shard layout.
    """

    dist = _require_initialized_distributed()
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than zero")
    if vocab_start_index < 0:
        raise ValueError("vocab_start_index must be non-negative")
    if local_logits.ndim < 1 or local_logits.size(-1) <= 0:
        raise ValueError("local_logits must have a non-empty vocab dimension")
    if local_logits.shape[:-1] != token_ids.shape:
        raise ValueError(
            f"local_logits leading shape {tuple(local_logits.shape[:-1])} must match "
            f"token_ids shape {tuple(token_ids.shape)}"
        )
    if mask is not None and mask.shape != token_ids.shape:
        raise ValueError("mask shape must match token_ids shape")

    device = local_logits.device
    token_ids_device = token_ids.to(device=device, dtype=torch.long)
    if mask is None:
        active_mask = torch.ones_like(token_ids_device, dtype=torch.bool, device=device)
    else:
        active_mask = _bool_mask(mask, device=device)

    scaled = local_logits.to(dtype=reduction_dtype) / float(temperature)
    global_max = scaled.amax(dim=-1)
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=group)

    global_sum = torch.exp(scaled - global_max.unsqueeze(-1)).sum(dim=-1)
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM, group=group)
    global_lse = global_max + torch.log(global_sum)

    shard_end = vocab_start_index + int(local_logits.size(-1))
    owns = active_mask & (token_ids_device >= vocab_start_index) & (token_ids_device < shard_end)
    safe_local_ids = (token_ids_device - vocab_start_index).clamp(
        min=0,
        max=int(local_logits.size(-1)) - 1,
    )
    gathered = torch.gather(scaled, dim=-1, index=safe_local_ids.unsqueeze(-1)).squeeze(-1)
    selected_logits = torch.where(owns, gathered, torch.zeros_like(global_lse))
    dist.all_reduce(selected_logits, op=dist.ReduceOp.SUM, group=group)

    coverage = owns.to(dtype=torch.int32)
    dist.all_reduce(coverage, op=dist.ReduceOp.SUM, group=group)
    bad_coverage = active_mask & (coverage != 1)
    if bool(bad_coverage.any().item()):
        first_bad = bad_coverage.nonzero(as_tuple=False)[0]
        token_id = int(token_ids_device[tuple(first_bad.tolist())].item())
        covered_by = int(coverage[tuple(first_bad.tolist())].item())
        raise ValueError(
            f"active token id {token_id} is covered by {covered_by} vocab shards; "
            "expected exactly one"
        )

    selected = selected_logits - global_lse
    selected = selected.masked_fill(~active_mask, 0.0)
    return selected.to(dtype=output_dtype)


def masked_sum(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Sum values while ignoring masked-out entries."""

    values_fp32 = values.float()
    if mask is None:
        return values_fp32.sum()
    return values_fp32.masked_fill(~_bool_mask(mask, device=values.device), 0.0).sum()


def sharded_masked_sum(
    value_shards: Sequence[torch.Tensor],
    mask_shards: Sequence[torch.Tensor] | None = None,
    *,
    reduction_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Global masked sum from token/micro-batch shards with fixed fp reduction state."""

    if not value_shards:
        raise ValueError("value_shards must contain at least one shard")
    if mask_shards is not None and len(mask_shards) != len(value_shards):
        raise ValueError("mask_shards length must match value_shards")

    device = value_shards[0].device
    total = torch.zeros((), device=device, dtype=reduction_dtype)
    for index, values in enumerate(value_shards):
        if values.device != device:
            raise ValueError("all value_shards must be on the same device")
        values_acc = values.to(dtype=reduction_dtype)
        if mask_shards is None:
            total = total + values_acc.sum()
            continue
        mask = mask_shards[index]
        if mask.shape != values.shape:
            raise ValueError("each mask shard shape must match its value shard")
        mask_bool = _bool_mask(mask, device=device)
        total = total + values_acc.masked_fill(~mask_bool, 0.0).sum()
    return total


def sharded_active_token_count(
    mask_shards: Sequence[torch.Tensor] | None = None,
    *,
    value_shards: Sequence[torch.Tensor] | None = None,
    reduction_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Global active-token count from distributed mask shards."""

    if mask_shards is None:
        if not value_shards:
            raise ValueError("value_shards must be provided when mask_shards is None")
        device = value_shards[0].device
        count = sum(int(values.numel()) for values in value_shards)
        return torch.tensor(count, device=device, dtype=reduction_dtype)
    if not mask_shards:
        raise ValueError("mask_shards must contain at least one shard")

    device = mask_shards[0].device
    total = torch.zeros((), device=device, dtype=reduction_dtype)
    for mask in mask_shards:
        if mask.device != device:
            raise ValueError("all mask_shards must be on the same device")
        total = total + _bool_mask(mask, device=device).sum().to(dtype=reduction_dtype)
    return total


def sharded_masked_mean(
    value_shards: Sequence[torch.Tensor],
    mask_shards: Sequence[torch.Tensor] | None = None,
    *,
    eps: float = 1e-8,
    reduction_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Global masked mean; never average local shard means."""

    denom = sharded_active_token_count(
        mask_shards,
        value_shards=value_shards,
        reduction_dtype=reduction_dtype,
    ).clamp_min(eps)
    return (
        sharded_masked_sum(
            value_shards,
            mask_shards,
            reduction_dtype=reduction_dtype,
        )
        / denom
    )


def distributed_masked_sum(
    local_values: torch.Tensor,
    local_mask: torch.Tensor | None = None,
    *,
    group: Any | None = None,
    reduction_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Distributed masked sum using real all-reduce collectives."""

    dist = _require_initialized_distributed()
    values = local_values.to(dtype=reduction_dtype)
    if local_mask is not None:
        if local_mask.shape != local_values.shape:
            raise ValueError("local_mask shape must match local_values shape")
        values = values.masked_fill(~_bool_mask(local_mask, device=local_values.device), 0.0)
    total = values.sum()
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    return total


def distributed_active_token_count(
    local_mask: torch.Tensor | None = None,
    *,
    local_values: torch.Tensor | None = None,
    group: Any | None = None,
    reduction_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Distributed active-token count using real all-reduce collectives."""

    dist = _require_initialized_distributed()
    if local_mask is None:
        if local_values is None:
            raise ValueError("local_values must be provided when local_mask is None")
        local_count = int(local_values.numel())
        device = local_values.device
    else:
        local_count = int(_bool_mask(local_mask, device=local_mask.device).sum().item())
        device = local_mask.device
    total = torch.tensor(local_count, device=device, dtype=reduction_dtype)
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    return total


def distributed_masked_mean(
    local_values: torch.Tensor,
    local_mask: torch.Tensor | None = None,
    *,
    group: Any | None = None,
    eps: float = 1e-8,
    reduction_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Distributed masked mean using global sum and global active-token count."""

    total = distributed_masked_sum(
        local_values,
        local_mask,
        group=group,
        reduction_dtype=reduction_dtype,
    )
    count = distributed_active_token_count(
        local_mask,
        local_values=local_values,
        group=group,
        reduction_dtype=reduction_dtype,
    )
    return total / count.clamp_min(eps)


def active_token_count(
    mask: torch.Tensor | None, values: torch.Tensor | None = None
) -> torch.Tensor:
    """Return the number of active tokens as an fp32 scalar tensor."""

    if mask is None:
        if values is None:
            raise ValueError("values must be provided when mask is None")
        return torch.tensor(values.numel(), device=values.device, dtype=torch.float32)
    return _bool_mask(mask, device=mask.device).sum().to(dtype=torch.float32)


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean values while ignoring masked-out entries."""

    denom = active_token_count(mask, values).clamp_min(eps)
    return masked_sum(values, mask) / denom


def compute_policy_ratio(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute exp(current - old) with masked entries set to zero."""

    ratio = torch.exp(current_logps.float() - old_logps.float())
    if mask is not None:
        ratio = ratio.masked_fill(~_bool_mask(mask, device=ratio.device), 0.0)
    return ratio


def compute_reference_kl(
    current_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the common GRPO/PPO reference KL approximation."""

    diff = ref_logps.float() - current_logps.float()
    kl = torch.exp(diff) - diff - 1.0
    if mask is not None:
        kl = kl.masked_fill(~_bool_mask(mask, device=kl.device), 0.0)
    return kl


def summarize_kernel_drift(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize candidate-vs-reference drift for benchmark/test output."""

    if candidate.shape != reference.shape:
        raise ValueError(
            f"candidate shape {tuple(candidate.shape)} must match reference shape "
            f"{tuple(reference.shape)}"
        )

    diff = (candidate.float() - reference.float()).abs()
    if mask is not None:
        active = _bool_mask(mask, device=diff.device)
        active_diff = diff[active]
        active_count = int(active.sum().item())
    else:
        active_diff = diff.reshape(-1)
        active_count = int(diff.numel())

    if active_count == 0:
        max_abs = 0.0
        mean_abs = 0.0
    else:
        max_abs = float(active_diff.max().item())
        mean_abs = float(active_diff.mean().item())

    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "active_count": active_count,
    }


def summarize_tp_logprob_drift(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    token_ids: torch.Tensor,
    shard_ranges: Sequence[tuple[int, int]],
    mask: torch.Tensor | None = None,
    *,
    backend: str = "reference",
    reduction_name: str = "tp_vocab_logsumexp",
    dtype: torch.dtype | str | None = None,
) -> dict[str, Any]:
    """Summarize TP logprob drift and identify the owning shard of the worst token."""

    summary = summarize_kernel_drift(candidate, reference, mask)
    if candidate.shape != token_ids.shape:
        raise ValueError("candidate shape must match token_ids shape")

    rel_denom = reference.float().abs().clamp_min(1e-12)
    rel_diff = (candidate.float() - reference.float()).abs() / rel_denom
    if mask is not None:
        active = _bool_mask(mask, device=rel_diff.device)
        active_rel_diff = rel_diff[active]
    else:
        active_rel_diff = rel_diff.reshape(-1)
    if summary["active_count"] == 0:
        max_rel = 0.0
        mean_rel = 0.0
    else:
        max_rel = float(active_rel_diff.max().item())
        mean_rel = float(active_rel_diff.mean().item())

    summary.update(
        {
            "max_rel_error": max_rel,
            "mean_rel_error": mean_rel,
            "backend": backend,
            "reduction_name": reduction_name,
            "dtype": str(dtype if dtype is not None else candidate.dtype),
        }
    )

    if summary["active_count"] == 0:
        summary.update(
            {
                "flat_index": None,
                "multi_index": None,
                "token_id": None,
                "owner_rank": None,
                "owner_vocab_start": None,
                "owner_vocab_end": None,
                "candidate_value": None,
                "reference_value": None,
                "signed_error": None,
                "tp_size": len(shard_ranges),
            }
        )
        return summary

    diff = (candidate.float() - reference.float()).abs()
    if mask is not None:
        active = _bool_mask(mask, device=diff.device)
        diff = diff.masked_fill(~active, -1.0)
    flat_index = int(diff.reshape(-1).argmax().item())
    multi_index_tensor = torch.unravel_index(
        torch.tensor(flat_index, device=diff.device),
        diff.shape,
    )
    multi_index = tuple(int(index.item()) for index in multi_index_tensor)
    token_id = int(token_ids.to(device=diff.device)[multi_index].item())
    owner_rank = None
    owner_start = None
    owner_end = None
    for rank, (start, end) in enumerate(shard_ranges):
        if int(start) <= token_id < int(end):
            owner_rank = rank
            owner_start = int(start)
            owner_end = int(end)
            break

    candidate_value = float(candidate.float().reshape(-1)[flat_index].item())
    reference_value = float(reference.float().reshape(-1)[flat_index].item())
    summary.update(
        {
            "flat_index": flat_index,
            "multi_index": multi_index,
            "token_id": token_id,
            "owner_rank": owner_rank,
            "owner_vocab_start": owner_start,
            "owner_vocab_end": owner_end,
            "candidate_value": candidate_value,
            "reference_value": reference_value,
            "signed_error": candidate_value - reference_value,
            "tp_size": len(shard_ranges),
        }
    )
    return summary
