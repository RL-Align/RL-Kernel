# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini speculative decoding and DSpark profile contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class SpeculativeContractError(ValueError):
    """Raised when speculative decoding metadata is inconsistent."""


@dataclass(frozen=True)
class SpeculativeProfile:
    name: str = "dspark_mtp"
    draft_length: int = 4
    acceptance: str = "exact_token"
    rollback_cache: bool = True
    enabled: bool = False

    def validate(self) -> None:
        if not self.name or self.draft_length <= 0:
            raise SpeculativeContractError("profile name and draft_length must be valid")
        if self.acceptance not in {"exact_token", "greedy_argmax"}:
            raise SpeculativeContractError("unsupported speculative acceptance rule")
        if self.enabled and not self.rollback_cache:
            raise SpeculativeContractError("enabled speculative decoding requires cache rollback")


@dataclass(frozen=True)
class SpeculativeResult:
    accepted_tokens: torch.Tensor
    accepted_lengths: torch.Tensor
    rollback_lengths: torch.Tensor


def accept_speculative_tokens(
    draft_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    eos_token_id: int | None = None,
) -> SpeculativeResult:
    if draft_tokens.ndim != 2 or target_tokens.ndim != 2:
        raise SpeculativeContractError("draft and target tokens must have shape [batch, steps]")
    if draft_tokens.shape != target_tokens.shape or draft_tokens.device != target_tokens.device:
        raise SpeculativeContractError("draft and target tokens must share shape and device")
    if (
        draft_tokens.dtype not in (torch.int32, torch.int64)
        or target_tokens.dtype != draft_tokens.dtype
    ):
        raise TypeError("speculative tokens must be int32 or int64")
    batch, steps = draft_tokens.shape
    accepted = torch.zeros_like(draft_tokens)
    lengths = torch.zeros(batch, device=draft_tokens.device, dtype=torch.int64)
    rollback = torch.full((batch,), steps, device=draft_tokens.device, dtype=torch.int64)
    for row in range(batch):
        count = 0
        for step in range(steps):
            if draft_tokens[row, step] != target_tokens[row, step]:
                break
            accepted[row, step] = draft_tokens[row, step]
            count += 1
            if eos_token_id is not None and int(draft_tokens[row, step]) == eos_token_id:
                break
        lengths[row] = count
        rollback[row] = steps - count
    return SpeculativeResult(accepted, lengths, rollback)


def commit_speculative_cache(
    cache: torch.Tensor, result: SpeculativeResult, *, draft_axis: int = 1
) -> torch.Tensor:
    if cache.ndim < 2 or draft_axis < 0 or draft_axis >= cache.ndim:
        raise SpeculativeContractError("cache must have a valid draft axis")
    if cache.shape[0] != result.accepted_lengths.shape[0]:
        raise SpeculativeContractError("cache batch and result batch must match")
    if torch.any(result.accepted_lengths < 0) or torch.any(
        result.accepted_lengths > cache.shape[draft_axis]
    ):
        raise SpeculativeContractError("accepted lengths exceed cache capacity")
    if draft_axis != 1:
        cache = cache.transpose(1, draft_axis)
    maximum = int(result.accepted_lengths.max().item()) if result.accepted_lengths.numel() else 0
    committed = cache[:, :maximum].clone()
    for row, length in enumerate(result.accepted_lengths.tolist()):
        if length < maximum:
            committed[row, length:] = 0
    return committed if draft_axis == 1 else committed.transpose(1, draft_axis)
