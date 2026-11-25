# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Portable deterministic embedding parameter-gradient reduction."""

from __future__ import annotations

import torch


def _deterministic_embedding_grad_weight(
    ids: torch.Tensor,
    grad_rows: torch.Tensor,
    *,
    weight_shape: tuple[int, ...],
    weight_dtype: torch.dtype,
) -> torch.Tensor:
    vocab_size = int(weight_shape[0])
    grad_weight = torch.zeros(
        weight_shape,
        device=grad_rows.device,
        dtype=grad_rows.dtype,
    )
    if ids.numel() == 0:
        return grad_weight.to(weight_dtype)

    valid = (ids >= 0) & (ids < vocab_size)
    ids = ids[valid]
    grad_rows = grad_rows[valid]
    if ids.numel() == 0:
        return grad_weight.to(weight_dtype)

    order = torch.argsort(ids, stable=True)
    sorted_ids = ids.index_select(0, order)
    sorted_rows = grad_rows.index_select(0, order)
    unique_ids, counts = torch.unique_consecutive(sorted_ids, return_counts=True)

    num_unique = int(unique_ids.numel())
    max_count = int(counts.max().item())
    hidden_size = sorted_rows.size(1)
    starts = torch.cumsum(counts, dim=0) - counts
    slot = torch.arange(sorted_rows.size(0), device=sorted_rows.device)
    slot = slot - starts.repeat_interleave(counts)
    segment = torch.arange(num_unique, device=sorted_rows.device).repeat_interleave(counts)
    padded = torch.zeros(
        (num_unique, max_count, hidden_size),
        device=sorted_rows.device,
        dtype=sorted_rows.dtype,
    )
    padded[segment, slot] = sorted_rows
    acc = padded[:, 0]
    for slot_idx in range(1, max_count):
        acc = acc + padded[:, slot_idx]

    grad_weight[unique_ids] = acc
    return grad_weight.to(weight_dtype)


deterministic_embedding_grad_weight = _deterministic_embedding_grad_weight

__all__ = [
    "_deterministic_embedding_grad_weight",
    "deterministic_embedding_grad_weight",
]
