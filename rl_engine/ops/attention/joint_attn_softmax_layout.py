# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Logical key layout shared by joint-attention softmax backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def validate_key_padding_mask(
    scores: torch.Tensor,
    key_padding_mask: torch.Tensor,
) -> None:
    if scores.dim() < 2:
        raise ValueError("scores must include a batch dimension when key_padding_mask is provided")
    if key_padding_mask.dtype != torch.bool:
        raise ValueError("key_padding_mask must use bool dtype")
    if key_padding_mask.device != scores.device:
        raise ValueError("scores and key_padding_mask must be on the same device")

    expected_shape = (scores.size(0), scores.size(-1))
    if key_padding_mask.shape != expected_shape:
        raise ValueError(
            f"key_padding_mask must have shape [B, K]={expected_shape}, "
            f"got {tuple(key_padding_mask.shape)}"
        )


@dataclass(frozen=True)
class LogicalKeyMapping:
    """Compact GPU mapping reused by fixed-order forward and backward."""

    logical_to_physical: torch.Tensor
    valid_key_counts: torch.Tensor
    rows_per_batch: int


@dataclass(frozen=True)
class KeyMaskLayout:
    """Map physical key positions to a padding-independent logical order."""

    physical_valid_mask: torch.Tensor
    logical_valid_mask: torch.Tensor

    @classmethod
    def from_mask(
        cls,
        scores: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> KeyMaskLayout:
        """Build a per-row layout from a shared ``[B, K]`` validity mask."""
        validate_key_padding_mask(scores, key_padding_mask)

        broadcast_shape = (scores.size(0),) + (1,) * (scores.dim() - 2) + (scores.size(-1),)
        physical_valid_mask = key_padding_mask.reshape(broadcast_shape).expand(scores.shape)
        valid_key_counts = key_padding_mask.sum(dim=-1, keepdim=True)
        key_positions = torch.arange(scores.size(-1), device=scores.device)
        logical_valid_mask = key_positions.unsqueeze(0) < valid_key_counts
        logical_valid_mask = logical_valid_mask.reshape(broadcast_shape).expand(scores.shape)
        return cls(
            physical_valid_mask=physical_valid_mask,
            logical_valid_mask=logical_valid_mask,
        )

    def compact(self, values: torch.Tensor, *, fill_value: float) -> torch.Tensor:
        """Move valid values to the front of each row without reordering them."""
        compacted = values.new_full(values.shape, fill_value)
        return compacted.masked_scatter(
            self.logical_valid_mask,
            values.masked_select(self.physical_valid_mask),
        )

    def restore(self, logical_values: torch.Tensor, *, fill_value: float) -> torch.Tensor:
        """Move logical values back to their original physical key positions."""
        restored = logical_values.new_full(logical_values.shape, fill_value)
        return restored.masked_scatter(
            self.physical_valid_mask,
            logical_values.masked_select(self.logical_valid_mask),
        )
