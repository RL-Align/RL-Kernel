# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini sequence, recompute, and long-context contracts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import torch


class SequenceContractError(ValueError):
    """Raised when packed sequence metadata is inconsistent."""


@dataclass(frozen=True)
class ChunkRange:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def chunked_prefill_ranges(sequence_length: int, chunk_size: int) -> tuple[ChunkRange, ...]:
    if sequence_length <= 0 or chunk_size <= 0:
        raise SequenceContractError("sequence_length and chunk_size must be positive")
    return tuple(
        ChunkRange(start, min(start + chunk_size, sequence_length))
        for start in range(0, sequence_length, chunk_size)
    )


def decode_position_ids(cache_lengths: torch.Tensor, *, step: int = 1) -> torch.Tensor:
    if cache_lengths.ndim != 1 or cache_lengths.numel() == 0:
        raise SequenceContractError("cache_lengths must be a nonempty rank-1 tensor")
    if cache_lengths.dtype not in (torch.int32, torch.int64):
        raise TypeError("cache_lengths must be integer")
    if torch.any(cache_lengths < 0) or step <= 0:
        raise SequenceContractError("cache lengths must be nonnegative and step must be positive")
    positions = cache_lengths[:, None] + torch.arange(step, device=cache_lengths.device)
    return positions.to(dtype=torch.int64)


@dataclass(frozen=True)
class PackedRows:
    values: torch.Tensor
    offsets: torch.Tensor
    row_ids: torch.Tensor
    position_ids: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.offsets.numel() - 1)


def pack_rows(rows: Sequence[torch.Tensor]) -> PackedRows:
    if not rows:
        raise SequenceContractError("at least one row is required")
    first = rows[0]
    if first.ndim < 1 or first.shape[0] == 0:
        raise SequenceContractError("rows must have a nonempty sequence dimension")
    if any(row.ndim != first.ndim or row.shape[1:] != first.shape[1:] for row in rows):
        raise SequenceContractError("rows must share trailing dimensions")
    if any(row.device != first.device or row.dtype != first.dtype for row in rows):
        raise SequenceContractError("rows must share device and dtype")
    values = torch.cat(tuple(row.contiguous() for row in rows), dim=0)
    lengths = torch.tensor([row.shape[0] for row in rows], device=first.device, dtype=torch.int64)
    offsets = torch.cat((torch.zeros(1, device=first.device, dtype=torch.int64), lengths.cumsum(0)))
    row_ids = torch.repeat_interleave(
        torch.arange(len(rows), device=first.device, dtype=torch.int64), lengths
    )
    position_ids = torch.cat(
        tuple(torch.arange(row.shape[0], device=first.device, dtype=torch.int64) for row in rows)
    )
    return PackedRows(values, offsets, row_ids, position_ids)


def unpack_rows(packed: PackedRows) -> tuple[torch.Tensor, ...]:
    if packed.offsets.ndim != 1 or packed.offsets.numel() < 2:
        raise SequenceContractError("offsets must contain row boundaries")
    if packed.offsets[0].item() != 0 or packed.offsets[-1].item() != packed.values.shape[0]:
        raise SequenceContractError("offsets do not cover packed values")
    rows = []
    for start, end in zip(packed.offsets[:-1], packed.offsets[1:], strict=True):
        rows.append(packed.values[int(start) : int(end)])
    return tuple(rows)


def causal_position_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    window: int | None = None,
) -> torch.Tensor:
    if query_positions.ndim != 2 or key_positions.ndim != 2:
        raise SequenceContractError("position IDs must have shape [batch, sequence]")
    if (
        query_positions.shape[0] != key_positions.shape[0]
        or query_positions.device != key_positions.device
    ):
        raise SequenceContractError("query and key positions must share batch and device")
    allowed = key_positions[:, None, :] <= query_positions[:, :, None]
    if window is not None:
        if window <= 0:
            raise SequenceContractError("window must be positive")
        allowed = allowed & (key_positions[:, None, :] > query_positions[:, :, None] - window)
    return allowed


@dataclass(frozen=True)
class LongContextPlan:
    context_lengths: tuple[int, ...] = (4096, 32768, 131072, 1048576)
    relative_position_limit: int = 1024

    def validate(self, sequence_length: int) -> None:
        if sequence_length not in self.context_lengths:
            raise SequenceContractError(
                f"sequence length {sequence_length} is outside the qualified contexts"
            )
        if self.relative_position_limit <= 0:
            raise SequenceContractError("relative position limit must be positive")

    def clipped_relative_positions(
        self, query_positions: torch.Tensor, key_positions: torch.Tensor
    ) -> torch.Tensor:
        relative = query_positions[..., None] - key_positions[..., None, :]
        return relative.clamp(-self.relative_position_limit, self.relative_position_limit)


def recompute_forward(
    function: Callable[..., torch.Tensor], *inputs: torch.Tensor, preserve_rng_state: bool = True
) -> torch.Tensor:
    if not callable(function):
        raise TypeError("function must be callable")
    if not inputs or any(not isinstance(value, torch.Tensor) for value in inputs):
        raise TypeError("recompute inputs must be tensors")
    return torch.utils.checkpoint.checkpoint(
        function, *inputs, use_reentrant=False, preserve_rng_state=preserve_rng_state
    )


@dataclass(frozen=True)
class VisitScheduleEntry:
    physical_layer: int
    pass_index: int
    logical_visit: int


def build_recompute_schedule(
    physical_layers: int, *, repeated_passes: int = 3
) -> tuple[VisitScheduleEntry, ...]:
    if physical_layers <= 0 or repeated_passes <= 0:
        raise SequenceContractError("physical_layers and repeated_passes must be positive")
    entries = []
    logical_visit = 0
    for pass_index in range(repeated_passes):
        for physical_layer in range(physical_layers):
            entries.append(VisitScheduleEntry(physical_layer, pass_index, logical_visit))
            logical_visit += 1
    return tuple(entries)


def validate_unrelated_row_invariance(
    reference: torch.Tensor, changed: torch.Tensor, row: int
) -> None:
    if reference.shape != changed.shape or reference.ndim == 0:
        raise SequenceContractError("reference and changed outputs must share a batch shape")
    if row < 0 or row >= reference.shape[0]:
        raise IndexError("row is outside the batch")
    keep = torch.arange(reference.shape[0], device=reference.device) != row
    if not torch.equal(reference[keep], changed[keep]):
        raise AssertionError("an unrelated packed row changed")
