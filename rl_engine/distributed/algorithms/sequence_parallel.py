# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Contiguous image-sequence sharding with padding confined to communication."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from rl_engine.distributed.transports.rccl import TorchDistributedDeterministicCollective

_IMAGE_DTYPES = (torch.float32, torch.bfloat16)


@dataclass(frozen=True)
class SequenceParallelLayout:
    """A rank's contiguous interval in the original, unpadded image sequence."""

    sequence_length: int
    world_size: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_length, int) or self.sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer")
        if not isinstance(self.world_size, int) or self.world_size not in (1, 2, 4, 8):
            raise ValueError("sequence parallelism requires world_size in (1, 2, 4, 8)")
        if not isinstance(self.rank, int) or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be an integer in [0, world_size)")

    @classmethod
    def from_group(
        cls, sequence_length: int, group: dist.ProcessGroup | None = None
    ) -> SequenceParallelLayout:
        if not dist.is_initialized():
            if group is not None:
                raise RuntimeError("a process group requires initialized torch.distributed")
            return cls(sequence_length)
        return cls(sequence_length, dist.get_world_size(group), dist.get_rank(group))

    @property
    def shard_length(self) -> int:
        """Equal buffer length used only while communicating."""
        return (self.sequence_length + self.world_size - 1) // self.world_size

    @property
    def start(self) -> int:
        return min(self.rank * self.shard_length, self.sequence_length)

    @property
    def local_length(self) -> int:
        return min(self.shard_length, self.sequence_length - self.start)


def _check_group(layout: SequenceParallelLayout, group: dist.ProcessGroup | None) -> None:
    actual = SequenceParallelLayout.from_group(layout.sequence_length, group)
    if actual != layout:
        raise ValueError("layout rank and world_size must match the process group")


def _sequence_dim(value: torch.Tensor, dim: int) -> int:
    if not -value.ndim <= dim < value.ndim:
        raise ValueError("sequence dimension is out of range")
    return dim % value.ndim


def _gather(
    value: torch.Tensor,
    layout: SequenceParallelLayout,
    dim: int,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    if layout.world_size == 1:
        return value.clone()

    # Collectives concatenate dimension 0. Move tokens ahead of batch so that
    # rank-major transport cannot interleave different samples' image rows.
    rows = value.movedim(dim, 0)
    padded = value.new_zeros((layout.shard_length, *rows.shape[1:]))
    padded[: layout.local_length].copy_(rows)
    with TorchDistributedDeterministicCollective(
        group=group,
        device=value.device,
        max_size_bytes=max(1, padded.numel() * padded.element_size()),
    ) as collective:
        gathered = collective.all_gather(padded)
    # Only the global tail is padding. No dummy token reaches a compute op.
    return gathered[: layout.sequence_length].movedim(0, dim).contiguous()


class _ShardSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, layout, group):
        ctx.layout, ctx.group = layout, group
        return value.narrow(1, layout.start, layout.local_length).contiguous()

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        # The input is replicated: each replica receives the complete gradient
        # assembled from the disjoint token intervals, without averaging.
        return _gather(gradient, ctx.layout, 1, ctx.group), None, None


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, layout, dim, group):
        ctx.layout, ctx.dim, ctx.group = layout, dim, group
        return _gather(value, layout, dim, group)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        layout, dim = ctx.layout, ctx.dim
        if layout.world_size == 1:
            return gradient.clone(), None, None, None

        rows = gradient.movedim(dim, 0)
        # Sum consumers' gradients in the existing balanced rank tree. BF16
        # contributions are accumulated in FP32 and cast only at the boundary.
        padded = torch.zeros(
            (layout.shard_length * layout.world_size, *rows.shape[1:]),
            dtype=torch.float32,
            device=gradient.device,
        )
        padded[: layout.sequence_length].copy_(rows)
        with TorchDistributedDeterministicCollective(
            group=ctx.group,
            device=gradient.device,
            max_size_bytes=max(1, padded.numel() * padded.element_size()),
        ) as collective:
            reduced = collective.reduce_scatter(padded)
        local = reduced[: layout.local_length].movedim(0, dim)
        return local.to(gradient.dtype).contiguous(), None, None, None


def shard_image_sequence(
    image: torch.Tensor,
    positions: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SequenceParallelLayout]:
    """Co-shard replicated ``[B,S,C]`` image rows and their three-axis positions.

    Positions are non-differentiable metadata shaped ``[S,3]`` or ``[B,S,3]``.
    Returned shards contain only valid rows; their global positions are unchanged.
    The image backward all-gathers local gradients into each replicated input.
    """
    if image.ndim != 3 or any(size < 1 for size in image.shape):
        raise ValueError("image must have nonempty shape [B,S,C]")
    if image.dtype not in _IMAGE_DTYPES:
        raise TypeError("image must have dtype float32 or bfloat16")
    if positions.ndim == 2:
        expected_shape = (image.size(1), 3)
        position_dim = 0
    elif positions.ndim == 3:
        expected_shape = (image.size(0), image.size(1), 3)
        position_dim = 1
    else:
        raise ValueError("positions must have shape [S,3] or [B,S,3]")
    if tuple(positions.shape) != expected_shape:
        raise ValueError("positions must match the image batch and sequence dimensions")
    if positions.device != image.device:
        raise ValueError("image and positions must be on the same device")
    if positions.requires_grad:
        raise ValueError("positions are metadata and must not require gradients")

    layout = SequenceParallelLayout.from_group(image.size(1), group)
    local_image = _ShardSequence.apply(image, layout, group)
    local_positions = positions.narrow(position_dim, layout.start, layout.local_length)
    return local_image, local_positions.contiguous(), layout


def gather_sequence(
    value: torch.Tensor,
    layout: SequenceParallelLayout,
    *,
    dim: int = 1,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Restore unpadded sequence order on every SP rank, preserving input bytes.

    Use ``dim=1`` for image rows or batched positions, and ``dim=0`` for shared
    ``[local_S,3]`` positions. Backward sums all consumers' gradients; callers
    own loss normalization. All ranks must participate in forward and backward.
    """
    dim = _sequence_dim(value, dim)
    _check_group(layout, group)
    if value.size(dim) != layout.local_length:
        raise ValueError("local sequence length does not match the layout")
    if value.requires_grad and value.dtype not in _IMAGE_DTYPES:
        raise TypeError("differentiable gather requires float32 or bfloat16")
    return _GatherSequence.apply(value, layout, dim, group)
