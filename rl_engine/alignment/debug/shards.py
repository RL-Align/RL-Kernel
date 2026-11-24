# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Lossless native tensor ownership; never infer a partition from its shape."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence

import torch

from .core import bitdiff, snapshot


@dataclass(frozen=True)
class TensorShard:
    """Cartesian global indices for each local axis, including CP zigzag rows.

    A full tensor can be represented with ``replicated``. All workers must use
    the same global shape. Replicas are checked bitwise, and missing elements
    cannot pass. This is diagnostic storage, not an online collective.
    """

    value: torch.Tensor
    global_shape: Sequence[int]
    indices: Sequence[Sequence[int]]

    @classmethod
    def replicated(cls, value: torch.Tensor) -> TensorShard:
        return cls(value, tuple(value.shape), [range(n) for n in value.shape])


@dataclass(frozen=True)
class ShardSet:
    """Multiple native fragments of ONE global tensor on a worker."""

    shards: Sequence[TensorShard]


def encode_shards(shards: Sequence[TensorShard]) -> dict:
    if not shards:
        raise ValueError("a shard set needs at least one typed fragment")
    parts = []
    for shard in shards:
        if not isinstance(shard.value, torch.Tensor):
            raise ValueError("native shard value must be a tensor")
        shape = list(shard.global_shape)
        indices = [list(axis) for axis in shard.indices]
        if (
            any(type(n) is not int or n < 1 for n in shape)
            or len(shape) != shard.value.ndim
            or len(indices) != len(shape)
            or any(
                len(axis) != shard.value.shape[d]
                or len(set(axis)) != len(axis)
                or any(type(i) is not int or not 0 <= i < shape[d] for i in axis)
                for d, axis in enumerate(indices)
            )
        ):
            raise ValueError("invalid global tensor ownership")
        parts.append({"value": snapshot(shard.value), "shape": shape, "indices": indices})
    return {"kind": "sharded", "parts": parts}


def merge_shards(trees: list[dict]) -> dict:
    parts = [part for tree in trees for part in tree["parts"]]
    if not parts:
        raise ValueError("missing shard fragments")
    # Revalidate serialized evidence, not only the live API's inputs.
    encode_shards([TensorShard(p["value"], p["shape"], p["indices"]) for p in parts])
    shape, dtype = parts[0]["shape"], parts[0]["value"].dtype
    if any(p["shape"] != shape or p["value"].dtype != dtype for p in parts):
        raise ValueError("global shard shapes/dtypes differ")
    result = torch.empty(prod(shape), dtype=dtype)
    covered = torch.zeros(prod(shape), dtype=torch.bool)
    for part in parts:
        axes = [torch.tensor(axis, dtype=torch.long) for axis in part["indices"]]
        if axes:
            grids = torch.meshgrid(*axes, indexing="ij")
            offsets = torch.zeros(part["value"].shape, dtype=torch.long)
            for d, grid in enumerate(grids):
                offsets += grid * prod(shape[d + 1 :])
            offsets = offsets.reshape(-1)
        else:
            offsets = torch.tensor([0])
        values = part["value"].reshape(-1)
        repeated = covered[offsets]
        if repeated.any() and not bitdiff(result[offsets[repeated]], values[repeated]).get("equal"):
            raise ValueError("global tensor replicas disagree")
        result[offsets] = values
        covered[offsets] = True
    if not covered.all():
        raise ValueError("incomplete global tensor coverage")
    return {"kind": "tensor", "value": result.reshape(shape)}


def parallel_tensor(
    value: torch.Tensor,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    tp_axis: int | None = None,
    tp_stride: int = 1,
    cp_rank: int = 0,
    cp_size: int = 1,
    cp_axis: int | None = None,
    cp_layout: str = "contiguous",
) -> TensorShard:
    """Map a declared native TP/CP layout without knowing the model architecture.

    ``tp_stride`` is Megatron's partition stride (e.g. separate Q/K/V chunks).
    ``zigzag`` CP owns chunks r and 2*CP-r-1. No process-global rank arithmetic
    is used: pass ranks within the actual TP/CP groups.
    """
    for rank, size in ((tp_rank, tp_size), (cp_rank, cp_size)):
        if type(size) is not int or size < 1 or type(rank) is not int or not 0 <= rank < size:
            raise ValueError("invalid parallel rank/size")
    if cp_layout not in ("contiguous", "zigzag"):
        raise ValueError("unknown CP layout")
    if type(tp_stride) is not int or tp_stride < 1:
        raise ValueError("invalid TP partition stride")
    for axis, size in ((tp_axis, tp_size), (cp_axis, cp_size)):
        if axis is None:
            if size != 1:
                raise ValueError("declare a partition axis for parallel execution")
        elif type(axis) is not int or not 0 <= axis < value.ndim:
            raise ValueError("invalid partition axis")
    if tp_axis is not None and tp_axis == cp_axis:
        raise ValueError("TP and CP partition axes must be distinct")
    shape = list(value.shape)
    indices = [list(range(n)) for n in shape]
    if tp_axis is not None:
        n = shape[tp_axis]
        if n % tp_stride:
            raise ValueError("local TP axis must divide into partition strides")
        width = n // tp_stride
        indices[tp_axis] = [
            chunk * n * tp_size // tp_stride + tp_rank * width + i
            for chunk in range(tp_stride)
            for i in range(width)
        ]
        shape[tp_axis] *= tp_size
    if cp_axis is not None:
        n = shape[cp_axis]
        shape[cp_axis] *= cp_size
        if cp_layout == "contiguous":
            indices[cp_axis] = list(range(cp_rank * n, (cp_rank + 1) * n))
        else:
            if n % 2:
                raise ValueError("zigzag CP requires two equal local chunks")
            indices[cp_axis] = [
                chunk * (n // 2) + i
                for chunk in (cp_rank, 2 * cp_size - cp_rank - 1)
                for i in range(n // 2)
            ]
    return TensorShard(value, shape, indices)
