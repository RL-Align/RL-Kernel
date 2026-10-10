# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Rank-ordered row gather shared by the H3 WS2 ops (a copy: no arithmetic)."""

from __future__ import annotations

import torch

# DeterministicCollective sends all-gathers whose output is at most this many bytes
# through a single-block byte-copy kernel (csrc/cuda/distributed/
# deterministic_collective.cu, kSingleBlockFastPathMaxBytes) that costs about 4 us per
# KiB on B200; larger outputs take the multi-block path (~45 us). Padding the shard
# past the threshold is cheaper for every non-trivial size.
_SINGLE_BLOCK_GATHER_MAX_BYTES = 256 * 1024


def gather_rows(collective, shard: torch.Tensor) -> torch.Tensor:
    """``(world * rows, ...)``: every rank's ``(rows, ...)`` shard, in rank order."""

    shard = shard.contiguous()
    rows, world = shard.shape[0], collective.world_size
    row_bytes = shard[0].numel() * shard.element_size() if rows else 0
    if world == 1:
        return shard
    if row_bytes == 0:
        return collective.all_gather(shard)
    padded = max(rows, _SINGLE_BLOCK_GATHER_MAX_BYTES // (world * row_bytes) + 1)
    if padded == rows:
        return collective.all_gather(shard)
    staged = torch.cat([shard, shard.new_zeros((padded - rows, *shard.shape[1:]))])
    out = collective.all_gather(staged).view(world, padded, *shard.shape[1:])
    return out[:, :rows].reshape(world * rows, *shard.shape[1:])
