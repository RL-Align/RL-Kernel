# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Rank-ordered row gather shared by the H3 WS2 ops (a copy: no arithmetic)."""

from __future__ import annotations

import torch

# DeterministicCollective sends all-gathers whose output is at most this many bytes
# through a single-block byte-copy kernel (csrc/cuda/distributed/
# deterministic_collective.cu, kSingleBlockFastPathMaxBytes) that costs about 4 us per
# KiB; larger outputs take the multi-block path, which costs a flat 55-65 us for 2 and 4 ranks
# (about 90 us for 8) at any output size. Padding a shard past the threshold therefore pays for
# outputs above the crossover and costs time below it. Measured on 8 x B200 (this kernel, 2, 4 and
# 8 ranks, outputs from 2 KiB to 512 KiB): the crossover is about 8 KiB of output (2 ranks:
# plain 54 us against padded 59 us at 8 KiB, 63 against 59 at 10 KiB; 8 ranks: padded is already
# faster at 8 KiB, the smallest output they can produce); at 256 KiB the plain gather takes
# 1.1-1.7 ms against 60-95 us padded. A tiny gather (an rstd or partial exchange of a handful of
# rows) is up to ~25 us slower when padded, so only outputs above this size are padded.
_SINGLE_BLOCK_GATHER_MAX_BYTES = 256 * 1024
_PAD_ABOVE_BYTES = 8 * 1024


def gather_rows(collective, shard: torch.Tensor) -> torch.Tensor:
    """``(world * rows, ...)``: every rank's ``(rows, ...)`` shard, in rank order."""

    shard = shard.contiguous()
    rows, world = shard.shape[0], collective.world_size
    row_bytes = shard[0].numel() * shard.element_size() if rows else 0
    if world == 1:
        return shard
    if row_bytes == 0 or world * rows * row_bytes <= _PAD_ABOVE_BYTES:
        return collective.all_gather(shard)
    padded = max(rows, _SINGLE_BLOCK_GATHER_MAX_BYTES // (world * row_bytes) + 1)
    if padded == rows:
        return collective.all_gather(shard)
    staged = torch.cat([shard, shard.new_zeros((padded - rows, *shard.shape[1:]))])
    out = collective.all_gather(staged).view(world, padded, *shard.shape[1:])
    return out[:, :rows].reshape(world * rows, *shard.shape[1:])
