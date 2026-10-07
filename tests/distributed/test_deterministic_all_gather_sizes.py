# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic all-gather is an exact rank-ordered copy at every size.

The copy has a single-block path (gathered output <= 64 KiB) and a multi-block
path, and moves 16-byte vectors when pointers allow. This sweeps byte counts
across the threshold, odd sizes that leave a byte tail, and misaligned input and
output pointers, and compares every result with NCCL's all-gather.
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.distributed import create_deterministic_collective

_WORLD_SIZE = max((n for n in (8, 4, 2) if n <= torch.cuda.device_count()), default=0)
_THRESHOLD = 64 * 1024  # kAllGatherSingleBlockMaxBytes

pytestmark = [
    pytest.mark.cuda_only,
    pytest.mark.skipif(
        int(os.environ.get("WORLD_SIZE", "1")) != 1,
        reason="this test owns its worker processes; run pytest directly",
    ),
    pytest.mark.skipif(_WORLD_SIZE < 2, reason="requires at least two visible CUDA GPUs"),
]


def _byte_sizes(world_size: int) -> list[int]:
    """Return per-rank byte counts covering copy boundaries and vector tails."""
    per_rank_threshold = _THRESHOLD // world_size
    return sorted(
        {
            1,
            15,
            16,
            17,
            4095,
            32 * 1024 + 3,
            96 * 1024,
            per_rank_threshold - 1,
            per_rank_threshold,
            per_rank_threshold + 1,
            per_rank_threshold + 16,
            1024 * 1024 + 5,
        }
    )


def _worker(rank: int, world_size: int, port: int) -> None:
    """Compare rank-ordered copies with NCCL in an owned process group."""
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5),
    )
    try:
        with create_deterministic_collective(
            device=device, max_size_bytes=2 * 1024 * 1024
        ) as collective:
            generator = torch.Generator(device="cpu").manual_seed(1234 + rank)
            for size in _byte_sizes(world_size):
                for input_offset, output_offset in ((0, 0), (1, 0), (0, 3), (5, 7)):
                    source = torch.randint(
                        0, 256, (size + input_offset,), generator=generator, dtype=torch.uint8
                    ).to(device)
                    shard = source[input_offset:]
                    expected = torch.empty(size * world_size, dtype=torch.uint8, device=device)
                    dist.all_gather_into_tensor(expected, shard.contiguous())

                    storage = torch.zeros(
                        size * world_size + output_offset, dtype=torch.uint8, device=device
                    )
                    out = storage[output_offset:]
                    returned = collective.all_gather(shard, out=out)
                    assert returned is out
                    assert torch.equal(out, expected), (size, input_offset, output_offset)
                    assert torch.count_nonzero(storage[:output_offset]) == 0
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _find_free_port() -> int:
    """Return an available loopback TCP port for the workers' rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_all_gather_is_exact_across_sizes_and_alignments() -> None:
    """Verify byte-exact output and prefix guards across sizes and offsets."""
    mp.spawn(_worker, args=(_WORLD_SIZE, _find_free_port()), nprocs=_WORLD_SIZE, join=True)
