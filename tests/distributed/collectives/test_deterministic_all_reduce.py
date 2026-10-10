# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.distributed import create_deterministic_collective

_MAX_WORLD_SIZE = 8
_TP_SIZES = (1, 2, 4, 8)
_EXTERNAL_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

pytestmark = [
    # The deterministic collectives are CUDA-IPC kernels, compiled out of ROCm
    # builds on purpose. ROCm reports GPUs through the CUDA device API, so the
    # device_count guard below does not exclude it.
    pytest.mark.cuda_only,
    pytest.mark.skipif(
        _EXTERNAL_WORLD_SIZE != 1,
        reason="this cross-TP test owns its worker processes; run pytest directly",
    ),
    pytest.mark.skipif(
        torch.cuda.device_count() < _MAX_WORLD_SIZE,
        reason="requires eight visible CUDA GPUs",
    ),
]


def _fixed_tree_reference(values: list[torch.Tensor]) -> torch.Tensor:
    level = values
    while len(level) > 1:
        level = [level[index] + level[index + 1] for index in range(0, len(level), 2)]
    return level[0]


def _all_gather_tensors(
    value: torch.Tensor,
    group: dist.ProcessGroup,
    world_size: int,
) -> list[torch.Tensor]:
    gathered = [torch.empty_like(value) for _ in range(world_size)]
    dist.all_gather(gathered, value, group=group)
    return gathered


def _worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_MAX_WORLD_SIZE,
        timeout=timedelta(minutes=5),
    )
    try:
        groups = {tp_size: dist.new_group(ranks=list(range(tp_size))) for tp_size in _TP_SIZES}
        for tp_size, group in groups.items():
            if rank < tp_size:
                with create_deterministic_collective(
                    group=group,
                    device=device,
                    max_size_bytes=1024 * 1024,
                ) as collective:
                    group_rank = dist.get_rank(group=group)
                    leaves_per_rank = _MAX_WORLD_SIZE // tp_size
                    start = group_rank * leaves_per_rank
                    for dtype in (torch.float32, torch.float16, torch.bfloat16):
                        generator = torch.Generator().manual_seed(20260815)
                        leaves_tensor = torch.randn(
                            _MAX_WORLD_SIZE,
                            257,
                            dtype=torch.float32,
                            generator=generator,
                        ).to(device=device, dtype=dtype)
                        leaves = list(leaves_tensor.unbind())
                        input = _fixed_tree_reference(leaves[start : start + leaves_per_rank])
                        expected = _fixed_tree_reference(leaves)

                        output = collective.all_reduce(input)
                        assert torch.equal(output, expected)

                        peer_outputs = _all_gather_tensors(output, group, tp_size)
                        assert all(torch.equal(peer_output, output) for peer_output in peer_outputs)

                        baseline = output.clone()
                        for _ in range(3):
                            repeated = collective.all_reduce(input)
                            assert torch.equal(repeated, baseline)

                        inplace = input.clone()
                        returned = collective.all_reduce(inplace, out=inplace)
                        assert returned is inplace
                        assert torch.equal(inplace, expected)

                        if dtype in (torch.float16, torch.bfloat16):
                            packed_input = torch.cat((input, input[:1]))
                            packed_expected = torch.cat((expected, expected[:1]))
                            packed_output = collective.all_reduce(packed_input)
                            assert torch.equal(packed_output, packed_expected)

                            output_storage = torch.empty(
                                packed_expected.numel() + 1,
                                dtype=dtype,
                                device=device,
                            )
                            misaligned_output = output_storage[1:].view_as(packed_expected)
                            returned = collective.all_reduce(
                                packed_input,
                                out=misaligned_output,
                            )
                            assert returned is misaligned_output
                            assert torch.equal(misaligned_output, packed_expected)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_deterministic_all_reduce_cross_tp_cuda() -> None:
    mp.spawn(
        _worker,
        args=(_find_free_port(),),
        nprocs=_MAX_WORLD_SIZE,
        join=True,
    )
