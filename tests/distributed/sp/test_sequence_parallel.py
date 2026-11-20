# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Independent indexing/VJP checks for Qwen-Image sequence partitioning."""

from __future__ import annotations

import os
import socket
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.distributed.algorithms.sequence_parallel import (
    SequenceParallelLayout,
    gather_sequence,
    shard_image_sequence,
)
from rl_engine.distributed.transports.rccl import TorchDistributedDeterministicCollective


def _assert_bytes(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if actual.numel() == 0:
        return
    assert torch.equal(
        actual.detach().cpu().contiguous().view(torch.uint8),
        expected.detach().cpu().contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize(
    "length,world_size,lengths",
    [
        (1, 4, [1, 0, 0, 0]),
        (7, 4, [2, 2, 2, 1]),
        (8, 4, [2, 2, 2, 2]),
        (6889, 2, [3445, 3444]),
        (6889, 4, [1723, 1723, 1723, 1720]),
        (6889, 8, [862, 862, 862, 862, 862, 862, 862, 855]),
    ],
)
def test_layout_covers_each_token_once(length, world_size, lengths):
    offset = 0
    for rank, local_length in enumerate(lengths):
        layout = SequenceParallelLayout(length, world_size, rank)
        assert layout.start == offset
        assert layout.local_length == local_length
        offset += local_length
    assert offset == length


@pytest.mark.parametrize("length,world_size,rank", [(0, 1, 0), (-1, 1, 0), (7, 3, 0), (7, 2, 2)])
def test_invalid_layout(length, world_size, rank):
    with pytest.raises(ValueError):
        SequenceParallelLayout(length, world_size, rank)


def test_single_rank_preserves_special_float_bit_patterns():
    # Include signed zero and a NaN payload: a numerical equality check alone
    # would not establish that this data movement preserves the input bytes.
    bits = torch.tensor([0, -2147483648, 2143289345, 1065353216], dtype=torch.int32)
    image = bits.view(torch.float32).reshape(1, 4, 1)
    positions = torch.arange(12).reshape(4, 3)
    local, local_positions, layout = shard_image_sequence(image, positions)
    _assert_bytes(local, image)
    _assert_bytes(gather_sequence(local, layout), image)
    _assert_bytes(local_positions, positions)


@pytest.mark.parametrize(
    "image,positions,error",
    [
        (torch.zeros(7, 3), torch.zeros(7, 3), ValueError),
        (torch.zeros(1, 0, 3), torch.zeros(0, 3), ValueError),
        (torch.zeros(1, 7, 3, dtype=torch.float64), torch.zeros(7, 3), TypeError),
        (torch.zeros(1, 7, 3), torch.zeros(6, 3), ValueError),
        (torch.zeros(1, 7, 3), torch.zeros(7, 2), ValueError),
        (torch.zeros(1, 7, 3), torch.zeros(2, 7, 3), ValueError),
        (torch.zeros(1, 7, 3), torch.zeros(7), ValueError),
        (torch.zeros(1, 7, 3), torch.zeros(7, 3, requires_grad=True), ValueError),
    ],
)
def test_invalid_image_inputs(image, positions, error):
    with pytest.raises(error):
        shard_image_sequence(image, positions)


def test_gather_rejects_wrong_length_dimension_and_group():
    with pytest.raises(ValueError, match="local sequence length"):
        gather_sequence(torch.zeros(1, 6, 3), SequenceParallelLayout(7))
    with pytest.raises(ValueError, match="dimension"):
        gather_sequence(torch.zeros(1, 7, 3), SequenceParallelLayout(7), dim=3)
    with pytest.raises(ValueError, match="process group"):
        gather_sequence(torch.zeros(1, 4, 3), SequenceParallelLayout(7, 2, 0))
    with pytest.raises(TypeError, match="differentiable gather"):
        gather_sequence(
            torch.zeros(1, 7, 3, dtype=torch.float64, requires_grad=True), SequenceParallelLayout(7)
        )


def _inputs(batch, length, channels, dtype, batched_positions, seed):
    generator = torch.Generator().manual_seed(seed)
    # Noncontiguous inputs exercise layout conversion without using the
    # production shard/gather helpers to construct the expected result.
    image = torch.randn(batch, channels, length, generator=generator).transpose(1, 2).to(dtype)
    token = torch.arange(length)
    positions = torch.stack((torch.zeros_like(token), token // 83, token % 83))
    positions = positions.transpose(0, 1)
    if batched_positions:
        positions = positions.unsqueeze(0).expand(batch, -1, -1).to(torch.float32)
    return image, positions


def _check_roundtrip(batch, length, channels, dtype, batched_positions, device, group, seed=386):
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    image_cpu, positions_cpu = _inputs(batch, length, channels, dtype, batched_positions, seed)
    image = image_cpu.to(device).detach().requires_grad_(True)
    positions = positions_cpu.to(device)
    local, local_positions, layout = shard_image_sequence(image, positions, group=group)
    # CPU index_select is the independent data-movement reference. Check each
    # shard before its inverse so two compensating permutation bugs cannot pass.
    width = (length + world_size - 1) // world_size
    indices = torch.arange(min(rank * width, length), min((rank + 1) * width, length))
    expected_local = image_cpu.index_select(1, indices)
    position_dim = 1 if batched_positions else 0
    _assert_bytes(local, expected_local)
    _assert_bytes(local_positions, positions_cpu.index_select(position_dim, indices))
    assert local.is_contiguous()
    assert local_positions.is_contiguous()
    restored = gather_sequence(local, layout, group=group)
    _assert_bytes(restored, image_cpu)
    _assert_bytes(
        gather_sequence(local_positions, layout, dim=position_dim, group=group), positions_cpu
    )

    original_gather = TorchDistributedDeterministicCollective.all_gather

    def poisoned_gather(collective, value, **kwargs):
        poisoned = value.clone()
        # Padding is deliberately unrelated to the image. It must be removed
        # even when it contains NaNs, rather than relying on zero-filled rows.
        poisoned[layout.local_length :] = float("nan")
        return original_gather(collective, poisoned, **kwargs)

    with patch.object(TorchDistributedDeterministicCollective, "all_gather", poisoned_gather):
        _assert_bytes(gather_sequence(local, layout, group=group), image_cpu)

    gradient_cpu = (torch.arange(image_cpu.numel()).remainder(13) - 6).reshape(image_cpu.shape)
    gradient_cpu = gradient_cpu.to(dtype)
    local.retain_grad()
    # One owner contributes the full loss; every other consumer contributes
    # zero but still enters backward's collectives in the same order.
    gradient = gradient_cpu.to(device) if rank == 0 else torch.zeros_like(restored)
    restored.backward(gradient)
    _assert_bytes(local.grad, gradient_cpu.index_select(1, indices))
    _assert_bytes(image.grad, gradient_cpu)

    # Test shard's backward independently of gather's backward as well.
    image.grad = None
    local, _, _ = shard_image_sequence(image, positions, group=group)
    local.backward(gradient_cpu.index_select(1, indices).to(device))
    _assert_bytes(image.grad, gradient_cpu)

    if batch == 2:
        # The same sample must survive moving to another batch position or
        # being run alone, including its own batched position metadata.
        for order in ([1, 0], [1]):
            selected_image = image_cpu[order].to(device)
            selected_positions = positions_cpu[order] if batched_positions else positions_cpu
            shard, _, selected_layout = shard_image_sequence(
                selected_image, selected_positions.to(device), group=group
            )
            _assert_bytes(gather_sequence(shard, selected_layout, group=group), image_cpu[order])


def _check_consumer_gradient_sum(dtype, device):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    layout = SequenceParallelLayout.from_group(7)
    local = torch.zeros(2, layout.local_length, 3, dtype=dtype, device=device, requires_grad=True)
    restored = gather_sequence(local, layout)
    values = (256.0, 1.0, -256.0, 1.0) if dtype == torch.bfloat16 else (1e20, 1.0, -1e20, 1.0)
    contributions = torch.tensor(values[:world_size], dtype=dtype).float()
    if world_size == 1:
        expected = contributions[0]
    elif world_size == 2:
        expected = contributions[0] + contributions[1]
    else:
        expected = (contributions[0] + contributions[1]) + (contributions[2] + contributions[3])
    restored.backward(torch.full_like(restored, values[rank]))
    _assert_bytes(local.grad, torch.full_like(local, expected.item()))


def _distributed_worker(rank, world_size, port, backend):
    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=90),
    )
    try:
        cases = [
            (1, 1, 3, False),
            (2, 3, 5, True),
            (2, 7, 5, False),
            (2, 8, 5, True),
            (1, 6889, 5, False),
        ]
        if backend == "nccl":
            cases += [
                (batch, length, 3072, True) for batch in (1, 2) for length in (4096, 6032, 6889)
            ]
        for dtype in (torch.float32, torch.bfloat16):
            for batch, length, channels, batched_positions in cases:
                _check_roundtrip(batch, length, channels, dtype, batched_positions, device, None)
            _check_consumer_gradient_sum(dtype, device)
        if world_size == 4:
            # Noncontiguous subgroups catch accidental use of global rank.
            groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
            _check_roundtrip(
                2, 7, 5, torch.float32, True, device, groups[rank % 2], seed=386 + rank % 2
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.ws2
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_sequence_parallel_distributed(world_size, backend):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("this test owns its workers; run pytest without torchrun")
    if backend == "nccl" and (
        torch.version.hip is not None or torch.cuda.device_count() < world_size
    ):
        pytest.skip(f"requires {world_size} NVIDIA GPUs")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    mp.spawn(_distributed_worker, args=(world_size, port, backend), nprocs=world_size, join=True)
