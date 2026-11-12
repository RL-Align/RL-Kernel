# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""TP4/CP2 bitwise acceptance for the dedicated ROCm FFN backward."""

from __future__ import annotations

import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_WORLD_SIZE = 8
_TP_SIZE = 4
_CP_SIZE = 2
_EXTERNAL_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

pytestmark = [
    pytest.mark.skipif(
        _EXTERNAL_WORLD_SIZE != 1,
        reason="this TP/CP test owns its worker processes; run pytest directly",
    ),
    pytest.mark.skipif(
        torch.version.hip is None or torch.cuda.device_count() < _WORLD_SIZE,
        reason="requires eight visible ROCm GPUs",
    ),
]


def _bf16(shape: tuple[int, ...], *, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=generator).to(device=device, dtype=torch.bfloat16)


def _run_ffn(ffn, *, rank: int, tp_rank: int, tp_group, cp_group, device: torch.device):
    hidden = _bf16((4, 64), seed=1000 + rank, device=device).requires_grad_(True)
    packed = _bf16((64, 64), seed=2000 + tp_rank, device=device).requires_grad_(True)
    down = _bf16((64, 32), seed=3000 + tp_rank, device=device).requires_grad_(True)
    grad_output = _bf16((4, 64), seed=4000 + rank, device=device)
    gate, up = packed.chunk(2, dim=0)

    output = ffn(
        hidden,
        gate,
        up,
        down,
        fused_gate_up_weight=packed,
        tp_group=tp_group,
        cp_group=cp_group,
        sequence_parallel=True,
        deterministic=True,
    )
    output.backward(grad_output)
    return output.detach(), hidden.grad.detach(), packed.grad.detach(), down.grad.detach()


def _worker(rank: int, port: int) -> None:
    from rl_engine.backends.rocm.ffn import qwen3_ffn_training
    from rl_engine.backends.rocm.ffn.ffn import qwen3_ffn

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(minutes=10),
    )
    try:
        tp_groups = [
            dist.new_group(ranks=list(range(cp_rank * _TP_SIZE, (cp_rank + 1) * _TP_SIZE)))
            for cp_rank in range(_CP_SIZE)
        ]
        cp_groups = [
            dist.new_group(ranks=[tp_rank, tp_rank + _TP_SIZE]) for tp_rank in range(_TP_SIZE)
        ]
        tp_rank = rank % _TP_SIZE
        cp_rank = rank // _TP_SIZE
        tp_group = tp_groups[cp_rank]
        cp_group = cp_groups[tp_rank]
        os.environ["RL_KERNEL_STRICT_CANONICAL_TP"] = str(_TP_SIZE)

        reference = _run_ffn(
            qwen3_ffn,
            rank=rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            cp_group=cp_group,
            device=device,
        )
        specialized = _run_ffn(
            qwen3_ffn_training,
            rank=rank,
            tp_rank=tp_rank,
            tp_group=tp_group,
            cp_group=cp_group,
            device=device,
        )
        labels = ("output", "input gradient", "gate/up gradient", "down gradient")
        for label, expected, actual in zip(labels, reference, specialized, strict=True):
            mismatch = int((expected != actual).sum().item())
            assert mismatch == 0, f"rank={rank} {label}: {mismatch} elements differ"
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_rocm_training_backward_matches_reference_at_tp4_cp2() -> None:
    mp.spawn(_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)
