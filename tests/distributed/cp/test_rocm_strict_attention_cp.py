# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""End-to-end CP coverage for the strict ROCm attention provider.

Acceptance is bitwise against a CP=1 run of the same core on the same logical
sequence. CP performs no arithmetic of its own here: the runtime all-gathers
Q/K/V, runs the core once over the full sequence, and scatters the root's
authoritative ``(out, lse)`` back to each rank's query range. Anything other
than bit equality means the CP path introduced a second merge order.

These run through ``attention_provider`` rather than the transport directly, so
they also pin that CP is reachable from the production dispatch path.
"""

from __future__ import annotations

import os
import socket
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.runtime.registry import _rocm_strict_attention_available

_MAX_WORLD_SIZE = 8
_CP_SIZES = (2, 4, 8)
_EXTERNAL_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

# Qwen3-8B dense head layout at TP=1.
_GLOBAL_Q_HEADS = 32
_GLOBAL_KV_HEADS = 8
_HEAD_DIM = 128
_GLOBAL_SEQ = 256

pytestmark = [
    pytest.mark.skipif(
        _EXTERNAL_WORLD_SIZE != 1,
        reason="this cross-CP test owns its worker processes; run pytest directly",
    ),
    pytest.mark.skipif(
        torch.cuda.device_count() < _MAX_WORLD_SIZE,
        reason="requires eight visible ROCm GPUs",
    ),
    pytest.mark.skipif(
        not _rocm_strict_attention_available(),
        reason="strict ROCm attention requires a ROCm device with aiter.ops.mha",
    ),
]


def _global_qkv(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identical global Q/K/V on every rank, generated on CPU for bit equality."""

    def _make(shape: tuple[int, ...], seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        value = torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.02
        return value.to(device=device, dtype=torch.bfloat16)

    q = _make((1, _GLOBAL_Q_HEADS, _GLOBAL_SEQ, _HEAD_DIM), 11)
    k = _make((1, _GLOBAL_KV_HEADS, _GLOBAL_SEQ, _HEAD_DIM), 22)
    v = _make((1, _GLOBAL_KV_HEADS, _GLOBAL_SEQ, _HEAD_DIM), 33)
    return q, k, v


def _request(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cp_world_size: int,
    cp_rank: int,
    cp_layout: str,
    token_start: int,
    cp_group: object | None,
) -> SimpleNamespace:
    kv_len = k.shape[2]
    positions = torch.arange(
        token_start,
        token_start + kv_len,
        device=q.device,
        dtype=torch.int64,
    )
    return SimpleNamespace(
        query=q,
        key=k,
        value=v,
        key_padding_mask=None,
        # TP=1 must be stated explicitly: with torch.distributed initialized,
        # a None TP group resolves to the global group, i.e. TP=8 here.
        tensor_parallel_group=SimpleNamespace(rank=lambda: 0, size=lambda: 1),
        context_parallel=SimpleNamespace(
            world_size=cp_world_size,
            rank=cp_rank,
            layout=cp_layout,
        ),
        context_parallel_group=cp_group,
        metadata={
            "global_q_heads": _GLOBAL_Q_HEADS,
            "global_kv_heads": _GLOBAL_KV_HEADS,
            "tp_rank": 0,
            "tp_world_size": 1,
            "attention_mode": "prefill",
            "role": "train",
            "causal": True,
            "key_position_ids": positions,
        },
    )


def _check_one_cp_degree(group: object, cp_size: int, rank: int, device: torch.device) -> None:
    from rl_engine.integrations.orchestrators.vime.providers.attention import attention_provider

    q_global, k_global, v_global = _global_qkv(device)

    # CP=1 on the whole logical sequence is the acceptance reference.
    reference = attention_provider(
        _request(
            q_global,
            k_global,
            v_global,
            cp_world_size=1,
            cp_rank=0,
            cp_layout="single",
            token_start=0,
            cp_group=None,
        )
    )

    local_seq = _GLOBAL_SEQ // cp_size
    lo, hi = rank * local_seq, (rank + 1) * local_seq
    result = attention_provider(
        _request(
            q_global[:, :, lo:hi].contiguous(),
            k_global[:, :, lo:hi].contiguous(),
            v_global[:, :, lo:hi].contiguous(),
            cp_world_size=cp_size,
            cp_rank=rank,
            cp_layout="allgather",
            token_start=lo,
            cp_group=group,
        )
    )

    assert result.out.shape == (1, _GLOBAL_Q_HEADS, local_seq, _HEAD_DIM)
    assert result.lse.shape == (1, _GLOBAL_Q_HEADS, local_seq)

    expected_out = reference.out[:, :, lo:hi]
    expected_lse = reference.lse[:, :, lo:hi]
    out_mismatch = int((result.out != expected_out).sum().item())
    lse_mismatch = int((result.lse != expected_lse).sum().item())
    assert out_mismatch == 0, f"CP={cp_size} rank={rank}: {out_mismatch} out elements differ"
    assert lse_mismatch == 0, f"CP={cp_size} rank={rank}: {lse_mismatch} lse elements differ"

    provenance = result.provenance
    assert provenance["cp_row_ownership"]["cp_is_merge_axis"] is True
    assert provenance["cp_row_ownership"]["cp_merge_owner"] == "rccl_ag_rs"
    assert provenance["cp_row_ownership"]["cp_world_size"] == cp_size
    # The core still runs once per (batch row, KV group), now over the gathered
    # global sequence rather than this rank's shard.
    assert provenance["execution"]["core_launches"] == _GLOBAL_KV_HEADS

    # Repeating the CP call must reproduce its own bits.
    repeated = attention_provider(
        _request(
            q_global[:, :, lo:hi].contiguous(),
            k_global[:, :, lo:hi].contiguous(),
            v_global[:, :, lo:hi].contiguous(),
            cp_world_size=cp_size,
            cp_rank=rank,
            cp_layout="allgather",
            token_start=lo,
            cp_group=group,
        )
    )
    assert torch.equal(repeated.out, result.out)
    assert torch.equal(repeated.lse, result.lse)


def _worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=_MAX_WORLD_SIZE,
        timeout=timedelta(minutes=10),
    )
    try:
        for cp_size in _CP_SIZES:
            group = dist.new_group(ranks=list(range(cp_size)))
            if rank < cp_size:
                _check_one_cp_degree(group, cp_size, rank, device)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_strict_rocm_attention_cp_is_bitwise_against_cp1() -> None:
    mp.spawn(
        _worker,
        args=(_find_free_port(),),
        nprocs=_MAX_WORLD_SIZE,
        join=True,
    )
