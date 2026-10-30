# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Exercise the self-owned deterministic AG/RS/AllReduce on an Attention shape.

This is intentionally a small transport probe, not a replacement for the CP
attention reference.  It checks the exact communication primitives required by
the table: AG for Q/SP, FP32 RS for `(Out, LSE)`, and AllReduce for the o_proj
partial sum.  Run under ``torchrun`` on one host with 2, 4, or 8 ranks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rows < 1:
        raise ValueError("rows must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("self-owned CUDA collective check requires CUDA")
    dist.init_process_group("nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4, 8):
        raise RuntimeError(f"expected 2, 4, or 8 ranks, got {world_size}")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        from rl_engine.distributed import DeterministicCollective

        with DeterministicCollective(device=device, max_size_bytes=16 * 1024 * 1024) as collective:
            operations = {
                "all_gather_q": _check_all_gather(collective, rank, world_size, args.rows, device),
                "reduce_scatter_out_lse": _check_reduce_scatter(
                    collective, rank, world_size, args.rows, device
                ),
                "all_reduce_o_proj": _check_all_reduce(
                    collective, rank, world_size, args.rows, device
                ),
            }
        passed = all(bool(item["passed"]) for item in operations.values())
        result = {
            "rank": rank,
            "device": str(device),
            "world_size": world_size,
            "transport": "self_owned_cuda_ag_rs",
            "allreduce_transport": "self_owned_cuda_allreduce",
            "accumulation_dtype": "fp32",
            "downcast_at": "final_write",
            "operations": operations,
            "passed": passed,
        }
        failures = torch.tensor([0 if passed else 1], dtype=torch.int32, device=device)
        dist.all_reduce(failures, op=dist.ReduceOp.SUM)
        reports: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(reports, result)
        if rank == 0:
            payload = {
                "schema_version": "ws2_deterministic_attention_collectives/v1",
                "world_size": world_size,
                "transport": "self_owned_cuda_ag_rs",
                "allreduce_transport": "self_owned_cuda_allreduce",
                "global_failure_count": int(failures.item()),
                "ranks": reports,
                "passed": int(failures.item()) == 0,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if int(failures.item()) == 0 else 1
    finally:
        dist.destroy_process_group()


def _check_all_gather(collective, rank: int, world_size: int, rows: int, device: torch.device):
    local = (
        torch.arange(rows * 8, device=device, dtype=torch.bfloat16).reshape(rows, 8) + rank * 100
    )
    expected = torch.cat(
        [
            torch.arange(rows * 8, device=device, dtype=torch.bfloat16).reshape(rows, 8)
            + peer_rank * 100
            for peer_rank in range(world_size)
        ],
        dim=0,
    )
    out = collective.all_gather(local)
    repeat = collective.all_gather(local)
    return {
        "dtype": "bf16",
        "passed": bool(torch.equal(out, expected) and torch.equal(out, repeat)),
    }


def _check_reduce_scatter(collective, rank: int, world_size: int, rows: int, device: torch.device):
    local = torch.full((rows * world_size, 9), float(rank + 1), device=device, dtype=torch.float32)
    expected = torch.full(
        (rows, 9),
        float(sum(range(1, world_size + 1))),
        device=device,
        dtype=torch.float32,
    )
    out = collective.reduce_scatter(local)
    repeat = collective.reduce_scatter(local)
    return {
        "dtype": "fp32",
        "passed": bool(torch.equal(out, expected) and torch.equal(out, repeat)),
    }


def _check_all_reduce(collective, rank: int, world_size: int, rows: int, device: torch.device):
    local = torch.full((rows, 11), float(rank + 1), device=device, dtype=torch.float32)
    expected = torch.full(
        (rows, 11),
        float(sum(range(1, world_size + 1))),
        device=device,
        dtype=torch.float32,
    )
    out = collective.all_reduce(local)
    repeat = collective.all_reduce(local)
    return {
        "dtype": "fp32",
        "passed": bool(torch.equal(out, expected) and torch.equal(out, repeat)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
