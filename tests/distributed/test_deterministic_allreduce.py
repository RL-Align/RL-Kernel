# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pytest
import torch
import torch.distributed as dist

from rl_engine.distributed import (
    DETERMINISTIC_NCCL_ENV,
    DeterministicAllReduceConfig,
    configure_deterministic_nccl_env,
    deterministic_all_reduce,
)


def test_configure_deterministic_nccl_env_preserves_existing_value(monkeypatch):
    monkeypatch.setenv("NCCL_ALGO", "Tree")
    with pytest.warns(RuntimeWarning, match="NCCL_ALGO"):
        previous = configure_deterministic_nccl_env()

    assert previous["NCCL_ALGO"] == "Tree"
    assert os.environ["NCCL_ALGO"] == "Tree"
    for key, value in DETERMINISTIC_NCCL_ENV.items():
        if key != "NCCL_ALGO":
            assert os.environ[key] == value


def test_configure_deterministic_nccl_env_can_overwrite(monkeypatch):
    monkeypatch.setenv("NCCL_ALGO", "Tree")
    configure_deterministic_nccl_env(overwrite=True)

    assert os.environ["NCCL_ALGO"] == "Ring"


def test_single_process_without_process_group_is_noop():
    tensor = torch.tensor([1.0, 2.0, 3.0])

    reduced = deterministic_all_reduce(
        tensor,
        DeterministicAllReduceConfig(mode="ordered_rank_fallback", op="mean"),
    )

    assert reduced is tensor
    assert torch.equal(tensor, torch.tensor([1.0, 2.0, 3.0]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic all-reduce smoke test")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument(
        "--mode",
        choices=("ordered_rank_fallback", "nccl_ring"),
        default="ordered_rank_fallback",
    )
    parser.add_argument("--op", choices=("sum", "mean"), default="sum")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--numel", type=int, default=257)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--configure-nccl-env", action="store_true")
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu" or args.backend == "gloo":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _make_input(rank: int, dtype: torch.dtype, device: torch.device, numel: int) -> torch.Tensor:
    base = torch.arange(numel, dtype=torch.float32, device=device)
    values = ((base % 17) - 8.0) / 17.0
    values = values + (rank + 1) * 0.03125
    return values.to(dtype=dtype)


def _tolerances(dtype: torch.dtype, args: argparse.Namespace) -> tuple[float, float]:
    if args.atol is not None and args.rtol is not None:
        return args.atol, args.rtol
    if dtype == torch.float32:
        return 0.0 if args.mode == "ordered_rank_fallback" else 1.0e-6, 0.0
    if dtype == torch.bfloat16:
        return 8.0e-3, 8.0e-3
    return 2.0e-3, 2.0e-3


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_f32 = actual.detach().to(torch.float32).cpu()
    expected_f32 = expected.detach().to(torch.float32).cpu()
    diff = (actual_f32 - expected_f32).abs()
    denom = expected_f32.abs().clamp_min(1.0e-12)
    rel = diff / denom
    return {
        "bitwise_equal": bool(torch.equal(actual.detach().cpu(), expected.detach().cpu())),
        "max_abs_diff": float(diff.max().item()),
        "max_rel_diff": float(rel.max().item()),
        "mismatch_count": int((diff != 0).sum().item()),
    }


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        stats = _diff_stats(actual, expected)
        raise AssertionError(
            "deterministic all-reduce mismatch: "
            f"max_abs_diff={stats['max_abs_diff']} "
            f"max_rel_diff={stats['max_rel_diff']} "
            f"mismatch_count={stats['mismatch_count']}"
        )


def _run_distributed_smoke(args: argparse.Namespace) -> None:
    if args.configure_nccl_env or (args.backend == "nccl" and args.mode == "nccl_ring"):
        configure_deterministic_nccl_env()
    device = _device(args)
    dist.init_process_group(backend=args.backend)
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dtype = _dtype(args.dtype)
        atol, rtol = _tolerances(dtype, args)
        previous: torch.Tensor | None = None
        final_stats: dict[str, Any] = {}

        for _ in range(args.iterations):
            original = _make_input(rank, dtype, device, args.numel)
            candidate = original.clone()
            reference = original.clone()

            deterministic_all_reduce(
                candidate,
                DeterministicAllReduceConfig(mode=args.mode, op=args.op),
            )
            deterministic_all_reduce(
                reference,
                DeterministicAllReduceConfig(mode="ordered_rank_fallback", op=args.op),
            )

            _assert_close(candidate, reference, atol=atol, rtol=rtol)
            if previous is not None:
                _assert_close(candidate, previous, atol=atol, rtol=rtol)
            previous = candidate.clone()
            final_stats = _diff_stats(candidate, reference)

        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "backend": args.backend,
                        "mode": args.mode,
                        "op": args.op,
                        "dtype": args.dtype,
                        "device": str(device),
                        "world_size": world_size,
                        "iterations": args.iterations,
                        **final_stats,
                    },
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _run_distributed_smoke(_parse_args())
