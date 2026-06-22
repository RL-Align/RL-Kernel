# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
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
        DeterministicAllReduceConfig(mode="ordered_rank_reference", op="mean"),
    )

    assert reduced is tensor
    assert torch.equal(tensor, torch.tensor([1.0, 2.0, 3.0]))


def test_ordered_rank_reference_gloo_smoke_runs_under_torchrun():
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("Gloo is unavailable")

    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}{os.pathsep}{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--backend",
        "gloo",
        "--mode",
        "ordered_rank_reference",
        "--dtype",
        "fp32",
        "--device",
        "cpu",
        "--iterations",
        "2",
    ]
    completed = subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert '"status": "pass"' in completed.stdout


def test_ordered_rank_reference_reverse_group_runs_under_torchrun():
    if not (dist.is_available() and dist.is_gloo_available()):
        pytest.skip("Gloo is unavailable")

    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo}{os.pathsep}{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--backend",
        "gloo",
        "--mode",
        "ordered_rank_reference",
        "--dtype",
        "fp32",
        "--device",
        "cpu",
        "--iterations",
        "2",
        "--reverse-group",
    ]
    completed = subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout
    assert '"status": "pass"' in completed.stdout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="all-reduce smoke test")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument(
        "--mode",
        choices=("ordered_rank_reference", "torch_all_reduce"),
        default="ordered_rank_reference",
    )
    parser.add_argument("--op", choices=("sum", "mean"), default="sum")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--numel", type=int, default=257)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--configure-nccl-env", action="store_true")
    parser.add_argument("--reverse-group", action="store_true")
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu" or args.backend == "gloo":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _make_input(rank: int, dtype: torch.dtype, device: torch.device, numel: int) -> torch.Tensor:
    base = torch.arange(numel, dtype=torch.float32, device=device)
    values = ((base % 17) - 8.0) / 17.0
    return (values + (rank + 1) * 0.03125).to(dtype=dtype)


def _group_rank_order(group: dist.ProcessGroup | None, device: torch.device) -> list[int]:
    rank = torch.tensor([dist.get_rank()], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(rank) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(gathered, rank, group=group)
    return [int(item.item()) for item in gathered]


def _expected_reduce(
    rank_order: list[int],
    dtype: torch.dtype,
    device: torch.device,
    numel: int,
    op: str,
) -> torch.Tensor:
    acc_dtype = torch.float32 if dtype != torch.float64 else torch.float64
    reduced = _make_input(rank_order[0], dtype, device, numel).to(dtype=acc_dtype)
    for rank in rank_order[1:]:
        reduced.add_(_make_input(rank, dtype, device, numel).to(dtype=acc_dtype))
    if op == "mean":
        reduced.div_(len(rank_order))
    return reduced.to(dtype=dtype)


def _tolerances(dtype: torch.dtype, args: argparse.Namespace) -> tuple[float, float]:
    if args.atol is not None and args.rtol is not None:
        return args.atol, args.rtol
    if dtype == torch.float32:
        return (0.0, 0.0) if args.mode == "ordered_rank_reference" else (1.0e-6, 0.0)
    if dtype == torch.bfloat16:
        return 8.0e-3, 8.0e-3
    return 2.0e-3, 2.0e-3


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_f32 = actual.detach().to(torch.float32).cpu()
    expected_f32 = expected.detach().to(torch.float32).cpu()
    diff = (actual_f32 - expected_f32).abs()
    rel = diff / expected_f32.abs().clamp_min(1.0e-12)
    return {
        "bitwise_equal": bool(torch.equal(actual.detach().cpu(), expected.detach().cpu())),
        "max_abs_diff": float(diff.max().item()),
        "max_rel_diff": float(rel.max().item()),
        "mismatch_count": int((diff != 0).sum().item()),
    }


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        stats = _diff_stats(actual, expected)
        raise AssertionError(f"all-reduce mismatch: {stats}")


def _run_distributed_smoke(args: argparse.Namespace) -> None:
    if args.configure_nccl_env or (args.backend == "nccl" and args.mode == "torch_all_reduce"):
        configure_deterministic_nccl_env()

    device = _device(args)
    dist.init_process_group(backend=args.backend)
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        group = None
        if args.reverse_group:
            group = dist.new_group(ranks=list(reversed(range(world_size))))

        dtype = _dtype(args.dtype)
        atol, rtol = _tolerances(dtype, args)
        rank_order = _group_rank_order(group, device)
        expected = _expected_reduce(rank_order, dtype, device, args.numel, args.op)
        previous: torch.Tensor | None = None
        stats: dict[str, Any] = {}

        for _ in range(args.iterations):
            candidate = _make_input(rank, dtype, device, args.numel)
            reference = candidate.clone()
            deterministic_all_reduce(
                candidate,
                DeterministicAllReduceConfig(mode=args.mode, op=args.op, group=group),
            )
            deterministic_all_reduce(
                reference,
                DeterministicAllReduceConfig(
                    mode="ordered_rank_reference",
                    op=args.op,
                    group=group,
                ),
            )
            _assert_close(candidate, expected, atol, rtol)
            _assert_close(reference, expected, atol, rtol)
            if previous is not None:
                _assert_close(candidate, previous, atol, rtol)
            previous = candidate.clone()
            stats = _diff_stats(candidate, expected)

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
                        **stats,
                    },
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _run_distributed_smoke(_parse_args())
