# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from rl_engine.distributed import (
    DeterministicAllReduceConfig,
    configure_deterministic_nccl_env,
    deterministic_all_reduce,
)


class TinyGradientModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@dataclass(frozen=True)
class GradientStats:
    bitwise_equal: bool
    max_abs_diff: float
    max_rel_diff: float
    mismatch_count: int


def test_fixed_batch_is_reproducible():
    first = _fixed_batch(global_batch_size=8, input_dim=3, output_dim=2)
    second = _fixed_batch(global_batch_size=8, input_dim=3, output_dim=2)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_dp_gradient_gloo_smoke_runs_under_torchrun():
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
    parser = argparse.ArgumentParser(description="DP gradient determinism smoke test")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument(
        "--mode",
        choices=("ordered_rank_reference", "torch_all_reduce"),
        default="ordered_rank_reference",
    )
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=13)
    parser.add_argument("--output-dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
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


def _set_deterministic_controls(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _fixed_batch(
    *,
    global_batch_size: int,
    input_dim: int,
    output_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.linspace(
        -1.0,
        1.0,
        steps=global_batch_size * input_dim,
        dtype=torch.float32,
    ).reshape(global_batch_size, input_dim)
    targets = torch.cos(
        torch.linspace(
            -0.7,
            0.9,
            steps=global_batch_size * output_dim,
            dtype=torch.float32,
        )
    ).reshape(global_batch_size, output_dim)
    return inputs, targets


def _make_model(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> TinyGradientModel:
    torch.manual_seed(seed)
    model = TinyGradientModel(input_dim, hidden_dim, output_dim)
    return model.to(device=device, dtype=dtype)


def _compute_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    batch_inputs = inputs.to(device=device, dtype=dtype)
    batch_targets = targets.to(device=device, dtype=dtype)
    predictions = model(batch_inputs)
    loss = F.mse_loss(predictions.float(), batch_targets.float(), reduction="mean")
    loss.backward()
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _reduce_gradients(model: torch.nn.Module, mode: str) -> dict[str, torch.Tensor]:
    reduced: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        deterministic_all_reduce(
            parameter.grad,
            DeterministicAllReduceConfig(mode=mode, op="mean"),
        )
        reduced[name] = parameter.grad.detach().clone()
    return reduced


def _stats(actual: torch.Tensor, expected: torch.Tensor) -> GradientStats:
    actual_f32 = actual.detach().to(torch.float32).cpu()
    expected_f32 = expected.detach().to(torch.float32).cpu()
    diff = (actual_f32 - expected_f32).abs()
    rel = diff / expected_f32.abs().clamp_min(1.0e-12)
    return GradientStats(
        bitwise_equal=bool(torch.equal(actual.detach().cpu(), expected.detach().cpu())),
        max_abs_diff=float(diff.max().item()),
        max_rel_diff=float(rel.max().item()),
        mismatch_count=int((diff != 0).sum().item()),
    )


def _tolerances(dtype: torch.dtype, args: argparse.Namespace) -> tuple[float, float]:
    if args.atol is not None and args.rtol is not None:
        return args.atol, args.rtol
    if dtype == torch.float32:
        return 1.0e-5, 1.0e-5
    if dtype == torch.bfloat16:
        return 2.0e-2, 2.0e-2
    return 5.0e-3, 5.0e-3


def _compare_gradients(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> tuple[GradientStats, list[dict[str, Any]]]:
    if set(actual) != set(expected):
        raise AssertionError(
            f"gradient key mismatch: actual={sorted(actual)}, expected={sorted(expected)}"
        )

    global_stats = GradientStats(True, 0.0, 0.0, 0)
    parameters: list[dict[str, Any]] = []
    for name in sorted(actual):
        param_stats = _stats(actual[name], expected[name])
        parameters.append({"name": name, **param_stats.__dict__})
        global_stats = GradientStats(
            bitwise_equal=global_stats.bitwise_equal and param_stats.bitwise_equal,
            max_abs_diff=max(global_stats.max_abs_diff, param_stats.max_abs_diff),
            max_rel_diff=max(global_stats.max_rel_diff, param_stats.max_rel_diff),
            mismatch_count=global_stats.mismatch_count + param_stats.mismatch_count,
        )
        if not torch.allclose(actual[name], expected[name], atol=atol, rtol=rtol):
            raise AssertionError(
                f"gradient mismatch for {name}: "
                f"max_abs_diff={param_stats.max_abs_diff} "
                f"max_rel_diff={param_stats.max_rel_diff} "
                f"mismatch_count={param_stats.mismatch_count}"
            )
    return global_stats, parameters


def _run_distributed_smoke(args: argparse.Namespace) -> None:
    if args.configure_nccl_env or (args.backend == "nccl" and args.mode == "torch_all_reduce"):
        configure_deterministic_nccl_env()
    device = _device(args)
    _set_deterministic_controls(args.seed)
    dist.init_process_group(backend=args.backend)
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if args.global_batch_size % world_size != 0:
            raise ValueError("global batch size must be divisible by world size")

        dtype = _dtype(args.dtype)
        atol, rtol = _tolerances(dtype, args)
        inputs, targets = _fixed_batch(
            global_batch_size=args.global_batch_size,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
        )

        baseline_model = _make_model(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            seed=args.seed,
            dtype=dtype,
            device=device,
        )
        baseline_grads = _compute_gradients(
            baseline_model,
            inputs,
            targets,
            dtype=dtype,
            device=device,
        )

        local_batch_size = args.global_batch_size // world_size
        start = rank * local_batch_size
        end = start + local_batch_size
        dp_model = _make_model(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            seed=args.seed,
            dtype=dtype,
            device=device,
        )
        _compute_gradients(
            dp_model,
            inputs[start:end],
            targets[start:end],
            dtype=dtype,
            device=device,
        )
        reduced_grads = _reduce_gradients(dp_model, args.mode)
        global_stats, parameter_stats = _compare_gradients(
            reduced_grads,
            baseline_grads,
            atol=atol,
            rtol=rtol,
        )

        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "backend": args.backend,
                        "mode": args.mode,
                        "dtype": args.dtype,
                        "device": str(device),
                        "world_size": world_size,
                        "global_batch_size": args.global_batch_size,
                        "atol": atol,
                        "rtol": rtol,
                        **global_stats.__dict__,
                        "parameters": parameter_stats,
                    },
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _run_distributed_smoke(_parse_args())
