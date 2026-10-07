# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Stage-by-stage replay of the MiniMax-H3 conditioning chain (RFC #420).

Each stage runs three ways on the same device:

* provider: the diffusers op sequence (``rl_engine.testing.h3_provider``);
* candidate: the dispatched RL-Kernel op, fed the previous candidate output
  (chained) and, separately, the previous provider output (isolated);
* golden: the FP32/FP64 reference, fed the previous golden output.

A stage also declares what it promises, so the end-to-end test and the
evidence JSON check the same thing: whether its isolated output is bitwise
equal to the provider, and its tolerance against the golden (rows of
``tolerance_contract.json``). ``first_drift`` is the first stage whose chained
output differs from the provider; ``first_isolated_drift`` the first stage
that differs even on provider inputs.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from rl_engine.kernels.registry import KernelRegistry
from rl_engine.testing.h3_cases import h3_packed_layout, h3_timesteps
from rl_engine.testing.h3_provider import (
    provider_adaln_modulation,
    provider_time_embedder,
    provider_time_proj,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Stage:
    name: str
    op_type: str
    # (op, ctx, upstream) -> outputs; ``upstream`` is the previous stage's output.
    candidate: Callable[[Any, dict, Any], Any]
    provider: Callable[[dict, Any], Any]
    golden: Callable[[Any, dict, Any], Any]
    # Promises checked by tests/h3/test_h3_conditioning_e2e.py.
    provider_bitwise_isolated: bool
    golden_atol: float
    golden_rtol: float


def mlp_params(ctx: dict[str, Any]) -> list[torch.Tensor]:
    weights = ctx["weights"]
    return [weights[f"time_embedder.linear_{i}.{p}"] for i in (1, 2) for p in ("weight", "bias")]


def adaln_params(ctx: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = "transformer_blocks.0.adaln_proj.linear"
    return ctx["weights"][f"{prefix}.weight"], ctx["weights"][f"{prefix}.bias"]


STAGES: list[Stage] = [
    Stage(
        name="timestep_sinusoid_h3",
        op_type="timestep_sinusoid_h3",
        candidate=lambda op, ctx, _up: op(ctx["timestep"]),
        provider=lambda ctx, _up: provider_time_proj(ctx["timestep"]),
        golden=lambda op, ctx, _up: op.forward_fp32(ctx["timestep"]),
        provider_bitwise_isolated=True,
        golden_atol=1e-5,  # elementwise / float32
        golden_rtol=1e-5,
    ),
    Stage(
        name="timestep_mlp_fp32",
        op_type="timestep_mlp_fp32",
        candidate=lambda op, ctx, up: op(up, *mlp_params(ctx)),
        provider=lambda ctx, up: provider_time_embedder(up, *mlp_params(ctx)),
        golden=lambda op, ctx, up: op.forward_fp32(up, *mlp_params(ctx)),
        provider_bitwise_isolated=False,  # reduction: different tree from cuBLAS
        golden_atol=1e-4,  # reduction / float32
        golden_rtol=1e-4,
    ),
    Stage(
        name="adaln_projection_3mod",
        op_type="adaln_projection_3mod",
        candidate=lambda op, ctx, up: op(up, *adaln_params(ctx)),
        provider=lambda ctx, up: provider_adaln_modulation(up, *adaln_params(ctx)),
        golden=lambda op, ctx, up: op.forward_fp32(up, *adaln_params(ctx)),
        provider_bitwise_isolated=False,  # reduction: tensor-core tree differs from cuBLAS
        golden_atol=5e-2,  # reduction / bfloat16
        golden_rtol=2e-2,
    ),
]


def _flat(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    return [tensor for item in value for tensor in _flat(item)]


def compare(lhs: Any, rhs: Any, atol: float = 0.0, rtol: float = 0.0) -> dict[str, Any]:
    lhs_t, rhs_t = _flat(lhs), _flat(rhs)
    bitwise = all(torch.equal(a, b) for a, b in zip(lhs_t, rhs_t, strict=True))
    max_abs = 0.0
    within = True
    for a, b in zip(lhs_t, rhs_t, strict=True):
        diff = (a.float() - b.float()).abs()
        if diff.numel():
            max_abs = max(max_abs, float(diff.max()))
            within = within and bool((diff <= atol + rtol * b.float().abs()).all())
    return {"bitwise_equal": bitwise, "max_abs": max_abs, "within_tolerance": within}


def make_context(weights, *, num_timesteps: int, seq_len: int, seed: int) -> dict[str, Any]:
    timestep_indices, token_tags = h3_packed_layout(seq_len, num_timesteps, seed=seed)
    return {
        "timestep": h3_timesteps(num_timesteps, seed=seed),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "weights": weights,
    }


def golden_op(registry: KernelRegistry, op_type: str):
    """The PyTorch reference: the last entry of the CPU priority list."""

    return registry._get_or_create_backend(registry._priority_map["cpu"][op_type][-1])


def run_case(
    registry: KernelRegistry,
    weights,
    *,
    num_timesteps: int,
    seq_len: int,
    seed: int = 0,
    stages: list[Stage] | None = None,
) -> dict[str, Any]:
    stages = STAGES if stages is None else stages
    ctx = make_context(weights, num_timesteps=num_timesteps, seq_len=seq_len, seed=seed)
    report: dict[str, Any] = {
        "num_timesteps": num_timesteps,
        "seq_len": seq_len,
        "seed": seed,
        "stages": [],
    }
    chained = provider = golden = None
    first_drift = first_isolated = None
    for stage in stages:
        op = registry.get_op(stage.op_type, device="cuda")
        gold = golden_op(registry, stage.op_type)
        with torch.no_grad():
            isolated = stage.candidate(op, ctx, provider)
            chained_input = chained
            chained = stage.candidate(op, ctx, chained_input)
            repeat = stage.candidate(op, ctx, chained_input)
            golden = stage.golden(gold, ctx, golden)
            provider = stage.provider(ctx, provider)
        entry = {
            "stage": stage.name,
            "backend": type(op).__name__,
            "kernel_id": getattr(op, "kernel_id", type(op).__name__),
            "repeat_bitwise_equal": compare(repeat, chained)["bitwise_equal"],
            "chained_vs_provider": compare(chained, provider),
            "isolated_vs_provider": compare(isolated, provider),
            "chained_vs_golden": compare(chained, golden, stage.golden_atol, stage.golden_rtol),
            "provider_vs_golden": compare(provider, golden, stage.golden_atol, stage.golden_rtol),
        }
        if first_drift is None and not entry["chained_vs_provider"]["bitwise_equal"]:
            first_drift = stage.name
        if first_isolated is None and not entry["isolated_vs_provider"]["bitwise_equal"]:
            first_isolated = stage.name
        report["stages"].append(entry)
    report["first_drift"] = first_drift
    report["first_isolated_drift"] = first_isolated
    return report


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()

    try:
        sha = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except Exception:  # noqa: BLE001
        sha, dirty = "unknown", True
    return {"rl_kernel_commit": sha, "tracked_tree_dirty": dirty}


def environment() -> dict[str, Any]:
    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }
