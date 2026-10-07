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
    provider_adaln_row_gather,
    provider_final_adaln_out,
    provider_gate_residual,
    provider_norm_modulate,
    provider_time_embedder,
    provider_time_proj,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HIDDEN = 5376


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
    Stage(
        name="adaln_row_gather",
        op_type="adaln_row_gather",
        candidate=lambda op, ctx, up: op.gather_chunks(
            up, ctx["timestep_indices"], ctx["token_tags"]
        ),
        provider=lambda ctx, up: provider_adaln_row_gather(
            up, ctx["timestep_indices"], ctx["token_tags"]
        ),
        golden=lambda op, ctx, up: op.forward_fp32(
            torch.cat(list(up), dim=1), ctx["timestep_indices"], ctx["token_tags"]
        ),
        provider_bitwise_isolated=True,  # a copy
        golden_atol=5e-2,  # carries the projection's BF16 rounding (reduction / bfloat16)
        golden_rtol=2e-2,
    ),
]


def adaln_indices(ctx: dict[str, Any]) -> torch.Tensor:
    return ctx["timestep_indices"] * 3 + ctx["token_tags"]


def _norm1_modulated(ctx: dict[str, Any], modulation, call):
    shift_msa, scale_msa = modulation[0], modulation[1]
    weight = ctx["weights"]["transformer_blocks.0.norm1.weight"]
    return call(ctx["hidden"], weight, shift_msa, scale_msa, adaln_indices(ctx))


STAGES.append(
    Stage(
        name="h3_rmsnorm",
        op_type="h3_rmsnorm",
        # Block norm1 followed by the MSA shift/scale of the projection stage.
        candidate=lambda op, ctx, _up: _norm1_modulated(
            ctx, ctx["history"]["adaln_projection_3mod"], op.forward_modulated
        ),
        provider=lambda ctx, _up: _norm1_modulated(
            ctx, ctx["history"]["adaln_projection_3mod"], provider_norm_modulate
        ),
        golden=lambda op, ctx, _up: _norm1_modulated(
            ctx, ctx["history"]["adaln_projection_3mod"], op.forward_modulated_fp32
        ),
        provider_bitwise_isolated=True,  # replays nn.RMSNorm's reduction order
        golden_atol=5e-2,  # reduction / bfloat16
        golden_rtol=2e-2,
    )
)

STAGES.append(
    Stage(
        name="adaln_gate_residual",
        op_type="adaln_gate_residual",
        # residual + gate_msa[row] * attention_output (the stand-in sublayer output).
        candidate=lambda op, ctx, _up: op(
            ctx["hidden"],
            ctx["sublayer"],
            ctx["history"]["adaln_projection_3mod"][2],
            adaln_indices(ctx),
        ),
        provider=lambda ctx, _up: provider_gate_residual(
            ctx["hidden"],
            ctx["history"]["adaln_projection_3mod"][2],
            adaln_indices(ctx),
            ctx["sublayer"],
        ),
        golden=lambda op, ctx, _up: op.forward_fp32(
            ctx["hidden"],
            ctx["sublayer"],
            ctx["history"]["adaln_projection_3mod"][2],
            adaln_indices(ctx),
        ),
        provider_bitwise_isolated=True,  # eager rounding order, elementwise
        golden_atol=5e-2,  # carries the gate's BF16 rounding (reduction / bfloat16)
        golden_rtol=2e-2,
    )
)


def _final(ctx: dict[str, Any], call):
    weights = ctx["weights"]
    history = ctx["history"]
    return call(
        history["adaln_gate_residual"],  # the residual stream after the gated sublayer
        weights["norm_out.norm.weight"],
        history["timestep_mlp_fp32"],  # temb
        weights["norm_out.linear.weight"],
        weights["norm_out.linear.bias"],
        ctx["timestep_indices"],
    )


STAGES.append(
    Stage(
        name="final_adaln_out",
        op_type="final_adaln_out",
        candidate=lambda op, ctx, _up: _final(ctx, op),
        provider=lambda ctx, _up: _final(ctx, provider_final_adaln_out),
        golden=lambda op, ctx, _up: _final(ctx, op.forward_fp32),
        provider_bitwise_isolated=False,  # shift/scale projection: tensor-core tree vs cuBLAS
        golden_atol=5e-2,  # reduction / bfloat16
        golden_rtol=2e-2,
    )
)

# The backward replay covers the conditioning chain up to the row gather.
CONDITIONING_STAGES = (
    "timestep_sinusoid_h3",
    "timestep_mlp_fp32",
    "adaln_projection_3mod",
    "adaln_row_gather",
)

# Parameters whose gradients the backward replay reports, in chain order.
GRAD_LEAVES = (
    "time_embedder.linear_1.weight",
    "time_embedder.linear_1.bias",
    "time_embedder.linear_2.weight",
    "time_embedder.linear_2.bias",
    "transformer_blocks.0.adaln_proj.linear.weight",
    "transformer_blocks.0.adaln_proj.linear.bias",
)
BACKWARD_MODES = ("candidate", "candidate_fused", "provider")


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
        # Stand-in for the packed hidden states (the patch/text projections are
        # other RFC rows): BF16 (1, S, H), seeded.
        "hidden": torch.randn(
            (1, seq_len, HIDDEN), generator=torch.Generator().manual_seed(seed + 2)
        )
        .to(torch.bfloat16)
        .cuda(),
        # Stand-in for the attention output the gate multiplies (the attention
        # rows belong to other contributors): BF16 (1, S, H), seeded.
        "sublayer": (
            torch.randn((1, seq_len, HIDDEN), generator=torch.Generator().manual_seed(seed + 3))
            * 3.0
        )
        .to(torch.bfloat16)
        .cuda(),
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
    # Each mode sees its own earlier outputs (a stage may read any earlier stage).
    history = {"candidate": {}, "provider": {}, "golden": {}}
    ctx_c = {**ctx, "history": history["candidate"]}
    ctx_p = {**ctx, "history": history["provider"]}
    ctx_g = {**ctx, "history": history["golden"]}
    for stage in stages:
        op = registry.get_op(stage.op_type, device="cuda")
        gold = golden_op(registry, stage.op_type)
        with torch.no_grad():
            isolated = stage.candidate(op, ctx_p, provider)
            chained_input = chained
            chained = stage.candidate(op, ctx_c, chained_input)
            repeat = stage.candidate(op, ctx_c, chained_input)
            golden = stage.golden(gold, ctx_g, golden)
            provider = stage.provider(ctx_p, provider)
        history["candidate"][stage.name] = chained
        history["provider"][stage.name] = provider
        history["golden"][stage.name] = golden
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


def chain_grads(mode: str, registry: KernelRegistry, ctx, upstream) -> list[torch.Tensor]:
    """Parameter gradients of the full chain for one execution mode.

    ``candidate`` runs the four RL-Kernel ops separately, ``candidate_fused``
    runs projection + gather as ``H3AdaLNModulationCudaOp``, ``provider`` the
    diffusers path, ``golden`` the FP64 references.
    """

    leaves = {
        name: ctx["weights"][name].detach().clone().requires_grad_(True) for name in GRAD_LEAVES
    }
    run_ctx = {**ctx, "weights": {**ctx["weights"], **leaves}, "history": {}}
    value = None
    if mode == "candidate_fused":
        from rl_engine.kernels.ops.cuda.h3.adaln_modulation import H3AdaLNModulationCudaOp

        for stage in STAGES[:2]:
            value = stage.candidate(registry.get_op(stage.op_type, device="cuda"), run_ctx, value)
        value = H3AdaLNModulationCudaOp()(
            value, *adaln_params(run_ctx), ctx["timestep_indices"], ctx["token_tags"]
        )
    else:
        for stage in (s for s in STAGES if s.name in CONDITIONING_STAGES):
            if mode == "provider":
                value = stage.provider(run_ctx, value)
            elif mode == "candidate":
                op = registry.get_op(stage.op_type, device="cuda")
                value = stage.candidate(op, run_ctx, value)
            else:
                value = stage.golden(golden_op(registry, stage.op_type), run_ctx, value)
    outputs = list(value)
    torch.autograd.backward(outputs, [g.to(out.dtype) for g, out in zip(upstream, outputs)])
    return [leaves[name].grad for name in GRAD_LEAVES]


def run_backward_case(
    registry: KernelRegistry, weights, *, num_timesteps: int, seq_len: int, seed: int = 0
) -> dict[str, Any]:
    """Determinism and accuracy of the chain's parameter gradients, per mode."""

    ctx = make_context(weights, num_timesteps=num_timesteps, seq_len=seq_len, seed=seed)
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    hidden = weights[GRAD_LEAVES[4]].shape[0] // 18
    upstream = [
        torch.randn(seq_len, hidden, device="cuda", generator=generator).bfloat16()
        for _ in range(6)
    ]
    golden = chain_grads("golden", registry, ctx, upstream)
    runs = {
        mode: [chain_grads(mode, registry, ctx, upstream) for _ in range(2)]
        for mode in BACKWARD_MODES
    }
    report: dict[str, Any] = {"num_timesteps": num_timesteps, "seq_len": seq_len, "leaves": {}}
    for index, name in enumerate(GRAD_LEAVES):
        gold = golden[index].float()
        entry: dict[str, Any] = {"golden_absmax": float(gold.abs().max())}
        for mode, (first, second) in runs.items():
            grad = first[index]
            err = (grad.float() - gold).abs().max()
            entry[mode] = {
                "repeat_bitwise_equal": bool(torch.equal(first[index], second[index])),
                "max_abs_vs_golden": float(err),
                "max_abs_vs_golden_over_absmax": float(err / gold.abs().max().clamp_min(1e-30)),
                "correctly_rounded_fraction": float(
                    (grad == golden[index].to(grad.dtype)).float().mean()
                ),
            }
        report["leaves"][name] = entry
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
