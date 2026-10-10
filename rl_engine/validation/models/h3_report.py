# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Per-operator performance and accuracy measurements for the H3 evidence.

``PERF_CASES[op]`` builds the timed cases that ``benchmarks/models/benchmark_h3_conditioning.py``
prints and ``tools/validation/models/h3_evidence.py`` stores. ``ACCURACY[op]`` measures the
op against the provider path and its golden. Each RFC #420 row registers both.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable

import torch

from rl_engine.runtime.registry import KernelRegistry
from rl_engine.validation.models.h3_cases import h3_timesteps
from rl_engine.validation.models.h3_provider import (
    provider_adaln_modulation,
    provider_time_embedder,
    provider_time_proj,
)
from rl_engine.validation.models.h3_weights import h3_weights_dir, load_h3_conditioning_weights

# Keys of a perf case that hold a timed callable, in display order.
TIMED_KEYS = (
    "candidate",
    "candidate_checked",
    "provider",
    "candidate_backward",
    "provider_backward",
)


def time_us(fn: Callable[[], Any], warmup: int = 20, iters: int = 200) -> float:
    """Median CUDA-event time of ``fn`` in microseconds (includes the Python wrapper)."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples)


def peak_mib(fn: Callable[[], Any]) -> float:
    """Synchronize one CUDA call and report peak allocated memory above baseline in MiB."""

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def measure(case: dict[str, Any], warmup: int = 20, iters: int = 200) -> dict[str, Any]:
    """Time case callables in rotating order after warmup and return their evidence.

    Record raw CUDA-event samples, executed orders, median microseconds,
    decimal GB/s from ``case['bytes']``, and peak allocated MiB per callable.
    """

    row = {key: case[key] for key in ("op", "case", "backend", "bytes") if key in case}
    keys = [key for key in TIMED_KEYS if key in case]
    if not keys:
        return row
    for _ in range(warmup):
        for key in keys:
            case[key]()
    torch.cuda.synchronize()
    samples = {key: [] for key in keys}
    orders = []
    for index in range(iters):
        # Rotate the first implementation so drift does not favor one backend.
        offset = index % len(keys)
        order = keys[offset:] + keys[:offset]
        orders.append(order)
        for key in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            case[key]()
            end.record()
            end.synchronize()
            samples[key].append(start.elapsed_time(end) * 1e3)
    row["timing_samples_us"] = samples
    row["timing_order"] = orders
    for key in keys:
        us = statistics.median(samples[key])
        row[f"{key}_us"] = us
        row[f"{key}_gbps"] = case["bytes"] / (us * 1e-6) / 1e9
        row[f"{key}_peak_mib"] = peak_mib(case[key])
    return row


# --------------------------------------------------------------------------- #
# timestep_sinusoid_h3
# --------------------------------------------------------------------------- #


def _sinusoid_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    """Build CUDA sinusoid timing cases with checked, unchecked, and provider calls."""

    op = registry.get_op("timestep_sinusoid_h3", device="cuda")
    cases = []
    for num in (1, 2, 4, 64):
        t = h3_timesteps(num)
        cases.append(
            {
                "op": "timestep_sinusoid_h3",
                "case": f"T={num}",
                "backend": type(op).__name__,
                "bytes": num * 4 + num * 256 * 4,
                "candidate": lambda t=t: op.forward(t, check_range=False),
                "candidate_checked": lambda t=t: op.forward(t),
                "provider": lambda t=t: provider_time_proj(t),
            }
        )
    return cases


def _sinusoid_accuracy(registry: KernelRegistry) -> dict[str, Any]:
    """Compare sinusoid outputs to provider and FP64 golden across timestep counts."""

    op = registry.get_op("timestep_sinusoid_h3", device="cuda")
    golden = registry._get_or_create_backend(
        registry._priority_map["cpu"]["timestep_sinusoid_h3"][-1]
    )
    rows = []
    for num in (1, 2, 3, 7, 64, 1000, 4097):
        t = h3_timesteps(num, seed=num)
        out, ref, gold = op(t), provider_time_proj(t), golden.forward_fp32(t)
        rows.append(
            {
                "num_timesteps": num,
                "bitwise_equal_to_provider": bool(torch.equal(out, ref)),
                "max_abs_vs_fp64": float((out - gold).abs().max()),
                "provider_max_abs_vs_fp64": float((ref - gold).abs().max()),
            }
        )
    return {"contract_atol": 1e-5, "cases": rows}


def h3_params(names: list[str], shapes: list[tuple[int, ...]]) -> list[torch.Tensor]:
    """Return CUDA checkpoint parameters, or seeded random tensors when weights are unset."""

    if h3_weights_dir() is not None:
        weights = load_h3_conditioning_weights("cuda", names)
        return [weights[name] for name in names]
    generator = torch.Generator(device="cuda").manual_seed(0)
    return [
        torch.randn(shape, device="cuda", generator=generator) * shape[-1] ** -0.5
        for shape in shapes
    ]


# --------------------------------------------------------------------------- #
# timestep_mlp_fp32
# --------------------------------------------------------------------------- #

MLP_NAMES = [f"time_embedder.linear_{i}.{p}" for i in (1, 2) for p in ("weight", "bias")]
MLP_SHAPES = [(5376, 256), (5376,), (2688, 5376), (2688,)]


def _mlp_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    """Build CUDA MLP candidate/provider timing cases for one to four timesteps."""

    op = registry.get_op("timestep_mlp_fp32", device="cuda")
    params = h3_params(MLP_NAMES, MLP_SHAPES)
    weight_bytes = sum(p.numel() * p.element_size() for p in params)
    cases = []
    for num in (1, 2, 3, 4):
        x = torch.rand(num, 256, device="cuda") * 2 - 1
        cases.append(
            {
                "op": "timestep_mlp_fp32",
                "case": f"T={num}",
                "backend": type(op).__name__,
                "bytes": weight_bytes,
                "candidate": lambda x=x: op(x, *params),
                "provider": lambda x=x: provider_time_embedder(x, *params),
            }
        )
    return cases


def _mlp_accuracy(registry: KernelRegistry, draws: int = 200) -> dict[str, Any]:
    """Measure per-draw MLP error against FP64 and test single-row/batched equality."""

    op = registry.get_op("timestep_mlp_fp32", device="cuda")
    golden = registry._get_or_create_backend(registry._priority_map["cpu"]["timestep_mlp_fp32"][-1])
    sinusoid = registry.get_op("timestep_sinusoid_h3", device="cuda")
    params = h3_params(MLP_NAMES, MLP_SHAPES)
    cuda_err, provider_err = [], []
    for seed in range(draws):
        x = sinusoid(h3_timesteps(4, seed=seed))
        gold = golden.forward_fp32(x, *params)
        cuda_err.append(float((op(x, *params) - gold).abs().max()))
        provider_err.append(float((provider_time_embedder(x, *params) - gold).abs().max()))
    # Row invariance: each timestep alone equals its row in the batch of 9.
    x = sinusoid(h3_timesteps(9, seed=1))
    full = op(x, *params)
    invariant = all(torch.equal(op(x[i : i + 1], *params)[0], full[i]) for i in range(9))
    return {
        "contract_atol": 1e-4,
        "draws": draws,
        "num_timesteps": 4,
        "cuda_max_abs_vs_fp64": cuda_err,
        "provider_max_abs_vs_fp64": provider_err,
        "rows_batch_invariant": invariant,
    }


# --------------------------------------------------------------------------- #
# adaln_projection_3mod
# --------------------------------------------------------------------------- #

ADALN_NAMES = [
    "transformer_blocks.0.adaln_proj.linear.weight",
    "transformer_blocks.0.adaln_proj.linear.bias",
]


def _adaln_params() -> tuple[torch.Tensor, torch.Tensor]:
    """Return block-0 AdaLN projection weights and bias as BF16 CUDA tensors."""

    weight, bias = h3_params(ADALN_NAMES, [(96768, 2688), (96768,)])
    return weight.bfloat16(), bias.bfloat16()


def _projection_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
    """Build CUDA AdaLN candidate/provider timing cases for one to four timesteps."""

    op = registry.get_op("adaln_projection_3mod", device="cuda")
    weight, bias = _adaln_params()
    cases = []
    for num in (1, 2, 3, 4):
        temb = torch.randn(num, 2688, device="cuda")
        cases.append(
            {
                "op": "adaln_projection_3mod",
                "case": f"T={num}",
                "backend": type(op).__name__,
                "bytes": weight.numel() * weight.element_size(),
                "candidate": lambda temb=temb: op(temb, weight, bias),
                "provider": lambda temb=temb: provider_adaln_modulation(temb, weight, bias),
            }
        )
    return cases


def _flat_cat(outputs) -> torch.Tensor:
    """Concatenate flattened modulation tensors in their output order."""

    return torch.cat([out.reshape(-1) for out in outputs])


def _projection_accuracy(registry: KernelRegistry, draws: int = 20) -> dict[str, Any]:
    """Report per-draw BF16 equality to FP64 goldens and single-row/batch invariance."""

    op = registry.get_op("adaln_projection_3mod", device="cuda")
    golden = registry._get_or_create_backend(
        registry._priority_map["cpu"]["adaln_projection_3mod"][-1]
    )
    weight, bias = _adaln_params()
    cuda_frac, provider_frac, early_frac = [], [], []
    for seed in range(draws):
        g = torch.Generator(device="cuda").manual_seed(seed)
        temb = torch.randn(4, 2688, device="cuda", generator=g) * 2
        gold = _flat_cat(golden.forward_fp32(temb, weight, bias)).bfloat16()
        early = _flat_cat(golden.forward_fp32(temb.bfloat16().float(), weight, bias)).bfloat16()
        ours = _flat_cat(op(temb, weight, bias))
        theirs = _flat_cat(provider_adaln_modulation(temb, weight, bias))
        cuda_frac.append(float((ours == gold).float().mean()))
        provider_frac.append(float((theirs == gold).float().mean()))
        early_frac.append(float((ours == early).float().mean()))
    temb = torch.randn(9, 2688, device="cuda")
    full = op(temb, weight, bias)
    invariant = all(
        all(
            torch.equal(s, f[3 * i : 3 * i + 3])
            for s, f in zip(op(temb[i : i + 1], weight, bias), full, strict=True)
        )
        for i in range(9)
    )
    return {
        "draws": draws,
        "num_timesteps": 4,
        "cuda_correctly_rounded": cuda_frac,
        "provider_correctly_rounded": provider_frac,
        "early_cast_golden_match": early_frac,
        "rows_batch_invariant": invariant,
    }


PERF_CASES: dict[str, Callable[[KernelRegistry], list[dict[str, Any]]]] = {
    "timestep_sinusoid_h3": _sinusoid_perf,
    "timestep_mlp_fp32": _mlp_perf,
    "adaln_projection_3mod": _projection_perf,
}
ACCURACY: dict[str, Callable[[KernelRegistry], dict[str, Any]]] = {
    "timestep_sinusoid_h3": _sinusoid_accuracy,
    "timestep_mlp_fp32": _mlp_accuracy,
    "adaln_projection_3mod": _projection_accuracy,
}
