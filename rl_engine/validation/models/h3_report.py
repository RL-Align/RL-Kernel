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
from rl_engine.validation.models.h3_provider import provider_time_proj

# Keys of a perf case that hold a timed callable, in display order.
TIMED_KEYS = (
    "candidate",
    "candidate_checked",
    "provider",
    "candidate_backward",
    "provider_backward",
)


def _prepared_call(
    fn: Callable[..., Any], setup: Callable[[], Any] | None = None
) -> Callable[[], Any]:
    if setup is None:
        return fn
    prepared = setup()
    return lambda: fn(prepared)


def _sample_us(fn: Callable[..., Any], setup: Callable[[], Any] | None = None) -> float:
    run = _prepared_call(fn, setup)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3


def time_us(
    fn: Callable[..., Any],
    warmup: int = 20,
    iters: int = 200,
    *,
    setup: Callable[[], Any] | None = None,
) -> float:
    """Median CUDA-event microseconds; optional setup runs outside the timed region.

    When supplied, ``setup`` runs before every call and its return value is passed
    to ``fn``. Backward measurements use it to build a fresh forward graph.
    """

    for _ in range(warmup):
        _prepared_call(fn, setup)()
    torch.cuda.synchronize()
    return statistics.median(_sample_us(fn, setup) for _ in range(iters))


def peak_mib(fn: Callable[..., Any], *, setup: Callable[[], Any] | None = None) -> float:
    """Incremental peak CUDA memory of the call, excluding optional setup."""

    run = _prepared_call(fn, setup)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    run()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def measure(case: dict[str, Any], warmup: int = 20, iters: int = 200) -> dict[str, Any]:
    """Interleave timed keys, reversing their execution order on alternate iterations.

    Record raw CUDA-event samples, executed orders, median microseconds,
    decimal GB/s from ``case['bytes']``, and peak allocated MiB per callable.
    """

    row = {key: case[key] for key in ("op", "case", "backend", "bytes") if key in case}
    keys = [key for key in TIMED_KEYS if key in case]
    if not keys:
        return row
    orders = (keys, list(reversed(keys)))
    row["execution_order"] = {
        "policy": "alternating",
        "iteration_0": orders[0],
        "iteration_1": orders[1],
    }
    if any(key.endswith("_backward") for key in keys):
        row["backward_timing_scope"] = "backward_only"
    for iteration in range(warmup):
        for key in orders[iteration % 2]:
            _prepared_call(case[key], case.get(f"{key}_setup"))()
    torch.cuda.synchronize()
    samples = {key: [] for key in keys}
    for iteration in range(iters):
        for key in orders[iteration % 2]:
            samples[key].append(_sample_us(case[key], case.get(f"{key}_setup")))
    row["timing_samples_us"] = samples
    row["timing_order"] = [orders[iteration % 2] for iteration in range(iters)]
    for key in keys:
        us = statistics.median(samples[key])
        row[f"{key}_us"] = us
        row[f"{key}_gbps"] = case["bytes"] / (us * 1e-6) / 1e9
        row[f"{key}_peak_mib"] = peak_mib(case[key], setup=case.get(f"{key}_setup"))
    return row


# --------------------------------------------------------------------------- #
# timestep_sinusoid_h3
# --------------------------------------------------------------------------- #


def _sinusoid_perf(registry: KernelRegistry) -> list[dict[str, Any]]:
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


PERF_CASES: dict[str, Callable[[KernelRegistry], list[dict[str, Any]]]] = {
    "timestep_sinusoid_h3": _sinusoid_perf,
}
ACCURACY: dict[str, Callable[[KernelRegistry], dict[str, Any]]] = {
    "timestep_sinusoid_h3": _sinusoid_accuracy,
}
