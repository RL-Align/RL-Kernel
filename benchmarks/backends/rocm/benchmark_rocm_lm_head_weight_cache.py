# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""A/B the ROCm LM-head transpose hot path against a prepared weight cache.

The baseline follows the current ``RocmDetGemmOp.linear`` path and therefore
materializes ``weight.T`` for every projection.  The candidate prepares that
same contiguous logical ``[K, N]`` right-hand side once, outside the timed
region, and passes it to ``linear_prepared``.  Both arms execute the same
deterministic Triton GEMM tree.  Blocks alternate AB/BA order to limit clock
drift.

The defaults reproduce one Qwen3-8B TP4 decode LM-head projection on MI300X::

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
      python benchmarks/backends/rocm/benchmark_rocm_lm_head_weight_cache.py \
      --blocks 10 --iterations 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch

from rl_engine.backends.rocm.gemm.det_gemm import RocmDetGemmOp, prepare_det_gemm_linear_weight
from rl_engine.integrations.engines.rollout.vllm.runtime import (
    _refresh_lm_head_weight_cache,
    _validated_lm_head_weight_cache,
)


def _raw_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _elapsed_ms(function: Callable[[], object], iterations: int) -> tuple[float, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1_000.0 / iterations
    return start.elapsed_time(end) / iterations, wall_ms


def _benchmark(
    *,
    m_size: int,
    n_size: int,
    k_size: int,
    warmup: int,
    iterations: int,
    blocks: int,
) -> dict[str, object]:
    activation = torch.randn(
        (m_size, k_size),
        device="cuda",
        dtype=torch.bfloat16,
    )

    class LmHeadLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(
                    (n_size, k_size),
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )

    layer = LmHeadLayer()
    weight = layer.weight
    operator = RocmDetGemmOp()

    # Preparation is deliberately outside the timed steady-state projection.
    # Weight synchronization refreshes this same storage before the next
    # projection.  Track the actual persistent allocator cost as well.
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    state = _refresh_lm_head_weight_cache(
        layer,
        prepare_det_gemm_linear_weight,
    )
    prepared_weight_t = state.weight_t
    allocated_after = torch.cuda.memory_allocated()
    reserved_after = torch.cuda.memory_reserved()

    @torch.inference_mode()
    def materialize_each_call() -> torch.Tensor:
        return operator.linear(activation, weight)

    @torch.inference_mode()
    def prepared_cache() -> torch.Tensor:
        return operator.linear_prepared(
            activation,
            _validated_lm_head_weight_cache(
                layer,
                prepare_det_gemm_linear_weight,
            ),
        )

    functions = {
        "materialize_each_call": materialize_each_call,
        "prepared_cache": prepared_cache,
    }
    for _ in range(warmup):
        materialize_each_call()
        prepared_cache()
    torch.cuda.synchronize()

    baseline_output = materialize_each_call()
    candidate_output = prepared_cache()
    torch.cuda.synchronize()
    raw_bytes_equal = torch.equal(
        baseline_output.view(torch.uint8),
        candidate_output.view(torch.uint8),
    )
    hashes = {
        "materialize_each_call": _raw_sha256(baseline_output),
        "prepared_cache": _raw_sha256(candidate_output),
    }
    if not raw_bytes_equal or len(set(hashes.values())) != 1:
        raise RuntimeError("prepared LM-head weight changed deterministic output bytes")

    gpu_samples: dict[str, list[float]] = {name: [] for name in functions}
    wall_samples: dict[str, list[float]] = {name: [] for name in functions}
    for block_index in range(blocks):
        order = (
            ("materialize_each_call", "prepared_cache")
            if block_index % 2 == 0
            else ("prepared_cache", "materialize_each_call")
        )
        for name in order:
            gpu_ms, wall_ms = _elapsed_ms(functions[name], iterations)
            gpu_samples[name].append(gpu_ms)
            wall_samples[name].append(wall_ms)

    refresh_gpu_samples = []
    refresh_wall_samples = []
    for _ in range(blocks):
        gpu_ms, wall_ms = _elapsed_ms(
            lambda: _refresh_lm_head_weight_cache(
                layer,
                prepare_det_gemm_linear_weight,
            ),
            1,
        )
        refresh_gpu_samples.append(gpu_ms)
        refresh_wall_samples.append(wall_ms)

    baseline_ms = statistics.median(gpu_samples["materialize_each_call"])
    candidate_ms = statistics.median(gpu_samples["prepared_cache"])
    baseline_wall_ms = statistics.median(wall_samples["materialize_each_call"])
    candidate_wall_ms = statistics.median(wall_samples["prepared_cache"])
    return {
        "shape": {
            "activation_mk": [m_size, k_size],
            "weight_nk": [n_size, k_size],
            "output_mn": [m_size, n_size],
        },
        "prepared_weight": {
            "shape": list(prepared_weight_t.shape),
            "stride": list(prepared_weight_t.stride()),
            "contiguous": prepared_weight_t.is_contiguous(),
            "bytes": prepared_weight_t.numel() * prepared_weight_t.element_size(),
            "allocated_delta_bytes": allocated_after - allocated_before,
            "reserved_delta_bytes": reserved_after - reserved_before,
        },
        "raw_bytes_equal": raw_bytes_equal,
        "sha256": hashes,
        "samples_ms": {
            "gpu": gpu_samples,
            "wall": wall_samples,
            "refresh_gpu": refresh_gpu_samples,
            "refresh_wall": refresh_wall_samples,
        },
        "median_ms": {
            "gpu": {
                "materialize_each_call": baseline_ms,
                "prepared_cache": candidate_ms,
                "refresh": statistics.median(refresh_gpu_samples),
            },
            "wall": {
                "materialize_each_call": baseline_wall_ms,
                "prepared_cache": candidate_wall_ms,
                "refresh": statistics.median(refresh_wall_samples),
            },
        },
        "latency_reduction_percent": 100.0 * (baseline_ms - candidate_ms) / baseline_ms,
        "wall_latency_reduction_percent": (
            100.0 * (baseline_wall_ms - candidate_wall_ms) / baseline_wall_ms
        ),
        "speedup": baseline_ms / candidate_ms,
        "wall_speedup": baseline_wall_ms / candidate_wall_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--m-size", type=int, default=1)
    parser.add_argument("--n-size", type=int, default=38_016)
    parser.add_argument("--k-size", type=int, default=4_096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires ROCm PyTorch and a visible GPU")
    if min(args.m_size, args.n_size, args.k_size) <= 0:
        raise ValueError("M, N, and K must be positive")
    if args.warmup < 0 or args.iterations <= 0 or args.blocks <= 0:
        raise ValueError("warmup must be non-negative; iterations and blocks must be positive")

    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    properties = torch.cuda.get_device_properties(args.device)
    result = _benchmark(
        m_size=args.m_size,
        n_size=args.n_size,
        k_size=args.k_size,
        warmup=args.warmup,
        iterations=args.iterations,
        blocks=args.blocks,
    )
    payload = {
        "environment": {
            "device": properties.name,
            "architecture": str(getattr(properties, "gcnArchName", "")),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "command": sys.argv,
        },
        "methodology": {
            "timing": (
                "GPU events and synchronized host wall time around complete "
                "projection calls; initial preparation is outside the steady-state "
                "region; refresh is reported separately"
            ),
            "order": "AB/BA alternating blocks",
            "warmup": args.warmup,
            "iterations_per_block": args.iterations,
            "blocks": args.blocks,
            "seed": args.seed,
        },
        "result": result,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
