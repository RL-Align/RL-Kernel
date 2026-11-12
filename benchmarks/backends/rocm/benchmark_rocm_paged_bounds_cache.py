# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark one page-bounds check per simulated decoder forward on ROCm.

This is an isolated strict-runtime benchmark, not an end-to-end vLLM timing.
Each timed iteration simulates every Qwen3 decoder layer over the same paged
metadata. The baseline validates physical page indices in every layer; the
candidate validates the first layer and reuses that runtime-owned proof for
the remaining layers. Page gathers, AITER calls, outputs, and LSE are otherwise
identical and must remain byte-exact. End-to-end benefit must still be measured
with the VIME rollout A/B.

Example on one MI300X:

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
      python benchmarks/backends/rocm/benchmark_rocm_paged_bounds_cache.py \
      --cached-lengths 32,128,512,2048 --layers 36 \
      --blocks 10 --iterations 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections.abc import Callable

import torch

from rl_engine.backends.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore
from rl_engine.backends.rocm.attention.strict_runtime import StrictRocmAttentionRuntime


class _CountingRuntime(StrictRocmAttentionRuntime):
    """Count validation rows outside the timed benchmark."""

    def __init__(self, *, core: StrictRocmAiterCKAttentionCore) -> None:
        super().__init__(core=core)
        self.bounds_validation_rows = 0
        self.core_launches = 0

    def _gather_paged_row(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_row: torch.Tensor,
        cached_length: int,
        *,
        validate_bounds: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if validate_bounds:
            self.bounds_validation_rows += 1
        return StrictRocmAttentionRuntime._gather_paged_row(
            k_cache,
            v_cache,
            page_row,
            cached_length,
            validate_bounds=validate_bounds,
        )

    def _run_core(self, *args, **kwargs):
        result = super()._run_core(*args, **kwargs)
        self.core_launches += result[3]
        return result


def _digest(tensor: torch.Tensor) -> str:
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


def _benchmark_length(
    *,
    cached_length: int,
    batch: int,
    layers: int,
    query_heads: int,
    key_heads: int,
    head_dim: int,
    page_size: int,
    warmup: int,
    blocks: int,
    iterations: int,
) -> dict[str, object]:
    pages_per_row = (cached_length + page_size - 1) // page_size
    total_pages = batch * pages_per_row
    generator = torch.Generator(device="cpu").manual_seed(390_700 + cached_length + batch)
    q = torch.randn(
        batch,
        query_heads,
        1,
        head_dim,
        dtype=torch.bfloat16,
        generator=generator,
    ).cuda()
    k_cache = torch.randn(
        total_pages,
        page_size,
        key_heads,
        head_dim,
        dtype=torch.bfloat16,
        generator=generator,
    ).cuda()
    v_cache = torch.randn(
        total_pages,
        page_size,
        key_heads,
        head_dim,
        dtype=torch.bfloat16,
        generator=generator,
    ).cuda()
    page_table = torch.arange(total_pages, device="cuda", dtype=torch.int32).reshape(
        batch,
        pages_per_row,
    )
    seqused_k = torch.full((batch,), cached_length, device="cuda", dtype=torch.int32)
    cached_lengths = (cached_length,) * batch
    core = StrictRocmAiterCKAttentionCore()
    counting_runtime = _CountingRuntime(core=core)
    baseline_runtime = StrictRocmAttentionRuntime(core=core)
    candidate_runtime = StrictRocmAttentionRuntime(core=core)
    counting_out = torch.empty_like(q)
    baseline_out = torch.empty_like(q)
    candidate_out = torch.empty_like(q)
    common = {
        "page_table": page_table,
        "seqused_k": seqused_k,
        "max_seqlen_k": pages_per_row * page_size,
        "scale": head_dim**-0.5,
        "cached_lengths": cached_lengths,
    }

    @torch.inference_mode()
    def run_layers(runtime, out, *, scoped):
        epoch = runtime.new_page_bounds_epoch() if scoped else None
        result = None
        for _ in range(layers):
            result = runtime.forward_paged_with_lse(
                q,
                k_cache,
                v_cache,
                out=out,
                page_bounds_epoch=epoch,
                **common,
            )
        if result is None:
            raise AssertionError("layers must be positive")
        return result

    counting_runtime.bounds_validation_rows = 0
    counting_runtime.core_launches = 0
    run_layers(counting_runtime, counting_out, scoped=False)
    baseline_validation_rows = counting_runtime.bounds_validation_rows
    baseline_core_launches = counting_runtime.core_launches
    counting_runtime.bounds_validation_rows = 0
    counting_runtime.core_launches = 0
    run_layers(counting_runtime, counting_out, scoped=True)
    candidate_validation_rows = counting_runtime.bounds_validation_rows
    candidate_core_launches = counting_runtime.core_launches
    expected_validation_rows = {"baseline": batch * layers, "candidate": batch}
    if {
        "baseline": baseline_validation_rows,
        "candidate": candidate_validation_rows,
    } != expected_validation_rows:
        raise AssertionError("page-bounds validation count changed")
    expected_core_launches = batch * key_heads * layers
    if baseline_core_launches != expected_core_launches:
        raise AssertionError("baseline AITER core launch count changed")
    if candidate_core_launches != expected_core_launches:
        raise AssertionError("candidate AITER core launch count changed")

    def baseline():
        return run_layers(baseline_runtime, baseline_out, scoped=False)

    def candidate():
        return run_layers(candidate_runtime, candidate_out, scoped=True)

    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()

    baseline_result = baseline()
    candidate_result = candidate()
    candidate_output_snapshot = candidate_result.out.clone()
    repeated_result = candidate()
    torch.cuda.synchronize()
    output_equal = torch.equal(
        baseline_result.out.view(torch.uint8),
        candidate_result.out.view(torch.uint8),
    ) and torch.equal(
        candidate_output_snapshot.view(torch.uint8),
        repeated_result.out.view(torch.uint8),
    )
    lse_equal = torch.equal(
        baseline_result.lse.view(torch.uint8),
        candidate_result.lse.view(torch.uint8),
    ) and torch.equal(
        candidate_result.lse.view(torch.uint8),
        repeated_result.lse.view(torch.uint8),
    )
    if not output_equal or not lse_equal:
        raise AssertionError("cached page-bounds validation changed output or LSE bytes")
    if baseline_result.provenance["page_bounds_validation_reused"]:
        raise AssertionError("baseline unexpectedly reused page validation")
    if not candidate_result.provenance["page_bounds_validation_reused"]:
        raise AssertionError("candidate did not reuse page validation")

    functions = {"every_layer": baseline, "once_per_forward": candidate}
    gpu_samples: dict[str, list[float]] = {name: [] for name in functions}
    wall_samples: dict[str, list[float]] = {name: [] for name in functions}
    for block in range(blocks):
        order = (
            ("every_layer", "once_per_forward")
            if block % 2 == 0
            else ("once_per_forward", "every_layer")
        )
        for name in order:
            gpu_ms, wall_ms = _elapsed_ms(functions[name], iterations)
            gpu_samples[name].append(gpu_ms)
            wall_samples[name].append(wall_ms)

    baseline_ms = statistics.median(gpu_samples["every_layer"])
    candidate_ms = statistics.median(gpu_samples["once_per_forward"])
    baseline_wall_ms = statistics.median(wall_samples["every_layer"])
    candidate_wall_ms = statistics.median(wall_samples["once_per_forward"])
    return {
        "cached_length": cached_length,
        "batch": batch,
        "layers_per_forward": layers,
        "query_heads": query_heads,
        "key_heads": key_heads,
        "head_dim": head_dim,
        "benchmark_scope": "strict_runtime_simulated_decoder_layers",
        "bounds_validation_rows": expected_validation_rows,
        "core_launches": {
            "every_layer": baseline_core_launches,
            "once_per_forward": candidate_core_launches,
        },
        "median_ms": {
            "gpu": {
                "every_layer": baseline_ms,
                "once_per_forward": candidate_ms,
            },
            "wall": {
                "every_layer": baseline_wall_ms,
                "once_per_forward": candidate_wall_ms,
            },
        },
        "latency_reduction_percent": 100.0 * (baseline_ms - candidate_ms) / baseline_ms,
        "wall_latency_reduction_percent": (
            100.0 * (baseline_wall_ms - candidate_wall_ms) / baseline_wall_ms
        ),
        "saved_ms_per_forward": baseline_ms - candidate_ms,
        "raw_output_equal": output_equal,
        "raw_lse_equal": lse_equal,
        "output_sha256": _digest(candidate_result.out),
        "lse_sha256": _digest(candidate_result.lse),
        "samples_ms": {"gpu": gpu_samples, "wall": wall_samples},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-lengths", default="32,128,512,2048")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--key-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires ROCm PyTorch and a visible GPU")
    if (
        min(
            args.batch,
            args.layers,
            args.query_heads,
            args.key_heads,
            args.head_dim,
            args.page_size,
            args.blocks,
            args.iterations,
        )
        <= 0
    ):
        raise ValueError("benchmark dimensions, blocks, and iterations must be positive")
    if args.layers < 2:
        raise ValueError("--layers must be at least 2 to measure cross-layer reuse")
    results = [
        _benchmark_length(
            cached_length=int(cached_length),
            batch=args.batch,
            layers=args.layers,
            query_heads=args.query_heads,
            key_heads=args.key_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
            warmup=args.warmup,
            blocks=args.blocks,
            iterations=args.iterations,
        )
        for cached_length in args.cached_lengths.split(",")
    ]
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "dtype": "bfloat16",
                "blocks": args.blocks,
                "iterations_per_block": args.iterations,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
