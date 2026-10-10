# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark removing singleton LSE concatenations from ROCm paged decode.

The legacy arm restores exactly the redundant ``torch.cat([lse], dim=0)``
copies that the production candidate skips. Both arms still execute the same
page gathers and AITER CK launches. This is an isolated strict-runtime timing;
end-to-end benefit must still be measured with the VIME rollout A/B.

Example on one MI300X:

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
      python benchmarks/backends/rocm/benchmark_rocm_singleton_lse.py \
      --cached-lengths 32,128,512,2048 --layers 36 \
      --blocks 12 --iterations 20
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
    if batch != 1:
        raise ValueError("this exact legacy singleton-cat reconstruction requires batch=1")
    pages_per_row = (cached_length + page_size - 1) // page_size
    total_pages = batch * pages_per_row
    generator = torch.Generator(device="cpu").manual_seed(390_800 + cached_length + batch)
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
    baseline_runtime = StrictRocmAttentionRuntime(core=core)
    candidate_runtime = StrictRocmAttentionRuntime(core=core)
    baseline_out = torch.empty_like(q)
    candidate_out = torch.empty_like(q)
    common = {
        "page_table": page_table,
        "seqused_k": seqused_k,
        "max_seqlen_k": pages_per_row * page_size,
        "scale": head_dim**-0.5,
        "cached_lengths": cached_lengths,
    }
    # The old path copied once for the singleton row passed into _run_core and
    # once for the singleton outer paged batch. With one KV head, its group
    # assembly was a third singleton copy.
    legacy_singleton_cats_per_layer = 2 + int(key_heads == 1)

    @torch.inference_mode()
    def run_layers(runtime, out, *, restore_legacy_cats):
        epoch = runtime.new_page_bounds_epoch()
        result = None
        returned_lse = None
        for _ in range(layers):
            result = runtime.forward_paged_with_lse(
                q,
                k_cache,
                v_cache,
                out=out,
                page_bounds_epoch=epoch,
                **common,
            )
            returned_lse = result.lse
            if restore_legacy_cats:
                if key_heads == 1:
                    returned_lse = torch.cat([returned_lse], dim=1)
                returned_lse = torch.cat([returned_lse], dim=0)
                returned_lse = torch.cat([returned_lse], dim=0)
        if result is None or returned_lse is None:
            raise AssertionError("layers must be positive")
        return result.out, returned_lse, result.provenance

    def baseline():
        return run_layers(baseline_runtime, baseline_out, restore_legacy_cats=True)

    def candidate():
        return run_layers(candidate_runtime, candidate_out, restore_legacy_cats=False)

    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()

    baseline_output, baseline_lse, baseline_provenance = baseline()
    candidate_output, candidate_lse, candidate_provenance = candidate()
    candidate_output_snapshot = candidate_output.clone()
    candidate_lse_snapshot = candidate_lse.clone()
    replay_output, replay_lse, _ = candidate()
    torch.cuda.synchronize()
    output_equal = torch.equal(
        baseline_output.view(torch.uint8),
        candidate_output.view(torch.uint8),
    ) and torch.equal(
        candidate_output_snapshot.view(torch.uint8),
        replay_output.view(torch.uint8),
    )
    lse_equal = torch.equal(
        baseline_lse.view(torch.uint8),
        candidate_lse.view(torch.uint8),
    ) and torch.equal(
        candidate_lse_snapshot.view(torch.uint8),
        replay_lse.view(torch.uint8),
    )
    if not output_equal or not lse_equal:
        raise AssertionError("singleton LSE fast path changed output or LSE bytes")
    if baseline_lse.shape != candidate_lse.shape or baseline_lse.stride() != candidate_lse.stride():
        raise AssertionError("singleton LSE fast path changed LSE layout")
    expected_launches = batch * key_heads
    if baseline_provenance["core_launch_count"] != expected_launches:
        raise AssertionError("baseline AITER launch count changed")
    if candidate_provenance["core_launch_count"] != expected_launches:
        raise AssertionError("candidate AITER launch count changed")

    functions = {"legacy_singleton_cat": baseline, "singleton_fast_path": candidate}
    gpu_samples: dict[str, list[float]] = {name: [] for name in functions}
    wall_samples: dict[str, list[float]] = {name: [] for name in functions}
    for block in range(blocks):
        order = (
            ("legacy_singleton_cat", "singleton_fast_path")
            if block % 2 == 0
            else ("singleton_fast_path", "legacy_singleton_cat")
        )
        for name in order:
            gpu_ms, wall_ms = _elapsed_ms(functions[name], iterations)
            gpu_samples[name].append(gpu_ms)
            wall_samples[name].append(wall_ms)

    baseline_ms = statistics.median(gpu_samples["legacy_singleton_cat"])
    candidate_ms = statistics.median(gpu_samples["singleton_fast_path"])
    baseline_wall_ms = statistics.median(wall_samples["legacy_singleton_cat"])
    candidate_wall_ms = statistics.median(wall_samples["singleton_fast_path"])
    return {
        "cached_length": cached_length,
        "batch": batch,
        "layers_per_forward": layers,
        "query_heads": query_heads,
        "key_heads": key_heads,
        "head_dim": head_dim,
        "benchmark_scope": "strict_runtime_simulated_decoder_layers",
        "legacy_singleton_cats_per_forward": legacy_singleton_cats_per_layer * layers,
        "aiter_core_launches_per_layer": expected_launches,
        "median_ms": {
            "gpu": {
                "legacy_singleton_cat": baseline_ms,
                "singleton_fast_path": candidate_ms,
            },
            "wall": {
                "legacy_singleton_cat": baseline_wall_ms,
                "singleton_fast_path": candidate_wall_ms,
            },
        },
        "latency_reduction_percent": 100.0 * (baseline_ms - candidate_ms) / baseline_ms,
        "wall_latency_reduction_percent": (
            100.0 * (baseline_wall_ms - candidate_wall_ms) / baseline_wall_ms
        ),
        "saved_ms_per_forward": baseline_ms - candidate_ms,
        "raw_output_equal": output_equal,
        "raw_lse_equal": lse_equal,
        "lse_shape": list(candidate_lse.shape),
        "lse_stride": list(candidate_lse.stride()),
        "output_sha256": _digest(candidate_output),
        "lse_sha256": _digest(candidate_lse),
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
    parser.add_argument("--blocks", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=20)
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
    if args.query_heads % args.key_heads:
        raise ValueError("--query-heads must be divisible by --key-heads")
    if args.batch != 1:
        raise ValueError("--batch must be 1 for the exact legacy singleton-cat A/B")
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
