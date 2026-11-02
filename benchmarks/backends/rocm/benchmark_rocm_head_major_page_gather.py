# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark one-pass head-major page gathers for ROCm paged decode.

By default both arms consume the pair-axis layout used by the vLLM revision in
the ROCm VIME image: ``[blocks, 2, block, kv_heads, head_dim]``. The adapter's
packed-last compatibility layout can be selected explicitly. The legacy arm
first copies pages in token-major order and then copies again to transpose
them; the candidate indexes a head-major view and materializes the final order
in one pass. AITER CK launch count/order and all arithmetic remain unchanged.
This is an isolated strict-runtime timing, not an end-to-end VIME result.

The candidate is active only for gradient-free ``key_heads > 1`` and
``cached_length > 1``. Singleton cases intentionally retain the legacy path
and are covered as fallback controls in the regression tests, not timed here.

Example on one MI300X:

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
      python benchmarks/backends/rocm/benchmark_rocm_head_major_page_gather.py \
      --cached-lengths 32,127,512,2048 --batch 1 --layers 36 \
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


class _LegacyGatherRuntime(StrictRocmAttentionRuntime):
    """Restore the two-pass page-major materialization for the baseline."""

    @staticmethod
    def _gather_paged_row(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_row: torch.Tensor,
        cached_length: int,
        *,
        validate_bounds: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        page_size = k_cache.size(1)
        page_count = (cached_length + page_size - 1) // page_size
        if page_count > page_row.numel():
            raise ValueError("page_table row is shorter than the cached length requires")
        pages = page_row[:page_count]
        if validate_bounds:
            bounds_ok = torch.all((pages >= 0) & (pages < k_cache.size(0)))
            if pages.is_cuda:
                torch._assert_async(bounds_ok, "page_table entries are outside the KV cache")
            elif not bool(bounds_ok.item()):
                raise ValueError("page_table entries are outside the KV cache")

        def gather(cache: torch.Tensor) -> torch.Tensor:
            selected = cache.index_select(0, pages)
            flat = selected.reshape(page_count * page_size, cache.size(2), cache.size(3))
            return flat[:cached_length].permute(1, 0, 2).unsqueeze(0).contiguous()

        return gather(k_cache), gather(v_cache)


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
    kv_layout: str,
    dtype: torch.dtype,
    warmup: int,
    blocks: int,
    iterations: int,
) -> dict[str, object]:
    if key_heads <= 1 or cached_length <= 1:
        raise ValueError("head-major gather benchmark requires key_heads > 1 and length > 1")
    pages_per_row = (cached_length + page_size - 1) // page_size
    total_pages = batch * pages_per_row
    generator = torch.Generator(device="cpu").manual_seed(390_900 + cached_length + batch)
    q = torch.randn(
        batch,
        query_heads,
        1,
        head_dim,
        dtype=dtype,
        generator=generator,
    ).cuda()
    if kv_layout == "vllm-rocm-5d":
        # vLLM ROCM_AITER_FA: [blocks, 2, block, kv_heads, head_dim]. Keep a
        # leading simulated-layer axis while preserving each layer's strides.
        packed_kv = torch.randn(
            layers,
            total_pages,
            2,
            page_size,
            key_heads,
            head_dim,
            dtype=dtype,
            generator=generator,
        ).cuda()
        k_caches, v_caches = packed_kv.unbind(2)
    elif kv_layout == "packed-last":
        # Adapter compatibility layout: [blocks, kv_heads, block, 2 * head_dim].
        packed_kv = torch.randn(
            layers,
            total_pages,
            key_heads,
            page_size,
            2 * head_dim,
            dtype=dtype,
            generator=generator,
        ).cuda()
        k_caches, v_caches = packed_kv.transpose(2, 3).split(head_dim, dim=-1)
    else:  # pragma: no cover - argparse owns the public validation.
        raise ValueError(f"unknown KV layout {kv_layout!r}")
    page_table = (
        torch.arange(total_pages, device="cuda", dtype=torch.int32)
        .reshape(batch, pages_per_row)
        .flip(1)
        .contiguous()
    )
    seqused_k = torch.full((batch,), cached_length, device="cuda", dtype=torch.int32)
    cached_lengths = (cached_length,) * batch
    core = StrictRocmAiterCKAttentionCore()
    baseline_runtime = _LegacyGatherRuntime(core=core)
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

    with torch.inference_mode():
        legacy_k, legacy_v = baseline_runtime._gather_paged_row(
            k_caches[0],
            v_caches[0],
            page_table[0],
            cached_length,
        )
        candidate_k, candidate_v = candidate_runtime._gather_paged_row(
            k_caches[0],
            v_caches[0],
            page_table[0],
            cached_length,
        )
    gather_k_equal = torch.equal(
        legacy_k.view(torch.uint8), candidate_k.contiguous().view(torch.uint8)
    )
    gather_v_equal = torch.equal(
        legacy_v.view(torch.uint8), candidate_v.contiguous().view(torch.uint8)
    )
    if not gather_k_equal or not gather_v_equal:
        raise AssertionError("head-major gather changed materialized K/V bytes")
    for group in range(key_heads):
        if not candidate_k[:, group : group + 1].transpose(1, 2).is_contiguous():
            raise AssertionError("candidate K group would add a pre-AITER copy")
        if not candidate_v[:, group : group + 1].transpose(1, 2).is_contiguous():
            raise AssertionError("candidate V group would add a pre-AITER copy")

    @torch.inference_mode()
    def run_layers(runtime, out):
        epoch = runtime.new_page_bounds_epoch()
        result = None
        for layer in range(layers):
            result = runtime.forward_paged_with_lse(
                q,
                k_caches[layer],
                v_caches[layer],
                out=out,
                page_bounds_epoch=epoch,
                **common,
            )
        if result is None:
            raise AssertionError("layers must be positive")
        return result

    def baseline():
        return run_layers(baseline_runtime, baseline_out)

    def candidate():
        return run_layers(candidate_runtime, candidate_out)

    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()

    baseline_result = baseline()
    candidate_result = candidate()
    candidate_output_snapshot = candidate_result.out.clone()
    candidate_lse_snapshot = candidate_result.lse.clone()
    replay_result = candidate()
    torch.cuda.synchronize()
    output_equal = torch.equal(
        baseline_result.out.view(torch.uint8),
        candidate_result.out.view(torch.uint8),
    ) and torch.equal(
        candidate_output_snapshot.view(torch.uint8),
        replay_result.out.view(torch.uint8),
    )
    lse_equal = torch.equal(
        baseline_result.lse.view(torch.uint8),
        candidate_result.lse.view(torch.uint8),
    ) and torch.equal(
        candidate_lse_snapshot.view(torch.uint8),
        replay_result.lse.view(torch.uint8),
    )
    if not output_equal or not lse_equal:
        raise AssertionError("head-major gather changed output or LSE bytes")
    expected_launches = batch * key_heads
    if baseline_result.provenance["core_launch_count"] != expected_launches:
        raise AssertionError("baseline AITER launch count changed")
    if candidate_result.provenance["core_launch_count"] != expected_launches:
        raise AssertionError("candidate AITER launch count changed")

    functions = {"page_major_two_pass": baseline, "head_major_one_pass": candidate}
    gpu_samples: dict[str, list[float]] = {name: [] for name in functions}
    wall_samples: dict[str, list[float]] = {name: [] for name in functions}
    for block in range(blocks):
        order = (
            ("page_major_two_pass", "head_major_one_pass")
            if block % 2 == 0
            else ("head_major_one_pass", "page_major_two_pass")
        )
        for name in order:
            gpu_ms, wall_ms = _elapsed_ms(functions[name], iterations)
            gpu_samples[name].append(gpu_ms)
            wall_samples[name].append(wall_ms)

    baseline_ms = statistics.median(gpu_samples["page_major_two_pass"])
    candidate_ms = statistics.median(gpu_samples["head_major_one_pass"])
    baseline_wall_ms = statistics.median(wall_samples["page_major_two_pass"])
    candidate_wall_ms = statistics.median(wall_samples["head_major_one_pass"])
    return {
        "cached_length": cached_length,
        "batch": batch,
        "layers_per_forward": layers,
        "query_heads": query_heads,
        "key_heads": key_heads,
        "head_dim": head_dim,
        "dtype": str(dtype).removeprefix("torch."),
        "kv_layout": kv_layout,
        "kv_cache_stride": list(k_caches[0].stride()),
        "head_major_active": key_heads > 1 and cached_length > 1,
        "distinct_kv_cache_per_layer": True,
        "benchmark_scope": "strict_runtime_native_vllm_kv_simulated_layers",
        "aiter_core_launches_per_layer": expected_launches,
        "median_ms": {
            "gpu": {
                "page_major_two_pass": baseline_ms,
                "head_major_one_pass": candidate_ms,
            },
            "wall": {
                "page_major_two_pass": baseline_wall_ms,
                "head_major_one_pass": candidate_wall_ms,
            },
        },
        "latency_reduction_percent": 100.0 * (baseline_ms - candidate_ms) / baseline_ms,
        "wall_latency_reduction_percent": (
            100.0 * (baseline_wall_ms - candidate_wall_ms) / baseline_wall_ms
        ),
        "saved_ms_per_forward": baseline_ms - candidate_ms,
        "raw_gather_k_equal": gather_k_equal,
        "raw_gather_v_equal": gather_v_equal,
        "raw_output_equal": output_equal,
        "raw_lse_equal": lse_equal,
        "gather_k_sha256": _digest(candidate_k),
        "gather_v_sha256": _digest(candidate_v),
        "output_sha256": _digest(candidate_result.out),
        "lse_sha256": _digest(candidate_result.lse),
        "samples_ms": {"gpu": gpu_samples, "wall": wall_samples},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-lengths", default="32,127,512,2048")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--key-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--kv-layout",
        choices=("vllm-rocm-5d", "packed-last"),
        default="vllm-rocm-5d",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
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
    if args.query_heads % args.key_heads:
        raise ValueError("--query-heads must be divisible by --key-heads")
    dtype = getattr(torch, args.dtype)
    results = [
        _benchmark_length(
            cached_length=int(cached_length),
            batch=args.batch,
            layers=args.layers,
            query_heads=args.query_heads,
            key_heads=args.key_heads,
            head_dim=args.head_dim,
            page_size=args.page_size,
            kv_layout=args.kv_layout,
            dtype=dtype,
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
                "dtype": args.dtype,
                "kv_layout": args.kv_layout,
                "blocks": args.blocks,
                "iterations_per_block": args.iterations,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
