# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark direct caller-output writes in strict ROCm paged decode.

Both arms run the full logical-page gather and the same one-row/one-KV-group
AITER CK schedule. The baseline hides the new direct-output entry point so the
runtime reproduces PR #390's staged group outputs plus ``torch.cat(..., out=)``.
The candidate lets AITER write each group directly into the caller's vLLM
output slice. Output and LSE must remain byte-identical.

Example on one MI300X:

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
      python benchmarks/benchmark_rocm_aiter_direct_output.py \
      --cached-lengths 32,128,512,2048 --blocks 10 --iterations 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable

import torch

from rl_engine.kernels.ops.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore
from rl_engine.kernels.ops.rocm.attention.strict_runtime import StrictRocmAttentionRuntime


class _StagedCore:
    """Expose only the pre-optimization core interface to the runtime."""

    core_id = StrictRocmAiterCKAttentionCore.core_id
    strict_schedule = StrictRocmAiterCKAttentionCore.strict_schedule
    backend_id = StrictRocmAiterCKAttentionCore.backend_id

    def __init__(self, delegate: StrictRocmAiterCKAttentionCore) -> None:
        self._delegate = delegate

    def forward_with_lse(self, *args, **kwargs):
        return self._delegate.forward_with_lse(*args, **kwargs)


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _elapsed_ms(function: Callable[[], object], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _benchmark_length(
    *,
    cached_length: int,
    batch: int,
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
    generator = torch.Generator(device="cpu").manual_seed(390_500 + cached_length)
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
    staged_runtime = StrictRocmAttentionRuntime(core=_StagedCore(core))
    direct_runtime = StrictRocmAttentionRuntime(core=core)
    staged_out = torch.empty_like(q)
    direct_out = torch.empty_like(q)
    common = {
        "page_table": page_table,
        "seqused_k": seqused_k,
        "max_seqlen_k": pages_per_row * page_size,
        "scale": head_dim**-0.5,
        "cached_lengths": cached_lengths,
    }

    @torch.inference_mode()
    def staged():
        return staged_runtime.forward_paged_with_lse(
            q,
            k_cache,
            v_cache,
            out=staged_out,
            **common,
        )

    @torch.inference_mode()
    def direct():
        return direct_runtime.forward_paged_with_lse(
            q,
            k_cache,
            v_cache,
            out=direct_out,
            **common,
        )

    for _ in range(warmup):
        staged()
        direct()
    torch.cuda.synchronize()

    staged_result = staged()
    direct_result = direct()
    torch.cuda.synchronize()
    output_equal = torch.equal(
        staged_result.out.view(torch.uint8),
        direct_result.out.view(torch.uint8),
    )
    lse_equal = torch.equal(
        staged_result.lse.view(torch.uint8),
        direct_result.lse.view(torch.uint8),
    )
    if not output_equal or not lse_equal:
        raise AssertionError("direct-output paged decode differs from the staged baseline")
    if staged_result.out.data_ptr() != staged_out.data_ptr():
        raise AssertionError("baseline did not return its staged caller output buffer")
    if staged_result.provenance["core_output_staging"] != "runtime_group_cat":
        raise AssertionError("baseline no longer reproduces PR #390 output staging")
    if direct_result.out.data_ptr() != direct_out.data_ptr():
        raise AssertionError("candidate did not return the caller's output buffer")
    if direct_result.provenance["core_output_staging"] != "aiter_direct_caller_group":
        raise AssertionError("candidate did not enter the direct AITER output path")

    samples: dict[str, list[float]] = {"staged": [], "direct": []}
    functions = {"staged": staged, "direct": direct}
    for block in range(blocks):
        order = ("staged", "direct") if block % 2 == 0 else ("direct", "staged")
        for name in order:
            samples[name].append(_elapsed_ms(functions[name], iterations))

    staged_ms = statistics.median(samples["staged"])
    direct_ms = statistics.median(samples["direct"])
    return {
        "cached_length": cached_length,
        "batch": batch,
        "query_heads": query_heads,
        "key_heads": key_heads,
        "head_dim": head_dim,
        "staged_ms": staged_ms,
        "direct_ms": direct_ms,
        "latency_reduction_percent": 100.0 * (staged_ms - direct_ms) / staged_ms,
        "speedup": staged_ms / direct_ms,
        "raw_output_equal": output_equal,
        "raw_lse_equal": lse_equal,
        "output_sha256": _digest(direct_result.out),
        "lse_sha256": _digest(direct_result.lse),
        "staged_samples_ms": samples["staged"],
        "direct_samples_ms": samples["direct"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-lengths", default="32,128,512,2048")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--key-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires ROCm PyTorch and a visible GPU")
    results = [
        _benchmark_length(
            cached_length=int(cached_length),
            batch=args.batch,
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
