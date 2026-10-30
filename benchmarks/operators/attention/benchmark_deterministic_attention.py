# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark for CUDA deterministic standard-softmax attention (issue #147).

Reports latency and peak memory for Qwen3-8B representative shapes,
including the cost of full scores/P materialization.
"""

import argparse
import time

import torch


def benchmark_attention(
    B: int, Hq: int, Hkv: int, Sq: int, Skv: int, D: int, dtype, warmup: int = 5, iters: int = 20
):
    from rl_engine.backends.cuda.attention.deterministic_attn import DeterministicAttentionOp

    op = DeterministicAttentionOp()
    device = "cuda"

    q = torch.randn(B, Hq, Sq, D, device=device, dtype=dtype)
    k = torch.randn(B, Hkv, Skv, D, device=device, dtype=dtype)
    v = torch.randn(B, Hkv, Skv, D, device=device, dtype=dtype)

    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(warmup):
            op.forward(q, k, v, causal=True)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            op.forward(q, k, v, causal=True)
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    scores_mem_mb = (B * Hq * Sq * Skv * 4) / (1024 * 1024)

    return {
        "latency_ms": elapsed * 1000,
        "peak_memory_mb": peak_mem_mb,
        "scores_materialization_mb": scores_mem_mb,
    }


QWEN3_8B_SHAPES = [
    {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 1, "Skv": 128, "D": 128, "label": "decode-128"},
    {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 1, "Skv": 1024, "D": 128, "label": "decode-1k"},
    {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 128, "Skv": 128, "D": 128, "label": "prefill-128"},
    {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 512, "Skv": 512, "D": 128, "label": "prefill-512"},
    {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 1024, "Skv": 1024, "D": 128, "label": "prefill-1k"},
    {"B": 4, "Hq": 32, "Hkv": 8, "Sq": 128, "Skv": 128, "D": 128, "label": "batch4-prefill-128"},
    {"B": 8, "Hq": 32, "Hkv": 8, "Sq": 64, "Skv": 64, "D": 128, "label": "batch8-prefill-64"},
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark deterministic attention")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"{'Shape':<25} {'Latency(ms)':>12} {'PeakMem(MB)':>12} {'Scores(MB)':>11}")
    print("-" * 65)

    for shape in QWEN3_8B_SHAPES:
        kwargs = shape.copy()
        label = kwargs.pop("label")
        try:
            result = benchmark_attention(
                **kwargs,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
            )
            print(
                f"{label:<25} {result['latency_ms']:>12.3f} "
                f"{result['peak_memory_mb']:>12.1f} "
                f"{result['scores_materialization_mb']:>11.1f}"
            )
        except RuntimeError as exc:
            print(f"{label:<25} {'OOM' if 'out of memory' in str(exc) else 'ERROR':>12}")
            if "out of memory" in str(exc):
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
