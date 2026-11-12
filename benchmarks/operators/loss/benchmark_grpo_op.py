# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import argparse
import time

import torch
from tabulate import tabulate

from rl_engine.ops.sampling.api import SamplerBackend as RL_Sampler
from rl_engine.platforms.device import device_ctx
from rl_engine.utils.logger import logger


def measure_extra_vram(fn, *args, warmup=3, iters=10):
    """
    Measures PEAK extra VRAM allocated by fn(*args),
    excluding the memory already occupied by the input tensors.
    Returns (extra_vram_gb, avg_latency_ms).
    """
    # Baseline: memory already in use BEFORE calling fn
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            fn(*args)
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            fn(*args)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak = torch.cuda.max_memory_allocated()
    extra_gb = (peak - baseline) / (1024**3)
    avg_ms = (t1 - t0) / iters * 1000
    return extra_gb, avg_ms


def native_logprob(logits, token_ids):
    """Standard full log_softmax + gather — O(G·L·V) extra memory."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def run_benchmark(args, return_data: bool = False):
    device = device_ctx.device
    dtype = device_ctx.get_preferred_dtype()

    sampler = RL_Sampler().to(device)
    g_sizes = [int(g) for g in args.g_sizes.split(",")]
    results = []
    raw_metrics = []

    logger.info(f"Starting Benchmark on {device} with dtype {dtype}")
    logger.info(f"Config: SeqLen={args.seq_len}, VocabSize={args.vocab_size}")

    # Warmup with small tensor to trigger JIT / kernel caching
    logger.info("Warming up CUDA kernels...")
    _w_logits = torch.randn(8, 64, 4096, device=device, dtype=dtype)
    _w_ids = torch.randint(0, 4096, (8, 64), device=device)
    for _ in range(5):
        sampler.compute_logp(_w_logits, _w_ids)
    torch.cuda.synchronize()
    del _w_logits, _w_ids
    torch.cuda.empty_cache()

    for g in g_sizes:
        logger.info(f"Running iteration for Group Size G={g}...")

        # Allocate input tensors ONCE — shared by both measurements
        try:
            logits = torch.randn(g, args.seq_len, args.vocab_size, device=device, dtype=dtype)
            token_ids = torch.randint(0, args.vocab_size, (g, args.seq_len), device=device)
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM allocating inputs at G={g}, skipping.")
            results.append([g, "OOM", "OOM", "N/A", "N/A", "N/A", "N/A"])
            continue

        try:
            native_extra_gb, native_ms = measure_extra_vram(native_logprob, logits, token_ids)
            native_mem_str = f"{native_extra_gb:.2f} GB"
        except torch.cuda.OutOfMemoryError:
            native_extra_gb = float("inf")
            native_ms = float("inf")
            native_mem_str = "OOM"
        torch.cuda.empty_cache()

        try:
            kernel_extra_gb, kernel_ms = measure_extra_vram(sampler.compute_logp, logits, token_ids)
            kernel_mem_str = f"{kernel_extra_gb:.2f} GB"
        except torch.cuda.OutOfMemoryError:
            kernel_extra_gb = float("inf")
            kernel_ms = float("inf")
            kernel_mem_str = "OOM"
        torch.cuda.empty_cache()

        if native_extra_gb != float("inf") and kernel_extra_gb != float("inf"):
            saved = native_extra_gb - kernel_extra_gb
            saved_str = f"-{abs(saved):.2f} GB" if saved < 0 else f"{saved:.2f} GB"
        else:
            saved = 0.0
            saved_str = "N/A"

        speedup = (
            native_ms / kernel_ms
            if native_ms != float("inf") and kernel_ms != float("inf") and kernel_ms > 0
            else 0.0
        )
        speedup_str = f"{speedup:.2f}x"

        if return_data:
            raw_metrics.append(
                {
                    "g": g,
                    "native_extra_gb": native_extra_gb,
                    "kernel_extra_gb": kernel_extra_gb,
                    "vram_saved": saved,
                    "native_ms": native_ms,
                    "kernel_ms": kernel_ms,
                    "speedup": speedup,
                }
            )

        results.append(
            [
                g,
                native_mem_str,
                kernel_mem_str,
                saved_str,
                f"{native_ms:.2f} ms" if native_ms != float("inf") else "N/A",
                f"{kernel_ms:.2f} ms" if kernel_ms != float("inf") else "N/A",
                speedup_str,
            ]
        )

        del logits, token_ids
        torch.cuda.empty_cache()

    if return_data:
        return raw_metrics

    headers = [
        "Group Size (G)",
        "Native extra VRAM",
        "RL-Kernel extra VRAM",
        "VRAM Saved",
        "Native Latency",
        "RL-Kernel Latency",
        "Speedup",
    ]

    print("\n" + "=" * 115)
    print(
        "RL-KERNEL GRPO OPERATOR BENCHMARK REPORT  (extra VRAM = peak overhead above input tensors)"
    )
    print(f"Platform: {torch.cuda.get_device_name()} | Dtype: {dtype}")
    print(f"Context: SeqLen={args.seq_len}, VocabSize={args.vocab_size}")
    print("=" * 115)
    print(tabulate(results, headers=headers, tablefmt="fancy_grid"))
    print("=" * 115)
    print("Note: 'extra VRAM' = peak memory allocated ABOVE the input logits tensor.\n")
    print(
        "Native uses full log_softmax (O(G·L·V)); RL-Kernel"
        "uses pre-allocated chunking (O(chunk)).\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Kernel GRPO Operator Benchmark")
    parser.add_argument(
        "--g-sizes", type=str, default="8,16,32,64,128,256", help="Comma-separated group sizes"
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=128256)
    args = parser.parse_args()
    run_benchmark(args, return_data=False)
