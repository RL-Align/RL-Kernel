# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch
import time
import argparse
from tabulate import tabulate
from rl_engine.utils.logger import logger
from rl_engine.kernels.sampling import SamplerBackend as RL_Sampler
from rl_engine.platforms.device import device_ctx


def run_benchmark(args, return_data: bool = False):
    """
    Standardized Benchmark for RL-Engine GRPO Operators.
    Focuses on VRAM efficiency and Latency.
    """
    device = device_ctx.device
    dtype = device_ctx.get_preferred_dtype()

    sampler = RL_Sampler().to(device)

    g_sizes = [int(g) for g in args.g_sizes.split(",")]
    results = []
    raw_metrics = []

    logger.info(f"Starting Benchmark on {device} with dtype {dtype}")
    logger.info(f"Config: SeqLen={args.seq_len}, VocabSize={args.vocab_size}")

    logger.info("Warming up CUDA kernels...")
    dummy_logits = torch.randn(16, 128, 4096, device=device, dtype=dtype)
    dummy_ids = torch.randint(0, 4096, (16, 128), device=device)
    for _ in range(5):
        _ = sampler.compute_logp(dummy_logits, dummy_ids)
    torch.cuda.synchronize()
    del dummy_logits, dummy_ids

    for g in g_sizes:
        logger.info(f"Running iteration for Group Size G={g}...")

        try:
            logits = torch.randn(g, args.seq_len, args.vocab_size, device=device, dtype=dtype)
            token_ids = torch.randint(0, args.vocab_size, (g, args.seq_len), device=device)
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM: Failed to allocate memory for G={g} at the start.")
            results.append([g, "OOM", "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                log_probs = torch.log_softmax(logits, dim=-1)
                _ = torch.gather(log_probs, dim=-1, index=token_ids.unsqueeze(-1))

            t1 = time.perf_counter()
            native_mem = torch.cuda.max_memory_allocated() / (1024**3)
            native_time = (t1 - t0) * 1000
        except torch.cuda.OutOfMemoryError:
            native_mem = "OOM"
            native_time = float("inf")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            t2 = time.perf_counter()
            with torch.no_grad():
                _ = sampler.compute_logp(logits, token_ids)

            t3 = time.perf_counter()
            engine_mem = torch.cuda.max_memory_allocated() / (1024**3)
            engine_time = (t3 - t2) * 1000
        except torch.cuda.OutOfMemoryError:
            engine_mem = "OOM"
            engine_time = float("inf")

        vram_diff = (
            native_mem - engine_mem
            if isinstance(native_mem, float) and isinstance(engine_mem, float)
            else 0.0
        )
        vram_saved_str = f"{vram_diff:.2f} GB" if vram_diff > 0 else "N/A"
        speedup_val = (
            native_time / engine_time
            if native_time != float("inf") and engine_time != float("inf")
            else 1.0
        )
        speedup_str = f"{speedup_val:.2f}x"

        if return_data:
            raw_metrics.append(
                {
                    "g": g,
                    "vram_saved_val": vram_diff,
                    "engine_ms": engine_time,
                    "native_ms": native_time,
                    "speedup": speedup_str,
                }
            )

        results.append(
            [
                g,
                f"{native_mem:.2f} GB" if isinstance(native_mem, float) else "OOM",
                f"{engine_mem:.2f} GB" if isinstance(engine_mem, float) else "OOM",
                vram_saved_str,
                f"{native_time:.2f} ms" if native_time != float("inf") else "N/A",
                f"{engine_time:.2f} ms" if engine_time != float("inf") else "N/A",
                speedup_str,
            ]
        )

        del logits, token_ids
        torch.cuda.empty_cache()

    # The decision to return data or print a table depends on the pattern.
    if return_data:
        return raw_metrics

    headers = [
        "Group Size (G)",
        "Native VRAM",
        "RL-Engine VRAM",
        "VRAM Saved",
        "Native Latency",
        "RL-Engine Latency",
        "Speedup",
    ]

    print("\n" + "=" * 115)
    print("RL-ENGINE GRPO OPERATOR BENCHMARK REPORT")
    print(f"Platform: {torch.cuda.get_device_name()} | Dtype: {dtype}")
    print(f"Context: SeqLen={args.seq_len}, VocabSize={args.vocab_size}")
    print("=" * 115)
    print(tabulate(results, headers=headers, tablefmt="fancy_grid"))
    print("=" * 115)
    print("Note: VRAM Saved indicates the reduction in peak HBM usage during forward pass.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-Engine Operator Benchmark Tool")
    parser.add_argument(
        "--g-sizes",
        type=str,
        default="8,16,32,64,128",
        help="Comma-separated group sizes to test (e.g., 8,16,32,64)",
    )
    parser.add_argument("--seq-len", type=int, default=512, help="Input sequence length")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=128256,
        help="Vocabulary size (Llama-3: 128256, Qwen: 151936)",
    )

    args = parser.parse_args()
    run_benchmark(args, return_data=False)
