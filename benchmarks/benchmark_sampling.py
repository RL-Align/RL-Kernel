# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch
import time
import argparse
from tabulate import tabulate
from rl_engine.utils.logger import logger
from rl_engine.kernels.sampling import SamplerBackend as RL_Sampler
from rl_engine.platforms.device import device_ctx


def native_sampling(logits, top_k=None, top_p=None, temperature=1.0):
    """
    Simulates standard PyTorch sampling logic (Top-K -> Top-P -> Softmax -> Multinomial)
    """
    if temperature != 1.0:
        logits = logits / temperature

    logits = logits.float()

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for i in range(logits.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i, indices_to_remove] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def run_benchmark(args, return_data: bool = False):
    device = device_ctx.device
    dtype = torch.float32

    sampler = RL_Sampler().to(device)
    g_sizes = [int(g) for g in args.g_sizes.split(",")]
    results = []
    raw_metrics = []

    logger.info_once(f"Starting Sampling Benchmark on {device}")
    logger.info_once(f"Config: VocabSize={args.vocab_size}, TopK={args.top_k}, TopP={args.top_p}")

    logger.info("Warming up kernels...")
    dummy_logits = torch.randn(16, args.vocab_size, device=device, dtype=dtype)
    for _ in range(10):
        _ = sampler.sample(dummy_logits, top_k=args.top_k, top_p=args.top_p)
    torch.cuda.synchronize()

    for g in g_sizes:
        logger.info(f"Testing Batch Size G={g}...")
        logits = torch.randn(g, args.vocab_size, device=device, dtype=dtype)

        # 1. Native PyTorch Latency
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = native_sampling(logits.clone(), top_k=args.top_k, top_p=args.top_p)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        native_time = (t1 - t0) * 1000

        # 2. RL-Kernel FlashInfer Latency
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        _ = sampler.sample(logits, top_k=args.top_k, top_p=args.top_p)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        engine_time = (t3 - t2) * 1000

        speedup_val = native_time / engine_time
        speedup_str = f"{speedup_val:.2f}x"

        if return_data:
            raw_metrics.append(
                {"g": g, "engine_ms": engine_time, "native_ms": native_time, "speedup": speedup_str}
            )

        results.append([g, f"{native_time:.2f} ms", f"{engine_time:.2f} ms", speedup_str])

    if return_data:
        return raw_metrics

    headers = ["Batch Size (G)", "Native Latency", "RL-Engine (FlashInfer)", "Speedup"]
    print("\n" + "=" * 80)
    print(f"RL-ENGINE SAMPLING BENCHMARK REPORT (TopK={args.top_k}, TopP={args.top_p})")
    print("=" * 80)
    print(tabulate(results, headers=headers, tablefmt="fancy_grid"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--g-sizes", type=str, default="32,64,128,256")
    parser.add_argument("--vocab-size", type=int, default=128256)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    run_benchmark(args, return_data=False)
