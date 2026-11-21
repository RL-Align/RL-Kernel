# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 indexer ``stable_topk_per_row`` against the non-invariant path (D7).

Baselines
---------
``torch.topk``   ``torch.topk(logits, k, dim=1)`` on the same FP32 rows: the
                 naive PyTorch top-k (radix select, not batch-invariant on ties,
                 ignores the row ranges, so it does strictly less work).
``vllm_bi``      vLLM's ``ds4_bi.top_k_per_row_prefill`` (the rollout's BI
                 selector), when ``--vllm-ref`` points at its
                 ``csrc/ds4_bi_topk_extension.cu``; JIT-compiled here.

Shapes follow DSv4's indexer: ``top_k = 512`` over the C4-compressed KV of a
request, so a row is up to ``seq_len / 4`` wide. Prefill rows (one per query
token) are causal: row ``r`` sees ``(r + 1) * width / rows`` candidates. Decode
rows see the whole width. Timing is CUDA-graph replay (median of ``--iters``).

Usage:
    python benchmarks/benchmark_stable_topk.py
    python benchmarks/benchmark_stable_topk.py --vllm-ref ~/vllm/csrc/ds4_bi_topk_extension.cu
"""

from __future__ import annotations

import argparse
import statistics

import torch

from rl_engine.kernels.ops.cuda.topk import stable_topk_per_row_cuda

SHAPES = (  # (label, rows, width, causal)
    ("decode b64 ctx16K", 64, 4096, False),
    ("decode b256 ctx64K", 256, 16384, False),
    ("decode b64 ctx256K", 64, 65536, False),
    ("prefill 4K tok ctx16K", 4096, 4096, True),
    ("prefill 4K tok ctx64K", 4096, 16384, True),
    ("prefill 8K tok ctx128K", 8192, 32768, True),
)


def _graph_time_us(fn, warmup: int, iters: int) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--vllm-ref", default=None, help="path to ds4_bi_topk_extension.cu")
    parser.add_argument(
        "--data", choices=("randn", "relu"), default="relu", help="relu: indexer-like ties"
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="block size (0: automatic, as in production)"
    )
    args = parser.parse_args()

    vllm = None
    if args.vllm_ref:
        from torch.utils.cpp_extension import load

        load(
            name="ds4_bi_topk_ref",
            sources=[args.vllm_ref],
            is_python_module=False,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        )
        vllm = torch.ops.ds4_bi.top_k_per_row_prefill

    k = args.top_k
    header = f"{'shape':<24}{'ours us':>10}{'topk us':>10}{'ratio':>8}"
    header += f"{'vllm_bi us':>12}" if vllm else ""
    print(f"top_k={k} data={args.data} threads={args.threads or 'auto'}")
    print(header)
    gen = torch.Generator(device="cuda").manual_seed(0)
    for label, rows, width, causal in SHAPES:
        logits = torch.randn(rows, width, device="cuda", generator=gen)
        if args.data == "relu":
            logits = torch.relu(logits - 1.0)
        if causal:
            ends = ((torch.arange(rows, device="cuda") + 1) * width // rows).int()
        else:
            ends = torch.full((rows,), width, device="cuda", dtype=torch.int32)
        starts = torch.zeros_like(ends)
        out = torch.empty(rows, k, device="cuda", dtype=torch.int32)

        ours = _graph_time_us(
            lambda: stable_topk_per_row_cuda(
                logits, starts, ends, k, out=out, threads=args.threads
            ),
            args.warmup,
            args.iters,
        )
        base = _graph_time_us(lambda: torch.topk(logits, k, dim=1), args.warmup, args.iters)
        line = f"{label:<24}{ours:>10.1f}{base:>10.1f}{ours / base:>8.2f}"
        if vllm:
            ref_out = torch.empty_like(out)
            ref = _graph_time_us(
                lambda: vllm(logits, starts, ends, ref_out, rows, logits.stride(0), 1, k),
                args.warmup,
                args.iters,
            )
            line += f"{ref:>12.1f}"
        print(line)


if __name__ == "__main__":
    main()
