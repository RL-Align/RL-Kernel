# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 indexer query RoPE + FP8 quant against the non-invariant path (D7).

Baselines: the eager PyTorch golden (``indexer_q_rope_fp8``, many small ops),
and ``--vllm`` adds vLLM's CuTe-DSL kernel when it imports. Every
implementation here is batch-invariant (each row is independent), so the
golden is the "naive PyTorch" reference of D7. 64 heads x 128, FP32 rotary
cache; CUDA-graph replay, median.

Usage:
    python benchmarks/benchmark_indexer_q.py [--vllm]
"""

from __future__ import annotations

import argparse
import statistics

import torch

from rl_engine.kernels.ops.cuda.indexer import indexer_q_rope_fp8_cuda
from rl_engine.kernels.ops.pytorch.indexer import indexer_q_rope_fp8

TOKENS = (1, 64, 512, 4096, 16384)


def _graph_us(fn, iters: int) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return statistics.median(times)


def _eager_us(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--vllm", action="store_true")
    args = parser.parse_args()
    vllm = None
    if args.vllm:
        try:
            from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
                fused_indexer_q_rope_quant_fp8_cutedsl as vllm,
            )
        except ImportError:
            from vllm.models_ref.fused_indexer_q_cutedsl import (
                fused_indexer_q_rope_quant_fp8_cutedsl as vllm,
            )

    heads, dim, rope, max_pos = 64, 128, 64, 1 << 16
    sm, hs = dim**-0.5, heads**-0.5
    inv = 1.0 / (10000 ** (torch.arange(0, rope, 2).double() / rope))
    ang = torch.arange(max_pos).double()[:, None] * inv[None]
    cache = torch.cat([ang.cos(), ang.sin()], -1).float().cuda()
    print(
        f"{'tokens':>8}{'cuda us':>10}{'GB/s':>8}{'golden us':>12}{'ratio':>8}"
        + ("{:>10}".format("vllm us") if vllm else "")
    )
    for t in TOKENS:
        q = torch.randn(t, heads, dim, device="cuda").to(torch.bfloat16)
        pos = torch.randint(0, max_pos, (t,), device="cuda")
        w = torch.randn(t, heads, device="cuda").to(torch.bfloat16)
        ours = _graph_us(lambda: indexer_q_rope_fp8_cuda(q, pos, cache, w, sm, hs), args.iters)
        gold = _eager_us(
            lambda: indexer_q_rope_fp8(q, pos, cache, w, sm, hs), max(3, args.iters // 10)
        )
        moved = t * heads * (dim * 2 + dim + 2 + 4)
        line = f"{t:>8}{ours:>10.1f}{moved / ours / 1e3:>8.0f}{gold:>12.1f}{ours / gold:>8.3f}"
        if vllm:
            rq = torch.empty(t, heads, dim, dtype=torch.float8_e4m3fn, device="cuda")
            rw = torch.empty(t, heads, dtype=torch.float32, device="cuda")
            vllm(pos, q, cache, w, sm, hs, rq, rw)
            line += (
                f"{_graph_us(lambda: vllm(pos, q, cache, w, sm, hs, rq, rw), args.iters):>10.1f}"
            )
        print(line)


if __name__ == "__main__":
    main()
