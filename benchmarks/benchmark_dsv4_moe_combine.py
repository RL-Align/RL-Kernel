# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark the DSv4 MoE route plan and fixed-order combine (H 4096, 256 experts, top-6).

Combine: our slot-ordered kernel (+ shared + residual, BF16 out) against the
non-invariant ``index_add_`` scatter (atomic FP32 adds) followed by the merge, and
against the eager golden. Plan: ``build_plan`` (stable sort + searchsorted) and
``build_plan_cuda`` (counting sort).
CUDA-graph replay, median, microseconds.
"""

from __future__ import annotations

import statistics

import torch

from rl_engine.kernels.ops.cuda.moe import build_plan_cuda, combine_cuda, dispatch_backward_cuda
from rl_engine.kernels.ops.pytorch.moe.route_plan import build_plan, combine

H, E = 4096, 256
TOKENS = (1, 64, 256, 1024, 4096, 16384)


def _graph_us(fn, iters=30):
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


def main():
    g = torch.Generator(device="cuda").manual_seed(0)
    print(
        f"{'T':>6} | {'plan':>7} {'plan cu':>7} | "
        f"{'combine':>8} {'index_add':>9} {'ratio':>5} {'eager':>8} | "
        f"{'fan-in':>7} {'index_add':>9} {'ratio':>5}"
    )
    for t in TOKENS:
        ids = torch.rand(t, E, generator=g, device="cuda").topk(6, dim=-1).indices.int()
        plan = build_plan(ids, E)
        y = torch.randn(t * 6, H, generator=g, device="cuda").bfloat16()
        dx = torch.randn(t * 6, H, generator=g, device="cuda")
        shared = torch.randn(t, H, generator=g, device="cuda").bfloat16()
        res = torch.randn(t, H, generator=g, device="cuda").bfloat16()
        token = plan.token
        acc = torch.zeros(t, H, device="cuda")

        def scatter_combine():
            acc.zero_()
            acc.index_add_(0, token, y.float())
            return ((acc + shared.float()) + res.float()).bfloat16()

        def scatter_fanin():
            acc.zero_()
            return acc.index_add_(0, token, dx)

        p = _graph_us(lambda: build_plan(ids, E))
        pc = _graph_us(lambda: build_plan_cuda(ids, E))
        c = _graph_us(lambda: combine_cuda(y, plan, shared, res))
        cb = _graph_us(scatter_combine)
        ce = _graph_us(lambda: combine(y, plan, shared, res))
        f = _graph_us(lambda: dispatch_backward_cuda(dx, plan))
        fb = _graph_us(scatter_fanin)
        print(
            f"{t:>6} | {p:7.1f} {pc:7.1f} | {c:8.1f} {cb:9.1f} {c / cb:5.2f} {ce:8.1f} | "
            f"{f:7.1f} {fb:9.1f} {f / fb:5.2f}"
        )


if __name__ == "__main__":
    main()
