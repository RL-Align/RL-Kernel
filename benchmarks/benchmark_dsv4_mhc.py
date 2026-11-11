# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark the DSv4 mHC kernels (dsv4-mhc-v1) against vLLM's rollout kernels on H200.

- **Ours (pre)**: ``mhc_pre`` with the RMSNorm (attn_norm) fused into its epilogue.
- **vLLM (pre)**: ``mhc_pre_tilelang`` with the RMSNorm fused (TF32 DeepGEMM GEMM + TileLang
  epilogue), what the rollout runs.
- **post**: ``mhc_post`` against ``mhc_post_tilelang``.
- **head**: ``hc_head`` alone (vLLM fuses it with the final norm; not compared).
- **Backward**: our block backward (post dX + pre dX through the fused norm, BF16 grads in
  and out, as autograd runs it); vLLM has no backward.
- **Train step** (T >= 1024): forward + backward of ``mhc_pre`` (fused norm) -> ``mhc_post``
  through autograd, ours against a naive PyTorch implementation of the same math (eager,
  CUDA events, median).

Shapes: H = 4096, 4 streams. CUDA-graph replay, median, microseconds.

    python benchmarks/benchmark_dsv4_mhc.py [--tokens 1,64,1024,4096,16384]
"""

from __future__ import annotations

import argparse
import statistics

import torch


def graph_us(fn, iters=20):
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


def eager_us(fn, iters=10):
    for _ in range(2):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return statistics.median(times)


def naive_block(res, fn, scale, base, nw, eps=1e-6, iters=20):
    """DeepSeek's mHC pre (+ attn_norm) -> post in plain PyTorch, sublayer = identity."""
    t = res.shape[0]
    rf = res.float()
    xf = rf.flatten(1)
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    mixes = (xf @ fn.t()) * r
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + eps
    post = 2.0 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = torch.softmax((mixes[:, 8:] * scale[2] + base[8:]).view(t, 4, 4), -1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    y = (pre[:, :, None] * rf).sum(1).bfloat16().float()
    yn = (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps) * nw.float()).bfloat16()
    mix = torch.einsum("tij,tih->tjh", comb, rf)
    return (post[:, :, None] * yn.float()[:, None, :] + mix).bfloat16()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tokens", default="1,64,1024,4096,16384")
    p.add_argument("--hidden", type=int, default=4096)
    args = p.parse_args()
    from rl_engine.kernels.ops.cuda import mhc_dsv4 as ours

    try:
        from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_tilelang
    except ImportError:
        mhc_pre_tilelang = mhc_post_tilelang = None
    h = args.hidden
    gen = torch.Generator(device="cuda").manual_seed(0)
    fn = torch.randn(24, 4 * h, generator=gen, device="cuda") * (1.0 / (4 * h) ** 0.5)
    hfn = torch.randn(4, 4 * h, generator=gen, device="cuda") * (1.0 / (4 * h) ** 0.5)
    scale = torch.rand(3, device="cuda") + 0.5
    base = torch.randn(24, device="cuda") * 0.5
    nw = torch.ones(h, device="cuda", dtype=torch.bfloat16)
    print(
        f"{'T':>6} | {'pre+norm':>9} {'vLLM pre':>9} {'ratio':>6} | {'post':>8} {'vLLM':>8} "
        f"{'ratio':>6} | {'head':>7} | {'block bwd':>9} | "
        f"{'step ours':>9} {'naive':>9} {'ratio':>6}"
    )
    for t in [int(v) for v in args.tokens.split(",")]:
        res = torch.randn(t, 4, h, generator=gen, device="cuda").bfloat16()
        x = torch.randn(t, h, generator=gen, device="cuda").bfloat16()

        a = graph_us(lambda: ours.mhc_pre_cuda(res, fn, scale, base, nw))
        y, post, comb, raw, ss = ours.mhc_pre_cuda(res, fn, scale, base, nw)
        c = graph_us(lambda: ours.mhc_post_cuda(x, res, post, comb))
        hd = graph_us(lambda: ours.hc_head_cuda(res, hfn, scale[:1], base[:4], nw))
        d_new = torch.randn(t, 4, h, generator=gen, device="cuda").bfloat16()
        dy = torch.randn(t, h, generator=gen, device="cuda").bfloat16()

        def block_bwd():
            bf = torch.bfloat16
            _, dres, dpost, dcomb = ours.mhc_post_backward_cuda(d_new, x, res, post, comb, bf)
            ours.mhc_pre_backward_cuda(dy, dpost, dcomb, res, fn, scale, base, raw, ss, nw, bf)

        bw = graph_us(block_bwd)
        if mhc_pre_tilelang is not None:
            args_pre = (res, fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
            b = graph_us(lambda: mhc_pre_tilelang(*args_pre, norm_weight=nw, norm_eps=1e-6))
            vpost, vcomb, vx = mhc_pre_tilelang(*args_pre, norm_weight=nw, norm_eps=1e-6)
            d = graph_us(lambda: mhc_post_tilelang(vx, res, vpost, vcomb))
        else:
            b = d = float("nan")
        train = ""
        if t >= 1024:
            rg = res.clone().requires_grad_(True)

            def ours_step():
                o, po, co = ours.mhc_pre(rg, fn, scale, base, nw)
                ours.mhc_post(o, rg, po, co).backward(d_new)
                rg.grad = None

            def naive_step():
                naive_block(rg, fn, scale, base, nw).backward(d_new)
                rg.grad = None

            e, f = eager_us(ours_step), eager_us(naive_step)
            train = f" | {e:9.1f} {f:9.1f} {e / f:6.2f}"
        print(
            f"{t:>6} | {a:9.1f} {b:9.1f} {a / b:6.2f} | {c:8.1f} {d:8.1f} {c / d:6.2f} | "
            f"{hd:7.1f} | {bw:9.1f}{train}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
