# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 attention layer forward (T07): wall time and where the GPU time goes.

Real DSv4 shapes (hidden 4096, 64 heads x 512, q / o LoRA 1024, 8 output
groups, window 128, indexer 64 x 128 top-512), random weights in checkpoint
formats. Prefill is timed eagerly; a decode step is captured in a CUDA graph
and replayed, as a serving engine runs it. ``--profile`` prints the top CUDA
kernels.

Usage:
    python benchmarks/benchmark_dsv4_attention_layer.py [--profile]
"""

from __future__ import annotations

import argparse

import torch

from rl_engine.models.deepseek_v4.attention import (
    Dsv4Attention,
    Dsv4AttentionConfig,
    Dsv4Batch,
    new_caches,
    random_layer,
)

MAX_LEN = 8192


def _cos_sin():
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2).double() / 64))
    ang = torch.arange(MAX_LEN).double()[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.sin()], -1).float().cuda()


def _time(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    for ratio in (4, 128, 1):
        config = Dsv4AttentionConfig(compress_ratio=ratio, max_model_len=MAX_LEN)
        layer = Dsv4Attention(config, random_layer(config), _cos_sin())
        for label, reqs, ctx, qlen in (
            ("prefill 1 x 4096", 1, 4096, 4096),
            ("decode 64 @ 4096", 64, 4096, 1),
        ):
            caches = new_caches(config, reqs, MAX_LEN)
            h_all = torch.randn(reqs * ctx, config.hidden, device="cuda").to(torch.bfloat16)
            # Fill the caches with the context first (prefill all but the measured tokens).
            pre = ctx - qlen
            if pre:
                pos = torch.arange(pre, device="cuda").repeat(reqs)
                req = torch.arange(reqs, device="cuda").repeat_interleave(pre)
                seq = torch.full((reqs,), pre, dtype=torch.int32, device="cuda")
                layer.forward(h_all[: reqs * pre], pos, req, seq, seq.clone(), caches)
            batch = Dsv4Batch.from_lens([ctx] * reqs, [qlen] * reqs, config.window)
            h = h_all[: reqs * qlen].contiguous()

            def step():
                return layer.forward_batch(h, batch, caches)

            if batch.decode:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    step()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    step()
                ms = _time(graph.replay, iters=20)
            else:
                ms = _time(step)
            mode = "graph" if batch.decode else "eager"
            print(f"ratio={ratio:<4} {label:<18} {ms:8.2f} ms ({mode})")
            if args.profile:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    step()
                    torch.cuda.synchronize()
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))


if __name__ == "__main__":
    main()
