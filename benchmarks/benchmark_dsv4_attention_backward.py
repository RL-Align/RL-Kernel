# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 attention layer: forward vs forward + backward time, and where the backward goes.

Real DSv4 shapes, random weights, one full-prefill request per layer type (window only,
C4, C128). Eager, CUDA events, median. ``--profile`` prints the top CUDA kernels of one
backward.

    PYTHONPATH=. python benchmarks/benchmark_dsv4_attention_backward.py [--profile]
"""

from __future__ import annotations

import argparse
import statistics

import torch

MAX_LEN = 16384


def _time(fn, iters=5):
    for _ in range(2):
        fn()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--profile", action="store_true")
    args = p.parse_args()
    from rl_engine.models.deepseek_v4.attention import (
        Dsv4Attention,
        Dsv4AttentionConfig,
        Dsv4Batch,
        new_caches,
        random_layer,
    )
    from rl_engine.models.deepseek_v4.attention_backward import attention_train

    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2).double() / 64))
    ang = torch.arange(MAX_LEN).double()[:, None] * inv[None]
    cos_sin = torch.cat([ang.cos(), ang.sin()], -1).float().cuda()
    print("ms, median; one request of T tokens")
    print(" ratio |     T |  forward | fwd + bwd |  bwd only")
    for ratio in (1, 4, 128):
        config = Dsv4AttentionConfig(compress_ratio=ratio, max_model_len=MAX_LEN)
        attn = Dsv4Attention(config, random_layer(config, seed=ratio), cos_sin)
        for t in (1024, 4096):
            g = torch.Generator(device="cuda").manual_seed(t)
            h = torch.randn(t, config.hidden, generator=g, device="cuda").bfloat16()
            dy = (torch.randn(t, config.hidden, generator=g, device="cuda") * 0.1).bfloat16()
            batch = Dsv4Batch.from_lens([t], [t], config.window)
            caches = new_caches(config, 1, MAX_LEN, seed=0)

            def fwd():
                with torch.no_grad():
                    attn.forward_batch(h, batch, caches)

            def fwd_bwd():
                x = h.detach().requires_grad_(True)
                attention_train(attn, x, batch, caches).backward(dy)

            f, fb = _time(fwd), _time(fwd_bwd)
            print(f" {ratio:5d} | {t:5d} | {f:8.2f} | {fb:9.2f} | {fb - f:9.2f}")
            if args.profile and t == 4096:
                from torch.profiler import ProfilerActivity, profile

                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    fwd_bwd()
                    torch.cuda.synchronize()
                events = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
                events.sort(key=lambda e: e.self_device_time_total, reverse=True)
                for e in events[:14]:
                    print(
                        f"    {e.self_device_time_total / 1e3:7.3f} ms x{e.count:<3d} {e.key[:70]}"
                    )


if __name__ == "__main__":
    main()
