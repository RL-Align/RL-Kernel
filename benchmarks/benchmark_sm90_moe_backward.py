# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark the SM90 routed-expert backward (P5, frozen MXFP4 base) on DSv4 shapes.

Ours: fc1 recompute (FP8 WGMMA) + ``dh`` GEMM with the fused clamp-SwiGLU backward +
``dx`` GEMM (BF16 WGMMA, MXFP4 expanded in shared memory). Baseline (not batch-
invariant): weights dequantized to BF16 ahead of time (not timed), per-expert cuBLAS
GEMMs for the fc1 recompute, ``dh`` and ``dx``, and the SwiGLU backward in PyTorch.
H 4096, F 2048, 32 local experts (EP 8 of 256), tokens spread evenly. CUDA-graph
replay, median, microseconds.
"""

from __future__ import annotations

import statistics

import torch

from rl_engine.moe.backends.sm90_fused_mlp import PROFILE, Sm90FusedMoeMlp
from rl_engine.moe.contract import ExpertBatch
from rl_engine.moe.mx_format import mx_dequantize, mx_quantize

H, F, E = 4096, 2048, 32
ROWS = (256, 2048, 8192, 32768)


def _graph_us(fn, iters=20):
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
    w1 = mx_quantize(torch.randn(E, 2 * F, H, generator=g, device="cuda") / H**0.5, "e2m1")
    w2 = mx_quantize(torch.randn(E, H, F, generator=g, device="cuda") / F**0.5, "e2m1")
    w1b = mx_dequantize(w1).bfloat16()
    w2b = mx_dequantize(w2).bfloat16()
    backend = Sm90FusedMoeMlp()
    print(f"{'rows':>6} | {'ours':>8} {'baseline':>9} {'ratio':>5}")
    for m in ROWS:
        per = m // E
        offsets = torch.arange(0, m + 1, per, dtype=torch.int32, device="cuda")
        x = (torch.randn(m, H, generator=g, device="cuda") * 0.5).bfloat16()
        dy = torch.randn(m, H, generator=g, device="cuda").bfloat16()
        p_s = torch.rand(m, generator=g, device="cuda")
        batch = ExpertBatch(
            x=x,
            expert_offsets=offsets,
            p_s=p_s,
            w1=w1,
            w2=w2,
            lora=None,
            output_slot=torch.arange(m, dtype=torch.int32, device="cuda"),
            numeric_profile=PROFILE,
        )
        x_q = mx_quantize(x, "e4m3")
        ours = _graph_us(lambda: backend.backward(batch, x_q, dy))
        dx = torch.empty(m, H, device="cuda", dtype=torch.bfloat16)

        def baseline():
            for e in range(E):
                lo, hi = e * per, (e + 1) * per
                z = x[lo:hi] @ w1b[e].t()
                gate, up = z[:, :F].float(), z[:, F:].float()
                gg = torch.clamp(gate, max=10.0)
                u = torch.clamp(up, min=-10.0, max=10.0)
                sig = torch.sigmoid(gg)
                dh = (dy[lo:hi] @ w2b[e]).float() * p_s[lo:hi, None]
                dgate = dh * u * sig * (1 + gg * (1 - sig)) * (gate < 10)
                dup = dh * gg * sig * ((up > -10) & (up < 10))
                dz = torch.cat([dgate, dup], 1).bfloat16()
                dx[lo:hi] = dz @ w1b[e]

        base = _graph_us(baseline)
        print(f"{m:>6} | {ours:8.1f} {base:9.1f} {ours / base:5.2f}")


if __name__ == "__main__":
    main()
