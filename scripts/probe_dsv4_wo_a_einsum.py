# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Probe DeepGEMM ``fp8_einsum('bhr,hdr->bhd')`` for DSv4's grouped ``wo_a``.

``OutputProjection`` runs ``wo_a`` as 8 per-group ``fp8_gemm_nt`` calls on side streams
(1.45-1.53x vLLM at T <= 64). vLLM runs the 8 groups in one ``fp8_einsum`` launch
(SM90: ``sm90_fp8_bmm``). This checks, for every T, whether the einsum gives the bytes of
the per-group loop (so it can replace it without breaking batch invariance), repeats each
call to catch intermittent differences, and times both (CUDA-graph replay).

    PYTHONPATH=. python scripts/probe_dsv4_wo_a_einsum.py
"""

from __future__ import annotations

import statistics

import torch

HEADS, HEAD_DIM, GROUPS, RANK = 64, 512, 8, 1024
MAX_POS = 32768
TOKENS = (
    1,
    2,
    3,
    7,
    8,
    16,
    31,
    63,
    64,
    65,
    100,
    128,
    129,
    200,
    257,
    300,
    513,
    1000,
    1024,
    2047,
    2049,
    4095,
    4096,
    8192,
    16384,
)
REPEATS = 5


def _graph_us(fn, iters: int = 20) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
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
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3)
    return statistics.median(times)


def main() -> int:
    from rl_engine.kernels.ops.cuda import dsv4_attention as core
    from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm
    from rl_engine.kernels.ops.cuda.fp8_linear import quantize_weight

    dg, _ = require_deepgemm(torch.device("cuda"))
    g = torch.Generator(device="cuda").manual_seed(0)
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    ang = torch.arange(MAX_POS, device="cuda").float()[:, None] * inv[None]
    cos_sin = torch.cat([ang.cos(), ang.sin()], -1).contiguous()
    wo_a = quantize_weight(
        (
            torch.randn(GROUPS * RANK, HEADS // GROUPS * HEAD_DIM, generator=g, device="cuda")
            * 0.02
        ).bfloat16()
    )
    w3 = wo_a.q.view(GROUPS, RANK, -1)
    s3 = wo_a.scale.view(GROUPS, RANK // 128, -1)
    per_group = [(w3[i].contiguous(), s3[i].contiguous()) for i in range(GROUPS)]

    def loop(a, sc):
        z = torch.empty(GROUPS, a.shape[1], RANK, dtype=torch.bfloat16, device="cuda")
        for i in range(GROUPS):
            dg.fp8_gemm_nt((a[i], sc[i]), per_group[i], z[i], disable_ue8m0_cast=True)
        return z

    def einsum(a, sc):
        z = torch.empty(a.shape[1], GROUPS, RANK, dtype=torch.bfloat16, device="cuda")
        dg.fp8_einsum("bhr,hdr->bhd", (a.permute(1, 0, 2), sc.permute(1, 0, 2)), (w3, s3), z)
        return z

    ok = True
    print("     T | same bytes | rel err  | loop us | einsum us")
    for t in TOKENS:
        pos = torch.randint(MAX_POS, (t,), generator=g, device="cuda")
        o = torch.randn(t, HEADS, HEAD_DIM, generator=g, device="cuda").bfloat16()
        a, sc = core.inv_rope_fp8(o, pos, cos_sin, GROUPS)
        want = loop(a, sc).transpose(0, 1)
        same, rel = True, 0.0
        for _ in range(REPEATS):
            got = einsum(a, sc)
            same &= torch.equal(got.view(torch.int16), want.view(torch.int16))
            rel = max(rel, float((got.float() - want.float()).norm() / want.float().norm()))
        ok &= same
        t_loop = _graph_us(lambda: loop(a, sc))
        t_ein = _graph_us(lambda: einsum(a, sc))
        print(f"{t:6d} | {'yes' if same else 'NO':>10} | {rel:8.1e} | {t_loop:7.1f} | {t_ein:9.1f}")
    print("ALL SAME" if ok else "SOME DIFF")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
