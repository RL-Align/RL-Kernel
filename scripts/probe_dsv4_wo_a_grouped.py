# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Probe DeepGEMM's single-launch grouped FP8 GEMMs for DSv4's grouped ``wo_a``.

``OutputProjection`` runs ``wo_a`` as 8 per-group ``fp8_gemm_nt`` calls (one launch each),
which cost about 2.2x vLLM's batched ``fp8_einsum`` at decode sizes. This checks whether
``m_grouped_fp8_gemm_nt_masked`` (all groups in one launch, A ``[G, T, K]`` as our
``inv_rope_fp8`` emits it) gives the same bytes as the per-group loop, and how fast it is.

Result (H200, 2026-11-11): it is 3-4x faster at decode sizes (13 us against 55 us at T = 1),
but not usable. Unpadded, it returns wrong values (relative error up to 80) unless T is a
multiple of 64. Zero-padded to whole 64-row blocks it matches at most T, but at some T
(257, 513, 300, 2049) it returns wrong values on some runs (relative error 5-12%), so it is
not reliable on these operands. ``OutputProjection`` instead runs the per-group GEMMs on
side streams.

    PYTHONPATH=. python scripts/probe_dsv4_wo_a_grouped.py
"""

from __future__ import annotations

import statistics

import torch

HEADS, HEAD_DIM, GROUPS, RANK = 64, 512, 8, 1024
MAX_POS = 32768
PAD = int(__import__("os").environ.get("WOA_PAD", "64"))
TOKENS = (1, 7, 63, 64, 65, 129, 200, 257, 300, 513, 1000, 1024, 2049, 4095, 4096, 16384)


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

    def masked(a, sc, m, expected):
        t = a.shape[1]
        z = torch.empty(GROUPS, t, RANK, dtype=torch.bfloat16, device="cuda")
        dg.m_grouped_fp8_gemm_nt_masked((a, sc), (w3, s3), z, m, expected, disable_ue8m0_cast=True)
        return z

    ok = True
    print("     T | zero-padded | unpadded | rel err unpadded | loop us | masked us")
    for t in TOKENS:
        pos = torch.randint(MAX_POS, (t,), generator=g, device="cuda")
        o = torch.randn(t, HEADS, HEAD_DIM, generator=g, device="cuda").bfloat16()
        a, sc = core.inv_rope_fp8(o, pos, cos_sin, GROUPS)
        m = torch.full((GROUPS,), t, dtype=torch.int32, device="cuda")
        want = loop(a, sc)
        raw = masked(a, sc, m, t)  # unpadded
        same_raw = torch.equal(raw.view(torch.int16), want.view(torch.int16))
        rel = float((raw.float() - want.float()).norm() / want.float().norm())
        tp = -(-t // PAD) * PAD
        o_pad = torch.zeros(tp, HEADS, HEAD_DIM, dtype=o.dtype, device="cuda")
        o_pad[:t] = o
        pos_pad = torch.zeros(tp, dtype=pos.dtype, device="cuda")
        pos_pad[:t] = pos
        ap, scp = core.inv_rope_fp8(o_pad, pos_pad, cos_sin, GROUPS)
        padded = masked(ap, scp, m, t)[:, :t]
        same_pad = torch.equal(padded.view(torch.int16), want.view(torch.int16))
        ok &= same_pad
        t_loop = _graph_us(lambda: loop(a, sc))
        t_mask = _graph_us(lambda: masked(ap, scp, m, t))
        print(
            f"{t:6d} | {'yes' if same_pad else 'NO':>11} | {'yes' if same_raw else 'NO':>8} | "
            f"{rel:16.1e} | {t_loop:7.1f} | {t_mask:9.1f}"
        )
    print("ALL SAME" if ok else "SOME DIFF")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
