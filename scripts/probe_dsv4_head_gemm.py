# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Probe DeepGEMM's BF16 GEMMs at the DSv4 LM-head shape (H = 4096, V = 129280).

Forward ``logits = h @ W.T`` (``bf16_gemm_nt``, FP32 out) and backward
``dh = dlogits @ W`` (``bf16_gemm_nn``, FP32 out). Checks, bytewise:

- batch invariance: each row of a sub-batch equals the full batch's row, for M from 1
  to 4096;
- vocab-shard invariance: a contiguous block of ``W``'s rows (a TP shard aligned to the
  64-tile contract) gives the same columns as the full ``W``;
- run-to-run determinism.

It also times both against cuBLAS (``torch.mm``, BF16 and FP32 output).

    python scripts/probe_dsv4_head_gemm.py
"""

from __future__ import annotations

import torch

H, V = 4096, 129280


def _same(a, b) -> bool:
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def _time(fn, iters=10) -> float:
    for _ in range(2):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3


def main() -> int:
    from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm

    dg, _ = require_deepgemm(torch.device("cuda"))
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(V, H, generator=g, device="cuda") / H**0.5).bfloat16()

    def nt(x, ww):
        out = torch.empty(x.shape[0], ww.shape[0], dtype=torch.float32, device="cuda")
        dg.bf16_gemm_nt(x.contiguous(), ww.contiguous(), out)
        return out

    def nn(d, ww):
        out = torch.empty(d.shape[0], ww.shape[1], dtype=torch.float32, device="cuda")
        dg.bf16_gemm_nn(d.contiguous(), ww.contiguous(), out)
        return out

    ok = True
    x = torch.randn(4096, H, generator=g, device="cuda").bfloat16()
    full = nt(x, w)
    for lo, hi in ((0, 1), (5, 12), (100, 164), (1000, 1513), (0, 2048), (3000, 4096)):
        same = _same(nt(x[lo:hi], w), full[lo:hi])
        ok &= same
        print(f"nt batch rows [{lo}, {hi}): {'same' if same else 'DIFF'}")
    for tp in (2, 4, 8):
        shard = V // tp
        for r in range(tp):
            part = nt(x[:513], w[r * shard : (r + 1) * shard])
            same = _same(part, full[:513, r * shard : (r + 1) * shard])
            ok &= same
            if not same:
                print(f"nt vocab shard tp={tp} rank={r}: DIFF")
        print(f"nt vocab shards tp={tp}: checked")
    ok &= _same(nt(x, w), full)
    print(f"nt rerun: {'same' if _same(nt(x, w), full) else 'DIFF'}")

    d = (torch.randn(1024, V, generator=g, device="cuda") * 1e-3).bfloat16()
    dfull = nn(d, w)
    for lo, hi in ((0, 1), (7, 70), (512, 1024)):
        same = _same(nn(d[lo:hi], w), dfull[lo:hi])
        ok &= same
        print(f"nn batch rows [{lo}, {hi}): {'same' if same else 'DIFF'}")
    same = _same(nn(d, w), dfull)
    ok &= same
    print(f"nn rerun: {'same' if same else 'DIFF'}")
    ref = d.float() @ w.float()
    rel = float((dfull - ref).norm() / ref.norm())
    print(f"nn vs FP32 reference rel L2 {rel:.2e}")

    print("\n     M |  dg nt fp32 | cublas bf16 | cublas fp32 |  dg nn fp32 | cublas nn bf16")
    for m in (1, 64, 1024, 4096, 16384):
        xm = torch.randn(m, H, generator=g, device="cuda").bfloat16()
        dm = (torch.randn(m, V, generator=g, device="cuda") * 1e-3).bfloat16()
        t_nt = _time(lambda: nt(xm, w))
        t_cb = _time(lambda: xm @ w.t())
        t_cf = _time(lambda: torch.mm(xm, w.t(), out_dtype=torch.float32))
        t_nn = _time(lambda: nn(dm, w))
        t_cn = _time(lambda: dm @ w)
        print(f"{m:6d} | {t_nt:11.1f} | {t_cb:11.1f} | {t_cf:11.1f} | {t_nn:11.1f} | {t_cn:13.1f}")
    print("\nALL SAME" if ok else "\nSOME DIFF")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
