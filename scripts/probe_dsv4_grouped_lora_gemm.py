# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Probe DeepGEMM's grouped BF16 GEMMs for per-expert MoE LoRA adapters.

Per-expert adapters need, per routed row r of expert e: ``h = BF16(x_r A[e]^T)`` and
``d = h B[e]^T``. This checks, against one dense ``bf16_gemm_nt`` per expert:

- ``m_grouped_bf16_gemm_nt_contiguous`` with the m-indices layout (each expert's rows
  zero-padded to the M alignment) and with the psum layout (``grouped_layout[i]`` is
  expert i's end row; each expert starts at the previous end aligned up to the M
  alignment, ``scheduler/gemm.cuh``; no zero padding needed): same bytes? Repeated
  calls, several row distributions (empty experts, 1-row experts).
- batch invariance: a row's bytes when it is alone in its expert.
- ``k_grouped_bf16_gemm_tn_contiguous`` (the per-expert weight gradient): close to
  FP32 torch.
- time against the per-expert loop (CUDA events, eager).

    PYTHONPATH=. python scripts/probe_dsv4_grouped_lora_gemm.py
"""

from __future__ import annotations

import statistics

import torch

H, R, N2 = 4096, 64, 4096


def _ms(fn, iters=10):
    for _ in range(2):
        fn()
    t = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        t.append(a.elapsed_time(b))
    return statistics.median(t)


def main() -> int:
    from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm

    dg, _ = require_deepgemm(torch.device("cuda"))
    align = dg.get_mk_alignment_for_contiguous_layout()
    print("mk alignment", align)
    g = torch.Generator(device="cuda").manual_seed(0)
    ok = True
    for name, counts in (
        ("uneven", [0, 1, 7, 130, 0, 64, 3, 257] * 4),
        ("decode", [1] * 6 + [0] * 26),
        ("big", [800] * 32),
    ):
        e = len(counts)
        m = sum(counts)
        x = torch.randn(m, H, generator=g, device="cuda").bfloat16()
        a = (torch.randn(e, R, H, generator=g, device="cuda") / 64).bfloat16()
        b = (torch.randn(e, N2, R, generator=g, device="cuda") / 8).bfloat16()
        offs = [0]
        for c in counts:
            offs.append(offs[-1] + c)

        def loop(inp, w, n):
            out = torch.empty(inp.shape[0], n, dtype=torch.bfloat16, device="cuda")
            for i in range(e):
                lo, hi = offs[i], offs[i + 1]
                if hi > lo:
                    dg.bf16_gemm_nt(inp[lo:hi].contiguous(), w[i], out[lo:hi])
            return out

        # m-indices layout: rows padded per expert to the alignment.
        pad_offs, idx_parts, src_parts = [0], [], []
        for i, c in enumerate(counts):
            cp = -(-c // align) * align
            pad_offs.append(pad_offs[-1] + cp)
            idx_parts.append(torch.tensor([i] * c + [-1] * (cp - c), dtype=torch.int32))
            src_parts.append(torch.arange(offs[i], offs[i] + c))
        m_idx = torch.cat(idx_parts).cuda()
        dst = torch.cat([torch.arange(pad_offs[i], pad_offs[i] + c) for i, c in enumerate(counts)])
        dst = dst.cuda()
        mp = pad_offs[-1]

        def grouped_idx(inp, w, n):
            xp = torch.zeros(mp, inp.shape[1], dtype=torch.bfloat16, device="cuda")
            xp[dst] = inp
            out = torch.empty(mp, n, dtype=torch.bfloat16, device="cuda")
            if mp:
                dg.m_grouped_bf16_gemm_nt_contiguous(xp, w, out, m_idx)
            return out[dst]

        # psum layout: same aligned starts, the end offset of each expert (no -1 rows).
        psum = torch.tensor(
            [pad_offs[i] + c for i, c in enumerate(counts)], dtype=torch.int32, device="cuda"
        )

        def grouped_psum(inp, w, n):
            xp = torch.empty(mp, inp.shape[1], dtype=torch.bfloat16, device="cuda")
            xp[dst] = inp
            out = torch.empty(mp, n, dtype=torch.bfloat16, device="cuda")
            if mp:
                dg.m_grouped_bf16_gemm_nt_contiguous(xp, w, out, psum, use_psum_layout=True)
            return out[dst]

        want_h = loop(x, a, R)
        want_d = loop(want_h, b, N2)
        res = {}
        for label, fn in (("m_indices", grouped_idx), ("psum", grouped_psum)):
            try:
                same = True
                for _ in range(5):
                    h = fn(x, a, R)
                    d = fn(want_h, b, N2)
                    same &= torch.equal(h.view(torch.int16), want_h.view(torch.int16))
                    same &= torch.equal(d.view(torch.int16), want_d.view(torch.int16))
                res[label] = "same" if same else "DIFF"
                ok &= same
            except Exception as exc:  # noqa: BLE001
                res[label] = f"error: {str(exc).splitlines()[0][:80]}"
        # Batch invariance of the dense per-expert GEMM: a row alone.
        i = max(range(e), key=lambda j: counts[j])
        r0 = offs[i] + counts[i] // 2
        alone = torch.empty(1, R, dtype=torch.bfloat16, device="cuda")
        dg.bf16_gemm_nt(x[r0 : r0 + 1].contiguous(), a[i], alone)
        bi = torch.equal(alone.view(torch.int16), want_h[r0 : r0 + 1].view(torch.int16))
        ok &= bi
        # Per-expert wgrad: dB[e] = dd_e^T h_e (FP32), k-grouped over tokens.
        dd = torch.randn(m, N2, generator=g, device="cuda").bfloat16()
        ref = torch.zeros(e, N2, R, device="cuda")
        for j in range(e):
            lo, hi = offs[j], offs[j + 1]
            ref[j] = dd[lo:hi].float().t() @ want_h[lo:hi].float()
        wg = "n/a"
        try:
            ks = [-(-c // align) * align for c in counts]
            ddp = torch.zeros(sum(ks), N2, dtype=torch.bfloat16, device="cuda")
            hp = torch.zeros(sum(ks), R, dtype=torch.bfloat16, device="cuda")
            ddp[dst], hp[dst] = dd, want_h
            out = torch.zeros(e, N2, R, device="cuda")
            ks_t = torch.tensor(ks, dtype=torch.int32, device="cuda")
            dg.k_grouped_bf16_gemm_tn_contiguous(ddp, hp, out, ks, ks_t, out)
            wg = f"rel {float((out - ref).norm() / ref.norm()):.1e}"
        except Exception as exc:  # noqa: BLE001
            wg = f"error: {str(exc).splitlines()[0][:80]}"
        t_loop = _ms(lambda: loop(loop(x, a, R), b, N2))
        t_grp = _ms(lambda: grouped_idx(grouped_idx(x, a, R), b, N2))
        try:
            t_psum = f"{_ms(lambda: grouped_psum(grouped_psum(x, a, R), b, N2)):.3f}"
        except Exception:  # noqa: BLE001
            t_psum = "-"
        print(
            f"{name:7s} rows {m:6d} | m_indices {res['m_indices']} | psum {res['psum']} | "
            f"row alone {'same' if bi else 'DIFF'} | wgrad {wg} | "
            f"ms loop {t_loop:.3f} m_idx {t_grp:.3f} psum {t_psum}"
        )
    print("ALL SAME" if ok else "SOME DIFF")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
