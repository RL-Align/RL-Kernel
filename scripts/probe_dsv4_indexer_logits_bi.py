# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Probe DeepGEMM's FP8 MQA-logits kernels for DSv4 batch invariance (SM90, T03).

The DSv4 indexer scores every compressed KV entry for every query with
DeepGEMM: ``fp8_fp4_mqa_logits`` for prefill (ragged ``[ks, ke)`` ranges over
a packed KV buffer) and ``fp8_fp4_paged_mqa_logits`` for decode (paged FP8
cache, per-token FP32 scales). On SM90 only the FP8 path exists (FP4 asserts
``arch_major == 10``), so that is what the rollout runs on Hopper.

Logits feed ``stable_topk_per_row``, so every logit of a query must be the same
bytes regardless of the batch. Checked here at the DSv4 indexer shape (64 heads,
head dim 128), FP32 logits, three seeds:

prefill
  * run to run;
  * prefixes of the query batch and single queries alone (with the same ranges);
  * a mid-batch window;
  * the request packed at another offset of a larger KV buffer, next to other
    requests (ks/ke shifted), and alone in its own buffer;
  * ``max_seqlen_k`` (compressed-logits layout) against the default layout;
decode (paged, next_n = 1)
  * run to run;
  * each request alone (batch 1) against inside batches of 8 and 64 requests
    with mixed context lengths (the SM schedule changes with the batch);
  * a different physical block assignment (block table permutation);
prefill vs decode
  * the paged logits of a query equal the prefill logits of the same query
    over the same KV bytes (chunked prefill and decode must pick the same KV).

Usage (SM90, CUDA 12.8+ JIT, pinned deep_gemm):
    python scripts/probe_dsv4_indexer_logits_bi.py [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

HEADS, DIM = 64, 128
BLOCK_KV = 64  # SM90 paged block size
SEEDS = (0, 1, 2)
SEQ_KV = 6000  # compressed KV entries of the probed request (C4 of a 24K context)
QUERIES = 700  # its queries (causal over the compressed KV)


def _dg():
    import deep_gemm

    return deep_gemm


def _quant_q(q: torch.Tensor) -> torch.Tensor:
    """Per (token, head) power-of-two FP8 scaling; the scale is folded away."""
    amax = q.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (q / scale).to(torch.float8_e4m3fn)


def _quant_kv(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = k.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (k / scale).to(torch.float8_e4m3fn), scale.squeeze(-1).float().contiguous()


def _make(seed: int, queries: int, seq_kv: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(queries, HEADS, DIM, generator=gen).cuda()
    k = torch.randn(seq_kv, DIM, generator=gen).cuda()
    w = torch.randn(queries, HEADS, generator=gen).cuda().contiguous()
    kq, ks = _quant_kv(k)
    # causal compressed ranges: query t sees the first ceil((t + 1) * seq_kv / queries)
    ends = torch.clamp(((torch.arange(queries) + 1) * seq_kv + queries - 1) // queries, max=seq_kv)
    return _quant_q(q).contiguous(), (kq, ks), w, ends.int().cuda()


def _prefill(q, kv, w, ks, ke, max_seqlen_k: int = 0) -> torch.Tensor:
    return _dg().fp8_fp4_mqa_logits(
        (q, None),
        kv,
        w,
        ks.contiguous(),
        ke.contiguous(),
        clean_logits=False,
        max_seqlen_k=max_seqlen_k,
        logits_dtype=torch.float32,
    )


def _rows_equal(a: torch.Tensor, b: torch.Tensor, ks, ke, b_offset=None) -> dict:
    """Compare logits row by row over each row's own [ks, ke) range."""
    bad = 0
    max_abs = 0.0
    for r in range(a.shape[0]):
        s, e = int(ks[r]), int(ke[r])
        bo = 0 if b_offset is None else int(b_offset[r])
        x = a[r, s:e]
        y = b[r, s - bo : e - bo] if b_offset is not None else b[r, s:e]
        if not torch.equal(x.view(torch.int32), y.view(torch.int32)):
            bad += 1
            max_abs = max(max_abs, float((x - y).abs().max()))
    return {"bitwise_equal": bad == 0, "rows_differing": bad, "max_abs": max_abs}


def probe_prefill(seed: int) -> dict:
    q, kv, w, ends = _make(seed, QUERIES, SEQ_KV)
    starts = torch.zeros_like(ends)
    full = _prefill(q, kv, w, starts, ends)
    checks = {"run_to_run": _rows_equal(_prefill(q, kv, w, starts, ends), full, starts, ends)}
    for m in (1, 3, 8, 64, 129, 256):
        sub = _prefill(q[:m], kv, w[:m], starts[:m], ends[:m])
        checks[f"prefix_{m}"] = _rows_equal(sub, full[:m], starts[:m], ends[:m])
    for r in (0, 1, 127, 128, 129, QUERIES - 1):
        one = _prefill(q[r : r + 1], kv, w[r : r + 1], starts[r : r + 1], ends[r : r + 1])
        checks[f"row{r}_alone"] = _rows_equal(
            one, full[r : r + 1], starts[r : r + 1], ends[r : r + 1]
        )
    lo, hi = 301, 301 + 77
    win = _prefill(q[lo:hi], kv, w[lo:hi], starts[lo:hi], ends[lo:hi])
    checks["window"] = _rows_equal(win, full[lo:hi], starts[lo:hi], ends[lo:hi])

    # Pack the request after another request's KV, with that request's queries first.
    gen = torch.Generator(device="cpu").manual_seed(seed + 100)
    other_kv = 1733
    other_q = 50
    ok, osc = _quant_kv(torch.randn(other_kv, DIM, generator=gen).cuda())
    oq = _quant_q(torch.randn(other_q, HEADS, DIM, generator=gen).cuda())
    ow = torch.randn(other_q, HEADS, generator=gen).cuda()
    packed_kv = (torch.cat([ok, kv[0]]).contiguous(), torch.cat([osc, kv[1]]).contiguous())
    o_end = torch.full((other_q,), other_kv, dtype=torch.int32, device="cuda")
    p_ks = torch.cat([torch.zeros_like(o_end), starts + other_kv])
    p_ke = torch.cat([o_end, ends + other_kv])
    packed = _prefill(
        torch.cat([oq, q]).contiguous(), packed_kv, torch.cat([ow, w]).contiguous(), p_ks, p_ke
    )
    checks["packed_offset"] = _rows_equal(
        packed[other_q:], full, p_ks[other_q:], p_ke[other_q:], b_offset=p_ks[other_q:]
    )
    # Sanity: the logits are the intended quantity, sum_h w_h * relu(q_h . k) * k_scale.
    rows = torch.tensor([0, 129, QUERIES - 1], device="cuda")
    dots = torch.einsum("rhd,nd->rhn", q[rows].float(), kv[0].float())
    ref = (torch.relu(dots) * w[rows].unsqueeze(-1)).sum(1) * kv[1]
    err = 0.0
    for i, r in enumerate(rows.tolist()):
        e = int(ends[r])
        err = max(err, float((full[r, :e] - ref[i, :e]).norm() / ref[i, :e].norm()))
    # SM90 FP8 WGMMA accumulates with reduced precision (~1.6e-4 relative here).
    checks["matches_reference"] = {"bitwise_equal": err < 1e-3, "rel_err": err}
    compressed = _prefill(q, kv, w, starts, ends, max_seqlen_k=SEQ_KV)
    checks["max_seqlen_k_layout"] = _rows_equal(compressed, full, starts, ends)
    return {
        "kind": "prefill",
        "seed": seed,
        "passed": all(c["bitwise_equal"] for c in checks.values()),
        "checks": checks,
    }


def _build_paged(requests, perm_seed: int):
    """requests: list of (kq [n, D] fp8, scale [n]). Returns cache, block_table, lens."""
    blocks_per = [-(-r[0].shape[0] // BLOCK_KV) for r in requests]
    total = sum(blocks_per) + 3
    gen = torch.Generator(device="cpu").manual_seed(perm_seed)
    order = torch.randperm(total, generator=gen)
    cache = torch.zeros(total, BLOCK_KV * (DIM + 4), dtype=torch.uint8, device="cuda")
    table = torch.zeros(len(requests), max(blocks_per), dtype=torch.int32, device="cuda")
    used = 0
    for i, (kq, sc) in enumerate(requests):
        n = kq.shape[0]
        for b in range(blocks_per[i]):
            phys = int(order[used])
            used += 1
            table[i, b] = phys
            lo, hi = b * BLOCK_KV, min(n, (b + 1) * BLOCK_KV)
            rows = hi - lo
            cache[phys, : rows * DIM] = kq[lo:hi].reshape(-1).view(torch.uint8)
            cache[phys, BLOCK_KV * DIM : BLOCK_KV * DIM + rows * 4] = (
                sc[lo:hi].contiguous().view(torch.uint8)
            )
    lens = torch.tensor([[r[0].shape[0]] for r in requests], dtype=torch.int32, device="cuda")
    return cache.view(total, BLOCK_KV, 1, DIM + 4), table, lens


def _paged(q, cache, table, lens, w, max_len: int) -> torch.Tensor:
    dg = _dg()
    meta = dg.get_paged_mqa_logits_metadata(lens, BLOCK_KV, dg.get_num_sms())
    return dg.fp8_fp4_paged_mqa_logits(
        (q.unsqueeze(1).contiguous(), None),
        cache,
        w,
        lens,
        table,
        meta,
        max_len,
        clean_logits=False,
        logits_dtype=torch.float32,
    )


def probe_paged(seed: int) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(seed + 7)
    batch = 64
    lens = torch.randint(1, 9000, (batch,), generator=gen).tolist()
    lens[0], lens[1], lens[2] = 1, BLOCK_KV, 8191
    reqs = [_quant_kv(torch.randn(n, DIM, generator=gen).cuda()) for n in lens]
    q = _quant_q(torch.randn(batch, HEADS, DIM, generator=gen).cuda())
    w = torch.randn(batch, HEADS, generator=gen).cuda().contiguous()
    max_len = 9216
    cache, table, clens = _build_paged(reqs, seed)
    full = _paged(q, cache, table, clens, w, max_len)
    zeros = torch.zeros(batch, dtype=torch.int32)
    ends = torch.tensor(lens, dtype=torch.int32)
    checks = {
        "run_to_run": _rows_equal(_paged(q, cache, table, clens, w, max_len), full, zeros, ends)
    }
    for size in (1, 8):
        for r in (0, 1, 2, 5, batch - 1):
            lo = min(r, batch - size)
            sub = _paged(
                q[lo : lo + size],
                cache,
                table[lo : lo + size],
                clens[lo : lo + size],
                w[lo : lo + size],
                max_len,
            )
            checks[f"batch{size}_req{r}"] = _rows_equal(
                sub, full[lo : lo + size], zeros[:size], ends[lo : lo + size]
            )
    cache2, table2, clens2 = _build_paged(reqs, seed + 1000)
    checks["block_permutation"] = _rows_equal(
        _paged(q, cache2, table2, clens2, w, max_len), full, zeros, ends
    )

    # Prefill over the same bytes: request i alone as one query.
    for r in (0, 1, 2, 3, 17, batch - 1):
        kq, sc = reqs[r]
        pre = _prefill(
            q[r : r + 1].contiguous(),
            (kq.contiguous(), sc.contiguous()),
            w[r : r + 1].contiguous(),
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            torch.tensor([lens[r]], dtype=torch.int32, device="cuda"),
        )
        checks[f"paged_vs_prefill_req{r}"] = _rows_equal(
            pre, full[r : r + 1], zeros[:1], ends[r : r + 1]
        )
    return {
        "kind": "paged",
        "seed": seed,
        "passed": all(c["bitwise_equal"] for c in checks.values()),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    dg = _dg()
    report = {
        "device": torch.cuda.get_device_name(0),
        "deep_gemm": getattr(dg, "__version__", "?"),
        "num_sms": int(dg.get_num_sms()),
        "cases": [],
    }
    for seed in SEEDS:
        for fn in (probe_prefill, probe_paged):
            case = fn(seed)
            report["cases"].append(case)
            failed = [k for k, v in case["checks"].items() if not v["bitwise_equal"]]
            print(f"{case['kind']:<8} seed={seed} {'PASS' if case['passed'] else 'FAIL'} {failed}")
            for k in failed:
                print("   ", k, case["checks"][k])
    ok = all(c["passed"] for c in report["cases"])
    print("ALL PASS" if ok else "SOME FAIL")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
