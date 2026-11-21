# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Probe FlashMLA's SM90 sparse kernel for DSv4 batch invariance (T05).

DSv4 attention on Hopper is ``flash_mla_sparse_fwd`` over a BF16 KV workspace
(dequantized compressed KV + the sliding window), with per-query index lists
(top-k compressed entries + window, ``-1`` padded) and a per-head attention
sink; vLLM runs the same kernel for prefill and decode under
``VLLM_BATCH_INVARIANT=1``. Shape: 64 query heads (padded), 1 KV head, head
dim 512, scale 512 ** -0.5, index width 640 (C4: 512 + 128 window) and 1152.

Checks (output and LSE bytes, 3 seeds):
  * run to run;
  * query prefixes, single queries and a window against the full batch;
  * the KV workspace with extra rows in front (indices shifted): position of
    a request's KV in the workspace must not matter;
  * ``topk_length`` against ``-1`` padding of the same list;
  * index order: the same set permuted. Expected to differ (accumulation
    follows index order), which is why the wrapper canonicalizes indices.
Timing (CUDA-graph replay) against an eager PyTorch gather + softmax.

Usage:
    python scripts/probe_dsv4_flashmla_bi.py [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

H, D = 64, 512
SCALE = D**-0.5
SEEDS = (0, 1, 2)


def _fm():
    from rl_engine.kernels.ops.cuda.flashmla import require_flashmla

    return require_flashmla()[0]


def _run(q, kv, idx, sink, topk_length=None):
    out, lse = _fm().flash_mla_sparse_fwd(q, kv, idx, SCALE, 512, sink, topk_length)[:2]
    return out, lse


def _same(a, b):
    view = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return a.shape == b.shape and torch.equal(a.view(view), b.view(view))


def _case(seed, sq, skv, width, valid_frac=0.85):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(sq, H, D, generator=gen).to(torch.bfloat16).cuda()
    kv = torch.randn(skv, 1, D, generator=gen).to(torch.bfloat16).cuda()
    sink = torch.randn(H, generator=gen).cuda()
    idx = torch.full((sq, 1, width), -1, dtype=torch.int32)
    lens = torch.zeros(sq, dtype=torch.int32)
    for t in range(sq):
        n = max(1, int(width * valid_frac * float(torch.rand(1, generator=gen))))
        idx[t, 0, :n] = torch.sort(torch.randperm(skv, generator=gen)[:n]).values.int()
        lens[t] = n
    return q, kv, idx.cuda(), sink, lens.cuda()


def probe(seed, width):
    sq, skv = 300, 6000
    q, kv, idx, sink, lens = _case(seed, sq, skv, width)
    out, lse = _run(q, kv, idx, sink)
    checks = {}
    o2, l2 = _run(q, kv, idx, sink)
    checks["run_to_run"] = _same(o2, out) and _same(l2, lse)
    for lo, hi in ((0, 1), (0, 7), (0, 64), (17, 18), (128, 129), (100, 229), (299, 300)):
        o, lse_ = _run(q[lo:hi].contiguous(), kv, idx[lo:hi].contiguous(), sink)
        checks[f"rows_{lo}_{hi}"] = _same(o, out[lo:hi]) and _same(lse_, lse[lo:hi])
    pad = 777
    kv_big = torch.cat([torch.randn(pad, 1, D, device="cuda").to(torch.bfloat16), kv])
    idx_shift = torch.where(idx >= 0, idx + pad, idx)
    o, lse_ = _run(q, kv_big, idx_shift, sink)
    checks["kv_offset"] = _same(o, out) and _same(lse_, lse)
    o, lse_ = _run(q, kv, idx, sink, topk_length=lens)
    checks["topk_length_vs_padding"] = _same(o, out) and _same(lse_, lse)
    perm = idx.clone()
    for t in range(sq):
        n = int(lens[t])
        perm[t, 0, :n] = idx[t, 0, :n][torch.randperm(n, device="cuda")]
    o, lse_ = _run(q, kv, perm, sink)
    order_matters = not (_same(o, out) and _same(lse_, lse))
    return {
        "seed": seed,
        "width": width,
        "passed": all(checks.values()),
        "failed": [k for k, v in checks.items() if not v],
        "index_order_changes_bytes": order_matters,
    }


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


def _eager_us(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / iters


def _torch_sparse(q, kv, idx, sink):
    """Naive gather + softmax with the sink (the non-kernel baseline)."""
    safe = idx.clamp(min=0).long()[:, 0]
    k = kv[:, 0][safe].float()  # [sq, width, D]
    s = torch.einsum("thd,twd->thw", q.float(), k) * SCALE
    s = s.masked_fill((idx[:, 0] < 0)[:, None, :], -float("inf"))
    lse = torch.logsumexp(s, -1)
    p = torch.softmax(s, -1)
    o = torch.einsum("thw,twd->thd", p, k)
    return (o * torch.sigmoid(lse - sink[None])[..., None]).to(torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = {"device": torch.cuda.get_device_name(0), "cases": [], "perf": []}
    for width in (640, 1152):
        for seed in SEEDS:
            case = probe(seed, width)
            report["cases"].append(case)
            print(
                f"width={width} seed={seed} {'PASS' if case['passed'] else 'FAIL'} "
                f"{case['failed']} index_order_changes_bytes={case['index_order_changes_bytes']}"
            )
    q, kv, idx, sink, _ = _case(0, 64, 8000, 640)
    ref = _torch_sparse(q, kv, idx, sink)
    out = _run(q, kv, idx, sink)[0]
    rel = float((out.float() - ref.float()).norm() / ref.float().norm())
    report["rel_err_vs_torch"] = rel
    print(f"rel err vs eager torch (sink applied): {rel:.2e}")
    for sq in (1, 64, 1024, 4096):
        q, kv, idx, sink, _ = _case(1, sq, 16384, 640, valid_frac=1.0)
        ours = _graph_us(lambda: _run(q, kv, idx, sink))
        base = _eager_us(lambda: _torch_sparse(q, kv, idx, sink)) if sq <= 1024 else float("nan")
        report["perf"].append({"s_q": sq, "flashmla_us": ours, "torch_us": base})
        print(f"perf s_q={sq:<5} flashmla {ours:9.1f} us   eager torch {base:10.1f} us")
    ok = all(c["passed"] for c in report["cases"])
    print("ALL BI CHECKS PASS" if ok else "SOME BI CHECKS FAIL")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
