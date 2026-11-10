# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Probe DeepGEMM's BF16 GEMM for DSv4 batch invariance and speed (SM90).

DSv4 computes three unquantized BF16 projections of the hidden state in every
compressed-attention layer (vLLM ``_run_parallel_input_projections``):

* ``weights_proj``: indexer per-head weights, ``[T, 4096] -> [T, 64]`` BF16;
* indexer compressor ``fused_wkv_wgate``: ``-> [T, 512]`` with FP32 output;
* main compressor ``fused_wkv_wgate``: ``-> [T, 2048]`` (C4) or ``[T, 1024]``
  (C128) with FP32 output;
* MoE router gate (every layer): ``-> [T, 256]`` with FP32 output.

cuBLAS picks kernels (and split-K) by M, so it is not batch-invariant. This
checks whether unmodified pinned DeepGEMM ``bf16_gemm_nt`` is, with BF16 and
FP32 output: rows of an M=4096 batch against prefixes, single rows and a
window, and run to run (3 seeds). A cuBLAS negative control shows the checks
can fail. Timing (CUDA-graph replay) is against cuBLAS on the same shapes.

Usage:
    python scripts/probe_dsv4_deepgemm_bf16_bi.py [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

SHAPES = (  # (name, N, K, output dtype)
    ("indexer_weights_proj", 64, 4096, torch.bfloat16),
    ("indexer_compressor_wkv_wgate", 512, 4096, torch.float32),
    ("compressor_c128_wkv_wgate", 1024, 4096, torch.float32),
    ("compressor_c4_wkv_wgate", 2048, 4096, torch.float32),
    ("moe_router_gate", 256, 4096, torch.float32),
)
M_MAX = 4096
M_LIST = (1, 3, 8, 64, 129, 256, 1000)
SINGLE_ROWS = (0, 1, 127, 128, 129, 2047, 4095)
WINDOW = (1000, 129)
PERF_M = (1, 64, 256, 1024, 4096)


def _dg():
    import deep_gemm

    return deep_gemm


def _deepgemm(x, w, out_dtype):
    d = torch.empty(x.shape[0], w.shape[0], device="cuda", dtype=out_dtype)
    if x.shape[0]:
        _dg().bf16_gemm_nt(x, w, d)
    return d


def _cublas(x, w, out_dtype):
    if out_dtype == torch.float32:
        return torch.mm(x, w.t(), out_dtype=torch.float32)
    return torch.nn.functional.linear(x, w)


def _same(a, b) -> bool:
    view = torch.int32 if a.dtype == torch.float32 else torch.int16
    return a.shape == b.shape and torch.equal(a.view(view), b.view(view))


def probe(fn, name, n, k, out_dtype, seed) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(M_MAX, k, generator=gen).to("cuda", torch.bfloat16)
    w = (torch.randn(n, k, generator=gen) / k**0.5).to("cuda", torch.bfloat16)
    full = fn(x, w, out_dtype)
    checks = {"run_to_run": _same(fn(x, w, out_dtype), full)}
    for m in M_LIST:
        checks[f"prefix_m{m}"] = _same(fn(x[:m].contiguous(), w, out_dtype), full[:m])
    for r in SINGLE_ROWS:
        checks[f"row{r}_alone"] = _same(
            fn(x[r : r + 1].contiguous(), w, out_dtype), full[r : r + 1]
        )
    off, m = WINDOW
    checks["window"] = _same(fn(x[off : off + m].contiguous(), w, out_dtype), full[off : off + m])
    ref = x.float() @ w.float().t()
    rel = float((full.float() - ref).norm() / ref.norm())
    return {
        "name": name,
        "n": n,
        "k": k,
        "out": str(out_dtype),
        "seed": seed,
        "passed": all(checks.values()),
        "failed": [c for c, ok in checks.items() if not ok],
        "rel_err_vs_fp32": rel,
    }


def _graph_us(fn, iters=50) -> float:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "device": torch.cuda.get_device_name(0),
        "num_sms": int(_dg().get_num_sms()),
        "cases": [],
        "negative_control": [],
        "perf": [],
    }
    for name, n, k, out in SHAPES:
        for seed in (0, 1, 2):
            case = probe(_deepgemm, name, n, k, out, seed)
            report["cases"].append(case)
            print(
                f"deepgemm {name:<30} seed={seed} {'PASS' if case['passed'] else 'FAIL'} "
                f"{case['failed']} rel_err={case['rel_err_vs_fp32']:.2e}"
            )
        neg = probe(_cublas, name, n, k, out, 0)
        report["negative_control"].append(neg)
        print(
            f"cublas   {name:<30} {'invariant' if neg['passed'] else 'VARIES'} {neg['failed'][:4]}"
        )
    for name, n, k, out in SHAPES:
        w = torch.randn(n, k, device="cuda").to(torch.bfloat16)
        for m in PERF_M:
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            d = torch.empty(m, n, device="cuda", dtype=out)
            ours = _graph_us(lambda: _dg().bf16_gemm_nt(x, w, d))
            base = _graph_us(lambda: _cublas(x, w, out))
            report["perf"].append({"name": name, "m": m, "deepgemm_us": ours, "cublas_us": base})
            print(
                f"perf {name:<30} m={m:<5} deepgemm {ours:7.1f} us  cublas {base:7.1f} us  "
                f"ratio {ours / base:.2f}"
            )
    ok = all(c["passed"] for c in report["cases"])
    print("ALL BI CHECKS PASS" if ok else "SOME BI CHECKS FAIL")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
