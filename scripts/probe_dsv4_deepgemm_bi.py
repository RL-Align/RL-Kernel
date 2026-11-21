#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DSv4 Phase 1 step 1: is pinned, unmodified DeepGEMM batch-invariant on SM90?

Probes the FP8 block-scaled GEMMs DSv4 runs through DeepGEMM at the
DeepSeek-V4-Flash-0731 projection shapes (TP=1, design D1/D3):

* dense ``fp8_gemm_nt``: rows computed inside an M=4096 batch must be
  byte-identical to the same rows computed alone, at other M, at other row
  offsets, and on a second run;
* grouped ``m_grouped_fp8_gemm_nt_{contiguous,masked}`` at the ``wo_a`` shape:
  a token's output must not change when other groups gain tokens, between the
  contiguous and masked layouts, or with the masked ``expected_m`` hint;
* backward: dgrad (``dX = dY @ W``, the same NT kernel with the transposed
  weight) gets the dense row checks; block-quantizing ``W.T`` must equal
  transposing the block-quantized ``W``; wgrad (``dW = dY.T @ X``, the 1D1D
  kernel, FP32 accumulate) reduces over tokens, so it only needs to be
  run-to-run identical (design D5).

It also times each dense shape against an eager BF16 cuBLAS matmul, the
non-invariant baseline of design D7, with CUDA-graph replay so host launch
overhead does not mask the kernels. Activations use 1x128 and weights 128x128
FP8 blocks with UE8M0-valued FP32 scales (SM90, rule 6). Quantization is
row-local and runs outside the timed region.

Run on an SM90 host with DeepGEMM installed (CUDA_HOME must point at the
toolkit DeepGEMM JIT-compiles with):
    python scripts/probe_dsv4_deepgemm_bi.py --output report.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import statistics
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_engine.kernels.ops.cuda.deepgemm import (  # noqa: E402
    DEEPGEMM_COMMIT,
    DeepGemmUnavailable,
    require_deepgemm,
)

# (name, N, K) from DeepSeek-V4-Flash-0731: hidden 4096, q_lora_rank 1024,
# head_dim 512 x 64 heads, index 128 x 64 heads, o_lora_rank 1024 x 8 groups,
# shared expert intermediate 2048.
DENSE_SHAPES = (
    ("fused_wqa_wkv", 1536, 4096),
    ("wq_b", 32768, 1024),
    ("wo_a_group", 1024, 4096),
    ("wo_b", 4096, 8192),
    ("indexer_wq_b", 8192, 1024),
    ("shared_gate_up", 4096, 4096),
    ("shared_down", 4096, 2048),
)
M_MAX = 4096
M_LIST = (1, 3, 8, 64, 129, 256, 1000)
SINGLE_ROWS = (0, 1, 127, 128, 129, 2047, 4095)
WINDOW = (1000, 129)  # (offset, rows): a mid-batch slice that crosses tiles
GROUPED = ("wo_a", 8, 1024, 4096)  # (name, groups, N, K)
PERF_M = (1, 64, 256, 4096)
# wgrad reduces over tokens; TMA needs the FP8 token stride to be 16-byte aligned.
WGRAD_TOKENS = (128, 1008, 4096)
WGRAD_UNALIGNED = 1000
WGRAD_REPEATS = 10


def _dg():
    import deep_gemm

    return deep_gemm


def _cast_a(x: torch.Tensor):
    from deep_gemm.utils import per_token_cast_to_fp8

    return per_token_cast_to_fp8(x, use_ue8m0=True)


def _cast_b(w: torch.Tensor):
    from deep_gemm.utils import per_block_cast_to_fp8

    return per_block_cast_to_fp8(w, use_ue8m0=True)


def _rows(a, start: int, stop: int):
    return a[0][start:stop].contiguous(), a[1][start:stop].contiguous()


def _gemm(a, b, n: int) -> torch.Tensor:
    d = torch.empty(a[0].shape[0], n, device="cuda", dtype=torch.bfloat16)
    _dg().fp8_gemm_nt(a, b, d, disable_ue8m0_cast=True)
    return d


def _compare(got: torch.Tensor, want: torch.Tensor) -> dict:
    same = torch.equal(got.view(torch.int16), want.view(torch.int16))
    if same:
        return {"bitwise_equal": True}
    rows = (got.view(torch.int16) != want.view(torch.int16)).any(dim=-1)
    return {
        "bitwise_equal": False,
        "rows_differing": int(rows.sum().item()),
        "max_abs": float((got.float() - want.float()).abs().max().item()),
    }


def probe_dense(name: str, n: int, k: int, seed: int, kind: str = "dense") -> dict:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(M_MAX, k, generator=gen).to("cuda", torch.bfloat16)
    w = torch.randn(n, k, generator=gen).to("cuda", torch.bfloat16)
    a, b = _cast_a(x), _cast_b(w)

    full = _gemm(a, b, n)
    checks = {"run_to_run": _compare(_gemm(a, b, n), full)}
    for m in M_LIST:
        checks[f"prefix_m{m}"] = _compare(_gemm(_rows(a, 0, m), b, n), full[:m])
    for row in SINGLE_ROWS:
        checks[f"row{row}_alone"] = _compare(
            _gemm(_rows(a, row, row + 1), b, n), full[row : row + 1]
        )
    off, m = WINDOW
    checks[f"window_{off}_{m}"] = _compare(_gemm(_rows(a, off, off + m), b, n), full[off : off + m])
    return {
        "kind": kind,
        "name": name,
        "n": n,
        "k": k,
        "seed": seed,
        "passed": all(c["bitwise_equal"] for c in checks.values()),
        "checks": checks,
    }


def probe_transpose(name: str, n: int, k: int, seed: int) -> dict:
    """dgrad needs ``W.T`` K-major; with 128x128 blocks it can be precomputed."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, k, generator=gen).to("cuda", torch.bfloat16)
    q, s = _cast_b(w)
    qt, st = _cast_b(w.t().contiguous())
    same = torch.equal(qt.view(torch.uint8), q.t().contiguous().view(torch.uint8)) and torch.equal(
        st, s.t().contiguous()
    )
    return {"kind": "transpose", "name": name, "n": n, "k": k, "seed": seed, "passed": same}


def _cast_tokens(x: torch.Tensor):
    """Quantize ``x`` [tokens, C] along tokens: 128-token blocks per channel."""
    return _cast_a(x.t().contiguous())


def _wgrad(a, b, n: int, k: int, init: torch.Tensor | None = None) -> torch.Tensor:
    d = torch.zeros(n, k, device="cuda", dtype=torch.float) if init is None else init.clone()
    _dg().fp8_gemm_nt(a, b, d, c=d, recipe=(1, 1, 128), disable_ue8m0_cast=True)
    return d


def _compare_fp32(got: torch.Tensor, want: torch.Tensor) -> dict:
    if torch.equal(got.view(torch.int32), want.view(torch.int32)):
        return {"bitwise_equal": True}
    return {
        "bitwise_equal": False,
        "elements_differing": int((got.view(torch.int32) != want.view(torch.int32)).sum().item()),
        "max_abs": float((got - want).abs().max().item()),
    }


def probe_wgrad(name: str, n: int, k: int, seed: int) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dy = torch.randn(max(WGRAD_TOKENS), n, generator=gen).to("cuda", torch.bfloat16)
    x = torch.randn(max(WGRAD_TOKENS), k, generator=gen).to("cuda", torch.bfloat16)
    acc = torch.randn(n, k, generator=gen).to("cuda")
    checks = {}
    for tokens in WGRAD_TOKENS:
        a, b = _cast_tokens(dy[:tokens]), _cast_tokens(x[:tokens])
        first = _wgrad(a, b, n, k)
        runs = [_compare_fp32(_wgrad(a, b, n, k), first) for _ in range(WGRAD_REPEATS - 1)]
        checks[f"tokens{tokens}_run_to_run_x{WGRAD_REPEATS}"] = {
            "bitwise_equal": all(r["bitwise_equal"] for r in runs)
        }
        first_acc = _wgrad(a, b, n, k, acc)
        checks[f"tokens{tokens}_accumulate_run_to_run"] = _compare_fp32(
            _wgrad(a, b, n, k, acc), first_acc
        )
    try:
        _wgrad(_cast_tokens(dy[:WGRAD_UNALIGNED]), _cast_tokens(x[:WGRAD_UNALIGNED]), n, k)
        torch.cuda.synchronize()
        unaligned = "supported"
    except RuntimeError as exc:
        unaligned = f"rejected: {str(exc).splitlines()[0][:120]}"
    return {
        "kind": "wgrad",
        "name": name,
        "n": n,
        "k": k,
        "seed": seed,
        "passed": all(c["bitwise_equal"] for c in checks.values()),
        "checks": checks,
        "info": {f"tokens{WGRAD_UNALIGNED}": unaligned},
    }


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    return float(((got.float() - want).norm() / want.norm()).item())


def precision(name: str, n: int, k: int, tokens: int = 4096) -> dict:
    """Relative L2 error of FP8 forward, dgrad and wgrad against FP32, next to BF16's."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(tokens, k, generator=gen).to("cuda", torch.bfloat16)
    w = torch.randn(n, k, generator=gen).to("cuda", torch.bfloat16)
    dy = torch.randn(tokens, n, generator=gen).to("cuda", torch.bfloat16)
    x32, w32, dy32 = x.float(), w.float(), dy.float()
    fwd, dgrad, wgrad = x32 @ w32.t(), dy32 @ w32, dy32.t() @ x32
    return {
        "name": name,
        "tokens": tokens,
        "fp8_forward": _rel_err(_gemm(_cast_a(x), _cast_b(w), n), fwd),
        "fp8_dgrad": _rel_err(_gemm(_cast_a(dy), _cast_b(w.t().contiguous()), k), dgrad),
        "fp8_wgrad": _rel_err(_wgrad(_cast_tokens(dy), _cast_tokens(x), n, k), wgrad),
        "bf16_forward": _rel_err(x @ w.t(), fwd),
        "bf16_dgrad": _rel_err(dy @ w, dgrad),
        "bf16_wgrad": _rel_err(dy.t() @ x, wgrad),
    }


def negative_control(name: str, n: int, k: int) -> dict:
    """Run the prefix checks on eager BF16 cuBLAS, which is not batch-invariant.

    At least one shape must differ, or the comparisons cannot detect a failure."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(M_MAX, k, generator=gen).to("cuda", torch.bfloat16)
    w = torch.randn(n, k, generator=gen).to("cuda", torch.bfloat16)
    full = x @ w.t()
    checks = {f"prefix_m{m}": _compare(x[:m] @ w.t(), full[:m]) for m in M_LIST}
    return {
        "name": name,
        "n": n,
        "k": k,
        "detected_variance": not all(c["bitwise_equal"] for c in checks.values()),
        "checks": checks,
    }


def _contiguous(tokens: list[torch.Tensor], k: int):
    """Stack per-group tokens into DeepGEMM's contiguous layout (128-row aligned)."""
    align = _dg().get_mk_alignment_for_contiguous_layout()
    segments, layout, starts = [], [], []
    row = 0
    for g, t in enumerate(tokens):
        padded = -(-t.shape[0] // align) * align
        seg = torch.zeros(padded, k, device="cuda", dtype=torch.bfloat16)
        seg[: t.shape[0]] = t
        segments.append(seg)
        layout += [g] * t.shape[0] + [-1] * (padded - t.shape[0])
        starts.append(row)
        row += padded
    x = torch.cat(segments) if segments else torch.zeros(0, k, device="cuda")
    return x, torch.tensor(layout, device="cuda", dtype=torch.int32), starts


def _grouped_contiguous(tokens, b, n: int, k: int) -> list[torch.Tensor]:
    x, layout, starts = _contiguous(tokens, k)
    d = torch.empty(x.shape[0], n, device="cuda", dtype=torch.bfloat16)
    _dg().m_grouped_fp8_gemm_nt_contiguous(_cast_a(x), b, d, layout, disable_ue8m0_cast=True)
    return [d[s : s + t.shape[0]] for s, t in zip(starts, tokens)]


def _grouped_masked(tokens, b, n: int, k: int, max_m: int, expected_m: int):
    groups = len(tokens)
    x = torch.zeros(groups, max_m, k, device="cuda", dtype=torch.bfloat16)
    for g, t in enumerate(tokens):
        x[g, : t.shape[0]] = t
    q, s = _cast_a(x.view(groups * max_m, k))
    a = (q.view(groups, max_m, k), s.view(groups, max_m, -1))
    masked_m = torch.tensor([t.shape[0] for t in tokens], device="cuda", dtype=torch.int32)
    d = torch.empty(groups, max_m, n, device="cuda", dtype=torch.bfloat16)
    _dg().m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m, disable_ue8m0_cast=True)
    return [d[g, : t.shape[0]] for g, t in enumerate(tokens)]


def _compare_groups(got: list[torch.Tensor], want: list[torch.Tensor]) -> dict:
    parts = [_compare(g, w[: g.shape[0]]) for g, w in zip(got, want) if g.shape[0]]
    if all(p["bitwise_equal"] for p in parts):
        return {"bitwise_equal": True}
    return {
        "bitwise_equal": False,
        "rows_differing": sum(p.get("rows_differing", 0) for p in parts),
        "max_abs": max(p.get("max_abs", 0.0) for p in parts),
    }


def probe_grouped(seed: int) -> dict:
    name, groups, n, k = GROUPED
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(groups, n, k, generator=gen).to("cuda", torch.bfloat16)
    wq = [_cast_b(w[g]) for g in range(groups)]
    b = (torch.stack([q for q, _ in wq]), torch.stack([s for _, s in wq]))

    counts = torch.randint(0, 300, (groups,), generator=gen).tolist()
    counts[0] = max(counts[0], 1)
    pool = torch.randn(sum(counts) + 512 * groups, k, generator=gen).to("cuda", torch.bfloat16)
    tokens, extra, cursor = [], [], 0
    for c in counts:
        tokens.append(pool[cursor : cursor + c])
        cursor += c
    for _ in range(groups):
        extra.append(pool[cursor : cursor + 200])
        cursor += 200

    base = _grouped_contiguous(tokens, b, n, k)
    grown = _grouped_contiguous([torch.cat([t, e]) for t, e in zip(tokens, extra)], b, n, k)
    only_first = [tokens[0][:1]] + [t[:0] for t in tokens[1:]]
    single = _grouped_contiguous(only_first, b, n, k)
    max_m = max(c + 200 for c in counts)
    checks = {
        "contiguous_run_to_run": _compare_groups(_grouped_contiguous(tokens, b, n, k), base),
        "contiguous_other_groups_grow": _compare_groups(base, grown),
        "contiguous_single_token": _compare_groups(single, base),
    }
    for expected_m in (1, max(1, sum(counts) // groups), max_m):
        masked = _grouped_masked(tokens, b, n, k, max_m, expected_m)
        checks[f"masked_vs_contiguous_expected_m{expected_m}"] = _compare_groups(masked, base)
    dense = [_gemm(_cast_a(t), wq[g], n) if t.shape[0] else t[:, :0] for g, t in enumerate(tokens)]
    info = {"grouped_vs_dense_per_group": _compare_groups(dense, base)}
    return {
        "kind": "grouped",
        "name": name,
        "groups": groups,
        "n": n,
        "k": k,
        "seed": seed,
        "counts": counts,
        "passed": all(c["bitwise_equal"] for c in checks.values()),
        "checks": checks,
        "info": info,
    }


def _graph_ms(fn, warmup: int, iters: int, reps: int = 20) -> float:
    """Median GPU time per call, replaying ``reps`` calls from one CUDA graph.

    Graph replay removes host launch overhead, which otherwise dominates the
    small-M cases and is not a property of either kernel."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / reps)
    return statistics.median(samples)


def perf_dense(name: str, n: int, k: int, warmup: int, iters: int) -> list[dict]:
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(max(PERF_M), k, generator=gen).to("cuda", torch.bfloat16)
    w = torch.randn(n, k, generator=gen).to("cuda", torch.bfloat16)
    b = _cast_b(w)
    rows = []
    for m in PERF_M:
        xm = x[:m].contiguous()
        a = _cast_a(xm)
        # The activation-quant kernel can emit scales in DeepGEMM's TMA-aligned
        # layout; time that case and the per-call layout conversion separately.
        a_laid_out = (a[0], _dg().get_mn_major_tma_aligned_tensor(a[1]))
        d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        dg_ms = _graph_ms(
            lambda: _dg().fp8_gemm_nt(a_laid_out, b, d, disable_ue8m0_cast=True), warmup, iters
        )
        dg_raw_ms = _graph_ms(
            lambda: _dg().fp8_gemm_nt(a, b, d, disable_ue8m0_cast=True), warmup, iters
        )
        bf16_ms = _graph_ms(lambda: torch.matmul(xm, w.t(), out=out), warmup, iters)
        rows.append(
            {
                "name": name,
                "m": m,
                "n": n,
                "k": k,
                "timing": "CUDA graph replay, median per call",
                "deepgemm_fp8_ms": dg_ms,
                "deepgemm_fp8_raw_sf_ms": dg_raw_ms,
                "torch_bf16_ms": bf16_ms,
                "deepgemm_over_bf16": dg_ms / bf16_ms,
                "deepgemm_tflops": 2 * m * n * k / dg_ms / 1e9,
            }
        )
    return rows


def perf_wgrad(name: str, n: int, k: int, warmup: int, iters: int, tokens: int = 4096) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(0)
    dy = torch.randn(tokens, n, generator=gen).to("cuda", torch.bfloat16)
    x = torch.randn(tokens, k, generator=gen).to("cuda", torch.bfloat16)
    a, b = _cast_tokens(dy), _cast_tokens(x)
    a = (a[0], _dg().get_mn_major_tma_aligned_tensor(a[1]))
    b = (b[0], _dg().get_mn_major_tma_aligned_tensor(b[1]))
    d = torch.zeros(n, k, device="cuda", dtype=torch.float)
    out = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    dyt = dy.t()
    dg_ms = _graph_ms(
        lambda: _dg().fp8_gemm_nt(a, b, d, c=d, recipe=(1, 1, 128), disable_ue8m0_cast=True),
        warmup,
        iters,
    )
    bf16_ms = _graph_ms(lambda: torch.matmul(dyt, x, out=out), warmup, iters)
    return {
        "name": name,
        "tokens": tokens,
        "n": n,
        "k": k,
        "timing": "CUDA graph replay, median per call",
        "deepgemm_fp8_ms": dg_ms,
        "torch_bf16_ms": bf16_ms,
        "deepgemm_over_bf16": dg_ms / bf16_ms,
    }


def environment() -> dict:
    dg, capability = require_deepgemm()
    return {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "deep_gemm": capability.version,
        "deep_gemm_pinned": capability.pinned,
        "deep_gemm_commit": DEEPGEMM_COMMIT if capability.pinned else "unpinned",
        "jit_nvcc": capability.nvcc,
        "jit_nvcc_version": ".".join(map(str, capability.nvcc_version)),
        "num_sms": dg.get_num_sms(),
        "mk_alignment": dg.get_mk_alignment_for_contiguous_layout(),
        "compiled_dims": "nk (default)",
        "scales": "FP32, UE8M0-valued; activations 1x128, weights 128x128",
        "python": platform.python_version(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-perf", action="store_true")
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--output", default="deepgemm_bi_report.json")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        print("an SM90 GPU is required")
        return 2

    try:
        env = environment()
    except DeepGemmUnavailable as exc:
        print(f"DeepGEMM is not usable: {exc}")
        return 2
    report = {
        "environment": env,
        "control": [],
        "cases": [],
        "precision": [],
        "perf": [],
        "perf_wgrad": [],
    }
    print(json.dumps(report["environment"], indent=2))
    torch.backends.cuda.matmul.allow_tf32 = False
    for name, n, k in DENSE_SHAPES:
        control = negative_control(name, n, k)
        report["control"].append(control)
        print(f"control {name:<15} BF16 cuBLAS varies with M: {control['detected_variance']}")
    if not any(c["detected_variance"] for c in report["control"]):
        print("warning: the negative control found no variance; checks may be ineffective")
    for seed in range(args.seeds):
        for name, n, k in DENSE_SHAPES:
            case = probe_dense(name, n, k, seed)
            report["cases"].append(case)
            failed = [c for c, v in case["checks"].items() if not v["bitwise_equal"]]
            print(
                f"dense   {name:<15} N={n:<6} K={k:<5} seed={seed}: "
                f"{'PASS' if case['passed'] else 'FAIL ' + ','.join(failed)}"
            )
        case = probe_grouped(seed)
        report["cases"].append(case)
        failed = [c for c, v in case["checks"].items() if not v["bitwise_equal"]]
        print(
            f"grouped {case['name']:<15} counts={case['counts']} seed={seed}: "
            f"{'PASS' if case['passed'] else 'FAIL ' + ','.join(failed)}; "
            f"vs dense: {case['info']['grouped_vs_dense_per_group']['bitwise_equal']}"
        )

    if not args.skip_backward:
        for seed in range(args.seeds):
            for name, n, k in DENSE_SHAPES:
                cases = [
                    probe_transpose(name, n, k, seed),
                    probe_dense(f"{name}_dgrad", k, n, seed, kind="dgrad"),
                    probe_wgrad(name, n, k, seed),
                ]
                report["cases"] += cases
                status = []
                for case in cases:
                    failed = [
                        c for c, v in case.get("checks", {}).items() if not v["bitwise_equal"]
                    ]
                    verdict = "PASS" if case["passed"] else "FAIL " + ",".join(failed)
                    status.append(f"{case['kind']} {verdict}")
                print(f"bwd     {name:<15} seed={seed}: " + "; ".join(status))
        for name, n, k in DENSE_SHAPES:
            row = precision(name, n, k)
            report["precision"].append(row)
            print(
                f"prec    {name:<15} rel err FP8 fwd/dgrad/wgrad "
                f"{row['fp8_forward']:.4f}/{row['fp8_dgrad']:.4f}/{row['fp8_wgrad']:.4f}"
                f"  BF16 {row['bf16_forward']:.4f}/{row['bf16_dgrad']:.4f}/{row['bf16_wgrad']:.4f}"
            )
        wgrad_info = {
            c["info"][f"tokens{WGRAD_UNALIGNED}"] for c in report["cases"] if c["kind"] == "wgrad"
        }
        print(f"wgrad with {WGRAD_UNALIGNED} tokens: {sorted(wgrad_info)}")

    if not args.skip_perf:
        perf_shapes = list(DENSE_SHAPES)
        if not args.skip_backward:
            perf_shapes += [(f"{name}_dgrad", k, n) for name, n, k in DENSE_SHAPES]
        for name, n, k in perf_shapes:
            for row in perf_dense(name, n, k, args.warmup, args.iters):
                report["perf"].append(row)
                print(
                    f"perf    {name:<15} M={row['m']:<5}"
                    f" DeepGEMM {row['deepgemm_fp8_ms'] * 1e3:7.1f} us"
                    f" (raw sf {row['deepgemm_fp8_raw_sf_ms'] * 1e3:7.1f} us)"
                    f"  BF16 {row['torch_bf16_ms'] * 1e3:7.1f} us"
                    f"  ratio {row['deepgemm_over_bf16']:.2f}"
                    f"  {row['deepgemm_tflops']:.0f} TFLOPS"
                )

        if not args.skip_backward:
            for name, n, k in DENSE_SHAPES:
                row = perf_wgrad(name, n, k, args.warmup, args.iters)
                report["perf_wgrad"].append(row)
                print(
                    f"perf    {name + '_wgrad':<21} tokens={row['tokens']}"
                    f" DeepGEMM {row['deepgemm_fp8_ms'] * 1e3:7.1f} us"
                    f"  BF16 {row['torch_bf16_ms'] * 1e3:7.1f} us"
                    f"  ratio {row['deepgemm_over_bf16']:.2f}"
                )

    report["passed"] = all(c["passed"] for c in report["cases"])
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"batch-invariant: {report['passed']}  report: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
