# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Operator-only cuBLAS + Transformer Engine 2.19 training baseline.

Native probabilities retain the dense [T,E] mask ABI; candidate retains [T,K].
Conversion for correctness and cotangent alignment is OUTSIDE timed work.
No expert GEMM, combine, auxiliary loss, or distributed communication included.
"""

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.router import fused_topk_with_score_function

import rl_engine.kernels.ops.triton.nemotron_router as core

NATIVE_MAP_TYPE = "mask"


def routing_mask(route, n):
    if route.dtype == torch.bool:
        return route
    return torch.zeros((n, 128), device=route.device, dtype=torch.bool).scatter_(
        1, route.long(), True
    )


def native(x, w, bias, full=True):
    logits = torch.nn.functional.linear(x.float(), w)
    indices = (
        torch.empty((x.shape[0], 6), device=x.device, dtype=torch.int32)
        if NATIVE_MAP_TYPE == "index"
        else None
    )
    probs, mask = fused_topk_with_score_function(
        logits, 6, False, None, None, 2.5, "sigmoid", bias, topk_indices=indices
    )
    packed = (
        te.moe_permute(x, mask, x.shape[0] * 6, max_token_num=x.shape[0], map_type=NATIVE_MAP_TYPE)[
            0
        ]
        if full
        else None
    )
    return probs, mask, packed


def candidate(x, w, bias, full=True):
    return core.nemotron_router_cuda(x, w, bias) if full else core._ProjectRoute.apply(x, w, bias)


def stats_error(a, b, atol, rtol):
    diff = (a.double() - b.double()).abs()
    return {
        "allclose": bool(torch.allclose(a.double(), b.double(), atol=atol, rtol=rtol)),
        "max_abs": diff.max().item(),
        "relative_l2": (diff.norm() / b.double().norm().clamp_min(1e-30)).item(),
        "atol": atol,
        "rtol": rtol,
    }


def qualify(x, w, bias, dense_grad, packed_grad, full):
    a, b = native(x, w, bias, full), candidate(x, w, bias, full)
    ids = b[0].long()
    native_mask = routing_mask(a[1], x.shape[0])
    native_ids = native_mask.nonzero(as_tuple=True)[1].reshape(x.shape[0], 6)
    same = (ids == native_ids).all(1)
    report = {"same_expert_rows": int(same.sum()), "rows": x.shape[0]}
    ga = dense_grad * native_mask
    gb = dense_grad.gather(1, ids)
    if full:
        da = torch.autograd.grad((a[0], a[2]), (x, w), (ga, packed_grad))
        db = torch.autograd.grad((b[1], b[4]), (x, w), (gb, packed_grad))
    else:
        da = torch.autograd.grad(a[0], (x, w), ga)
        db = torch.autograd.grad(b[1], (x, w), gb)
    # Cross-provider differences are meaningful only for an identical branch.
    # Every provider is still qualified against FP64 on its OWN selected branch.
    report["gradients_compared"] = bool(same.all())
    if bool(same.all()):
        report["weights"] = stats_error(b[1], a[0].gather(1, ids), 2e-6, 2e-5)
        report["dx"] = stats_error(
            db[0],
            da[0],
            (0.0625 if full else 2e-4) if x.dtype == torch.bfloat16 else 2e-5,
            0.02 if x.dtype == torch.bfloat16 else 2e-4,
        )
        report["dw"] = stats_error(db[1], da[1], 2e-4, 3e-4)
    for name, selected, actual, actual_prob, payload in (
        ("native", native_ids, da, a[0].gather(1, native_ids), a[2]),
        ("candidate", ids, db, b[1], b[4] if full else None),
    ):
        xx = x.detach().double().requires_grad_()
        ww = w.detach().double().requires_grad_()
        scores = torch.nn.functional.linear(xx, ww).sigmoid().gather(1, selected)
        probs = scores / (scores.sum(1, keepdim=True) + 1e-20) * 2.5
        targets, cots = (probs,), (dense_grad.gather(1, selected).double(),)
        if full:
            perm = torch.argsort(selected.flatten(), stable=True)
            targets += (xx[perm // 6],)
            cots += (packed_grad.double(),)
            report[name + "_payload_equal"] = bool(torch.equal(payload, x[perm // 6]))
        ref = torch.autograd.grad(targets, (xx, ww), cots)
        report[name + "_fp64_weights"] = stats_error(actual_prob, probs, 2e-6, 2e-5)
        report[name + "_fp64_dx"] = stats_error(
            actual[0],
            ref[0],
            (0.0625 if full else 2e-4) if x.dtype == torch.bfloat16 else 2e-5,
            0.02 if x.dtype == torch.bfloat16 else 2e-4,
        )
        report[name + "_fp64_dw"] = stats_error(actual[1], ref[1], 2e-4, 3e-4)
    assert all(report["candidate_fp64_" + key]["allclose"] for key in ("weights", "dx", "dw"))
    if full:
        assert report["candidate_payload_equal"] and report["native_payload_equal"]
    return report


def capture(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    once = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del once
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return graph, outputs, peak


def paired_measure(calls, reverse, mode):
    graphs = {}
    if mode == "graph":
        graphs = {name: capture(fn) for name, fn in calls.items()}
    else:
        for name, fn in calls.items():
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            _once = fn()
            assert _once is not None
            torch.cuda.synchronize()
            graphs[name] = (None, None, torch.cuda.max_memory_allocated() - before)
            del _once
    samples = {name: [] for name in calls}
    order = list(calls)
    if reverse:
        order.reverse()
    for i in range(20):
        for name in order if i % 2 == 0 else order[::-1]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                if mode == "graph":
                    graphs[name][0].replay()
                else:
                    calls[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / 5)
    return {
        name: {
            "median_ms": statistics.median(v),
            "samples_ms": v,
            "peak_increment_bytes": graphs[name][2],
        }
        for name, v in samples.items()
    }


def main():
    global NATIVE_MAP_TYPE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 16, 128, 1024, 4096, 8192, 32768]
    )
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--distributions", nargs="+", default=["random", "concentrated"])
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--map-type", choices=["mask", "index"], default="mask")
    parser.add_argument("--mode", choices=["graph", "eager"], default="graph")
    args = parser.parse_args()
    NATIVE_MAP_TYPE = args.map_type
    if args.map_type == "index" and args.mode == "graph":
        parser.error("TE 2.19 index radix sort uses the default stream; select --mode eager")
    if args.output.exists():
        parser.error("refusing to overwrite results")
    if importlib.metadata.version("transformer_engine") != "2.19.0":
        parser.error("requires pinned Transformer Engine 2.19.0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(434)
    result = {
        "torch": torch.__version__,
        "te": "2.19.0",
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "20 alternating paired samples, 5 executions per sample; mode="
        + args.mode
        + "; native dense ABI; no canonicalization timed; eager includes host launch gaps",
        "scope": (
            "cuBLAS FP32 + TE fused sigmoid/top6 + TE "
            + args.map_type
            + " permutation and autograd; no expert/combine/aux loss/EP"
        ),
        "arguments": {**vars(args), "output": args.output.name},
        "rows": [],
    }
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    # TE 2.19 index-map radix sort uses the default CUDA stream and is not
    # safe to compare under a side-stream CUDA graph. Eager mode uses the
    # default stream for BOTH providers, preserving upstream implementation.
    execution_stream = torch.cuda.default_stream() if args.mode == "eager" else torch.cuda.Stream()
    with torch.cuda.stream(execution_stream):
        for distribution in args.distributions:
            for n in args.tokens:
                x = torch.randn(n, 2688, device="cuda", dtype=dtype).requires_grad_()
                w = (torch.randn(128, 2688, device="cuda") * 0.02).requires_grad_()
                bias = torch.randn(128, device="cuda") * 0.01
                if distribution == "concentrated":
                    bias.zero_()
                    bias[:6] = 4
                dense_grad = torch.randn(n, 128, device="cuda")
                packed_grad = torch.randn(n * 6, 2688, device="cuda", dtype=dtype)
                for full in (False, True):
                    qualification = qualify(x, w, bias, dense_grad, packed_grad, full)
                    out_a, out_b = native(x, w, bias, full), candidate(x, w, bias, full)
                    grads_a = (
                        (dense_grad * routing_mask(out_a[1], n), packed_grad)
                        if full
                        else (dense_grad * routing_mask(out_a[1], n),)
                    )
                    grads_b = (
                        (dense_grad.gather(1, out_b[0].long()), packed_grad)
                        if full
                        else (dense_grad.gather(1, out_b[0].long()),)
                    )
                    del out_a, out_b

                    def fa():
                        o = native(x, w, bias, full)
                        return (o[0], o[2]) if full else (o[0],)

                    def fb():
                        o = candidate(x, w, bias, full)
                        return (o[1], o[4]) if full else (o[1],)

                    for phase in ("forward", "forward_backward"):
                        calls = (
                            {"native": fa, "candidate": fb}
                            if phase == "forward"
                            else {
                                "native": lambda: torch.autograd.grad(fa(), (x, w), grads_a),
                                "candidate": lambda: torch.autograd.grad(fb(), (x, w), grads_b),
                            }
                        )
                        measured = paired_measure(calls, args.reverse, args.mode)
                        row = {
                            "tokens": n,
                            "dtype": args.dtype,
                            "distribution": distribution,
                            "scope": "full" if full else "route",
                            "phase": phase,
                            "qualification": qualification,
                            **measured,
                        }
                        row["speedup"] = (
                            measured["native"]["median_ms"] / measured["candidate"]["median_ms"]
                        )
                        result["rows"].append(row)
                        args.output.write_text(json.dumps(result, indent=2))
                        print(
                            n,
                            args.dtype,
                            distribution,
                            row["scope"],
                            phase,
                            "native",
                            round(measured["native"]["median_ms"], 6),
                            "candidate",
                            round(measured["candidate"]["median_ms"], 6),
                            "speedup",
                            round(row["speedup"], 4),
                            flush=True,
                        )
    print("TRAINING_BASELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
