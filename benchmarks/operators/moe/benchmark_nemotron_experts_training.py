# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Nemotron grouped experts + combine: training-path timing and FP64 qualification.

Baseline: the same math expert by expert with `torch.matmul` in FP32 and a
weighted, slot-ascending combine - the obvious eager formulation. Its expert span
list is read once outside the timed region.

Candidates: the provider objects themselves (`NemotronExpertsOp`,
`TritonNemotronExpertsOp`), so the public validation wrapper is excluded from
both sides of the comparison. Timing uses CUDA Graphs by default so host launch
gaps do not dominate; `--mode eager` measures the eager path including them.
Cotangents are prepared outside timing.

The FP64 qualification rebuilds the pipeline in FP64 and reports the relative L2
error of the forward and every gradient. It is memory bound, so
`--qualify-max-tokens` bounds which rows carry a verdict; larger rows record why
they carry none.

Transformer Engine is not used: the pinned `transformer_engine==2.19.0` grouped
GEMM is not available in this environment, and the documented baseline is the
one that actually ran.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

import rl_engine.ops.moe.nemotron_experts as core
from rl_engine.reference.moe.nemotron_experts import routed_experts, shared_experts
from rl_engine.reference.moe.nemotron_router import dispatch_tokens, route_scores

HIDDEN = 2688
INTERMEDIATE = 1856
EXPERTS = 128
SHARED = 3712
SLOTS = 6
GRADIENT_NAMES = ("packed", "x", "weights", "w_up", "w_down", "w_sh1", "w_sh2")


def stats_error(actual, expected):
    difference = (actual.double() - expected.double()).abs()
    return {
        "relative_l2": (difference.norm() / expected.double().norm().clamp_min(1e-30)).item(),
        "max_abs": difference.max().item(),
    }


def build(tokens, distribution, seed=434):
    """Router output, payload, stacked weights and a fixed cotangent."""
    generator = torch.Generator().manual_seed(seed)
    payload = torch.randn(tokens, HIDDEN, generator=generator) * 0.5
    scores = torch.rand(tokens, EXPERTS, generator=generator)
    bias = torch.zeros(EXPERTS)
    if distribution == "concentrated":
        bias[:SLOTS] = 4.0
    ids, route_weights = route_scores(scores, bias, SLOTS)
    permutation, offsets, packed = dispatch_tokens(payload, ids, EXPERTS)
    device = "cuda"
    case = {
        "packed": packed.to(device=device, dtype=torch.bfloat16),
        "x": payload.to(device=device, dtype=torch.bfloat16),
        "weights": route_weights.to(device=device),
        "permutation": permutation.to(device=device),
        "offsets": offsets.to(device=device),
        "w_up": (torch.randn(EXPERTS, INTERMEDIATE, HIDDEN, generator=generator) * 0.05).to(
            device=device, dtype=torch.bfloat16
        ),
        "w_down": (torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, generator=generator) * 0.02).to(
            device=device, dtype=torch.bfloat16
        ),
        "w_sh1": (torch.randn(SHARED, HIDDEN, generator=generator) * 0.05).to(
            device=device, dtype=torch.bfloat16
        ),
        "w_sh2": (torch.randn(HIDDEN, SHARED, generator=generator) * 0.02).to(
            device=device, dtype=torch.bfloat16
        ),
        "ids": ids,
    }
    counts = torch.bincount(ids.reshape(-1), minlength=EXPERTS)
    active = int((counts > 0).sum())
    steps = torch.Generator().manual_seed(seed + 1)
    meta = {
        "spans": offsets.tolist(),
        # Only experts with rows are ever read: two BF16 weight matrices each.
        "weights_bytes": active * (2 * INTERMEDIATE * HIDDEN) * 2,
        "n_e": {
            "min": int(counts.min()),
            "max": int(counts.max()),
            "empty_experts": int((counts == 0).sum()),
            "active_experts": active,
        },
        "cotangent": (torch.randn(tokens, HIDDEN, generator=steps) * 0.01).to(
            device=device, dtype=torch.bfloat16
        ),
    }
    return case, meta


def leaves_of(case):
    return {name: case[name].detach().clone().requires_grad_(True) for name in GRADIENT_NAMES}


def baseline_calls(case, spans, leaves, cotangent):
    """Per-expert `torch.matmul` in FP32 plus a slot-ascending weighted combine."""
    packed = case["packed"]
    weights = case["weights"]
    tokens = case["x"].shape[0]
    rows, width = packed.shape
    packed_rows = case["permutation"].argsort().reshape(tokens, SLOTS)

    def body(source, source_x, source_weights, up, down, sh1, sh2):
        wide = torch.empty((rows, width), device=packed.device, dtype=torch.float32)
        for expert in range(EXPERTS):
            lo, hi = spans[expert], spans[expert + 1]
            if lo == hi:
                continue
            gate = torch.relu(source[lo:hi].float() @ up[expert].float().t()).square()
            wide[lo:hi] = gate @ down[expert].float().t()
        routed = torch.zeros((tokens, width), device=packed.device, dtype=torch.float32)
        for slot in range(SLOTS):
            row = wide[packed_rows[:, slot]]
            routed = routed + row * source_weights[:, slot : slot + 1]
        shared = torch.relu(source_x.float() @ sh1.float().t()).square() @ sh2.float().t()
        return (routed + shared).to(torch.bfloat16)

    def forward():
        return body(
            packed, case["x"], weights, case["w_up"], case["w_down"], case["w_sh1"], case["w_sh2"]
        )

    def backward():
        return torch.autograd.grad(
            body(
                leaves["packed"],
                leaves["x"],
                leaves["weights"],
                leaves["w_up"],
                leaves["w_down"],
                leaves["w_sh1"],
                leaves["w_sh2"],
            ),
            [leaves[name] for name in GRADIENT_NAMES],
            cotangent,
        )

    return forward, backward


def provider_calls(op, case, leaves, cotangent):
    def forward():
        return op(
            case["packed"],
            case["x"],
            case["weights"],
            case["permutation"],
            case["offsets"],
            case["w_up"],
            case["w_down"],
            case["w_sh1"],
            case["w_sh2"],
        )

    def backward():
        return torch.autograd.grad(
            op(
                leaves["packed"],
                leaves["x"],
                leaves["weights"],
                case["permutation"],
                case["offsets"],
                leaves["w_up"],
                leaves["w_down"],
                leaves["w_sh1"],
                leaves["w_sh2"],
            ),
            [leaves[name] for name in GRADIENT_NAMES],
            cotangent,
        )

    return forward, backward


def qualify(op, case, meta, token_limit):
    """FP64 comparison of the forward and every gradient on this provider."""
    tokens = case["x"].shape[0]
    if tokens > token_limit:
        return {"skipped": f"above --qualify-max-tokens={token_limit}"}
    leaves = leaves_of(case)
    cotangent = meta["cotangent"]
    args = (case["permutation"], case["offsets"])
    actual = op(
        leaves["packed"],
        leaves["x"],
        leaves["weights"],
        *args,
        leaves["w_up"],
        leaves["w_down"],
        leaves["w_sh1"],
        leaves["w_sh2"],
    )
    actual_grads = torch.autograd.grad(actual, [leaves[name] for name in GRADIENT_NAMES], cotangent)
    exact = {name: case[name].detach().double().requires_grad_() for name in GRADIENT_NAMES}
    wide = routed_experts(exact["packed"], exact["weights"], *args, exact["w_up"], exact["w_down"])
    wide = wide + shared_experts(exact["x"], exact["w_sh1"], exact["w_sh2"])
    exact_grads = torch.autograd.grad(
        wide, [exact[name] for name in GRADIENT_NAMES], cotangent.double()
    )
    report = {"forward": stats_error(actual, wide)}
    for position, name in enumerate(GRADIENT_NAMES):
        report[name] = stats_error(actual_grads[position], exact_grads[position])
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
    measured = {}
    if mode == "graph":
        measured = {name: capture(fn) for name, fn in calls.items()}
    else:
        for name, fn in calls.items():
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            once = fn()
            assert once is not None
            torch.cuda.synchronize()
            measured[name] = (None, None, torch.cuda.max_memory_allocated() - before)
            del once
    samples = {name: [] for name in calls}
    order = list(calls)
    if reverse:
        order.reverse()
    for index in range(20):
        for name in order if index % 2 == 0 else order[::-1]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                if mode == "graph":
                    measured[name][0].replay()
                else:
                    calls[name]()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / 5)
    return {
        name: {
            "median_ms": statistics.median(values),
            "samples_ms": values,
            "peak_increment_bytes": measured[name][2],
        }
        for name, values in samples.items()
    }


def provider_op(name):
    if name == "native":
        from rl_engine.backends.cuda.moe.nemotron_experts import NemotronExpertsOp

        return NemotronExpertsOp()
    from rl_engine.backends.shared.triton.moe.nemotron_experts import TritonNemotronExpertsOp

    return TritonNemotronExpertsOp()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 16, 128, 513, 1024, 4096, 8192, 32768]
    )
    parser.add_argument("--dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--distributions", nargs="+", default=["random", "concentrated"])
    parser.add_argument("--providers", nargs="+", default=["native", "triton"])
    parser.add_argument("--mode", choices=["graph", "eager"], default="graph")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--qualify-max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite results")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(434)

    providers = {}
    for name in args.providers:
        try:
            providers[name] = provider_op(name)
        except RuntimeError as error:
            print(f"provider {name} unavailable: {error}", flush=True)
    if not providers:
        parser.error("no provider available")

    result = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provider_ids": {name: op.provider_id for name, op in providers.items()},
        "method": (
            "20 alternating paired samples, 5 executions per sample, median reported; "
            f"mode={args.mode}; baseline is per-expert FP32 torch.matmul plus a weighted "
            "slot-ascending combine; cotangents prepared outside timing; the public "
            "validation wrapper is outside both sides"
        ),
        "scope": (
            "grouped up/relu2/down + fixed-order combine + shared expert, forward and "
            "forward+backward; no optimizer, no auxiliary loss, no communication"
        ),
        "baseline_notes": (
            "Transformer Engine 2.19 grouped GEMM is not installed, so the baseline is the "
            "eager expert-by-expert formulation that actually ran"
        ),
        "arguments": {**vars(args), "output": args.output.name},
        "rows": [],
    }

    stream = torch.cuda.default_stream() if args.mode == "eager" else torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for distribution in args.distributions:
            for tokens in args.tokens:
                case, meta = build(tokens, distribution)
                cotangent = meta["cotangent"]
                leaves = leaves_of(case)
                calls = {}
                base_forward, base_backward = baseline_calls(case, meta["spans"], leaves, cotangent)
                calls["baseline"] = (base_forward, base_backward)
                for name, op in providers.items():
                    calls[name] = provider_calls(op, case, leaves, cotangent)
                qualification = {
                    name: qualify(op, case, meta, args.qualify_max_tokens)
                    for name, op in providers.items()
                }
                diagnostics = {
                    "weights_bytes": meta["weights_bytes"],
                    "n_e": meta["n_e"],
                    "qualification": qualification,
                }
                for phase, index in (("forward", 0), ("forward_backward", 1)):
                    timed = {name: closures[index] for name, closures in calls.items()}
                    measured = paired_measure(timed, args.reverse, args.mode)
                    row = {
                        "tokens": tokens,
                        "dtype": args.dtype,
                        "distribution": distribution,
                        "phase": phase,
                        "reverse": args.reverse,
                        "providers": measured,
                        "diagnostics": diagnostics,
                    }
                    for name in providers:
                        row[f"speedup_{name}_vs_baseline"] = (
                            measured["baseline"]["median_ms"] / measured[name]["median_ms"]
                        )
                    result["rows"].append(row)
                    args.output.write_text(json.dumps(result, indent=2))
                    summary = " ".join(
                        f"{name}={measured[name]['median_ms']:.4f}" for name in measured
                    )
                    print(tokens, distribution, phase, summary, flush=True)
                del calls, leaves, case, meta
                torch.cuda.empty_cache()
    print("TRAINING_BASELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
