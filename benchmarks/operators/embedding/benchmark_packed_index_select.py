# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Eager index_select/autograd vs ordered Triton, with all-element accuracy gates.

Run from the checkout with PYTHONPATH=.; no extension build is required.
JSON includes every sample, source hashes, environment and failing comparisons.
Native PyTorch and its deterministic-mode variant are performance baselines;
the CPU FP64 reference is an accuracy oracle and is never timed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.ops.autograd.backward_runtime import snapshot_backward_runtime
from rl_engine.reference.minimax_h3.packed_index_select import NativePackedIndexSelectOp

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = (
    "rl_engine/reference/minimax_h3/packed_index_select.py",
    "rl_engine/backends/shared/triton/embedding/packed_index_select.py",
    "benchmarks/operators/embedding/benchmark_packed_index_select.py",
    "rl_engine/contracts/profiles/precision/ws1.json",
)


def make_inputs(batch, sequence, width, duplicate_fraction, dtype, device, seed):
    """Select about half the mixed rows; replace a declared fraction with repeats.

    Selection counts and repeat fractions are synthetic operator workloads, not
    complete FL2VA/Ref2VA layouts. Sequence counts and widths come from H3.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = max(1, sequence // 2)
    unique = max(1, round(count * (1 - duplicate_fraction)))
    destinations = torch.randperm(sequence, generator=generator)[:unique]
    index = destinations[torch.arange(count) % unique]
    index = index[torch.randperm(count, generator=generator)]
    x = torch.randn(batch, sequence, width, generator=generator).to(dtype=dtype, device=device)
    grad = torch.randn(batch, count, width, generator=generator).to(dtype=dtype, device=device)
    return x, index.to(device), grad


def run_vjp(op, x, index, grad):
    leaf = x.detach().requires_grad_(True)
    output = op(leaf, index)
    return output.detach(), torch.autograd.grad(output, leaf, grad)[0]


def bit_mismatches(actual, expected):
    integer = torch.int32 if actual.element_size() == 4 else torch.int16
    return int((actual.contiguous().view(integer) != expected.contiguous().view(integer)).sum())


def check_result(op, x, index, grad, gold, tolerance):
    output, dx = run_vjp(op, x, index, grad)
    repeated_y, repeated_dx = run_vjp(op, x, index, grad)
    error = (dx.float() - gold).abs()
    bad = ~torch.isfinite(error) | (error > tolerance.atol + tolerance.rtol * gold.abs())
    return {
        "forward_bit_mismatches": bit_mismatches(output, x.index_select(1, index)),
        "gradient_accuracy_failures": int(bad.sum()),
        "gradient_max_abs_error": float(error.max()),
        "repeat_forward_bit_mismatches": bit_mismatches(output, repeated_y),
        "repeat_gradient_bit_mismatches": bit_mismatches(dx, repeated_dx),
        "gradient_atol": tolerance.atol,
        "gradient_rtol": tolerance.rtol,
    }


def measure(op, x, index, grad, warmup, samples):
    """Include allocations, validation, wrapping and sorting in the owning phase.

    Forward runs with autograd enabled. Backward reuses a retained forward graph;
    F+B constructs a new graph each time. Input creation, gold, warmups and JIT
    compilation are excluded. No .grad buffers accumulate between repetitions.
    """
    leaf = x.detach().requires_grad_(True)
    output = op(leaf, index)

    def forward():
        return op(leaf, index)

    def backward():
        return torch.autograd.grad(output, leaf, grad, retain_graph=True)[0]

    def combined():
        return torch.autograd.grad(op(leaf, index), leaf, grad)[0]

    timings = {}
    for phase, fn in (("forward", forward), ("backward", backward), ("forward_backward", combined)):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        event_ms, wall_ms = [], []
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(samples):
            start = time.perf_counter()
            begin.record()
            fn()
            end.record()
            end.synchronize()
            wall_ms.append((time.perf_counter() - start) * 1000)
            event_ms.append(begin.elapsed_time(end))
        timings[phase] = {
            "event_ms": event_ms,
            "wall_ms": wall_ms,
            "median_event_ms": statistics.median(event_ms),
            "median_wall_ms": statistics.median(wall_ms),
        }
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    combined()
    torch.cuda.synchronize()
    timings["incremental_peak_bytes"] = torch.cuda.max_memory_allocated() - allocated
    return timings


def command_output(*args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def environment():
    import triton

    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "sm": f"{props.major}{props.minor}",
        "total_memory_bytes": props.total_memory,
        "driver": command_output(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ),
        "commit": command_output("git", "rev-parse", "HEAD"),
        "git_status": command_output("git", "status", "--porcelain"),
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[129, 4097, 38734, 110750, 367986])
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 96])
    parser.add_argument("--duplicates", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows + args.widths + [args.batch, args.samples]) <= 0 or args.warmup < 1:
        parser.error("shapes, samples and warmup must be positive")
    if any(fraction < 0 or fraction >= 1 for fraction in args.duplicates):
        parser.error("duplicate fractions must be in [0, 1)")
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("benchmark requires NVIDIA CUDA")
    from rl_engine.backends.shared.triton.embedding.packed_index_select import (
        TritonPackedIndexSelectOp,
    )

    dtype = getattr(torch, args.dtype)
    tolerance = resolve_tolerance(
        load_contract(), judgment="gradient_accuracy", op_class="reduction", dtype=dtype
    )
    implementations = (
        ("torch", lambda x, index: x.index_select(1, index), False),
        ("torch_deterministic", lambda x, index: x.index_select(1, index), True),
        ("triton", TritonPackedIndexSelectOp(), False),
    )
    report = {
        "environment": environment(),
        "arguments": {**vars(args), "output": str(args.output)},
        "method": {
            "mode": "eager, autograd-enabled forward; no CUDA Graph replay",
            "included": "validation, allocations, wrappers, backward stable sort and zero fill",
            "excluded": "inputs, reference, JIT compilation, warmups",
            "backward": "prebuilt retained graph; sorting repeated on each backward",
            "wall_ms": "host dispatch plus execution and end synchronization",
            "event_ms": "CUDA events around the complete call, including device idle dispatch gaps",
            "reference": "CPU FP64 index_add accuracy oracle, not timed",
            "comparison": "all elements; shared WS1 gradient_accuracy/reduction thresholds",
            "selection": "synthetic half-row selection at H3 mixed-row counts and head widths",
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        for sequence in args.rows:
            for width in args.widths:
                for fraction in args.duplicates:
                    x, index, grad = make_inputs(
                        args.batch, sequence, width, fraction, dtype, "cuda", args.seed
                    )
                    gold = run_vjp(
                        NativePackedIndexSelectOp().forward_fp32, x.float(), index, grad.float()
                    )[1]
                    for name, op, deterministic in implementations:
                        torch.use_deterministic_algorithms(deterministic)
                        result = {
                            "implementation": name,
                            "deterministic_algorithms": deterministic,
                            "shape": list(x.shape),
                            "selected_rows": index.numel(),
                            "duplicate_fraction_requested": fraction,
                            "duplicate_fraction_actual": 1 - index.unique().numel() / index.numel(),
                            "checks": check_result(op, x, index, grad, gold, tolerance),
                            "timings": measure(op, x, index, grad, args.warmup, args.samples),
                        }
                        report["results"].append(result)
                        report["backward_runtime"] = snapshot_backward_runtime()
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(
                            json.dumps({k: v for k, v in result.items() if k != "timings"}),
                            flush=True,
                        )
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
    failed = any(
        result["implementation"] == "triton"
        and any(
            result["checks"][key]
            for key in (
                "forward_bit_mismatches",
                "gradient_accuracy_failures",
                "repeat_forward_bit_mismatches",
                "repeat_gradient_bit_mismatches",
            )
        )
        for result in report["results"]
    )
    if failed:
        raise RuntimeError(f"candidate checks failed; complete evidence retained in {args.output}")


if __name__ == "__main__":
    main()
