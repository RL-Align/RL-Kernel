# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Time eager PyTorch and Triton index selection on the same unique indices.

Run from the checkout with PYTHONPATH=.; no extension build is required.
JSON includes every sample, source hashes, environment and failing comparisons.
Native PyTorch and its deterministic-mode variant are performance baselines.
A CPU gather checks the copied bits and a CPU FP64 scatter-add checks gradients.
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
from functools import partial
from math import prod
from pathlib import Path

import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance

ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = (
    "rl_engine/contracts/operators/index_select.py",
    "rl_engine/reference/embedding/index_select.py",
    "rl_engine/backends/shared/triton/embedding/index_select.py",
    "benchmarks/operators/embedding/benchmark_index_select.py",
    "rl_engine/contracts/profiles/precision/ws1.json",
)


def strided_copy(tensor):
    """Place the same values in every other storage element on the target device."""
    view = torch.empty((*tensor.shape, 2), dtype=tensor.dtype, device=tensor.device)[..., 0]
    return view.copy_(tensor)


def make_inputs(shape, dim, dtype, device, seed, *, x_contiguous=True, grad_y_contiguous=True):
    """Keep index and logical values identical when changing x or grad_y layout."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    axis_size = shape[dim]
    count = max(1, axis_size // 2)
    index = torch.randperm(axis_size, generator=generator)[:count]
    output_shape = list(shape)
    output_shape[dim] = count
    x = torch.randn(shape, generator=generator).to(dtype=dtype, device=device)
    grad = torch.randn(output_shape, generator=generator).to(dtype=dtype, device=device)
    if not x_contiguous:
        x = strided_copy(x)
    if not grad_y_contiguous:
        grad = strided_copy(grad)
    return x, index.to(device), grad


def layout_cases(shape, dim):
    """Vary x for forward, grad_y for backward, and both for combined timing."""
    x_elements = prod(shape)
    selected = max(1, shape[dim] // 2)
    grad_elements = x_elements // shape[dim] * selected
    # PyTorch considers every one-element tensor contiguous, whatever its strides.
    x_layouts = (True, False) if x_elements > 1 else (True,)
    grad_layouts = (True, False) if grad_elements > 1 else (True,)
    for x_contiguous in x_layouts:
        for grad_y_contiguous in grad_layouts:
            phases = []
            if grad_y_contiguous:
                phases.append("forward")
            if x_contiguous:
                phases.append("backward")
            phases.append("forward_backward")
            yield {
                "shape": list(shape),
                "dim": dim,
                "unique_indices": True,
                "duplicate_fraction": 0.0,
                "x_contiguous": x_contiguous,
                "grad_y_contiguous": grad_y_contiguous,
                "phases": phases,
            }


def run_vjp(op, x, index, grad):
    """Run one forward and differentiate it with the supplied upstream gradient."""
    leaf = x.detach().requires_grad_(True)
    output = op(leaf, index)
    return output.detach(), torch.autograd.grad(output, leaf, grad)[0]


def reference_outputs(x, index, grad_y, *, dim):
    """Build forward and gradient references on CPU before timing any backend."""
    cpu_index = index.to(device="cpu", dtype=torch.int64)
    expected_y = x.cpu().index_select(dim, cpu_index)
    # Scatter from zero so the sign of a zero gradient follows the same rule as autograd.
    grad_x = torch.zeros(x.shape, dtype=torch.float64, device="cpu")
    grad_x.index_add_(dim, cpu_index, grad_y.to(device="cpu", dtype=torch.float64))
    return expected_y.to(x.device), grad_x.to(device=x.device, dtype=torch.float32)


def bit_mismatches(actual, expected):
    """Count different element bit patterns, including signed zeros and NaN payloads."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(
            f"expected shape {tuple(expected.shape)} and dtype {expected.dtype}, "
            f"got {tuple(actual.shape)} and {actual.dtype}"
        )
    integer = torch.int32 if actual.element_size() == 4 else torch.int16
    return int((actual.contiguous().view(integer) != expected.contiguous().view(integer)).sum())


def check_result(op, x, index, grad, reference, tolerance, *, check_gradient_bits=False):
    """Check copied bits, gradient accuracy, and repeatability on the same inputs."""
    expected_y, expected_dx = reference
    output, dx = run_vjp(op, x, index, grad)
    repeated_y, repeated_dx = run_vjp(op, x, index, grad)
    error = (dx.float() - expected_dx).abs()
    bad = ~torch.isfinite(error) | (error > tolerance.atol + tolerance.rtol * expected_dx.abs())
    checks = {
        "forward_bit_mismatches": bit_mismatches(output, expected_y),
        "gradient_accuracy_failures": int(bad.sum()),
        "gradient_max_abs_error": float(error.max()),
        "repeat_forward_bit_mismatches": bit_mismatches(output, repeated_y),
        "repeat_gradient_bit_mismatches": bit_mismatches(dx, repeated_dx),
        "gradient_atol": tolerance.atol,
        "gradient_rtol": tolerance.rtol,
    }
    if check_gradient_bits:
        # With one contribution per selected slice, the oracle also gives the exact result.
        checks["gradient_bit_mismatches"] = bit_mismatches(dx, expected_dx.to(x.dtype))
    return checks


def checks_failed(checks):
    """Fail on mismatch counts, while keeping error magnitudes for reporting."""
    return any(value for name, value in checks.items() if name.endswith(("mismatches", "failures")))


def measure(
    op, x, index, grad, warmup, samples, *, phases=("forward", "backward", "forward_backward")
):
    """Include allocations, validation and scatter in the owning phase.

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
        if phase not in phases:
            continue
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
    # Subtract the live inputs and retained graph to report only the extra bytes for F+B.
    combined()
    torch.cuda.synchronize()
    timings["incremental_peak_bytes"] = torch.cuda.max_memory_allocated() - allocated
    return timings


def command_output(*args):
    """Collect optional driver and checkout metadata even when a CLI is unavailable."""
    try:
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def environment():
    """Record the active device and hash the sources used by this benchmark."""
    import triton

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    is_rocm = torch.version.hip is not None
    driver_command = (
        ("rocm-smi", "--showdriverversion")
        if is_rocm
        else ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "gpu": props.name,
        "device_index": device_index,
        "sm": None if is_rocm else f"{props.major}{props.minor}",
        "gcn_arch": getattr(props, "gcnArchName", None),
        "total_memory_bytes": props.total_memory,
        "driver": command_output(*driver_command),
        "commit": command_output("git", "rev-parse", "HEAD"),
        "git_status": command_output("git", "status", "--porcelain"),
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS
        },
    }


def parse_shape(value):
    """Parse explicit dimensions such as 2x129x96 or 129x28x128."""
    try:
        shape = tuple(int(size) for size in value.split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must contain integers separated by x") from exc
    if not shape or min(shape) < 1:
        raise argparse.ArgumentTypeError("benchmark dimensions must be positive")
    return shape


def parse_args(argv=None):
    """Validate each requested shape and its selection axis before allocating tensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes", type=parse_shape, nargs="+", default=[(1, 129, 32), (1, 4097, 96)]
    )
    parser.add_argument(
        "--dim", type=int, required=True, help="Shared selection axis; negative allowed."
    )
    parser.add_argument(
        "--duplicates",
        type=float,
        choices=[0.0],
        default=0.0,
        help="This benchmark supports unique indices only (zero duplicate fraction).",
    )
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--output", type=Path, default=Path("reports/index-select/results.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print the case plan without a GPU.")
    args = parser.parse_args(argv)
    if args.samples < 1 or args.warmup < 1:
        parser.error("samples and warmup must be positive")
    if any(not -len(shape) <= args.dim < len(shape) for shape in args.shapes):
        parser.error("dim must be a valid axis for every shape")
    return args


def main():
    """Validate each baseline and candidate before recording time and peak memory."""
    args = parse_args()
    case_plan = [case for shape in args.shapes for case in layout_cases(shape, args.dim)]
    if args.dry_run:
        measurement_count = 3 * sum(len(case["phases"]) for case in case_plan)
        print(json.dumps({"cases": case_plan, "measurement_count": measurement_count}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark timing requires a CUDA or ROCm device")
    from rl_engine.backends.shared.triton.embedding.index_select import TritonIndexSelectOp

    dtype = getattr(torch, args.dtype)
    tolerance = resolve_tolerance(
        load_contract(), judgment="gradient_accuracy", op_class="reduction", dtype=dtype
    )
    implementations = (
        ("torch", lambda x, index: x.index_select(args.dim, index), False),
        ("torch_deterministic", lambda x, index: x.index_select(args.dim, index), True),
        ("triton", partial(TritonIndexSelectOp(), dim=args.dim, unique_indices=True), False),
    )
    report = {
        "environment": environment(),
        "arguments": {**vars(args), "output": str(args.output)},
        "method": {
            "mode": "eager, autograd-enabled forward; no CUDA Graph replay",
            "included": "validation, allocations, wrappers, backward zero fill and scatter",
            "excluded": "inputs, reference, JIT compilation, warmups",
            "backward": "prebuilt retained graph, direct scatter to unique rows",
            "wall_ms": "host dispatch plus execution and end synchronization",
            "event_ms": "CUDA events around the complete call, including device idle dispatch gaps",
            "reference": "CPU input-dtype gather and FP64 scatter-add for the input gradient",
            "comparison": "all elements; shared WS1 gradient_accuracy/reduction thresholds",
            "selection": "synthetic half-axis selection without replacement",
            "layouts": "contiguous index; independent contiguous/stride-two x and grad_y",
        },
        "complete": False,
        "case_plan": case_plan,
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        for case in case_plan:
            x, index, grad = make_inputs(
                case["shape"],
                args.dim,
                dtype,
                "cuda",
                args.seed,
                x_contiguous=case["x_contiguous"],
                grad_y_contiguous=case["grad_y_contiguous"],
            )
            duplicate_fraction_actual = 1 - index.unique().numel() / index.numel()
            if duplicate_fraction_actual != 0:
                raise ValueError("benchmark produced repeated indices")
            reference = reference_outputs(x, index, grad, dim=args.dim)
            for name, op, deterministic in implementations:
                torch.use_deterministic_algorithms(deterministic)
                checks = check_result(
                    op,
                    x,
                    index,
                    grad,
                    reference,
                    tolerance,
                    check_gradient_bits=name == "triton",
                )
                result = {
                    **case,
                    "implementation": name,
                    "deterministic_algorithms": deterministic,
                    "selected_count": index.numel(),
                    "duplicate_fraction_actual": duplicate_fraction_actual,
                    "x_strides": list(x.stride()),
                    "grad_y_strides": list(grad.stride()),
                    "index_stride": index.stride(0),
                    "checks": checks,
                }
                report["results"].append(result)
                if checks_failed(checks):
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    raise RuntimeError(f"{name} checks failed before timing: {args.output}")
                result["timings"] = measure(
                    op, x, index, grad, args.warmup, args.samples, phases=case["phases"]
                )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({k: v for k, v in result.items() if k != "timings"}), flush=True)
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
