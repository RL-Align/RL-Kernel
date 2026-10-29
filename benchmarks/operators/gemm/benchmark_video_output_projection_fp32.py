# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Measure H3 output projection on all mixed rows after accuracy and BI checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch

from rl_engine.contracts.numerical import load_contract
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.reference.gemm.video_output_projection_fp32 import (
    NativeVideoOutputProjectionFP32Op,
    _reference_matmul_fp32,
)


def _candidate(name):
    if name == "triton":
        from rl_engine.backends.shared.triton.gemm.video_output_projection_fp32 import (
            TritonVideoOutputProjectionFP32Op,
        )

        return TritonVideoOutputProjectionFP32Op()
    return None


def _tool_version(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _forward(op, x, weight, bias, keys):
    if op is None:
        return torch.nn.functional.linear(x, weight, bias)
    return op(x, weight, bias, logical_keys=keys, parameter_id="benchmark.proj_out")


def _vjp(op, x, weight, bias, grad, keys):
    x, weight, bias = [t.detach().requires_grad_() for t in (x, weight, bias)]
    with canonical_backward_session() as session:
        output = _forward(op, x, weight, bias, keys)
        output.backward(grad)
        session.validate_complete()
    return output.detach(), x.grad, weight.grad, bias.grad


def _reference(x, weight, bias, grad):
    """Check all outputs with the existing CPU FP32 full-reduction reference."""
    rows, features = x.shape
    cpu_weight, cpu_bias = weight.detach().cpu(), bias.detach().cpu()
    cpu_grad = grad.detach().cpu()
    y = torch.empty((rows, weight.shape[0]), dtype=torch.float32)
    dx = torch.empty((rows, features), dtype=torch.float32)
    # Tile output rows to avoid a full CPU copy of the hidden states. Each
    # reference dot still reduces all features with the existing torch.sum.
    for start in range(0, rows, 4096):
        stop = start + 4096
        cpu_x = x[start:stop].detach().cpu()
        y[start:stop] = _reference_matmul_fp32(cpu_x, cpu_weight.t()) + cpu_bias
        dx[start:stop] = _reference_matmul_fp32(cpu_grad[start:stop], cpu_weight)
    dw = torch.empty_like(cpu_weight)
    grad_t = cpu_grad.t().contiguous()
    # Tile dW's output columns, not its row-reduction axis. This keeps the
    # reference independent of the candidate's 128-term reduction chunks.
    for start in range(0, features, 16):
        stop = start + 16
        dw[:, start:stop] = _reference_matmul_fp32(grad_t, x[:, start:stop].detach().cpu())
    return y, dx, dw, cpu_grad.sum(dim=0)


def _accuracy(actual, gold):
    judgments = load_contract()["judgments"]
    result = {}
    for i, (name, tensor, reference) in enumerate(
        zip(("y", "dx", "dw", "dbias"), actual, gold, strict=True)
    ):
        policy = judgments["forward_accuracy" if i == 0 else "gradient_accuracy"]["by_op_class"][
            "reduction"
        ]["float32"]
        max_error, violations, non_finite, checked = 0.0, 0, 0, 0
        finite_errors = True
        if tensor.shape != reference.shape:
            raise ValueError(f"{name} reference shape must match the candidate")
        for start in range(0, tensor.shape[0], 4096):
            ac = tensor[start : start + 4096].detach().cpu()
            bc = reference[start : start + 4096].detach().cpu()
            error = (ac - bc).abs()
            finite = torch.isfinite(ac) & torch.isfinite(bc)
            limit = policy["atol"] + policy["rtol"] * bc.abs()
            if torch.isfinite(error).all():
                max_error = max(max_error, error.max().item())
            else:
                finite_errors = False
            violations += int(((error > limit) | ~finite).sum())
            non_finite += int((~finite).sum())
            checked += ac.numel()
        result[name] = {
            "max_abs_error": max_error if finite_errors else None,
            "out_of_tolerance": violations,
            "non_finite_elements": non_finite,
            "checked_elements": checked,
            "total_elements": tensor.numel(),
            **policy,
        }
    return result


def _permutation_mismatches(actual, permuted):
    result = {}
    for i, (label, a, b) in enumerate(
        zip(("y", "dx", "dw", "dbias"), actual, permuted, strict=True)
    ):
        count = 0
        for start in range(0, a.shape[0], 4096):
            stop = min(start + 4096, a.shape[0])
            # The permutation is a reversal. Undo it one slice at a time.
            aligned = b[a.shape[0] - stop : a.shape[0] - start].flip(0) if i < 2 else b[start:stop]
            count += int(
                (
                    a[start:stop].contiguous().view(torch.int32)
                    != aligned.contiguous().view(torch.int32)
                ).sum()
            )
        result[label] = count
    return result


def _sample(fn, prepare, samples, warmup):
    values = []
    for i in range(warmup + samples):
        call = prepare() if prepare is not None else fn
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        if i >= warmup:
            values.append(start.elapsed_time(end) * 1000)
        del call
    return {"median_us": statistics.median(values), "samples_us": values}


def _backward_call(op, x, weight, bias, grad, keys):
    x.grad = weight.grad = bias.grad = None
    cm = canonical_backward_session()
    session = cm.__enter__()
    try:
        output = _forward(op, x, weight, bias, keys)
    except BaseException:
        cm.__exit__(None, None, None)
        raise

    def run():
        nonlocal output, session
        try:
            output.backward(grad)
            session.validate_complete()
        finally:
            cm.__exit__(None, None, None)
            output = session = None

    return run


def _forward_backward(op, x, weight, bias, grad, keys):
    _backward_call(op, x, weight, bias, grad, keys)()


def _forward_backward_memory(op, x, weight, bias, grad, keys):
    """Measure peak allocated bytes above the warmed, gradient-free baseline."""
    x.grad = weight.grad = bias.grad = None
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _forward_backward(op, x, weight, bias, grad, keys)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    x.grad = weight.grad = bias.grad = None
    return {
        "baseline_allocated_bytes": baseline,
        "peak_allocated_bytes": peak,
        "incremental_peak_allocated_bytes": peak - baseline,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("pytorch", "triton"),
        default=["pytorch", "triton"],
    )
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=[1, 128, 1024, 4096, 13104, 38734, 74410, 104990, 110750, 367986],
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not torch.cuda.is_available()
        or torch.version.hip is not None
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        raise RuntimeError("benchmark requires a CUDA SM90 GPU")
    if args.samples < 1 or args.warmup < 0 or min(args.rows) < 1:
        raise ValueError("rows/samples must be positive and warmup must be non-negative")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    weight = torch.randn(96, 5376, device="cuda", dtype=torch.float32) * 0.0625
    bias = torch.randn(96, device="cuda", dtype=torch.float32) * 0.125
    root = Path(__file__).resolve().parents[3]
    sources = [
        "benchmarks/operators/gemm/benchmark_video_output_projection_fp32.py",
        "rl_engine/ops/gemm/canonical_linear.py",
        "rl_engine/ops/autograd/canonical_backward.py",
        "rl_engine/ops/gemm/video_output_projection_fp32.py",
        "rl_engine/backends/shared/triton/gemm/det_gemm.py",
        "rl_engine/backends/shared/triton/gemm/video_output_projection_fp32.py",
        "rl_engine/reference/gemm/video_input_projection_fp32.py",
        "rl_engine/reference/gemm/video_output_projection_fp32.py",
    ]
    # Source hashes also identify exported workspaces without Git metadata.
    git_commit = _tool_version(["git", "-C", str(root), "rev-parse", "HEAD"])
    git_status = (
        _tool_version(["git", "-C", str(root), "status", "--porcelain"])
        if git_commit is not None
        else None
    )
    report = {
        "scope": "single H3 FP32 output projection; eager training and no-grad forward graph",
        "reference_id": NativeVideoOutputProjectionFP32Op.reference_id,
        "seed": args.seed,
        "benchmark": {
            "backends": args.backends,
            "rows": args.rows,
            "warmup": args.warmup,
            "samples": args.samples,
            "inputs": "seeded FP32 normal samples; x/bias scale 0.125, weight/grad scale 0.0625",
            "timed_keys": "canonical row order; reversal BI is checked separately",
            "throughput": "all projected mixed-sequence rows per eager forward/backward second",
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "cpu_threads": torch.get_num_threads(),
            "tf32": False,
            "driver": _tool_version(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ),
            "nvcc": _tool_version(
                [shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc", "--version"]
            ),
        },
        "git_commit": git_commit,
        "git_dirty": bool(git_status) if git_status is not None else None,
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        "timing": "CUDA event interval for complete operator calls, including host launch gaps",
        "memory": (
            "Warmed eager F+B peak allocated bytes minus the pre-call baseline, "
            "excluding preallocated inputs, persistent caches and allocator reservations"
        ),
        "results": [],
    }
    for rows in args.rows:
        x = torch.randn(rows, 5376, device="cuda", dtype=torch.float32) * 0.125
        grad = torch.randn(rows, 96, device="cuda", dtype=torch.float32) * 0.0625
        keys = torch.stack(
            (
                torch.zeros(rows, dtype=torch.int64, device="cuda"),
                torch.arange(rows, device="cuda"),
            ),
            dim=1,
        )
        print(f"M={rows}: computing full CPU FP32 reference", flush=True)
        gold = _reference(x, weight, bias, grad)
        for name in args.backends:
            op = _candidate(name)
            actual = _vjp(op, x, weight, bias, grad, keys)
            accuracy = _accuracy(actual, gold)
            if op is not None and any(v["out_of_tolerance"] for v in accuracy.values()):
                raise RuntimeError(f"{name} failed FP32 accuracy at M={rows}: {accuracy}")
            order = torch.arange(rows - 1, -1, -1, device="cuda")
            permuted = _vjp(
                op,
                x[order].contiguous(),
                weight,
                bias,
                grad[order].contiguous(),
                keys[order].contiguous(),
            )
            mismatches = _permutation_mismatches(actual, permuted)
            if op is not None and any(mismatches.values()):
                raise RuntimeError(f"{name} failed row permutation BI: {mismatches}")
            del actual, permuted
            with torch.no_grad():
                forward = _sample(
                    partial(_forward, op, x, weight, bias, keys),
                    None,
                    args.samples,
                    args.warmup,
                )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        expected = _forward(op, x, weight, bias, keys)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = _forward(op, x, weight, bias, keys)
                graph.replay()
                if not torch.equal(output.view(torch.int32), expected.view(torch.int32)):
                    raise RuntimeError("forward graph changed output bytes")
                graph_time = _sample(graph.replay, None, args.samples, args.warmup)
                del graph, output, expected
            tx, tw, tb = [t.detach().requires_grad_() for t in (x, weight, bias)]
            backward = _sample(
                None,
                partial(_backward_call, op, tx, tw, tb, grad, keys),
                args.samples,
                args.warmup,
            )
            combined = _sample(
                partial(_forward_backward, op, tx, tw, tb, grad, keys),
                None,
                args.samples,
                args.warmup,
            )
            memory = _forward_backward_memory(op, tx, tw, tb, grad, keys)
            del tx, tw, tb
            item = {
                "backend": name,
                "rows": rows,
                "accuracy": accuracy,
                "accuracy_coverage": "all elements of y, dx, dw and dbias; complete reductions",
                "permutation_bit_mismatches": mismatches,
                "permutation_coverage": "all elements of y, dx, dw and dbias",
                "actual_backend": (
                    op.backend_info()
                    if op is not None
                    else {
                        "family": "pytorch",
                        "operator": "torch.nn.functional.linear",
                        "strict_bi": False,
                    }
                ),
                "forward_eager": forward,
                "forward_graph": graph_time,
                "backward_eager": backward,
                "forward_backward_eager": combined,
                "forward_backward_memory": memory,
                "throughput_rows_per_second": rows * 1e6 / combined["median_us"],
            }
            report["results"].append(item)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(
                f"{name} M={rows}: F={forward['median_us']:.1f} us, "
                f"B={backward['median_us']:.1f} us, "
                f"F+B={combined['median_us']:.1f} us, "
                f"peak delta={memory['incremental_peak_allocated_bytes'] / 2**20:.2f} MiB, "
                f"throughput={item['throughput_rows_per_second']:.0f} rows/s"
            )
        del gold, x, grad, keys
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
