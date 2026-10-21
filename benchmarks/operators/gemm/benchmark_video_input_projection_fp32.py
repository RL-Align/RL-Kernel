# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Measure complete H3 FP32 projection calls after reference accuracy checks."""

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
from rl_engine.reference.gemm.video_input_projection_fp32 import NativeVideoInputProjectionFP32Op


def _candidate(name):
    if name == "triton":
        from rl_engine.backends.shared.triton.gemm.video_input_projection_fp32 import (
            TritonVideoInputProjectionFP32Op,
        )

        return TritonVideoInputProjectionFP32Op()
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
    return op(x, weight, bias, logical_keys=keys, parameter_id="benchmark.proj_in")


def _vjp(op, x, weight, bias, grad, keys):
    x, weight, bias = [t.detach().clone().requires_grad_() for t in (x, weight, bias)]
    with canonical_backward_session() as session:
        output = _forward(op, x, weight, bias, keys)
        output.backward(grad)
        session.validate_complete()
    return output.detach(), x.grad, weight.grad, bias.grad


def _accuracy(actual, gold):
    judgments = load_contract()["judgments"]
    result = {}
    for i, (name, a, b) in enumerate(zip(("y", "dx", "dw", "dbias"), actual, gold, strict=True)):
        policy = judgments["forward_accuracy" if i == 0 else "gradient_accuracy"]["by_op_class"][
            "reduction"
        ]["float32"]
        a, b = a.detach().cpu(), b.detach().cpu()
        error = (a - b).abs()
        limit = policy["atol"] + policy["rtol"] * b.abs()
        finite = torch.isfinite(a) & torch.isfinite(b)
        result[name] = {
            "max_abs_error": error.max().item() if torch.isfinite(error).all() else None,
            "out_of_tolerance": int(((error > limit) | ~finite).sum()),
            "non_finite_elements": int((~finite).sum()),
            **policy,
        }
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
        try:
            output.backward(grad)
            session.validate_complete()
        finally:
            cm.__exit__(None, None, None)

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
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 128, 1024, 4096, 13104])
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
    weight = torch.randn(5376, 96, device="cuda") * 0.0625
    bias = torch.randn(5376, device="cuda") * 0.125
    root = Path(__file__).resolve().parents[3]
    sources = [
        "rl_engine/ops/gemm/canonical_linear.py",
        "rl_engine/ops/autograd/canonical_backward.py",
        "rl_engine/ops/gemm/video_input_projection_fp32.py",
        "rl_engine/backends/shared/triton/gemm/det_gemm.py",
        "rl_engine/backends/shared/triton/gemm/video_input_projection_fp32.py",
        "rl_engine/reference/gemm/video_input_projection_fp32.py",
    ]
    # Source hashes also identify exported workspaces without Git metadata.
    git_commit = _tool_version(["git", "-C", str(root), "rev-parse", "HEAD"])
    git_status = (
        _tool_version(["git", "-C", str(root), "status", "--porcelain"])
        if git_commit is not None
        else None
    )
    report = {
        "scope": "single H3 FP32 input projection; eager training and no-grad forward graph",
        "reference_id": NativeVideoInputProjectionFP32Op.reference_id,
        "seed": args.seed,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
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
        x = torch.randn(rows, 96, device="cuda") * 0.125
        grad = torch.randn(rows, 5376, device="cuda") * 0.0625
        keys = torch.stack(
            (
                torch.zeros(rows, dtype=torch.int64, device="cuda"),
                torch.arange(rows, device="cuda"),
            ),
            dim=1,
        )
        cx, cw, cb = [t.detach().cpu().requires_grad_() for t in (x, weight, bias)]
        cy = NativeVideoInputProjectionFP32Op()(cx, cw, cb)
        cy.backward(grad.cpu())
        gold = (cy, cx.grad, cw.grad, cb.grad)
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
            mismatches = {
                label: int(
                    (
                        a.contiguous().view(torch.int32)
                        != (b[order] if i < 2 else b).contiguous().view(torch.int32)
                    ).sum()
                )
                for i, (label, a, b) in enumerate(
                    zip(("y", "dx", "dw", "dbias"), actual, permuted, strict=True)
                )
            }
            if op is not None and any(mismatches.values()):
                raise RuntimeError(f"{name} failed row permutation BI: {mismatches}")
            with torch.no_grad():
                forward = _sample(
                    partial(_forward, op, x, weight, bias, keys), None, args.samples, args.warmup
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
            tx, tw, tb = [t.detach().clone().requires_grad_() for t in (x, weight, bias)]
            backward = _sample(
                None, partial(_backward_call, op, tx, tw, tb, grad, keys), args.samples, args.warmup
            )
            combined = _sample(
                partial(_forward_backward, op, tx, tw, tb, grad, keys),
                None,
                args.samples,
                args.warmup,
            )
            memory = _forward_backward_memory(op, tx, tw, tb, grad, keys)
            item = {
                "backend": name,
                "rows": rows,
                "accuracy": accuracy,
                "permutation_bit_mismatches": mismatches,
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
            }
            report["results"].append(item)
            print(
                f"{name} M={rows}: F={forward['median_us']:.1f} us, "
                f"B={backward['median_us']:.1f} us, "
                f"F+B={combined['median_us']:.1f} us, "
                f"peak delta={memory['incremental_peak_allocated_bytes'] / 2**20:.2f} MiB"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
