# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare native and Triton SwiGLU on Muse-Glimmer-30B shapes.

RFC #501 asks for a benchmark on the model's own geometry rather than the Qwen3
width the existing tests use. Muse-Glimmer-30B has hidden size 6656 and separate
``gate_proj`` / ``up_proj`` projections whose output width is 19968, so the
elementwise contract here runs over ``(rows, 19968)``.

Both sides are driven through ``forward_fp32``. ``NativeSwiGLUOp.forward`` and
``TritonSwiGLUOp.forward`` already agree on dtype, but the FP32 entry points make
the comparison unambiguous and match what the accuracy harness uses.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profiler import PerformanceProfiler  # noqa: E402
from rl_engine.kernels.gtest.tolerance import (  # noqa: E402
    load_contract,
    resolve_tolerance,
    tolerance_contract_fingerprint,
)
from rl_engine.reference.activation.swiglu import NativeSwiGLUOp  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
MODES = ("forward", "backward", "forward_backward")
# Muse-Glimmer-30B gate/up output width; see RFC #501.
INTERMEDIATE = 19968
DEFAULT_SHAPES = ("1x19968", "8x19968", "64x19968", "2x3x19968")


def _shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value.lower().split("x"))
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError
        return shape
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shapes must contain positive dimensions, e.g. 8x19968"
        ) from exc


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _make_workload(op, gate, up, dy, mode):
    """Prepare inputs and graphs outside timing; return a callable that returns None."""
    if mode == "forward":

        @torch.no_grad()
        def forward():
            op(leaf_gate, leaf_up)

        leaf_gate = gate.detach()
        leaf_up = up.detach()
        return forward

    leaf_gate = gate.detach().requires_grad_(True)
    leaf_up = up.detach().requires_grad_(True)

    if mode == "backward":
        out = op(leaf_gate, leaf_up)

        def backward():
            torch.autograd.grad(out, (leaf_gate, leaf_up), grad_outputs=dy, retain_graph=True)

        return backward

    if mode == "forward_backward":

        def forward_backward():
            torch.autograd.grad(op(leaf_gate, leaf_up), (leaf_gate, leaf_up), grad_outputs=dy)

        return forward_backward

    raise ValueError(f"unknown mode: {mode}")


def _check_accuracy(native, candidate, gate, up, dy):
    """Compare output and both input gradients against the native gold."""

    def evaluate(op):
        gate_leaf = gate.detach().requires_grad_(True)
        up_leaf = up.detach().requires_grad_(True)
        out = op.forward_fp32(gate_leaf, up_leaf)
        d_gate, d_up = torch.autograd.grad(out, (gate_leaf, up_leaf), grad_outputs=dy)
        return out.detach(), d_gate, d_up

    expected, actual = evaluate(native), evaluate(candidate)
    contract = load_contract()
    errors = {}
    judgments = ("forward_accuracy", "gradient_accuracy", "gradient_accuracy")
    dtypes = (torch.float32, gate.dtype, up.dtype)
    labels = ("output", "grad_input", "grad_up")
    for label, judgment, dtype, reference, result in zip(
        labels, judgments, dtypes, expected, actual, strict=True
    ):
        tolerance = resolve_tolerance(
            contract, judgment=judgment, op_class="elementwise", dtype=dtype
        )
        torch.testing.assert_close(result, reference, atol=tolerance.atol, rtol=tolerance.rtol)
        errors[label] = {
            "max_abs_error": (result.to(torch.float32) - reference.to(torch.float32))
            .abs()
            .max()
            .item(),
            "atol": tolerance.atol,
            "rtol": tolerance.rtol,
        }
    return errors


def _measure(profiler, fn):
    _, median_ms, std_ms = profiler._time_kernel(fn)
    if not math.isfinite(median_ms) or median_ms <= 0:
        raise RuntimeError(f"invalid accelerator-event latency: {median_ms}")
    # The profiler's own helper warms up, resets peak stats and takes one
    # measured invocation, so the figure cannot come out negative.
    peak_gb = profiler._measure_peak_memory_once(fn)
    return {
        "median_ms": median_ms,
        "std_ms": std_ms,
        "peak_extra_gib": max(peak_gb, 0.0),
    }


def _command_output(command):
    try:
        return subprocess.check_output(
            command, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _environment(device, args, triton_version, shapes):
    target = PerformanceProfiler(device=device).gpu_info
    is_rocm = target.compute_capability is None
    props = torch.cuda.get_device_properties(device)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_tracked_changes": _command_output(
            ["git", "status", "--porcelain", "--untracked-files=no"]
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": triton_version,
        "backend": target.backend,
        "runtime": target.driver_version,
        "gpu": props.name,
        "architecture": target.architecture,
        "compute_capability": None if is_rocm else list(torch.cuda.get_device_capability(device)),
        "device": str(device),
        "total_memory_gib": round(props.total_memory / 1024**3, 2),
        "multiprocessors": props.multi_processor_count,
        "shapes": ["x".join(map(str, s)) for s in shapes],
        "intermediate_width": INTERMEDIATE,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "tolerance_contract_sha256": tolerance_contract_fingerprint(),
        "baseline": "eager NativeSwiGLUOp driven through forward_fp32 on the same GPU",
        "scope": "Public wrappers, including allocation and autograd dispatch; no graph capture",
    }


def _markdown(report):
    env = report["environment"]
    lines = [
        "# SwiGLU on Muse-Glimmer-30B shapes",
        "",
        f"GPU: {env['gpu']}; PyTorch: {env['torch']}; Triton: {env['triton']}; "
        f"{env['backend']} runtime: {env['runtime']}.",
        f"Commit: `{env['git_commit']}`; tracked changes: "
        f"`{env['git_tracked_changes'] or 'none'}`.",
        f"Warmup: {env['warmup']}; measured repetitions: {env['repeat']}.",
        "",
        "Both sides run through `forward_fp32`. Median accelerator-event latency for "
        "public wrappers, including allocation and autograd dispatch.",
        "Speedup = eager PyTorch / Triton; values below 1 mean Triton was slower. All "
        "reported cases passed the output and gradient tolerance checks first.",
        "",
        "| Input | Shape | Mode | Native ms | Triton ms | Speedup | Native GiB | Triton GiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        n, t = row["native"], row["triton"]
        lines.append(
            f"| {row['dtype']} | {'x'.join(map(str, row['shape']))} | {row['mode']} | "
            f"{n['median_ms']:.6f} | {t['median_ms']:.6f} | {row['speedup']:.2f}x | "
            f"{n['peak_extra_gib']:.4f} | {t['peak_extra_gib']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument(
        "--shapes", nargs="+", type=_shape, default=list(map(_shape, DEFAULT_SHAPES))
    )
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--repeat", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=415)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks/swiglu"))
    return parser


def run_benchmark(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires an NVIDIA CUDA or AMD ROCm GPU; no CPU fallback is timed"
        )
    import triton

    from rl_engine.backends.shared.triton.activation.swiglu import TritonSwiGLUOp

    native, candidate = NativeSwiGLUOp(), TritonSwiGLUOp()
    with torch.cuda.device(device):
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        report = {
            "environment": _environment(device, args, triton.__version__, args.shapes),
            "results": [],
        }
        for dtype_name in args.dtypes:
            dtype = DTYPES[dtype_name]
            for shape in args.shapes:
                generator = torch.Generator(device=device).manual_seed(args.seed)
                gate = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
                up = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
                dy = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
                gate, up = gate.to(dtype), up.to(dtype)
                dy = dy.to(dtype)

                accuracy = _check_accuracy(native, candidate, gate, up, dy)
                for mode in MODES:
                    measurements = {}
                    for name, op in (("native", native), ("triton", candidate)):
                        fn = _make_workload(op, gate, up, dy, mode)
                        measurements[name] = _measure(profiler, fn)
                        del fn  # Release the retained graph before the next backend.
                    row = {
                        "dtype": dtype_name,
                        "shape": list(shape),
                        "numel": gate.numel(),
                        "mode": mode,
                        "accuracy": accuracy,
                        **measurements,
                        "speedup": measurements["native"]["median_ms"]
                        / measurements["triton"]["median_ms"],
                    }
                    report["results"].append(row)
                    print(f"{dtype_name} {shape} {mode}: {row['speedup']:.2f}x", flush=True)
                del gate, up, dy
    return report


def main():
    args = build_arg_parser().parse_args()
    report = run_benchmark(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown = _markdown(report)
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
