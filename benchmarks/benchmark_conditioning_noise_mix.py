# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compare the public native and CUDA conditioning-noise-mix wrappers."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tabulate import tabulate

if TYPE_CHECKING or __package__:
    from benchmarks.profiler import PerformanceProfiler
else:
    from profiler import PerformanceProfiler

from rl_engine.kernels.ops.cuda.conditioning_noise_mix import ConditioningNoiseMixCudaOp
from rl_engine.kernels.ops.pytorch.conditioning_noise_mix import NativeConditioningNoiseMixOp
from rl_engine.testing import summarize_kernel_drift

DEFAULT_SHAPES = ("3x257", "1x32x400", "1x24x1x48x80", "1x24x17x48x80")
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODES = ("forward", "backward", "forward_backward")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value.split("x"))
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError
        return shape
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shapes must contain positive dimensions, e.g. 1x24x17x48x80"
        ) from exc


def _make_workload(op, sample, timestep, noise, grad_y, mode):
    """Prepare leaves/graphs outside timing; discard outputs and avoid .grad accumulation."""
    if mode == "forward":

        @torch.no_grad()
        def forward():
            op(sample, timestep, noise)

        return forward

    leaves = (sample.detach().requires_grad_(True), noise.detach().requires_grad_(True))
    if mode == "backward":
        y = op(leaves[0], timestep, leaves[1])

        def backward():
            torch.autograd.grad(y, leaves, grad_outputs=grad_y, retain_graph=True)

        return backward
    if mode == "forward_backward":

        def forward_backward():
            torch.autograd.grad(op(leaves[0], timestep, leaves[1]), leaves, grad_outputs=grad_y)

        return forward_backward
    raise ValueError(f"unknown mode: {mode}")


def _check_accuracy(native, candidate, sample, timestep, noise, grad_y):
    def evaluate(op):
        leaves = (sample.detach().requires_grad_(True), noise.detach().requires_grad_(True))
        y = op(leaves[0], timestep, leaves[1])
        gradients = torch.autograd.grad(y, leaves, grad_outputs=grad_y)
        with torch.no_grad():
            inference = op(sample, timestep, noise)
        return (y.detach(), *gradients, inference)

    expected, actual = evaluate(native), evaluate(candidate)
    accuracy = {}
    for label, reference, result in zip(
        ("output", "d_sample", "d_noise", "inference_output"), expected, actual, strict=True
    ):
        # This arithmetic profile checks exact bits, including signed zero.
        if result.shape != reference.shape or result.dtype != reference.dtype:
            raise AssertionError(f"{label} shape or dtype differs from the eager reference")
        if not torch.equal(
            result.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)
        ):
            raise AssertionError(f"{label} differs from the eager dtype-staged reference")
        accuracy[label] = {
            "bitwise_equal": True,
            "max_abs_error": summarize_kernel_drift(result, reference)["max_abs_error"],
        }
    return accuracy


def _measure(profiler, fn, numel):
    # Use per-invocation CUDA events from the shared timer. The workload returns
    # None so an earlier output does not stay live.
    _, median_ms, std_ms = profiler._time_kernel(fn)
    if not math.isfinite(median_ms) or median_ms <= 0:
        raise RuntimeError(f"invalid accelerator-event latency: {median_ms}")
    profiler._sync()
    baseline = torch.cuda.memory_allocated(profiler.device)
    torch.cuda.reset_peak_memory_stats(profiler.device)
    fn()
    profiler._sync()
    peak = torch.cuda.max_memory_allocated(profiler.device)
    return {
        "elements_per_sec": numel * 1000 / median_ms,
        "median_ms": median_ms,
        "peak_extra_mib": (peak - baseline) / 1024**2,
        "peak_vram_gb": peak / 1024**3,
        "std_ms": std_ms,
    }


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _reproduction_command(args: argparse.Namespace) -> str:
    return shlex.join(
        [
            sys.executable,
            "benchmarks/benchmark_conditioning_noise_mix.py",
            "--device",
            args.device,
            "--dtypes",
            *args.dtypes,
            "--shapes",
            *("x".join(map(str, shape)) for shape in args.shapes),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(args.output_dir),
        ]
    )


def _environment(device, args, target):
    return {
        "architecture": target.architecture,
        "command": _reproduction_command(args),
        "compute_capability": target.compute_capability,
        "device": str(device),
        "driver": _command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_tracked_changes": _command_output(
            ["git", "status", "--porcelain", "--untracked-files=no"]
        ),
        "gpu": target.name,
        "limitations": [
            "Synthetic resident tensors; input generation and transfers are outside timing",
            "CPU scheduling and concurrent GPU work can affect measurements",
            "Operator API latency only; no full H3 or cross-GPU performance claim",
        ],
        "repeat": args.repeat,
        "runtime": torch.version.cuda,
        "seed": args.seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "total_memory_gib": target.total_memory_gb,
        "warmup": args.warmup,
    }


def _markdown(report: dict) -> str:
    """Render the saved measurements without recomputing or filtering results."""
    env = report["environment"]
    rows = []
    for result in report["results"]:
        native, cuda = result["native"], result["cuda"]
        rows.append(
            (
                result["dtype"],
                "x".join(map(str, result["shape"])),
                result["mode"],
                native["median_ms"],
                cuda["median_ms"],
                f"{result['speedup']:.2f}x",
                native["peak_extra_mib"],
                cuda["peak_extra_mib"],
                result["status"],
            )
        )
    headers = (
        "Dtype",
        "Shape",
        "Mode",
        "Native ms",
        "CUDA ms",
        "Speedup",
        "Native extra MiB",
        "CUDA extra MiB",
        "Status",
    )
    driver = (env["driver"] or "unavailable").replace("\n", "; ")
    sections = [
        "# Conditioning noise mix benchmark",
        f"{env['gpu']} ({env['architecture']}, compute capability {env['compute_capability']}); "
        f"driver {driver}, CUDA {env['runtime']}, PyTorch {env['torch']}. "
        f"Commit: `{env['git_commit'] or 'unavailable'}`.",
        f"PerformanceProfiler CUDA-event medians; {env['warmup']} warmups and "
        f"{env['repeat']} measured calls per backend. Seed {env['seed']}; contiguous inputs; "
        "per-sample timestep 0.37 rounded to the input dtype. Outputs and both input "
        "gradients pass bitwise checks against eager PyTorch before timing.",
        "Public wrappers are timed, including allocation and autograd dispatch. "
        "Forward uses no_grad; backward reuses a prebuilt retained graph; "
        "forward+backward builds a fresh graph. Both input gradients are computed. "
        "Speedup is eager PyTorch / CUDA; below 1 means CUDA is slower.",
        "Extra peak MiB excludes inputs and any prebuilt graph. Status pass means the "
        "workload completed after accuracy checks.",
        tabulate(
            rows,
            headers=headers,
            tablefmt="github",
            floatfmt=("", "", "", ".6f", ".6f", "", ".3f", ".3f", ""),
        ),
        "Reproduce from the repository root after building the CUDA extension:",
        f"```bash\n{env['command']}\n```",
        "\n".join(f"- {item}." for item in env["limitations"]),
        "[results.json](results.json) includes backend paths, arithmetic policy, accuracy, "
        "timing standard deviation, elements/s, extra peak MiB and total peak allocated GiB "
        "(PyTorch allocations, not reserved memory or device usage).",
    ]
    return "\n\n".join(sections) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--shapes", nargs="+", type=_shape, default=list(map(_shape, DEFAULT_SHAPES))
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/conditioning-noise-mix"))
    return parser


def run_benchmark(args):
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError("This benchmark requires an NVIDIA CUDA GPU; no fallback is timed")
    with torch.cuda.device(device):
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        native, candidate = NativeConditioningNoiseMixOp(), ConditioningNoiseMixCudaOp()
        report = {
            "backends": {
                "cuda": {
                    "actual_backend": "cuda",
                    "fallback": False,
                    "kernel_id": (
                        "rl_engine._C.conditioning_noise_mix_forward+"
                        "rl_engine._C.conditioning_noise_mix_backward"
                    ),
                    "requested_backend": "cuda",
                    "wrapper": f"{type(candidate).__module__}.{type(candidate).__qualname__}",
                },
                "native": {
                    "actual_backend": "pytorch",
                    "fallback": False,
                    "kernel_id": f"{native._mix.__module__}.{native._mix.__qualname__}",
                    "requested_backend": "pytorch",
                    "wrapper": f"{type(native).__module__}.{type(native).__qualname__}",
                },
            },
            "environment": _environment(device, args, profiler.gpu_info),
            "methodology": {
                "cast_policy": "input dtype after subtraction, each product, and final addition",
                "compute_dtype": "fp32",
                "layout": "contiguous",
                "reduction_order": "none; independent elements",
                "split_policy": "none",
                "timestep": 0.37,
            },
            "results": [],
        }
        for dtype_name in args.dtypes:
            dtype = DTYPES[dtype_name]
            for shape in args.shapes:
                generator = torch.Generator(device=device).manual_seed(args.seed)
                sample, noise, grad_y = [
                    torch.randn(shape, device=device, generator=generator, dtype=dtype)
                    for _ in range(3)
                ]
                timestep = torch.full((shape[0],), 0.37, device=device, dtype=dtype)
                accuracy = _check_accuracy(native, candidate, sample, timestep, noise, grad_y)
                for mode in MODES:
                    measurements = {}
                    for name, op in (("native", native), ("cuda", candidate)):
                        fn = _make_workload(op, sample, timestep, noise, grad_y, mode)
                        measurements[name] = _measure(profiler, fn, sample.numel())
                        del fn  # Release any retained graph before the next path.
                    row = {
                        "accuracy": accuracy,
                        "cuda": measurements["cuda"],
                        "dtype": dtype_name,
                        "mode": mode,
                        "native": measurements["native"],
                        "numel": sample.numel(),
                        "shape": list(shape),
                        "speedup": measurements["native"]["median_ms"]
                        / measurements["cuda"]["median_ms"],
                        "status": "pass",
                    }
                    report["results"].append(row)
                    print(f"{dtype_name} {shape} {mode}: {row['speedup']:.2f}x", flush=True)
                del sample, noise, grad_y, timestep
    return report


def main() -> None:
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
