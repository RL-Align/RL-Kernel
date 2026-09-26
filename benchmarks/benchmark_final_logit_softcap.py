# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare the public native and Triton softcap wrappers on a CUDA or ROCm GPU."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profiler import GPUTargetInfo, PerformanceProfiler  # noqa: E402
from rl_engine.kernels.gtest.tolerance import (  # noqa: E402
    load_contract,
    resolve_tolerance,
    tolerance_contract_fingerprint,
)
from rl_engine.kernels.ops.pytorch.activation import NativeFinalLogitSoftcapOp  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
MODES = ("forward", "backward", "forward_backward")
DEFAULT_SHAPES = ("1x1x1025", "1x1x262144", "1x16x262144", "1x64x262144")


def _shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value.split("x"))
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError
        return shape
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shapes must contain positive dimensions, e.g. 1x16x262144"
        ) from exc


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _make_workload(op, x, grad_y, mode):
    """Prepare inputs/graphs outside timing; discard results without accumulating .grad."""
    if mode == "forward":

        @torch.no_grad()
        def forward():
            op(x)

        return forward

    leaf = x.detach().requires_grad_(True)
    if mode == "backward":
        y = op(leaf)

        def backward():
            torch.autograd.grad(y, leaf, grad_outputs=grad_y, retain_graph=True)

        return backward
    if mode == "forward_backward":

        def forward_backward():
            torch.autograd.grad(op(leaf), leaf, grad_outputs=grad_y)

        return forward_backward
    raise ValueError(f"unknown mode: {mode}")


def _check_accuracy(native, candidate, x, grad_y):
    def evaluate(op):
        leaf = x.detach().requires_grad_(True)
        y = op(leaf)
        (dx,) = torch.autograd.grad(y, leaf, grad_outputs=grad_y)
        return y.detach(), dx

    expected, actual = evaluate(native), evaluate(candidate)
    contract = load_contract()
    errors = {}
    for label, judgment, dtype, reference, result in zip(
        ("output", "gradient"),
        ("forward_accuracy", "gradient_accuracy"),
        (torch.float32, x.dtype),
        expected,
        actual,
    ):
        assert result.shape == x.shape and result.dtype == dtype
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
    # The existing profiler warms up before recording per-invocation accelerator
    # events (torch.cuda.Event, which is a HIP event on ROCm).
    # fn returns None so a previous invocation's output cannot remain live.
    _, median_ms, std_ms = profiler._time_kernel(fn)
    if not math.isfinite(median_ms) or median_ms <= 0:
        raise RuntimeError(f"invalid accelerator-event latency: {median_ms}")
    # Measure one additional invocation. Persistent inputs and the retained
    # backward graph are in the baseline, not in this incremental allocation.
    profiler._sync()
    baseline = torch.cuda.memory_allocated(profiler.device)
    torch.cuda.reset_peak_memory_stats(profiler.device)
    fn()
    profiler._sync()
    peak = torch.cuda.max_memory_allocated(profiler.device) - baseline
    return {"median_ms": median_ms, "std_ms": std_ms, "peak_extra_mib": peak / 1024**2}


def _command_output(command):
    try:
        return subprocess.check_output(
            command, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _environment(device, args, triton_version, block_size, target: GPUTargetInfo):
    """Fingerprint the run on the active accelerator.

    ``target`` is the profiler's ``GPUTargetInfo``, which already tells CUDA and
    ROCm apart (``backend``, ``architecture``, and ``compute_capability=None`` on
    ROCm). Without it a ROCm host would be reported as a CUDA host whose runtime
    failed to load, with an emulated compute capability attached.
    """
    props = torch.cuda.get_device_properties(device)
    is_rocm = target.backend == "rocm"
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
        # torch.version.cuda on NVIDIA, torch.version.hip on ROCm.
        "runtime": target.driver_version,
        "gpu": props.name,
        "architecture": target.architecture,
        "compute_capability": None if is_rocm else list(torch.cuda.get_device_capability(device)),
        "gpu_name_driver": _command_output(
            ["rocm-smi", "--showproductname", "--showdriverversion"]
            if is_rocm
            else ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        ),
        "device": str(device),
        "total_memory_gib": props.total_memory / 1024**3,
        "multiprocessors": props.multi_processor_count,
        "block_size": block_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "layout": "contiguous",
        "tolerance_contract_sha256": tolerance_contract_fingerprint(),
        "timer": (
            f"PerformanceProfiler {'HIP' if is_rocm else 'CUDA'} events; "
            "median and sample standard deviation"
        ),
        "baseline": "eager NativeFinalLogitSoftcapOp on the same GPU; no torch.compile",
        "scope": "Public wrappers, including allocation and autograd dispatch; no graph capture",
        "backward": "Random FP32 upstream gradient; prebuilt retained graph; no forward timing",
        "memory": "Extra peak allocated MiB per call; excludes inputs and any prebuilt graph",
    }


def _markdown(report):
    env = report["environment"]
    event_api = "HIP" if env["backend"] == "rocm" else "CUDA"
    lines = [
        "# Final logit softcap benchmark",
        "",
        f"GPU: {env['gpu']}; PyTorch: {env['torch']}; Triton: {env['triton']}; "
        f"{env['backend']} runtime: {env['runtime']}.",
        f"Commit: `{env['git_commit']}`; tracked changes: "
        f"`{env['git_tracked_changes'] or 'none'}`.",
        f"Warmup: {env['warmup']}; measured repetitions: {env['repeat']}; "
        f"BLOCK_SIZE: {env['block_size']}.",
        "",
        f"Contiguous inputs; median {event_api} event latency for public wrappers, including "
        "allocation and autograd dispatch. Input generation, correctness checks and JIT "
        "compilation are outside timing. Small cases can be dominated by host dispatch gaps.",
        "Backward reuses a prebuilt graph; forward+backward builds a fresh graph per call. "
        "Forward runs under no_grad. Extra peak allocation excludes inputs and prebuilt graphs.",
        "Speedup = eager PyTorch / Triton; values below 1 mean Triton was slower. "
        "All reported cases passed the output and gradient tolerance checks first.",
        "",
        "| Input | Shape | Mode | Native ms | Triton ms | Speedup | "
        "Native extra MiB | Triton extra MiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        n, t = row["native"], row["triton"]
        lines.append(
            f"| {row['dtype']} | {'x'.join(map(str, row['shape']))} | {row['mode']} | "
            f"{n['median_ms']:.6f} | {t['median_ms']:.6f} | {row['speedup']:.2f}x | "
            f"{n['peak_extra_mib']:.2f} | {t['peak_extra_mib']:.2f} |"
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
    parser.add_argument("--output-dir", type=Path, default=Path("reports/final-logit-softcap"))
    return parser


def run_benchmark(args):
    device = torch.device(args.device)
    # CPU is rejected outright: its fallback timing must never be reported as GPU
    # performance. ROCm is accepted; AMD GPUs live under torch.cuda, so the vendor is
    # only distinguishable through torch.version.hip (which GPUProfiler reads).
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires an NVIDIA CUDA or AMD ROCm GPU; no CPU fallback is timed"
        )
    # Import only after the device check, so --help works without Triton/GPU.
    import triton

    from rl_engine.kernels.ops.triton.activation.final_logit_softcap import (
        _BLOCK,
        TritonFinalLogitSoftcapOp,
    )

    with torch.cuda.device(device):
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        native, candidate = NativeFinalLogitSoftcapOp(), TritonFinalLogitSoftcapOp()
        report = {
            "environment": _environment(
                device, args, triton.__version__, _BLOCK, profiler.gpu_info
            ),
            "results": [],
        }
        for dtype_name in args.dtypes:
            dtype = DTYPES[dtype_name]
            for shape in args.shapes:
                generator = torch.Generator(device=device).manual_seed(args.seed)
                x = (30.0 * torch.randn(shape, device=device, generator=generator)).to(dtype)
                grad_y = torch.randn(shape, device=device, generator=generator, dtype=torch.float32)
                accuracy = _check_accuracy(native, candidate, x, grad_y)
                for mode in MODES:
                    measurements = {}
                    for name, op in (("native", native), ("triton", candidate)):
                        fn = _make_workload(op, x, grad_y, mode)
                        measurements[name] = _measure(profiler, fn)
                        del fn  # Release the retained graph before measuring the next backend.
                    row = {
                        "dtype": dtype_name,
                        "shape": list(shape),
                        "numel": x.numel(),
                        "mode": mode,
                        "accuracy": accuracy,
                        **measurements,
                        "speedup": measurements["native"]["median_ms"]
                        / measurements["triton"]["median_ms"],
                    }
                    report["results"].append(row)
                    print(f"{dtype_name} {shape} {mode}: {row['speedup']:.2f}x", flush=True)
                del x, grad_y
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
