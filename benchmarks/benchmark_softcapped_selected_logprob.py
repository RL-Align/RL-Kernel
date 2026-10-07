# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare eager PyTorch and the automatic Triton softcapped selected logprob Op."""

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
from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance  # noqa: E402
from rl_engine.kernels.ops.pytorch.loss import NativeSoftcappedSelectedLogprobOp  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
MODES = ("forward", "backward", "forward_backward")
DEFAULT_SHAPES = ("1x1025", "1x262144", "16x262144", "128x262144", "1024x262144")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _shape(value: str) -> tuple[int, int]:
    try:
        rows, vocab = map(int, value.split("x"))
        if rows <= 0 or vocab <= 0:
            raise ValueError
        return rows, vocab
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be positive MxV, e.g. 16x262144") from exc


def _make_workload(op, logits, token_ids, upstream, mode):
    """Prepare inputs/graphs outside timing; discard results without accumulating .grad."""
    if mode == "forward":

        @torch.no_grad()
        def forward():
            op(logits, token_ids)

        return forward

    leaf = logits.detach().requires_grad_(True)
    if mode == "backward":
        output = op(leaf, token_ids)

        def backward():
            torch.autograd.grad(output, leaf, grad_outputs=upstream, retain_graph=True)

        return backward
    if mode == "forward_backward":

        def forward_backward():
            torch.autograd.grad(op(leaf, token_ids), leaf, grad_outputs=upstream)

        return forward_backward
    raise ValueError(f"unknown mode: {mode}")


def _check_accuracy(native, candidate, logits, token_ids, upstream):
    def evaluate(op):
        leaf = logits.detach().requires_grad_(True)
        output = op(leaf, token_ids)
        gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]
        return output.detach(), gradient

    expected, actual = evaluate(native), evaluate(candidate)
    errors = {}
    for label, judgment, shape, dtype, reference, result in zip(
        ("output", "gradient"),
        ("forward_accuracy", "gradient_accuracy"),
        (logits.shape[:1], logits.shape),
        (torch.float32, logits.dtype),
        expected,
        actual,
        strict=True,
    ):
        assert result.shape == shape and result.dtype == dtype
        tolerance = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="logprob", dtype=dtype
        )
        torch.testing.assert_close(result, reference, atol=tolerance.atol, rtol=tolerance.rtol)
        errors[label] = {
            "max_abs_error": (result.float() - reference.float()).abs().max().item(),
            "atol": tolerance.atol,
            "rtol": tolerance.rtol,
        }
    return errors


def _case_plan(args):
    return {
        "dtypes": args.dtypes,
        "shapes": [list(shape) for shape in args.shapes],
        "modes": args.modes,
        "case_count": len(args.dtypes) * len(args.shapes),
        "measurement_count": len(args.dtypes) * len(args.shapes) * len(args.modes),
    }


def _measure(profiler, fn):
    """Use the shared event timer, then measure allocation in a separate call."""
    _, median_ms, std_ms = profiler._time_kernel(fn)
    if not math.isfinite(median_ms) or median_ms <= 0:
        raise RuntimeError(f"invalid accelerator-event latency: {median_ms}")
    # Inputs and any retained backward graph are already part of the baseline.
    # fn returns None, so outputs do not remain live between invocations.
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


def _environment(device, args, triton_version, block_v, target: GPUTargetInfo):
    """Record this benchmark's settings and the profiler's CUDA or ROCm target."""
    props = torch.cuda.get_device_properties(device)
    is_rocm = target.backend == "rocm"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": triton_version,
        "backend": target.backend,
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
        "block_v": block_v,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "layout": "contiguous",
        "timer": (
            f"PerformanceProfiler {'HIP' if is_rocm else 'CUDA'} events; "
            "median and sample standard deviation"
        ),
        "operator": "softcapped_selected_logprob",
        "softcap": 30.0,
        "baseline": "eager NativeSoftcappedSelectedLogprobOp on the same GPU; no torch.compile",
        "candidate": "TritonSoftcappedSelectedLogprobOp with automatic forward selection",
        "cache_policy": "reused contiguous inputs, warm selector cache, no explicit eviction",
        "scope": "Public wrappers, including allocation and autograd dispatch; no graph capture",
        "backward": "Random FP32 upstream gradient; prebuilt retained graph; no forward timing",
        "memory": "Extra peak allocated MiB per call; excludes inputs and any prebuilt graph",
    }


def _markdown(report):
    env = report["environment"]
    lines = [
        "# Softcapped selected logprob benchmark",
        "",
        f"GPU: {env['gpu']}; PyTorch: {env['torch']}; Triton: {env['triton']}; "
        f"{env['backend']} runtime: {env['runtime']}.",
        f"Status: {'complete' if report['complete'] else 'incomplete'}; "
        f"{len(report['results'])}/{report['case_plan']['measurement_count']} measurements.",
        f"Warmup: {env['warmup']}; measured repetitions: {env['repeat']}; seed: {env['seed']}.",
        "",
        "Eager NativeSoftcappedSelectedLogprobOp vs TritonSoftcappedSelectedLogprobOp "
        "with automatic strategy selection; no torch.compile or CUDA Graph capture.",
        "Contiguous inputs are reused without explicit cache eviction. Median accelerator-event "
        "latency includes public wrapper allocation and autograd dispatch. Input generation, "
        "accuracy checks and JIT compilation are outside timing; the selector cache is warm. "
        "Small cases can be dominated by host dispatch gaps.",
        "Forward uses no_grad. Backward reuses a prebuilt graph; forward+backward builds a "
        "fresh graph per call. Extra peak allocation excludes inputs and prebuilt graphs.",
        "Speedup = eager PyTorch / Triton; values below 1 mean Triton was slower. Every "
        "reported case passed output and random-upstream gradient checks before timing. "
        "JSON also contains sample standard deviations and error bounds.",
        "",
        "| Input | Shape | Triton strategy | Mode | Native ms | Triton ms | Speedup | "
        "Native extra MiB | Triton extra MiB |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        native, triton = row["native"], row["triton"]
        lines.append(
            f"| {row['dtype']} | {'x'.join(map(str, row['shape']))} | "
            f"{row['triton_strategy']} | {row['mode']} | {native['median_ms']:.6f} | "
            f"{triton['median_ms']:.6f} | {row['speedup']:.2f}x | "
            f"{native['peak_extra_mib']:.2f} | {triton['peak_extra_mib']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument(
        "--shapes", nargs="+", type=_shape, default=list(map(_shape, DEFAULT_SHAPES))
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--repeat", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=415)
    parser.add_argument(
        "--list-cases", action="store_true", help="Print the plan without a GPU run"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/softcapped-selected-logprob")
    )
    return parser


def run_benchmark(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires an NVIDIA CUDA or AMD ROCm GPU; no CPU fallback is timed"
        )
    # Delay GPU imports so --help and --list-cases also work on CPU-only hosts.
    import triton

    from rl_engine.kernels.ops.triton.loss.softcapped_selected_logprob import (
        _BLOCK_V,
        TritonSoftcappedSelectedLogprobOp,
        select_softcapped_logprob_strategy,
        softcapped_logprob_device_key,
    )

    with torch.cuda.device(device):
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        native, candidate = NativeSoftcappedSelectedLogprobOp(), TritonSoftcappedSelectedLogprobOp()
        env = _environment(device, args, triton.__version__, _BLOCK_V, profiler.gpu_info)
        report = {
            "environment": env,
            "case_plan": _case_plan(args),
            "complete": False,
            "results": [],
        }
        _write_report(report, args.output_dir)
        for dtype_name in args.dtypes:
            for shape in args.shapes:
                print(f"Checking {dtype_name} {shape} before timing...", flush=True)
                generator = torch.Generator(device=device).manual_seed(args.seed)
                logits = (30.0 * torch.randn(shape, device=device, generator=generator)).to(
                    DTYPES[dtype_name]
                )
                ids = torch.randint(shape[1], (shape[0],), device=device, generator=generator)
                upstream = torch.randn(
                    shape[0], device=device, generator=generator, dtype=torch.float32
                )
                accuracy = _check_accuracy(native, candidate, logits, ids, upstream)
                # Descriptive metadata only: the timed Op still selects its strategy itself.
                strategy = select_softcapped_logprob_strategy(
                    softcapped_logprob_device_key(logits.device), logits.dtype, *shape
                ).value
                for mode in args.modes:
                    measurements = {}
                    for name, op in (("native", native), ("triton", candidate)):
                        fn = _make_workload(op, logits, ids, upstream, mode)
                        measurements[name] = _measure(profiler, fn)
                        del fn  # Free the retained graph before measuring the next backend.
                    row = {
                        "dtype": dtype_name,
                        "shape": list(shape),
                        "mode": mode,
                        "triton_strategy": strategy,
                        "accuracy": accuracy,
                        **measurements,
                        "speedup": measurements["native"]["median_ms"]
                        / measurements["triton"]["median_ms"],
                    }
                    report["results"].append(row)
                    # Retain completed cases on interruption; disk I/O is outside timing.
                    _write_report(report, args.output_dir)
                    print(
                        f"{dtype_name} {shape} {mode} ({strategy}): {row['speedup']:.2f}x",
                        flush=True,
                    )
                del logits, ids, upstream
        report["complete"] = True
        _write_report(report, args.output_dir)
    return report


def _write_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    pending = json_path.with_suffix(".json.tmp")
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(json_path)
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")


def main():
    args = build_arg_parser().parse_args()
    print(json.dumps(_case_plan(args), indent=2), flush=True)
    if args.list_cases:
        return
    report = run_benchmark(args)
    print(_markdown(report))
    print(f"Reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
