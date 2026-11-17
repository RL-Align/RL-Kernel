# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Measure Qwen3.6 partial RoPE latency and extra allocated memory on a real GPU.

Compare eager NativeQwen3_6PartialRoPEOp with the public Triton wrapper for Q/K shapes,
BF16/FP32, and forward/backward/combined calls. --dry-run only lists workloads;
it does not require Triton or a GPU and never emits fabricated measurements.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.profiler import PerformanceProfiler  # noqa: E402
from rl_engine.contracts.numerical import load_contract  # noqa: E402
from rl_engine.contracts.numerical import (  # noqa: E402
    resolve_tolerance,
    tolerance_contract_fingerprint,
)
from rl_engine.reference.rope.qwen3_6_partial_rope import NativeQwen3_6PartialRoPEOp  # noqa: E402
from rl_engine.validation.operators.operator_inputs import make_operator_inputs  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
MODES = ("forward", "backward", "forward_backward")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_arg_parser() -> argparse.ArgumentParser:
    """Expose workload sizes and repetitions; no device work happens while parsing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=_positive_int, default=[1, 32, 256, 2048, 8192])
    parser.add_argument("--heads", nargs="+", type=_positive_int, default=[2, 16])
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--repeat", type=_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=642)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen3-6-partial-rope"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _make_workload(op, inputs, grad_y, mode):
    """Return one public call; input creation and backward graph setup are untimed."""
    if mode == "forward":

        @torch.no_grad()
        def forward():
            op(**inputs)

        return forward
    leaf = inputs["x"].detach().requires_grad_(True)
    arguments = {**inputs, "x": leaf}
    if mode == "backward":
        y = op(**arguments)

        def backward():
            torch.autograd.grad(y, leaf, grad_y, retain_graph=True)

        return backward
    if mode == "forward_backward":

        def forward_backward():
            torch.autograd.grad(op(**arguments), leaf, grad_y)

        return forward_backward
    raise ValueError(f"unknown mode: {mode}")


def _check_accuracy(op, inputs, grad_y):
    """Gate both output and dx against FP32 math using unmodified contract rows."""
    leaf = inputs["x"].detach().requires_grad_(True)
    y = op(**{**inputs, "x": leaf})
    (dx,) = torch.autograd.grad(y, leaf, grad_y)
    gold_leaf = leaf.detach().float().requires_grad_(True)
    gold_y = NativeQwen3_6PartialRoPEOp().forward_fp32(**{**inputs, "x": gold_leaf})
    (gold_dx,) = torch.autograd.grad(gold_y, gold_leaf, grad_y.float())
    errors = {}
    for label, actual, gold, judgment in (
        ("output", y, gold_y, "forward_accuracy"),
        ("gradient", dx, gold_dx, "gradient_accuracy"),
    ):
        tolerance = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="elementwise", dtype=leaf.dtype
        )
        assert actual.shape == leaf.shape and actual.dtype == leaf.dtype
        torch.testing.assert_close(actual.float(), gold, atol=tolerance.atol, rtol=tolerance.rtol)
        errors[label] = {
            "max_abs_error": (actual.float() - gold).abs().max().item(),
            "atol": tolerance.atol,
            "rtol": tolerance.rtol,
        }
    return errors


def _measure_interleaved(profiler, functions):
    """Alternate providers per round; retain each sample and its execution position."""
    names = tuple(functions)
    samples = {name: [] for name in names}
    for index in range(profiler.warmup):
        for name in names if index % 2 == 0 else names[::-1]:
            functions[name]()
    profiler._sync()
    torch.cuda.empty_cache()
    for index in range(profiler.repeat):
        order = names if index % 2 == 0 else names[::-1]
        for position, name in enumerate(order):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            elapsed_us = start.elapsed_time(end) * 1000.0
            if not math.isfinite(elapsed_us) or elapsed_us <= 0:
                raise RuntimeError(f"invalid measured latency: {elapsed_us}")
            samples[name].append(
                {"round": index, "order_index": position, "latency_us": elapsed_us}
            )
    profiler._sync()
    return {
        name: {
            "median_us": statistics.median(sample["latency_us"] for sample in values),
            "std_us": (
                statistics.stdev(sample["latency_us"] for sample in values)
                if len(values) > 1
                else None
            ),
            "samples": values,
        }
        for name, values in samples.items()
    }


def _measure_peak_memory(profiler, fn):
    """Measure one provider separately, after reclaiming retired autograd objects."""
    fn()  # Warm the rebuilt workload/graph; this call is outside the measurement.
    profiler._sync()
    gc.collect()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(profiler.device)
    torch.cuda.reset_peak_memory_stats(profiler.device)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        fn()
        profiler._sync()
        peak = torch.cuda.max_memory_allocated(profiler.device) - baseline
    finally:
        if gc_enabled:
            gc.enable()
    gc.collect()
    profiler._sync()
    after = torch.cuda.memory_allocated(profiler.device)
    if after != baseline or peak <= 0:
        raise RuntimeError(
            f"unstable allocation probe: baseline={baseline}, after={after}, peak={peak}"
        )
    return peak / 1024**2


def _revision():
    """Best-effort metadata: extracted source distributions need not contain Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_benchmark(args):
    """Run accuracy gates before timing; reject CPU rather than timing a fallback."""
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("requires a real CUDA/ROCm GPU; use --dry-run to inspect workloads")
    import triton

    from rl_engine.backends.shared.triton.rope.qwen3_6_partial_rope import (
        TritonQwen3_6PartialRoPEOp,
    )

    source_paths = (
        "rl_engine/backends/shared/triton/rope/qwen3_6_partial_rope.py",
        "rl_engine/reference/rope/qwen3_6_partial_rope.py",
        "benchmarks/operators/rope/benchmark_qwen3_6_partial_rope.py",
        "benchmarks/common/profiler.py",
        "rl_engine/validation/operators/operator_inputs.py",
    )
    with torch.get_device_module(device).device(device):
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        report = {
            "environment": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "gpu": torch.cuda.get_device_name(device),
                "architecture": profiler.gpu_info.architecture,
                "total_memory_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
                "device": str(device),
                "backend": "rocm" if torch.version.hip else "cuda",
                "runtime": torch.version.hip or torch.version.cuda,
                "torch": torch.__version__,
                "triton": triton.__version__,
                "python": platform.python_version(),
                "git_commit": _revision(),
                "source_sha256": {
                    p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest()
                    for p in source_paths
                },
                "tolerance_sha256": tolerance_contract_fingerprint(),
                "warmup": args.warmup,
                "repeat": args.repeat,
                "seed": args.seed,
                "timer": "interleaved CUDA/HIP events, median and sample SD; no CUDA Graph",
                "provider_order": "alternating forward/reverse order per warmup/sample round",
                "baseline": "eager NativeQwen3_6PartialRoPEOp, explicit FP32 VJP; no torch.compile",
                "scope": "public wrappers; allocations/autograd included; JIT/input setup excluded",
                "backward": "random input-dtype dy; retained forward graph excluded from timing",
                "memory": (
                    "extra peak allocated MiB from a post-GC baseline; "
                    "each provider measured separately; excludes inputs and its prebuilt graph"
                ),
                "tables": "FP32 [T+11,32], positions 11..T+10; generated once outside timing",
            },
            "results": [],
        }
        for dtype_name in args.dtypes:
            for heads in args.heads:
                for tokens in args.tokens:
                    workload = argparse.Namespace(
                        batch=1,
                        seq=tokens,
                        n_heads=heads,
                        input_mode="random",
                        seed=args.seed,
                        position_offset=11,
                    )
                    inputs = make_operator_inputs(
                        "qwen3_6_partial_rope", workload, DTYPES[dtype_name], device
                    )
                    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
                    grad_y = torch.randn(
                        inputs["x"].shape,
                        generator=generator,
                        device=device,
                        dtype=DTYPES[dtype_name],
                    )
                    providers = {
                        "pytorch": NativeQwen3_6PartialRoPEOp(),
                        "triton": TritonQwen3_6PartialRoPEOp(),
                    }
                    accuracy = {
                        name: _check_accuracy(op, inputs, grad_y) for name, op in providers.items()
                    }
                    for mode in args.modes:
                        functions = {
                            name: _make_workload(op, inputs, grad_y, mode)
                            for name, op in providers.items()
                        }
                        measurements = _measure_interleaved(profiler, functions)
                        del functions
                        gc.collect()  # Release both timing graphs before isolated memory probes.
                        for name, measurement in measurements.items():
                            fn = _make_workload(providers[name], inputs, grad_y, mode)
                            measurement["peak_extra_mib"] = _measure_peak_memory(profiler, fn)
                            del fn
                            row = {
                                "provider": name,
                                "dtype": dtype_name,
                                "shape": list(inputs["x"].shape),
                                "mode": mode,
                                "accuracy": accuracy[name],
                                **measurement,
                            }
                            report["results"].append(row)
                            print(json.dumps(row), flush=True)
    return report


def _markdown(report):
    """Render absolute times and allocations; raw JSON retains noise/error details."""
    env = report["environment"]
    lines = [
        "# Qwen3.6-35B-A3B partial RoPE benchmark",
        "",
        f"{env['gpu']}; PyTorch {env['torch']}; Triton {env['triton']}; {env['runtime']}.",
        f"Timer: {env['timer']}. Warmup {env['warmup']}; repeat {env['repeat']}.",
        "All rows passed forward/dx accuracy checks before measurement.",
        "",
        "| Shape [T,H,256] | Dtype | Mode | Provider | Median us | Std us | Extra MiB |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in report["results"]:
        std = "n/a" if r["std_us"] is None else f"{r['std_us']:.3f}"
        lines.append(
            f"| {r['shape']} | {r['dtype']} | {r['mode']} | {r['provider']} | "
            f"{r['median_us']:.3f} | {std} | {r['peak_extra_mib']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    """List the matrix or write measured JSON/Markdown under the requested directory."""
    args = build_arg_parser().parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "shapes": [[t, h, 256] for h in args.heads for t in args.tokens],
                    "dtypes": args.dtypes,
                    "modes": args.modes,
                    "providers": ["pytorch", "triton"],
                    "measured": False,
                },
                indent=2,
            )
        )
        return
    report = run_benchmark(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
