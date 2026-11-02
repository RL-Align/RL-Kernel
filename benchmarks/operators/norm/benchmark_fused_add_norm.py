# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare native, Triton and CUDA fused add-norm on Muse-Glimmer-30B rows.

RFC #501 [F1]: post-norm, residual add, pre-norm as one operator. Muse-Glimmer-30B
has hidden size 6656, so the contract here runs over ``(rows, 6656)``; the row
counts span a decode step, a small batch, and a prefill-sized chunk.

All three sides are timed through the public ``forward``, which returns the input
dtype everywhere, so the timed workloads write the same number of bytes. The
accuracy gate runs on that same ``forward`` path and compares both outputs and all
four input gradients against the native ``forward_fp32`` gold at the contract's
``reduction`` rows before anything is timed.

The Triton and CUDA backwards are one kernel each plus the FP32 row fold for the
two weight gradients; the native backward is whatever autograd records over two
norms and an add. The backward speedup compares those complete paths.

Reporting is shared with ``benchmark_swiglu.py``: a clock-settling phase before
each measurement, repeated passes with a 95% interval and a verdict only when the
interval excludes parity, a generated spread note, and a determinism check over
memory and accuracy. The reasons for each are recorded there and in the commit
history; this file only states what applies to this operator.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profiler import PerformanceProfiler  # noqa: E402
from rl_engine.contracts.numerical import resolve_tolerance_support  # noqa: E402
from rl_engine.kernels.gtest.tolerance import (  # noqa: E402
    load_contract,
    resolve_tolerance,
    tolerance_contract_fingerprint,
)
from rl_engine.reference.norm.fused_add_norm import (  # noqa: E402
    HIDDEN_SIZE,
    POST_NORM_EPS,
    RMS_NORM_EPS,
    NativeFusedAddNormOp,
)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
MODES = ("forward", "backward", "forward_backward")
DEFAULT_SHAPES = ("1x6656", "8x6656", "64x6656", "512x6656")
EPS = {"post_eps": POST_NORM_EPS, "pre_eps": RMS_NORM_EPS}
OUTPUT_NAMES = ("residual_out", "pre_normed", "d_hidden", "d_residual", "d_post_w", "d_pre_w")
OUTPUT_JUDGMENTS = ("forward_accuracy",) * 2 + ("gradient_accuracy",) * 4


def _shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value.lower().split("x"))
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError
        return shape
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "shapes must contain positive dimensions, e.g. 8x6656"
        ) from exc


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


# --------------------------------------------------------------------------- #
# Workloads and the accuracy gate
# --------------------------------------------------------------------------- #


def _make_workload(op, hidden, residual, post_w, pre_w, g_res, g_pre, mode):
    """Prepare inputs and graphs outside timing; return a callable that returns None.

    Both outputs are live in the model, so the backward drives both cotangents.
    """
    if mode == "forward":
        h, r = hidden.detach(), residual.detach()

        @torch.no_grad()
        def forward():
            op(h, r, post_w, pre_w, **EPS)

        return forward

    leaves = [hidden.detach().requires_grad_(True), residual.detach().requires_grad_(True)]
    weights = [post_w.detach().requires_grad_(True), pre_w.detach().requires_grad_(True)]
    inputs = leaves + weights

    if mode == "backward":
        outs = op(*leaves, *weights, **EPS)

        def backward():
            torch.autograd.grad(outs, inputs, grad_outputs=(g_res, g_pre), retain_graph=True)

        return backward

    if mode == "forward_backward":

        def forward_backward():
            torch.autograd.grad(op(*leaves, *weights, **EPS), inputs, grad_outputs=(g_res, g_pre))

        return forward_backward

    raise ValueError(f"unknown mode: {mode}")


def _one_ulp(reference: torch.Tensor) -> float:
    peak = float(reference.abs().max())
    if peak == 0.0:
        return 0.0
    hi = torch.nextafter(
        torch.tensor(peak, dtype=reference.dtype), torch.tensor(float("inf"), dtype=reference.dtype)
    )
    return float(hi - peak)


def _check_accuracy(native, candidate, hidden, residual, post_w, pre_w, g_res, g_pre):
    """Gate the timed ``forward`` of a candidate against the FP32 gold.

    Both outputs and all four input gradients, each at the contract's ``reduction``
    row for its own dtype. FP16 is ``optional`` in the contract and is held to one
    ULP of the reference instead of the declared floor, the same rule and for the
    same measured reason as the test files: the kernels land on the far side of an
    FP16 tie in the residual add on a handful of elements.
    """

    def evaluate(fn):
        leaves = [hidden.detach().requires_grad_(True), residual.detach().requires_grad_(True)]
        weights = [post_w.detach().requires_grad_(True), pre_w.detach().requires_grad_(True)]
        outs = fn(*leaves, *weights, **EPS)
        grads = torch.autograd.grad(outs, leaves + weights, grad_outputs=(g_res, g_pre))
        return [o.detach() for o in outs] + list(grads)

    expected = evaluate(native.forward_fp32)
    actual = evaluate(candidate.forward)
    contract = load_contract()
    errors = {}
    for name, judgment, reference, result in zip(
        OUTPUT_NAMES, OUTPUT_JUDGMENTS, expected, actual, strict=True
    ):
        tolerance = resolve_tolerance(
            contract, judgment=judgment, op_class="reduction", dtype=result.dtype
        )
        atol = tolerance.atol
        support = resolve_tolerance_support(
            contract, judgment=judgment, op_class="reduction", dtype=result.dtype
        )
        if support.status != "applicable":
            atol = max(atol, _one_ulp(reference.to(result.dtype)))
        torch.testing.assert_close(
            result.to(torch.float32),
            reference.to(torch.float32),
            atol=atol,
            rtol=tolerance.rtol,
            msg=lambda built, name=name: f"{name}: {built}",
        )
        errors[name] = {
            "max_abs_error": (result.to(torch.float32) - reference.to(torch.float32))
            .abs()
            .max()
            .item(),
            "atol": atol,
            "rtol": tolerance.rtol,
        }
    return errors


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #

# A row-norm call at this width runs for tens of microseconds, while an idle H100
# sits at 345 MHz against a 1980 MHz maximum and only ramps under sustained load.
# Ten warmup iterations do not lift the clock, so a median taken right after them
# lands at an arbitrary point on the ramp. Drive the device for a fixed wall-clock
# span first so the measurement happens at a settled clock.
CLOCK_SETTLE_SECONDS = 1.0


def _settle_clocks(profiler, fn):
    deadline = time.perf_counter() + CLOCK_SETTLE_SECONDS
    while time.perf_counter() < deadline:
        for _ in range(32):
            fn()
        profiler._sync()


def _measure(profiler, fn):
    _settle_clocks(profiler, fn)
    _, median_ms, std_ms = profiler._time_kernel(fn)
    if not math.isfinite(median_ms) or median_ms <= 0:
        raise RuntimeError(f"invalid accelerator-event latency: {median_ms}")
    # Collect before reading the baseline. An autograd graph from an earlier cell
    # can sit in a reference cycle, get freed inside fn(), and leave the peak below
    # where the baseline said it started.
    gc.collect()
    profiler._sync()
    baseline = torch.cuda.memory_allocated(profiler.device)
    torch.cuda.reset_peak_memory_stats(profiler.device)
    fn()
    profiler._sync()
    extra = torch.cuda.max_memory_allocated(profiler.device) - baseline
    if extra < 0:
        raise RuntimeError(f"peak allocation fell below the live baseline: {extra} bytes")
    return {"median_ms": median_ms, "std_ms": std_ms, "peak_extra_gib": extra / 1024**3}


def _command_output(command):
    try:
        return subprocess.check_output(
            command, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _environment(device, args, triton_version, shapes, candidates):
    target = PerformanceProfiler(device=device).gpu_info
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
        "compute_capability": target.compute_capability,
        "device": str(device),
        "total_memory_gib": round(props.total_memory / 1024**3, 2),
        "multiprocessors": props.multi_processor_count,
        "shapes": ["x".join(map(str, s)) for s in shapes],
        "widths_run": sorted({s[-1] for s in shapes}),
        "muse_hidden_size": HIDDEN_SIZE,
        "all_shapes_at_muse_width": all(s[-1] == HIDDEN_SIZE for s in shapes),
        "candidates": list(candidates),
        "post_eps": POST_NORM_EPS,
        "pre_eps": RMS_NORM_EPS,
        "warmup": args.warmup,
        "clock_settle_seconds": CLOCK_SETTLE_SECONDS,
        "repeat": args.repeat,
        "runs": args.runs,
        "seed": args.seed,
        "tolerance_contract_sha256": tolerance_contract_fingerprint(),
        "baseline": (
            "eager NativeFusedAddNormOp.forward on the same GPU; every side emits the input dtype"
        ),
        "accuracy": (
            "candidate forward vs native forward_fp32 gold, reduction rows, "
            "both outputs and four gradients"
        ),
        "scope": "Public wrappers, including allocation and autograd dispatch; no graph capture",
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

NOISY_STD_FRACTION = 0.25


def _sides(row):
    return [name for name in ("native", "triton", "cuda") if name in row]


def _noise_note(report):
    spreads = [
        row[side]["std_ms"] / row[side]["median_ms"]
        for row in report["results"]
        for side in _sides(row)
        if row[side]["median_ms"] > 0 and row[side]["std_ms"] is not None
    ]
    if not spreads:
        return (
            "Spread: unavailable. Every measurement here is a single repetition "
            "(--repeat 1), so there is a median but no deviation to report; the "
            "latency cells below print the one sample bare."
        )
    noisy = sum(1 for s in spreads if s > NOISY_STD_FRACTION)
    heavy_tail = sum(1 for s in spreads if s > 1.0)
    worst = max(spreads)
    repeat = report["environment"]["repeat"]
    return (
        f"Spread: {noisy} of {len(spreads)} timed samples have a standard deviation "
        f"above {NOISY_STD_FRACTION:.0%} of their median, the worst at {worst:.0%}, and "
        f"{heavy_tail} exceed 100%. Latencies are positive and the median is central, "
        f"so a standard deviation above 100% of the median means a handful of stalled "
        f"iterations out of {repeat}, not broad jitter: each median stays usable, but "
        f"do not read the standard deviation as a confidence interval on it. These are "
        f"launch-overhead-bound sizes, so treat the table as shape qualification rather "
        f"than a performance claim."
    )


_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
}  # fmt: skip


def _mean_ci95(values):
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, None, None
    sd = statistics.stdev(values)
    t = _T95.get(len(values) - 1, 1.96)
    half = t * sd / math.sqrt(len(values))
    return mean, mean - half, mean + half


def _verdict(low, high):
    if low is None or high is None:
        return "single pass, no interval"
    if low > 1.0:
        return "faster"
    if high < 1.0:
        return "slower"
    return "indeterminate"


def _reproducibility(passes):
    """Speedup per (dtype, mode, candidate) across passes, one sample per pass."""
    groups = {}
    for rows in passes:
        per_group = {}
        for row in rows:
            for cand in ("triton", "cuda"):
                key = f"speedup_{cand}"
                if key in row:
                    per_group.setdefault((row["dtype"], row["mode"], cand), []).append(row[key])
        for key, values in per_group.items():
            groups.setdefault(key, []).append(statistics.mean(values))

    summary = []
    for (dtype, mode, cand), per_pass in groups.items():
        mean, low, high = _mean_ci95(per_pass)
        summary.append(
            {
                "dtype": dtype,
                "mode": mode,
                "candidate": cand,
                "passes": len(per_pass),
                "mean_speedup": mean,
                "ci95_low": low,
                "ci95_high": high,
                "min_speedup": min(per_pass),
                "max_speedup": max(per_pass),
                "verdict": _verdict(low, high),
            }
        )
    return summary


_MEMORY_DECIMALS = 4


def _determinism(passes):
    if len(passes) < 2:
        return {
            "compared_rows": 0,
            "memory_mismatches": 0,
            "accuracy_mismatches": 0,
            "max_memory_deviation_gib": 0.0,
        }

    def key(row):
        return (row["dtype"], tuple(row["shape"]), row["mode"])

    base = {key(row): row for row in passes[0]}
    memory = accuracy = compared = 0
    worst = 0.0
    for rows in passes[1:]:
        for row in rows:
            reference = base[key(row)]
            compared += 1
            for side in _sides(row):
                a = row[side]["peak_extra_gib"]
                b = reference[side]["peak_extra_gib"]
                worst = max(worst, abs(a - b))
                if round(a, _MEMORY_DECIMALS) != round(b, _MEMORY_DECIMALS):
                    memory += 1
                    break
            if row["accuracy"] != reference["accuracy"]:
                accuracy += 1
    return {
        "compared_rows": compared,
        "memory_mismatches": memory,
        "accuracy_mismatches": accuracy,
        "max_memory_deviation_gib": worst,
    }


def _interval(item):
    if item["ci95_low"] is None or item["ci95_high"] is None:
        return "n/a"
    return f"[{item['ci95_low']:.3f}, {item['ci95_high']:.3f}]"


def _ms(side):
    if side is None:
        return "n/a"
    if side["std_ms"] is None:
        return f"{side['median_ms']:.6f}"
    return f"{side['median_ms']:.6f} ± {side['std_ms']:.6f}"


def _gib(side):
    return "n/a" if side is None else f"{side['peak_extra_gib']:.4f}"


def _speedup(row, cand):
    return f"{row[f'speedup_{cand}']:.2f}x" if f"speedup_{cand}" in row else "n/a"


def _title(env):
    if env["all_shapes_at_muse_width"]:
        return "# Fused add-norm on Muse-Glimmer-30B rows"
    widths = ", ".join(str(w) for w in env["widths_run"])
    return (
        f"# Fused add-norm at widths {widths} (not the Muse-Glimmer-30B hidden size "
        f"{env['muse_hidden_size']})"
    )


def _reproducibility_lines(report):
    summary = report.get("reproducibility") or []
    passes = summary[0]["passes"] if summary else 1
    if passes < 2:
        return [
            "Single pass over the matrix. The speedup columns below are one measurement "
            "each, not a result: re-run with `--runs 10` for an interval.",
            "",
        ]
    det = report.get("determinism") or {}
    lines = [
        f"### Reproducibility across {passes} independent passes",
        "",
        "Each pass re-runs the whole matrix on the same commit and GPU. One sample per "
        "pass, so the interval below describes how much the reported figure moves "
        "between runs. A verdict is only given when the interval excludes parity.",
        "",
        "| Input | Mode | Candidate | Mean speedup | 95% CI | Min | Max | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    order = {"fp16": 0, "bf16": 1, "fp32": 2}
    modes = {"forward": 0, "backward": 1, "forward_backward": 2}
    cands = {"triton": 0, "cuda": 1}
    for item in sorted(
        summary,
        key=lambda d: (
            order.get(d["dtype"], 9),
            modes.get(d["mode"], 9),
            cands.get(d["candidate"], 9),
        ),
    ):
        lines.append(
            f"| {item['dtype']} | {item['mode']} | {item['candidate']} | "
            f"{item['mean_speedup']:.3f}x | {_interval(item)} | "
            f"{item['min_speedup']:.3f}x | {item['max_speedup']:.3f}x | {item['verdict']} |"
        )
    if det.get("compared_rows"):
        lines += [
            "",
            f"Memory and accuracy hold still: across {det['compared_rows']} row "
            f"comparisons between passes, {det['memory_mismatches']} peak-memory and "
            f"{det['accuracy_mismatches']} accuracy figures differed at the printed "
            f"precision, with a largest raw memory deviation of "
            f"{det['max_memory_deviation_gib']:.2e} GiB. The detail table below is the "
            f"last pass.",
        ]
    lines += [
        "",
        "These intervals are within one invocation. Separate invocations shift the "
        "whole measurement by a few percent on unchanged code, so treat a verdict that "
        "rests on a few percent as provisional until it survives a second session.",
    ]
    return lines + [""]


def _markdown(report):
    env = report["environment"]
    lines = [
        _title(env),
        "",
        f"GPU: {env['gpu']}; PyTorch: {env['torch']}; Triton: {env['triton']}; "
        f"{env['backend']} runtime: {env['runtime']}.",
        f"Commit: `{env['git_commit']}`; tracked changes: "
        f"`{env['git_tracked_changes'] or 'none'}`.",
        f"Candidates: {', '.join(env['candidates'])}. Warmup: {env['warmup']} iterations "
        f"after a {env['clock_settle_seconds']}s clock-settling phase; measured "
        f"repetitions: {env['repeat']}.",
        "",
        "Every side is timed through the public `forward`, which returns the input dtype. "
        "Median accelerator-event latency for public wrappers, including allocation and "
        "autograd dispatch. The Triton and CUDA backwards are one kernel each plus the FP32 "
        "row fold for the two weight gradients; the native backward is autograd over two "
        "norms and an add. The backward speedup compares complete paths.",
        "Speedup = eager PyTorch / candidate; values below 1 mean the candidate was slower. "
        "All reported cases passed both outputs and all four gradient tolerance checks "
        "first, at the contract's reduction rows.",
        "",
        _noise_note(report),
        "",
        *_reproducibility_lines(report),
        "| Input | Shape | Mode | Native ms | Triton ms | CUDA ms | vs Triton | vs CUDA | "
        "Native GiB | Triton GiB | CUDA GiB |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        n, t, c = row["native"], row.get("triton"), row.get("cuda")
        lines.append(
            f"| {row['dtype']} | {'x'.join(map(str, row['shape']))} | {row['mode']} | "
            f"{_ms(n)} | {_ms(t)} | {_ms(c)} | "
            f"{_speedup(row, 'triton')} | {_speedup(row, 'cuda')} | "
            f"{_gib(n)} | {_gib(t)} | {_gib(c)} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument(
        "--shapes", nargs="+", type=_shape, default=list(map(_shape, DEFAULT_SHAPES))
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=("triton", "cuda"),
        default=None,
        help="defaults to every candidate the build makes available",
    )
    parser.add_argument("--warmup", type=_positive_int, default=10)
    parser.add_argument("--repeat", type=_positive_int, default=50)
    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=3,
        help=(
            "independent passes over the whole matrix; one speedup sample each. "
            "Use 10 for a figure worth quoting."
        ),
    )
    parser.add_argument("--seed", type=int, default=415)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/benchmarks/fused_add_norm")
    )
    return parser


def _resolve_candidates(requested):
    """Build the candidate ops, skipping CUDA when the extension lacks the kernels."""
    from rl_engine.backends.cuda.norm.fused_add_norm import (
        CudaFusedAddNormOp,
        cuda_fused_add_norm_available,
    )
    from rl_engine.backends.shared.triton.norm.fused_add_norm import TritonFusedAddNormOp

    available = {"triton": TritonFusedAddNormOp}
    if cuda_fused_add_norm_available():
        available["cuda"] = CudaFusedAddNormOp
    names = requested or list(available)
    missing = [n for n in names if n not in available]
    if missing:
        raise RuntimeError(
            f"requested candidates unavailable in this build: {missing}; "
            f"available: {list(available)}"
        )
    return {n: available[n]() for n in names}


def _sweep(args, profiler, native, candidates, device):
    rows = []
    for dtype_name in args.dtypes:
        dtype = DTYPES[dtype_name]
        for shape in args.shapes:
            width = shape[-1]
            gen = torch.Generator(device=device).manual_seed(args.seed)

            def rand(s):
                return torch.randn(s, device=device, generator=gen, dtype=torch.float32).to(dtype)

            hidden, residual = rand(shape), rand(shape)
            post_w, pre_w = rand((width,)), rand((width,))
            g_res, g_pre = rand(shape), rand(shape)

            accuracy = {
                name: _check_accuracy(native, op, hidden, residual, post_w, pre_w, g_res, g_pre)
                for name, op in candidates.items()
            }
            for mode in MODES:
                measurements = {}
                for name, op in [("native", native), *candidates.items()]:
                    fn = _make_workload(op, hidden, residual, post_w, pre_w, g_res, g_pre, mode)
                    measurements[name] = _measure(profiler, fn)
                    del fn
                row = {
                    "dtype": dtype_name,
                    "shape": list(shape),
                    "numel": hidden.numel(),
                    "mode": mode,
                    "accuracy": accuracy,
                    **measurements,
                }
                for name in candidates:
                    row[f"speedup_{name}"] = (
                        measurements["native"]["median_ms"] / measurements[name]["median_ms"]
                    )
                rows.append(row)
                print(
                    f"{dtype_name} {shape} {mode}: "
                    + ", ".join(f"{n} {row[f'speedup_{n}']:.2f}x" for n in candidates),
                    flush=True,
                )
            del hidden, residual, post_w, pre_w, g_res, g_pre
    return rows


def run_benchmark(args):
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "This benchmark requires an NVIDIA CUDA or AMD ROCm GPU; no CPU fallback is timed"
        )
    import triton

    native = NativeFusedAddNormOp()
    candidates = _resolve_candidates(args.candidates)
    with torch.cuda.device(device):
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        report = {
            "environment": _environment(device, args, triton.__version__, args.shapes, candidates),
            "results": [],
        }
        passes = []
        for index in range(args.runs):
            passes.append(_sweep(args, profiler, native, candidates, device))
            print(f"--- pass {index + 1}/{args.runs} complete", flush=True)
        report["results"] = passes[-1]
        report["reproducibility"] = _reproducibility(passes)
        report["determinism"] = _determinism(passes)
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
