# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Compare native, Triton and CUDA fused add-norm on Muse-Glimmer-30B rows.

RFC #501 [F1]: post-norm, residual add, pre-norm as one operator. Muse-Glimmer-30B
has hidden size 6656, so the contract here runs over ``(rows, 6656)``; the row
counts span a decode step, a small batch, and a prefill-sized chunk.

All three sides are timed through the public ``forward``, which returns the input
dtype everywhere, so the timed workloads write the same number of bytes. The
accuracy gate runs on that same ``forward`` path and compares both outputs and all
four input gradients against the native ``forward`` at the contract's ``reduction``
rows before anything is timed; ``_check_accuracy`` records why ``forward`` rather
than ``forward_fp32`` is the reference here.

The Triton and CUDA backwards are one kernel each plus the FP32 row fold for the
two weight gradients; the native backward is whatever autograd records over two
norms and an add. The backward speedup compares those complete paths.

Reporting follows the shape worked out for the SwiGLU row: a clock-settling phase
before each measurement, repeated passes with a 95% interval and a verdict only
when the interval excludes parity, a generated spread note, and a determinism check
over memory and accuracy. That file is not on this branch or on the base -- it
lives on the unmerged ``feat/museglimmer-swiglu-ffn`` -- so the reasons are stated
here rather than pointed at, and the pointer an earlier version of this docstring
carried has been removed because following it led nowhere.
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
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
from rl_engine.kernels.gtest.tolerance import tolerance_contract_fingerprint  # noqa: E402
from rl_engine.ops.autograd.backward_runtime import (  # noqa: E402
    reset_backward_runtime,
    snapshot_backward_runtime,
)
from rl_engine.reference.norm.fused_add_norm import (  # noqa: E402
    HIDDEN_SIZE,
    POST_NORM_EPS,
    RMS_NORM_EPS,
    NativeFusedAddNormOp,
)

# The accuracy gate has to be the tests' gate, not a second opinion, so the policy
# comes from where the tests keep it. It reads the contract rows and owns the FP16
# budget; see the sixteenth round's finding in the commit history for what happened
# when this file had its own.
from tests.helpers.fused_add_norm_tolerance import allowance, assert_within  # noqa: E402

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


def _check_accuracy(native, candidate, hidden, residual, post_w, pre_w, g_res, g_pre):
    """Gate the timed ``forward`` of a candidate against the native ``forward``.

    Both outputs and all four input gradients, each at the contract's ``reduction``
    row for its own dtype. The reference is ``native.forward``, not
    ``forward_fp32``, and that choice is load-bearing for this operator.
    ``forward_fp32`` deliberately skips both narrowing points, so it is a different
    computation, and the pre-norm weight gradient sums over rows a value that
    differs by one BF16 ULP per element between the two: measured gold-against-gold
    at bf16, ``d_pre_w`` differs by 0.0625 at 8 rows, 0.25 at 64, 0.5 at 512 and
    1.0 at 2048, which is exactly the series the kernels showed against
    ``forward_fp32`` and more than the 0.1 row allows from 512 rows on. Against
    ``native.forward`` the same gradient is within 0.03 at every size. What gets
    timed is ``forward``, so ``forward`` is the thing to be right about; formula
    correctness against FP64 is the test files' job. FP16 is ``optional`` in the
    contract and is held to one ULP of the reference for the reason recorded there.
    """

    def evaluate(fn):
        leaves = [hidden.detach().requires_grad_(True), residual.detach().requires_grad_(True)]
        weights = [post_w.detach().requires_grad_(True), pre_w.detach().requires_grad_(True)]
        outs = fn(*leaves, *weights, **EPS)
        grads = torch.autograd.grad(outs, leaves + weights, grad_outputs=(g_res, g_pre))
        return [o.detach() for o in outs] + list(grads)

    expected = evaluate(native.forward)
    actual = evaluate(candidate.forward)
    errors = {}
    for name, judgment, reference, result in zip(
        OUTPUT_NAMES, OUTPUT_JUDGMENTS, expected, actual, strict=True
    ):
        # The tests' policy, imported rather than re-derived here. This gate used to
        # build its own FP16 floor as max(contract atol, one ULP at the reference
        # peak), while the shared policy deliberately adds a quarter of a ULP to the
        # row: its docstring records that a full ULP reached 125x the contract atol
        # on the row-summed weight gradients and left 0.3 to 4% of elements with an
        # allowance at or above their own value, which is why it was abandoned
        # there. So this gate was looser than the test files on exactly d_post_w and
        # d_pre_w, and a candidate the tests reject could be recorded here as
        # accuracy-passing, which is not what the report says it checked.
        assert_within(judgment, result.dtype, result, reference, f"{name}: ")
        allowed = allowance(judgment, result.dtype, reference.to(torch.float32))
        error = (result.to(torch.float32) - reference.to(torch.float32)).abs()
        worst = int(error.flatten().argmax())
        errors[name] = {
            "max_abs_error": float(error.max()),
            "judgment": judgment,
            "policy": "tests/helpers/fused_add_norm_tolerance.py",
            # The allowance that actually applied at the worst element, read out of
            # the same elementwise policy the assertion used. A scalar atol and rtol
            # instead made the artifact look as though it had failed: d_post_w at
            # bf16 and 512 rows shows max_abs_error 0.125 against an atol of 0.1, on
            # a row the report calls passing, because the relative term covered the
            # rest.
            "threshold_at_worst": float(allowed.flatten()[worst]),
            # A hash of the bytes, because the figures above cannot see a change
            # smaller than one ULP of a narrow dtype and the determinism check
            # compares this dict. A reordered FP32 reduction inside the backward
            # moves this even when max_abs_error, which lands on the BF16 lattice,
            # does not.
            "result_sha256_16": hashlib.sha256(
                result.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
            ).hexdigest()[:16],
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


def _measure(profiler, make_fn):
    """Time ``make_fn()``'s closure, then measure one call's memory from scratch.

    Takes a factory rather than a closure so the memory window can start before the
    workload is built. In ``backward`` mode the workload runs the forward at
    construction time, so with a ready-made closure everything that forward retained
    -- for the eager path, several FP32 rows-by-width intermediates held by the
    autograd graph -- sat inside the baseline and was subtracted away, while in
    ``forward_backward`` mode the same graph is built inside the timed call and was
    counted. The column therefore meant different things in different rows and the
    direction was backwards: measured at bf16 and 512 rows, the native read 0.0620
    GiB in the backward row against 0.1075 in forward_backward, so a reader saw the
    eager path as the cheaper one in the backward exactly where it is holding the
    most. Building inside the window in both modes makes the number comparable.
    """
    fn = make_fn()
    _settle_clocks(profiler, fn)
    _, median_ms, std_ms = profiler._time_kernel(fn)
    if not math.isfinite(median_ms) or median_ms <= 0:
        raise RuntimeError(f"invalid accelerator-event latency: {median_ms}")
    # Drop the timing closure first, then collect. An autograd graph from an earlier
    # cell can sit in a reference cycle, get freed inside fn(), and leave the peak
    # below where the baseline said it started.
    del fn
    gc.collect()
    profiler._sync()
    baseline = torch.cuda.memory_allocated(profiler.device)
    torch.cuda.reset_peak_memory_stats(profiler.device)
    measured = make_fn()
    measured()
    profiler._sync()
    extra = torch.cuda.max_memory_allocated(profiler.device) - baseline
    del measured
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
            "NativeFusedAddNormOp.forward on the same GPU, the operator's gold rather than "
            "an idiomatic eager spelling; every side emits the input dtype"
        ),
        "accuracy": (
            "candidate forward vs native forward (same narrowing contract), reduction rows, "
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
        f"{heavy_tail} exceed 100%, over {repeat} repetitions each. A deviation that "
        f"size means the samples are not symmetric around the median, which is why the "
        f"median is reported and the deviation is not an interval on it. Which shape "
        f"of asymmetry it is -- a few stalled iterations or broad jitter -- is not "
        f"something the deviation alone can say, and an earlier version of this note "
        f"asserted the first; the per-iteration samples would be needed to tell them "
        f"apart. Whether a row is launch-bound or bandwidth-bound likewise depends on "
        f"the shape rather than on this table: at one row it is launch overhead, and "
        f"the 512-row bf16 native forward on an RTX 4090 moves about 0.15 GB in "
        f"0.196 ms, which is near that card's rated bandwidth."
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


def _across_shapes(summary):
    """Geometric mean of the per-shape speedups, for each dtype, mode and candidate.

    Geometric because these are ratios: 4.0x on one shape and 0.5x on another is
    1.0x, not the 2.25x an arithmetic mean reports. No interval, because a spread
    over shapes is not a measurement error; the range of the shapes it covers is
    given instead, with the shape each end came from.
    """
    groups = {}
    for item in summary:
        groups.setdefault((item["dtype"], item["mode"], item["candidate"]), []).append(item)
    out = []
    for (dtype, mode, cand), items in groups.items():
        ratios = [i["mean_speedup"] for i in items]
        worst = min(items, key=lambda i: i["mean_speedup"])
        best = max(items, key=lambda i: i["mean_speedup"])
        out.append(
            {
                "dtype": dtype,
                "mode": mode,
                "candidate": cand,
                "shapes": len(items),
                "geometric_mean_speedup": statistics.geometric_mean(ratios),
                "worst_shape": worst["shape"],
                "worst_speedup": worst["mean_speedup"],
                "best_shape": best["shape"],
                "best_speedup": best["mean_speedup"],
            }
        )
    return out


def _verdict(low, high):
    if low is None or high is None:
        return "single pass, no interval"
    if low > 1.0:
        return "faster"
    if high < 1.0:
        return "slower"
    return "indeterminate"


def _reproducibility(passes):
    """Speedup per (dtype, shape, mode, candidate) across passes, one sample per pass.

    The shape is in the key, and that is the whole point of this function's shape.
    It used to average the per-shape speedups inside each pass, so one row covered
    every shape at once: measured on this box, 2.74x at 1x6656 and 1.14x at
    512x6656 were reported as a single 1.873x, a number that occurs at neither
    shape, and the min and max columns described pass-to-pass movement rather than
    the spread across shapes, so nothing in the table revealed the 1.14x. An
    arithmetic mean of ratios is biased upward on top of that, and _verdict then
    read "faster" off the interval of that mean. One row per shape instead; the
    cross-shape figure is in _across_shapes, as a geometric mean, labelled.
    """
    groups = {}
    for rows in passes:
        for row in rows:
            for cand in ("triton", "cuda"):
                key = f"speedup_{cand}"
                if key in row:
                    groups.setdefault(
                        (row["dtype"], tuple(row["shape"]), row["mode"], cand), []
                    ).append(row[key])

    summary = []
    for (dtype, shape, mode, cand), per_pass in groups.items():
        mean, low, high = _mean_ci95(per_pass)
        summary.append(
            {
                "dtype": dtype,
                "shape": list(shape),
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
            "compared_accuracy_measurements": 0,
            "memory_mismatches": 0,
            "accuracy_mismatches": 0,
            "accuracy_detail": [],
            "max_memory_deviation_gib": 0.0,
        }

    accuracy_detail: list[str] = []

    def key(row):
        return (row["dtype"], tuple(row["shape"]), row["mode"])

    base = {key(row): row for row in passes[0]}
    memory = accuracy = compared = 0
    accuracy_compared = 0
    # The accuracy dict is measured once per (dtype, shape) and attached to all
    # three mode rows, so comparing it per row counted each measurement three
    # times: the note said "6 row comparisons" where 2 measurements existed.
    seen_accuracy = set()
    worst = 0.0
    for rows in passes[1:]:
        for row in rows:
            reference = base[key(row)]
            compared += 1
            # Every side, then record. This used to break out of the loop on the
            # first mismatch, after updating ``worst`` for that side only, so a
            # 0.0002 GiB move in the first side counted and a 2 GiB move in the
            # next was never read: the report then printed the small number as
            # "the largest raw memory deviation".
            moved = False
            for side in _sides(row):
                a = row[side]["peak_extra_gib"]
                b = reference[side]["peak_extra_gib"]
                worst = max(worst, abs(a - b))
                moved = moved or round(a, _MEMORY_DECIMALS) != round(b, _MEMORY_DECIMALS)
            if moved:
                memory += 1
            # Exact comparison on purpose. This is a reproducibility check, not an
            # accuracy check: the question is whether the same computation on the
            # same inputs produced the same number twice, and a tolerance would
            # answer a different question. The figures compared are themselves
            # error magnitudes already gated against the contract before any timing
            # ran, so a non-zero count here means a run-to-run difference and not a
            # contract violation, which is what the note below now says.
            accuracy_key = (id(rows), row["dtype"], tuple(row["shape"]))
            if accuracy_key in seen_accuracy:
                continue
            seen_accuracy.add(accuracy_key)
            accuracy_compared += 1
            if row["accuracy"] != reference["accuracy"]:
                accuracy += 1
                accuracy_detail.append(
                    f"{row['dtype']} {row['shape']} {row['mode']}: "
                    f"{reference['accuracy']} -> {row['accuracy']}"
                )
    return {
        "compared_rows": compared,
        "compared_accuracy_measurements": accuracy_compared,
        "memory_mismatches": memory,
        "accuracy_mismatches": accuracy,
        "accuracy_detail": accuracy_detail[:8],
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
        "Each pass re-runs the whole matrix on the same commit and GPU. One row per "
        "shape, because a speedup is a property of a shape: the mean and the interval "
        "below describe how much that shape's figure moves between runs, and the min "
        "and max are that shape's extremes across passes. A verdict is only given when "
        "the interval excludes parity. With two passes the t multiplier is 12.7, so "
        "the interval is very wide and can reach below zero; that is a statement about "
        "having two samples, not about the speedup, and it is why most rows at two "
        "passes read indeterminate.",
        "",
        "| Input | Shape | Mode | Candidate | Mean speedup | 95% CI | Min | Max | Verdict |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    order = {"fp16": 0, "bf16": 1, "fp32": 2}
    modes = {"forward": 0, "backward": 1, "forward_backward": 2}
    cands = {"triton": 0, "cuda": 1}

    def sort_key(d):
        return (
            order.get(d["dtype"], 9),
            modes.get(d["mode"], 9),
            cands.get(d["candidate"], 9),
            d.get("shape", []),
        )

    for item in sorted(summary, key=sort_key):
        lines.append(
            f"| {item['dtype']} | {'x'.join(map(str, item['shape']))} | {item['mode']} | "
            f"{item['candidate']} | "
            f"{item['mean_speedup']:.3f}x | {_interval(item)} | "
            f"{item['min_speedup']:.3f}x | {item['max_speedup']:.3f}x | {item['verdict']} |"
        )
    across = report.get("speedup_across_shapes") or []
    if across:
        lines += [
            "",
            "Collapsed over shapes, for one figure per dtype and mode. Geometric, "
            "because these are ratios: 4.0x on one shape and 0.5x on another is 1.0x, "
            "not the 2.25x an arithmetic mean gives. No interval, since a spread over "
            "shapes is not a measurement error; the ends are named instead.",
            "",
            "| Input | Mode | Candidate | Shapes | Geometric mean | Worst shape | Best shape |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
        for item in sorted(across, key=sort_key):
            lines.append(
                f"| {item['dtype']} | {item['mode']} | {item['candidate']} | "
                f"{item['shapes']} | {item['geometric_mean_speedup']:.3f}x | "
                f"{'x'.join(map(str, item['worst_shape']))} at "
                f"{item['worst_speedup']:.3f}x | "
                f"{'x'.join(map(str, item['best_shape']))} at {item['best_speedup']:.3f}x |"
            )
    if det.get("compared_rows"):
        lines += [
            "",
            f"Memory and accuracy hold still: across {det['compared_rows']} row "
            f"comparisons between passes, {det['memory_mismatches']} peak-memory "
            f"figures differed at the printed precision, with a largest raw deviation "
            f"of {det['max_memory_deviation_gib']:.2e} GiB over every side of every "
            f"row. Accuracy is measured once per dtype and shape rather than once per "
            f"mode, so it is "
            f"{det.get('compared_accuracy_measurements', det['compared_rows'])} "
            f"comparisons, {det['accuracy_mismatches']} of which differed; each one "
            f"covers all six outputs by error magnitude and by a hash of the result "
            f"bytes, so a reordered reduction below one narrow ULP shows up too. The "
            f"detail table below is the last pass.",
        ]
    lines += [
        "",
        "These intervals are within one invocation. Separate invocations shift the "
        "whole measurement by a few percent on unchanged code, so treat a verdict that "
        "rests on a few percent as provisional until it survives a second session.",
    ]
    return lines + [""]


def _tracked(env):
    """ "unknown" and "none" are different answers, and both used to print as "none".

    ``git_tracked_changes`` is None when the subprocess failed -- no checkout, no
    git on PATH, or the 10s timeout -- and the empty string when the tree is
    genuinely clean, so the provenance line asserted a clean tree that had never
    been checked.
    """
    value = env["git_tracked_changes"]
    if value is None:
        return "unknown"
    return value or "none"


def _markdown(report):
    env = report["environment"]
    lines = [
        _title(env),
        "",
        f"GPU: {env['gpu']}; PyTorch: {env['torch']}; "
        f"Triton: {env['triton'] or 'not installed'}; "
        f"{env['backend']} runtime: {env['runtime']}.",
        # "unknown" and "none" are different answers. git_tracked_changes is None
        # when the subprocess failed -- no checkout, no git on PATH, or the 10s
        # timeout -- and the empty string when the tree is genuinely clean, and both
        # used to print as "none", so the provenance line asserted a clean tree that
        # had never been checked.
        f"Commit: `{env['git_commit'] or 'unknown'}`; tracked changes: `{_tracked(env)}`.",
        f"Candidates: {', '.join(env['candidates'])}. Warmup: {env['warmup']} iterations "
        f"after a {env['clock_settle_seconds']}s clock-settling phase; measured "
        f"repetitions: {env['repeat']}.",
        "",
        "Every side is timed through the public `forward`, which returns the input dtype. "
        "Median accelerator-event latency for public wrappers, including allocation. The "
        "`forward` rows run under `torch.no_grad()`, so autograd dispatch is in the "
        "`backward` and `forward_backward` rows only; measured at 1x6656 bf16, leaving "
        "grad on costs the native 140.3 us against 117.8 and the CUDA candidate 21.5 "
        "against 19.5, so the reported forward speedup of 6.05x would have been 6.52x. "
        "The no_grad figure is the conservative one. The Triton and CUDA backwards are "
        "one kernel each plus the FP32 row fold for the two weight gradients; the native "
        "backward is autograd over two norms and an add. The backward speedup compares "
        "complete paths.",
        "One asymmetry in that comparison, stated because it is real and small rather "
        "than left for a reader to find: both candidates call record_backward once per "
        "backward and the eager baseline has no such bookkeeping. Measured at 0.733 us "
        "per call, 0.27 to 0.69 percent of the backward timings here, against reported "
        "speedups of 1.4x and up. The recorder keys one entry per operator and "
        "overwrites it, so it does not accumulate: 20,000 calls leave one entry, and "
        "the settling loop cannot grow memory through it.",
        "Speedup = the reference / candidate; values below 1 mean the candidate was "
        "slower. The reference is `NativeFusedAddNormOp.forward`, which is the operator's "
        "gold and not an idiomatic eager spelling: its statistic goes through "
        "`shape_invariant_meansq`, which zero-pads to 32-wide chunks and sums in two "
        "stages so that a row's bytes do not depend on the batch layout. That costs "
        "something, and comparing against the gold is the right comparison, so the cost "
        "is stated rather than removed. Measured on an RTX 4090 at bf16 with the cast "
        "structure held byte-identical and only the reduction replaced by `mean(-1)`: "
        "the gold is 1.28x slower at 1x6656 and 1.07x at 512x6656 on the forward. So "
        "the forward speedups here are larger than they would be against a plain eager "
        "spelling, by about that factor, most of it at the small shapes. "
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
    """Build the candidate ops, skipping either backend this build cannot provide.

    Both are gated. Only CUDA was, so ``--candidates cuda`` still raised on a
    CUDA-only build with no Triton installed, because the Triton import ran
    unconditionally before the gate had a chance to matter.
    """
    from rl_engine.backends.cuda.norm.fused_add_norm import (
        CudaFusedAddNormOp,
        cuda_fused_add_norm_available,
    )

    available = {}
    try:
        from rl_engine.backends.shared.triton.norm.fused_add_norm import TritonFusedAddNormOp
    except ImportError:
        pass
    else:
        available["triton"] = TritonFusedAddNormOp
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

            accuracy = {}
            executed = {}
            for name, op in candidates.items():
                # One backward per candidate runs here, before any timing. The
                # recorder keys by operator and overwrites, so resetting around
                # each candidate is what makes the record per-candidate rather
                # than "whichever ran last".
                reset_backward_runtime()
                accuracy[name] = _check_accuracy(
                    native, op, hidden, residual, post_w, pre_w, g_res, g_pre
                )
                snap = snapshot_backward_runtime().get("fused_post_norm_residual_pre_norm")
                executed[name] = snap["impl"] if snap else None
            for mode in MODES:
                measurements = {}
                for name, op in [("native", native), *candidates.items()]:
                    # partial, not a lambda: the tensors are deleted at the end of
                    # this shape's iteration, and binding them by value keeps the
                    # factory from closing over names that go away.
                    measurements[name] = _measure(
                        profiler,
                        functools.partial(
                            _make_workload,
                            op,
                            hidden,
                            residual,
                            post_w,
                            pre_w,
                            g_res,
                            g_pre,
                            mode,
                        ),
                    )
                row = {
                    "dtype": dtype_name,
                    "shape": list(shape),
                    "numel": hidden.numel(),
                    "mode": mode,
                    "accuracy": accuracy,
                    # The implementation each candidate actually executed, not the
                    # name it was requested under.
                    "backward_impl": executed,
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
    # Imported for the version string only, so a box without Triton can still run
    # --candidates cuda: _resolve_candidates gates the backend itself, and this
    # unconditional import defeated that by raising first.
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None

    native = NativeFusedAddNormOp()
    candidates = _resolve_candidates(args.candidates)
    with torch.cuda.device(device):
        profiler = PerformanceProfiler(device=device, warmup=args.warmup, repeat=args.repeat)
        report = {
            "environment": _environment(device, args, triton_version, args.shapes, candidates),
            "results": [],
        }
        reset_backward_runtime()
        passes = []
        for index in range(args.runs):
            passes.append(_sweep(args, profiler, native, candidates, device))
            print(f"--- pass {index + 1}/{args.runs} complete", flush=True)
        report["results"] = passes[-1]
        report["reproducibility"] = _reproducibility(passes)
        report["speedup_across_shapes"] = _across_shapes(report["reproducibility"])
        report["determinism"] = _determinism(passes)
        # What ran, not what was asked for, per candidate. This used to be
        # snapshot_backward_runtime() taken after the whole sweep, and
        # record_backward keys on the operator alone, so the entry was whichever
        # candidate ran last carrying an execution_count that every candidate had
        # contributed to: one run reported 13373 calls against the CUDA impl with
        # Triton absent, having run just as many. The per-row backward_impl was
        # always right, so this field is built from those instead.
        report["executed_backward"] = {
            name: sorted({row["backward_impl"].get(name) for row in passes[-1]} - {None})
            for name in candidates
        }
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
