# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark the h3_sigma_schedule operator across its backends (RFC #420).

Operator-only microbenchmark: builds sigma grids for N in {25, 50, 1000,
1000000} x shift in {3.0, 12.0} on three paths:

* path A — CPU reference, grid built on CPU, outputs stay on CPU;
* path B — CPU reference, grid built on CPU, sigmas and timesteps moved to
  the target GPU via the operator's ``device`` argument (the registry default
  path on every platform: build + host-to-device copy);
* path C — Triton opt-in backend (semantic ``requested_backend`` or direct
  construction), grid built on the device.

B vs C is the comparison that matters when the consumer lives on the GPU; A
is an auxiliary CPU baseline and A/C ratios must not be presented as a
speedup under equal output conditions.

Timing semantics (deliberately conservative): **per-call end-to-end host
wall latency**. Each call is timed individually with ``time.perf_counter``
around the whole ``op()`` call — including Python dispatch, host
allocations, kernel launches, and the device path's single host
synchronization (the deduplicated-length read) — and, for the GPU paths,
one ``torch.cuda.synchronize()`` before starting each sample (so prior
async work cannot leak into the window) and one after the call (so the
call's device work is complete before the clock stops). Reported per case:
raw per-call samples, mean, median, standard deviation, min, and max. The
median is the headline number; mean and raw samples are recorded so
outliers stay visible. No CUDA events are used, because they cannot see the
host-side part of the call. Triton kernel compilation happens inside the
pre-timing correctness gate (the gate's calls are this process's first
Triton invocations): the gate's total wall time is reported once as
``correctness_gate_wall_ms`` and no later call is labeled "first" or
"compile". Timed samples start only after per-case warmup, so steady-state
numbers exclude compilation by construction. The benchmark calls the op
classes directly (registry bypassed for timing); registry-entry identity
and correctness are pinned by tests/runtime/ and tests/backends/cuda/.

Before any timing on a GPU host where the Triton backend is available, path C
is verified bitwise against path A for every measured case (sigmas,
timesteps, and num_model_evals); a mismatch aborts with exit code 2 and no
performance numbers are produced.

Exit codes: 0 on success; 2 on the pre-timing correctness gate; nonzero on
invalid arguments, and on any Triton compile, launch, or runtime failure
after the backend was confirmed available. Path B needs only torch and a
device, so it is measured on every GPU host. Path C (and its gate) is
skipped only for a *confirmed* absent backend — a missing Triton import on a
GPU host — or recorded as unsupported on a HIP/ROCm build, where the
NVIDIA-only kernels (inline PTX) refuse construction. Without any CUDA/HIP
device (``torch.cuda.is_available()`` false) both B and C are skipped. The
script runs from an uninstalled checkout (it bootstraps the repository root
onto ``sys.path``).

The grid is tiny (4 KB at N=1000), so GPU launch overhead may exceed the
arithmetic — the numbers are what they are, and this benchmark does not
assume the device path wins. N=1000000 is a dedup stress case, not a
typical workload. No model, checkpoint, or tokenizer is loaded.

Example:

    python benchmarks/models/benchmark_h3_sigma_schedule.py --samples 100
    CUDA_VISIBLE_DEVICES=0 python benchmarks/models/benchmark_h3_sigma_schedule.py \
        --samples 100 --json /tmp/h3_sigma_bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # uninstalled checkout

import torch  # noqa: E402

_SIZES = [25, 50, 1000, 1000000]
_SHIFTS = [3.0, 12.0]
_SPEEDUP_NOTE = "speedup_b_over_c = latency(B) / latency(C); > 1 means C (Triton) is faster"


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _int32_bits(tensor: torch.Tensor) -> list[int]:
    return tensor.detach().contiguous().view(torch.int32).tolist()


def _time_calls(call, samples: int, *, device_sync: bool) -> dict:
    """Per-call end-to-end host wall latency; raw samples plus summary stats."""

    raw_ms: list[float] = []
    for _ in range(samples):
        if device_sync:
            torch.cuda.synchronize()  # no prior async work leaks into this window
        start = time.perf_counter()
        call()
        if device_sync:
            torch.cuda.synchronize()  # the call's device work is done before stop
        raw_ms.append((time.perf_counter() - start) * 1e3)
    return {
        "samples_raw_ms": [round(value, 6) for value in raw_ms],
        "mean_ms": statistics.fmean(raw_ms),
        "median_ms": statistics.median(raw_ms),
        "stdev_ms": statistics.stdev(raw_ms) if len(raw_ms) > 1 else 0.0,
        "min_ms": min(raw_ms),
        "max_ms": max(raw_ms),
    }


def _reference_op():
    from rl_engine.reference.minimax_h3.sigma_schedule import NativeH3SigmaScheduleOp

    return NativeH3SigmaScheduleOp()


def _time_reference(samples: int, warmup: int, *, device) -> dict[str, dict]:
    # Path A when device is None (outputs stay on CPU); path B when device is
    # "cuda" (same CPU build, then sigmas/timesteps moved by the operator).
    op = _reference_op()
    results = {}
    for n in _SIZES:
        for shift in _SHIFTS:
            for _ in range(warmup):
                op(n, shift=shift, device=device)
            if device is not None:
                torch.cuda.synchronize()
            call = partial(op, n, shift=shift, device=device)
            results[f"N={n},shift={shift:g}"] = _time_calls(
                call, samples, device_sync=device is not None
            )
    return results


def _verify_triton_matches_reference() -> float:
    """Bitwise gate: path C must equal path A before any timing happens.

    This gate's calls are the FIRST Triton invocations of the process — the
    kernels are compiled here, not later. Returns the gate's wall time in ms
    (compilation + all case comparisons) so the cold-start cost stays on the
    record instead of being mislabeled somewhere else.
    """

    from rl_engine.backends.cuda.model_specific.minimax_h3.sigma_schedule import (
        H3SigmaScheduleCudaOp,
    )

    cpu_op = _reference_op()
    gpu_op = H3SigmaScheduleCudaOp()
    start = time.perf_counter()
    for n in _SIZES:
        for shift in _SHIFTS:
            expected = cpu_op(n, shift=shift)
            actual = gpu_op(n, shift=shift)
            if (
                _int32_bits(actual.sigmas.cpu()) != _int32_bits(expected.sigmas)
                or _int32_bits(actual.timesteps.cpu()) != _int32_bits(expected.timesteps)
                or actual.num_model_evals != expected.num_model_evals
            ):
                first_sigma = next(
                    (
                        idx
                        for idx, (a, e) in enumerate(
                            zip(_int32_bits(actual.sigmas.cpu()), _int32_bits(expected.sigmas))
                        )
                        if a != e
                    ),
                    None,
                )
                print(
                    f"correctness gate FAILED at N={n}, shift={shift}: Triton result "
                    f"differs from the CPU reference (first mismatching sigma index: "
                    f"{first_sigma}); refusing to benchmark"
                )
                sys.exit(2)  # documented gate exit code; SystemExit(str) would exit 1
    return (time.perf_counter() - start) * 1e3


def _time_triton(samples: int, warmup: int) -> dict[str, dict]:
    # The caller ran the correctness gate first, so every kernel has already
    # been compiled and executed; no "first call" exists here to re-time and
    # nothing in this function is labeled as compilation. Any failure below
    # (launch, host read) propagates and fails the command.
    from rl_engine.backends.cuda.model_specific.minimax_h3.sigma_schedule import (
        H3SigmaScheduleCudaOp,
    )

    op = H3SigmaScheduleCudaOp()

    results = {}
    for n in _SIZES:
        for shift in _SHIFTS:
            for _ in range(warmup):
                op(n, shift=shift)
            torch.cuda.synchronize()
            call = partial(op, n, shift=shift)
            results[f"N={n},shift={shift:g}"] = _time_calls(call, samples, device_sync=True)
    return results


def _summarize(report: dict) -> dict:
    """Speedup B/C per case from medians and means; A medians for context."""

    summary = {"definition": _SPEEDUP_NOTE, "cases": {}}
    path_b = report.get("path_b_reference_cpu_build_then_copy_to_gpu")
    path_c = report.get("path_c_triton_device_build")
    # A path that was skipped or recorded as unsupported carries no per-case
    # timings, so there is no ratio to compute (B alone is still reported).
    if (
        not path_b
        or not path_c
        or "skipped" in path_b
        or "unsupported" in path_b
        or "skipped" in path_c
        or "unsupported" in path_c
    ):
        return summary
    for key, case_b in path_b.items():
        case_c = path_c[key]
        summary["cases"][key] = {
            "b_median_ms": case_b["median_ms"],
            "c_median_ms": case_c["median_ms"],
            "speedup_b_over_c_median": case_b["median_ms"] / case_c["median_ms"],
            "speedup_b_over_c_mean": case_b["mean_ms"] / case_c["mean_ms"],
            "separated_from_dispersion": abs(case_b["median_ms"] - case_c["median_ms"])
            > (case_b["stdev_ms"] + case_c["stdev_ms"]),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=_positive_int, default=100)
    parser.add_argument("--warmup", type=_positive_int, default=5)
    parser.add_argument("--json", default=None, help="optional output path")
    args = parser.parse_args(argv)

    has_gpu = torch.cuda.is_available()
    report: dict = {
        "operator": "h3_sigma_schedule",
        "timing": (
            "per-call end-to-end host wall latency; GPU paths sync before each "
            "sample and after each call; Triton first-call compile excluded"
        ),
        "entry": "direct op construction; registry bypassed for timing (identity pinned by tests)",
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "hip": torch.version.hip,
        "triton": None,
        "device": torch.cuda.get_device_name(0) if has_gpu else "cpu-only",
        "n_gpu_devices": torch.cuda.device_count() if has_gpu else 0,
        "samples": args.samples,
        "warmup": args.warmup,
        "sizes": _SIZES,
        "shifts": _SHIFTS,
        "stress_case_note": "N=1000000 is a dedup stress case, not a typical H3 workload",
    }

    print(
        f"path A (reference, cpu build, cpu outputs): {args.samples} calls x {_SIZES} x {_SHIFTS}"
    )
    report["path_a_reference_cpu"] = _time_reference(args.samples, args.warmup, device=None)

    if not has_gpu:
        skip = "no CUDA/HIP device (torch.cuda.is_available() is False)"
        report["path_b_reference_cpu_build_then_copy_to_gpu"] = {"skipped": skip}
        report["path_c_triton_device_build"] = {"skipped": skip}
        print("paths B and C: skipped — no CUDA/HIP device in this process")
    else:
        try:
            import triton

            report["triton"] = triton.__version__
        except ImportError as exc:
            # Path B is the registry default path and needs only torch plus a
            # device, so a missing Triton skips path C alone.
            skip = f"triton import failed on a GPU host: {exc}"
            print(f"path B (reference, cpu build, outputs moved to gpu): {args.samples} calls")
            report["path_b_reference_cpu_build_then_copy_to_gpu"] = _time_reference(
                args.samples, args.warmup, device="cuda"
            )
            report["path_c_triton_device_build"] = {"skipped": skip}
            print(f"path C: skipped — triton missing ({exc})")
        else:
            if torch.version.hip is not None:
                # Triton may import on a HIP host, but the backend is NVIDIA-only
                # (inline PTX) and refuses construction: record that instead of
                # letting the ImportError escape from inside the correctness gate.
                unsupported = (
                    "HIP/ROCm build: the Triton backend is NVIDIA-only (inline PTX "
                    "div.rn.f32) and refuses construction; the bitwise correctness "
                    "gate was not run"
                )
                print(f"path B (reference, cpu build, outputs moved to gpu): {args.samples} calls")
                report["path_b_reference_cpu_build_then_copy_to_gpu"] = _time_reference(
                    args.samples, args.warmup, device="cuda"
                )
                report["path_c_triton_device_build"] = {"unsupported": unsupported}
                print("path C: unsupported — NVIDIA-only backend on a HIP/ROCm build")
            else:
                print("verifying path C against path A bitwise before timing...")
                gate_ms = _verify_triton_matches_reference()
                report["correctness_gate_wall_ms"] = gate_ms
                print(
                    "correctness gate passed "
                    f"(includes Triton kernel compilation: {gate_ms:.1f} ms wall, "
                    "reported once, excluded from all timed samples)"
                )

                print(f"path B (reference, cpu build, outputs moved to gpu): {args.samples} calls")
                report["path_b_reference_cpu_build_then_copy_to_gpu"] = _time_reference(
                    args.samples, args.warmup, device="cuda"
                )
                print(f"path C (triton, device build): {args.samples} calls")
                report["path_c_triton_device_build"] = _time_triton(args.samples, args.warmup)

    report["summary"] = _summarize(report)

    for section in (
        "path_a_reference_cpu",
        "path_b_reference_cpu_build_then_copy_to_gpu",
        "path_c_triton_device_build",
    ):
        for key, value in report[section].items():
            if isinstance(value, dict) and "median_ms" in value:
                print(f"{section:>44s} {key:>20s}: {value['median_ms']:.4f} ms/call (median)")
    for key, case in report["summary"]["cases"].items():
        separated = case["separated_from_dispersion"]
        note = "separated from dispersion" if separated else "within dispersion"
        print(f"speedup B/C {key:>20s}: {case['speedup_b_over_c_median']:.2f}x ({note})")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
