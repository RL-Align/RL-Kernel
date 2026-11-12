# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Measurement and report-writing helpers shared by the operator benchmark scripts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def measure(profiler, fn):
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


def command_output(command):
    try:
        return subprocess.check_output(
            command, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_and_save_report(parser, run_benchmark, render_markdown):
    args = parser.parse_args()
    report = run_benchmark(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report)
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Reports saved to {args.output_dir}")
