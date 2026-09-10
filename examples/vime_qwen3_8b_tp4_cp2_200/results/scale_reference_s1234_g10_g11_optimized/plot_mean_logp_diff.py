#!/usr/bin/env python3
"""Plot the PR #377 mean absolute train/rollout logp difference."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

RECORD_RE = re.compile(r"(?:perf|step|rollout)\s+(\d+):\s+(\{.*\})")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRIC = "train/train_rollout_logprob_abs_diff"
BLUE = "#2F67D8"
LIGHT_BLUE = "#9AB8EE"
RED = "#E45756"
LIGHT_RED = "#F3AAA7"
GRID = "#D7DCE2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g10-log", type=Path, required=True)
    parser.add_argument("--g11-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_metric(path: Path) -> np.ndarray:
    rows: dict[int, dict[str, Any]] = {}
    for raw_line in path.open(encoding="utf-8", errors="replace"):
        match = RECORD_RE.search(ANSI_RE.sub("", raw_line))
        if not match:
            continue
        try:
            payload = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, dict):
            rows.setdefault(int(match.group(1)), {}).update(payload)
    if sorted(rows) != list(range(200)):
        raise RuntimeError(f"{path} has {len(rows)} steps; expected 0..199")
    return np.asarray([float(rows[index][METRIC]) for index in range(200)])


def moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
    result = np.empty_like(values)
    for index in range(values.size):
        result[index] = np.mean(values[max(0, index - window + 1) : index + 1])
    return result


def main() -> None:
    args = parse_args()
    steps = np.arange(1, 201)
    fig, axis = plt.subplots(figsize=(14, 7.2))
    for label, path, raw, strong in (
        ("G10", args.g10_log, LIGHT_RED, RED),
        ("G11", args.g11_log, LIGHT_BLUE, BLUE),
    ):
        values = read_metric(path)
        axis.plot(steps, values, color=raw, linewidth=0.9, alpha=0.7)
        axis.plot(
            steps,
            moving_average(values),
            color=strong,
            linewidth=2.4,
            label=f"{label} 10-step MA",
        )
    axis.set_title(
        "Mean Absolute Train/Rollout Logp Difference Across 200 Steps",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    axis.set_xlabel("Training step")
    axis.set_ylabel("Mean absolute log-probability difference")
    axis.set_xlim(1, 200)
    axis.set_ylim(bottom=-0.0005)
    axis.grid(True, color=GRID, linestyle="--", linewidth=0.7)
    axis.legend(frameon=True)
    fig.text(
        0.5,
        0.02,
        "G10 production P/P vs optimized G11 strict R/R - Qwen3-8B - NVIDIA H100",
        ha="center",
        fontsize=10,
        color="#4B525B",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
