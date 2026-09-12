# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render consistency, learning, and throughput figures from rounds.csv."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


COLORS = {"G00": "#6b7280", "G10": "#2563eb", "G01": "#dc2626", "G11": "#059669"}
MARKERS = {"G00": "o", "G10": "x", "G01": "D", "G11": "+"}
LABELS = {
    "G00": "G00: production",
    "G10": "G10: framework",
    "G01": "G01: operator",
    "G11": "G11: framework + operator",
}


def _parse(value: str) -> Any:
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def _load(path: Path, phase: str | None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            {key: _parse(value) for key, value in row.items()} for row in csv.DictReader(handle)
        ]
    if phase is not None:
        rows = [row for row in rows if row["phase"] == phase]
    if not rows:
        raise ValueError("no rows match the requested phase")
    return rows


def _series(
    rows: list[dict[str, Any]], metric: str
) -> dict[str, tuple[list[int], list[float], list[float]]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            grouped[(str(row["group"]), int(row["step"]))].append(float(value))
    result = {}
    for group in sorted({key[0] for key in grouped}):
        steps = sorted(step for candidate, step in grouped if candidate == group)
        centers = [mean(grouped[(group, step)]) for step in steps]
        spreads = [pstdev(grouped[(group, step)]) for step in steps]
        result[group] = (steps, centers, spreads)
    return result


def _moving_average(values: list[float], window: int) -> list[float]:
    return [mean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


def _plot_series(axis, rows, metric, title, *, moving_average=1, symlog=False):
    for group, (steps, values, spreads) in _series(rows, metric).items():
        centers = _moving_average(values, moving_average)
        color = COLORS.get(group)
        axis.plot(
            steps,
            centers,
            label=LABELS.get(group, group),
            color=color,
            linewidth=2,
            marker=MARKERS.get(group, "o"),
            markersize=5,
        )
        if any(spreads):
            lower = [center - spread for center, spread in zip(centers, spreads, strict=True)]
            upper = [center + spread for center, spread in zip(centers, spreads, strict=True)]
            axis.fill_between(steps, lower, upper, color=color, alpha=0.15)
    if symlog:
        axis.set_yscale("symlog", linthresh=1e-9)
    axis.set_title(title)
    axis.set_xlabel("training step")
    axis.grid(alpha=0.25)


def _mismatch_rate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        updated = dict(row)
        mismatch = row.get("mismatch_count")
        tokens = row.get("active_token_count")
        updated["mismatch_rate"] = (
            float(mismatch) / float(tokens)
            if isinstance(mismatch, (int, float)) and isinstance(tokens, (int, float)) and tokens
            else None
        )
        result.append(updated)
    return result


def _save_consistency(rows, output, dpi):
    import matplotlib.pyplot as plt

    derived = _mismatch_rate_rows(rows)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    _plot_series(
        axes[0, 0],
        derived,
        "mean_abs_dlogp",
        "Mean |train logp - rollout logp|",
        symlog=True,
    )
    _plot_series(axes[0, 1], derived, "max_abs_dlogp", "Maximum |Δlogp|", symlog=True)
    _plot_series(axes[1, 0], derived, "mismatch_rate", "Bitwise mismatch rate")
    _plot_series(
        axes[1, 1],
        derived,
        "mismatch_count",
        "Bitwise mismatch count per step",
        symlog=True,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=max(1, len(labels)))
    figure.savefig(output / "consistency.png", dpi=dpi)
    plt.close(figure)


def _save_learning(rows, output, dpi, window):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    _plot_series(
        axes[0, 0],
        rows,
        "raw_reward",
        f"Raw reward ({window}-step MA)",
        moving_average=window,
    )
    _plot_series(axes[0, 1], rows, "ppo_kl", "Training PPO KL", symlog=True)
    _plot_series(axes[1, 0], rows, "entropy", "Token entropy")
    _plot_series(axes[1, 1], rows, "truncated_ratio", "Truncated response ratio")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=max(1, len(labels)))
    figure.savefig(output / "learning.png", dpi=dpi)
    plt.close(figure)


def _save_optimization(rows, output, dpi):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    _plot_series(axes[0, 0], rows, "pg_loss", "GRPO policy-gradient loss", symlog=True)
    _plot_series(axes[0, 1], rows, "pg_clipfrac", "Policy ratio clipped fraction")
    _plot_series(axes[1, 0], rows, "ppo_kl", "Training PPO KL", symlog=True)
    _plot_series(axes[1, 1], rows, "grad_norm", "Gradient norm", symlog=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=max(1, len(labels)))
    figure.savefig(output / "optimization.png", dpi=dpi)
    plt.close(figure)


def _save_performance(rows, output, dpi):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    _plot_series(axes[0], rows, "step_time", "End-to-end step time (s)")
    _plot_series(axes[1], rows, "rollout_time", "Rollout time (s)")
    _plot_series(axes[2], rows, "actor_tokens_per_second", "Actor tokens/s")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=max(1, len(labels)))
    figure.savefig(output / "performance.png", dpi=dpi)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--moving-average", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.moving_average <= 0:
        raise ValueError("--moving-average must be positive")
    rows = _load(args.rounds_csv, args.phase)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_consistency(rows, args.output_dir, args.dpi)
    _save_learning(rows, args.output_dir, args.dpi, args.moving_average)
    _save_optimization(rows, args.output_dir, args.dpi)
    _save_performance(rows, args.output_dir, args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
