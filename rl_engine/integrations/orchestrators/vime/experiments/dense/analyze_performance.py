# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Compare stage timing and token-normalized throughput for sealed G10/G11 logs.

The historical runs used different RL-Kernel, VIME, and Transformer Engine
revisions.  The output is therefore descriptive diagnostic evidence, not a
causal estimate of the cost of the RL-Kernel consistency operators.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

RECORD_RE = re.compile(r"\b(rollout|step|perf)\s+(\d+):\s+(\{.*\})\s*$")
GENERATION_RE = re.compile(r"Rollout generation:.*100%.*\[(\d+):(\d+)<")
BLUE = "#2F67D8"
LIGHT_BLUE = "#93B4F4"
RED = "#E53935"
LIGHT_RED = "#F4A3A0"
GRID = "#D7DCE2"
TEXT = "#20242A"


def parse_log(path: Path) -> tuple[dict[str, dict[int, dict[str, Any]]], np.ndarray]:
    records: dict[str, dict[int, dict[str, Any]]] = {
        "rollout": {},
        "step": {},
        "rollout_perf": {},
        "train_perf": {},
    }
    progress_seconds: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = RECORD_RE.search(line)
        if match:
            try:
                payload = ast.literal_eval(match.group(3))
            except (SyntaxError, ValueError):
                payload = None
            if isinstance(payload, dict):
                kind = match.group(1)
                step = int(match.group(2))
                if kind == "perf":
                    if "perf/rollout_time" in payload:
                        kind = "rollout_perf"
                    elif "perf/step_time" in payload:
                        kind = "train_perf"
                    else:
                        continue
                records[kind][step] = payload
        if "Rollout generation:" in line and "100%" in line:
            match = GENERATION_RE.search(line)
            if match:
                progress_seconds.append(60 * int(match.group(1)) + int(match.group(2)))

    expected = list(range(200))
    for kind, values in records.items():
        if sorted(values) != expected:
            raise RuntimeError(f"{path}: {kind} has {len(values)} records; expected steps 0..199")
    if len(progress_seconds) != 400:
        raise RuntimeError(
            f"{path}: expected 400 duplicated rollout progress events, got "
            f"{len(progress_seconds)}"
        )
    pairs = np.asarray(progress_seconds, dtype=float).reshape(200, 2)
    if not np.array_equal(pairs[:, 0], pairs[:, 1]):
        raise RuntimeError(f"{path}: duplicated rollout progress events do not match")
    return records, pairs[:, 0]


def array(records: dict[int, dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(records[index][key]) for index in range(200)], dtype=float)


def rows_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "sum": float(np.sum(values)),
        "std": float(np.std(values)),
    }


def moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
    totals = np.convolve(values, np.ones(window, dtype=float), mode="full")[: len(values)]
    counts = np.minimum(np.arange(1, len(values) + 1), window)
    return totals / counts


def regression(x: np.ndarray, y: np.ndarray, common_x: float) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * common_x
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "prediction_at_common_x": float(prediction),
        "common_x": float(common_x),
    }


def bootstrap_mean_difference(
    left: np.ndarray, right: np.ndarray, *, draws: int = 20_000
) -> dict[str, float]:
    rng = np.random.default_rng(1234)
    left_idx = rng.integers(0, len(left), size=(draws, len(left)))
    right_idx = rng.integers(0, len(right), size=(draws, len(right)))
    gaps = left[left_idx].mean(axis=1) - right[right_idx].mean(axis=1)
    return {
        "estimate": float(left.mean() - right.mean()),
        "ci95_low": float(np.percentile(gaps, 2.5)),
        "ci95_high": float(np.percentile(gaps, 97.5)),
        "draws": draws,
    }


def build_rows(
    group: str,
    records: dict[str, dict[int, dict[str, Any]]],
    progress_seconds: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rollout = records["rollout"]
    rollout_perf = records["rollout_perf"]
    train_perf = records["train_perf"]
    response_length = array(rollout, "rollout/response_lengths")
    total_length = array(rollout, "rollout/total_lengths")
    rollout_time = array(rollout_perf, "perf/rollout_time")
    train_wait = array(train_perf, "perf/train_wait_time")
    train_time = array(train_perf, "perf/train_time")
    actor_train = array(train_perf, "perf/actor_train_time")
    update_weights = array(train_perf, "perf/update_weights_time")

    rows: list[dict[str, float | int | str]] = []
    for step in range(200):
        rows.append(
            {
                "group": group,
                "step": step,
                "response_length_mean": response_length[step],
                "prompt_length_mean": total_length[step] - response_length[step],
                "total_length_mean": total_length[step],
                "response_tokens_total": 8 * response_length[step],
                "rollout_time_s": rollout_time[step],
                "rollout_progress_time_s_rounded": progress_seconds[step],
                "rollout_tok_per_gpu_s": float(rollout_perf[step]["perf/tokens_per_gpu_per_sec"]),
                "rollout_aggregate_tok_s": 8 * response_length[step] / rollout_time[step],
                "rollout_truncated_ratio": float(rollout_perf[step]["rollout/truncated_ratio"]),
                "update_weights_time_s": update_weights[step],
                "wait_residual_time_s": (
                    train_wait[step] - rollout_time[step] - update_weights[step]
                ),
                "train_wait_time_s": train_wait[step],
                "actor_train_time_s": actor_train[step],
                "train_residual_time_s": train_time[step] - actor_train[step],
                "train_time_s": train_time[step],
                "data_preprocess_time_s": float(train_perf[step]["perf/data_preprocess_time"]),
                "step_time_s": float(train_perf[step]["perf/step_time"]),
                "actor_train_tok_s": float(train_perf[step]["perf/actor_train_tok_per_s"]),
                "actor_train_tflops": float(train_perf[step]["perf/actor_train_tflops"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(True, color=GRID, linestyle="--", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(TEXT)
        spine.set_linewidth(1.0)
    axis.tick_params(colors=TEXT, labelsize=10)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def plot_decomposition(group_rows: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    stages = [
        ("rollout_time_s", "Rollout generation", "#E15759"),
        ("update_weights_time_s", "Weight update", "#F2B134"),
        ("wait_residual_time_s", "Wake/offload residual", "#BAB0AC"),
        ("actor_train_time_s", "Actor training", "#4E79A7"),
        ("train_residual_time_s", "Train residual", "#76B7B2"),
    ]
    groups = ["G11", "G10"]
    fig, axis = plt.subplots(figsize=(9.5, 6.5))
    style_axis(axis)
    bottoms = np.zeros(2)
    for key, label, color in stages:
        values = np.asarray(
            [statistics.fmean(float(row[key]) for row in group_rows[group]) for group in groups]
        )
        axis.bar(groups, values, bottom=bottoms, label=label, color=color, width=0.58)
        for index, value in enumerate(values):
            if value >= 2:
                axis.text(
                    index,
                    bottoms[index] + value / 2,
                    f"{value:.1f}s",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=TEXT,
                )
        bottoms += values
    for index, total in enumerate(bottoms):
        axis.text(index, total + 2, f"{total:.1f}s / step", ha="center", fontweight="bold")
    axis.set_title(
        "G11 vs G10 · Mean Step-Time Decomposition",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    axis.set_ylabel("Seconds per training step")
    axis.legend(ncol=2, loc="upper right", frameon=True)
    axis.set_ylim(0, 1.16 * float(np.max(bottoms)))
    fig.tight_layout()
    save_figure(fig, output_dir, "performance-decomposition")
    plt.close(fig)


def plot_scaling(group_rows: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.7))
    configs = {"G11": (BLUE, "o"), "G10": (RED, "s")}
    panels = [
        (
            "response_length_mean",
            "rollout_time_s",
            "Rollout generation",
            "Mean response length (tokens/sample)",
            "Seconds",
        ),
        (
            "total_length_mean",
            "actor_train_time_s",
            "Actor training",
            "Mean total length (tokens/sample)",
            "Seconds",
        ),
    ]
    for axis, (x_key, y_key, title, xlabel, ylabel) in zip(axes, panels, strict=True):
        style_axis(axis)
        for group, (color, marker) in configs.items():
            x = rows_array(group_rows[group], x_key)
            y = rows_array(group_rows[group], y_key)
            axis.scatter(
                x,
                y,
                s=23,
                alpha=0.30,
                color=color,
                marker=marker,
                label=f"{group} steps",
            )
            slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            axis.plot(
                grid,
                intercept + slope * grid,
                color=color,
                linewidth=2.3,
                label=f"{group} OLS",
            )
        axis.set_title(title, fontsize=12.5, pad=8)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.legend(ncol=2, frameon=True, fontsize=9.3)
    fig.suptitle(
        "G11 vs G10 · Length-Controlled Stage Scaling",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_dir, "length-controlled-scaling")
    plt.close(fig)


def plot_time_series(group_rows: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    steps = np.arange(200)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.8), sharex=True)
    for group, raw_color, ma_color in (
        ("G11", LIGHT_BLUE, BLUE),
        ("G10", LIGHT_RED, RED),
    ):
        for axis, key in zip(axes, ("rollout_time_s", "actor_train_time_s"), strict=True):
            values = rows_array(group_rows[group], key)
            axis.plot(steps, values, color=raw_color, alpha=0.7, linewidth=0.9)
            axis.plot(
                steps,
                moving_average(values),
                color=ma_color,
                linewidth=2.2,
                label=f"{group} 10-step MA",
            )
    for axis, title in zip(axes, ("Rollout generation", "Actor training"), strict=True):
        style_axis(axis)
        axis.set_title(title, fontsize=12.5, pad=8)
        axis.set_ylabel("Seconds")
        axis.legend(frameon=True)
    axes[-1].set_xlabel("Training step")
    axes[-1].set_xlim(0, 199)
    axes[-1].set_xticks(np.arange(0, 200, 25))
    fig.suptitle(
        "G11 vs G10 · Performance Across 200 Steps",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_dir, "stage-time-series")
    plt.close(fig)


def plot_throughput(group_rows: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3))
    panels = [
        ("rollout_tok_per_gpu_s", "Rollout throughput", "Tokens / GPU / s"),
        ("actor_train_tok_s", "Actor training throughput", "Tokens / s"),
    ]
    for axis, (key, title, ylabel) in zip(axes, panels, strict=True):
        style_axis(axis)
        means = [rows_array(group_rows[group], key).mean() for group in ("G11", "G10")]
        bars = axis.bar(
            ["G11", "G10"],
            means,
            width=0.5,
            color=["#A9C7ED", "#F2B566"],
            edgecolor=[BLUE, "#D66A27"],
            linewidth=1.2,
        )
        for bar, value in zip(bars, means, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.018,
                f"{value:,.0f}",
                ha="center",
                color=TEXT,
            )
        speedup = means[1] / means[0]
        axis.text(
            0.5,
            max(means) * 0.88,
            f"G10 / G11 = {speedup:.2f}×",
            ha="center",
            fontweight="bold",
            color=TEXT,
        )
        axis.set_title(title, fontsize=12.5, pad=8)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, max(means) * 1.14)
    fig.suptitle(
        "G11 vs G10 · Token-Normalized Throughput",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir, "token-normalized-throughput")
    plt.close(fig)


def summarize(group_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    keys = [
        "response_length_mean",
        "prompt_length_mean",
        "total_length_mean",
        "rollout_time_s",
        "rollout_progress_time_s_rounded",
        "rollout_tok_per_gpu_s",
        "rollout_aggregate_tok_s",
        "rollout_truncated_ratio",
        "update_weights_time_s",
        "wait_residual_time_s",
        "train_wait_time_s",
        "actor_train_time_s",
        "train_residual_time_s",
        "train_time_s",
        "step_time_s",
        "actor_train_tok_s",
        "actor_train_tflops",
    ]
    all_rows = group_rows["G11"] + group_rows["G10"]
    common_response_length = statistics.fmean(
        float(row["response_length_mean"]) for row in all_rows
    )
    common_total_length = statistics.fmean(float(row["total_length_mean"]) for row in all_rows)
    summary: dict[str, Any] = {"groups": {}}
    for group, rows in group_rows.items():
        summary["groups"][group] = {key: describe(rows_array(rows, key)) for key in keys}
        summary["groups"][group]["rollout_length_regression"] = regression(
            rows_array(rows, "response_length_mean"),
            rows_array(rows, "rollout_time_s"),
            common_response_length,
        )
        summary["groups"][group]["actor_length_regression"] = regression(
            rows_array(rows, "total_length_mean"),
            rows_array(rows, "actor_train_time_s"),
            common_total_length,
        )
        saturated = [
            row
            for row in rows
            if float(row["response_length_mean"]) == 7168.0
            and float(row["rollout_truncated_ratio"]) == 1.0
        ]
        summary["groups"][group]["fully_truncated_subset"] = {
            "count": len(saturated),
            "rollout_time_s": describe(rows_array(saturated, "rollout_time_s")),
            "actor_train_time_s": describe(rows_array(saturated, "actor_train_time_s")),
        }
        steady = rows[1:]
        summary["groups"][group]["steady_steps_1_199"] = {
            "step_time_s": describe(rows_array(steady, "step_time_s")),
            "rollout_time_s": describe(rows_array(steady, "rollout_time_s")),
            "actor_train_time_s": describe(rows_array(steady, "actor_train_time_s")),
        }

    g11 = summary["groups"]["G11"]
    g10 = summary["groups"]["G10"]
    step_gap = g11["step_time_s"]["mean"] - g10["step_time_s"]["mean"]
    stages = [
        "rollout_time_s",
        "update_weights_time_s",
        "wait_residual_time_s",
        "actor_train_time_s",
        "train_residual_time_s",
    ]
    summary["gap"] = {
        "g11_minus_g10_step_time_s": step_gap,
        "g11_minus_g10_total_hours": (g11["step_time_s"]["sum"] - g10["step_time_s"]["sum"]) / 3600,
        "g10_step_time_reduction_fraction": 1
        - g10["step_time_s"]["mean"] / g11["step_time_s"]["mean"],
        "g10_end_to_end_speedup": g11["step_time_s"]["mean"] / g10["step_time_s"]["mean"],
        "g10_rollout_throughput_speedup": g10["rollout_tok_per_gpu_s"]["mean"]
        / g11["rollout_tok_per_gpu_s"]["mean"],
        "g10_actor_throughput_speedup": g10["actor_train_tok_s"]["mean"]
        / g11["actor_train_tok_s"]["mean"],
        "rollout_common_length_gap_s": g11["rollout_length_regression"]["prediction_at_common_x"]
        - g10["rollout_length_regression"]["prediction_at_common_x"],
        "actor_common_length_gap_s": g11["actor_length_regression"]["prediction_at_common_x"]
        - g10["actor_length_regression"]["prediction_at_common_x"],
        "stage_contributions": {},
        "bootstrap_mean_gap_ci95": {},
    }
    for stage in stages:
        gap = g11[stage]["mean"] - g10[stage]["mean"]
        summary["gap"]["stage_contributions"][stage] = {
            "g11_minus_g10_s": gap,
            "share_of_step_gap": gap / step_gap,
        }
    for key in ("step_time_s", "rollout_time_s", "actor_train_time_s"):
        summary["gap"]["bootstrap_mean_gap_ci95"][key] = bootstrap_mean_difference(
            rows_array(group_rows["G11"], key), rows_array(group_rows["G10"], key)
        )
    summary["method_notes"] = {
        "rollout_time": (
            "Exact perf/rollout_time emitted by RolloutManager; "
            "tqdm integer seconds are retained only as an audit cross-check."
        ),
        "wait_residual": (
            "train_wait - exact rollout_time - update_weights_time; "
            "captures wake/offload/orchestration outside the two named timers."
        ),
        "throughput": (
            "Uses emitted perf/tokens_per_gpu_per_sec and perf/actor_train_tok_per_s, "
            "avoiding response-length confounding."
        ),
        "common_length": (
            "Separate OLS fits for each arm evaluated at the pooled mean length; "
            "descriptive rather than causal because the two runs use different operator stacks, "
            "TE versions, revisions and generated sequences."
        ),
        "bootstrap": (
            "Independent non-parametric bootstrap of per-step mean gaps "
            "with fixed seed 1234 and 20,000 draws."
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="directory containing g10.run.log and g11.run.log",
    )
    parser.add_argument("--g10-log", type=Path, help="sealed G10 run.log")
    parser.add_argument("--g11-log", type=Path, help="sealed G11 run.log")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.data_dir is not None and (args.g10_log is not None or args.g11_log is not None):
        parser.error("use either --data-dir or both --g10-log/--g11-log")
    if args.data_dir is not None:
        g10_log = args.data_dir / "g10.run.log"
        g11_log = args.data_dir / "g11.run.log"
    elif args.g10_log is not None and args.g11_log is not None:
        g10_log = args.g10_log
        g11_log = args.g11_log
    else:
        parser.error("provide --data-dir or both --g10-log and --g11-log")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parsed = {
        "G11": parse_log(g11_log),
        "G10": parse_log(g10_log),
    }
    group_rows = {
        group: build_rows(group, records, progress) for group, (records, progress) in parsed.items()
    }
    all_rows = group_rows["G11"] + group_rows["G10"]
    write_csv(args.output_dir / "step-metrics.csv", all_rows)
    summary = summarize(group_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    comparison_rows: list[dict[str, Any]] = []
    for key, g11_value in summary["groups"]["G11"].items():
        g10_value = summary["groups"]["G10"].get(key)
        if not isinstance(g11_value, dict) or "mean" not in g11_value:
            continue
        g11_mean = g11_value["mean"]
        g10_mean = g10_value["mean"]
        comparison_rows.append(
            {
                "metric": key,
                "g11_mean": g11_mean,
                "g10_mean": g10_mean,
                "g11_minus_g10": g11_mean - g10_mean,
                "g10_over_g11": g10_mean / g11_mean if g11_mean else None,
            }
        )
    write_csv(args.output_dir / "summary.csv", comparison_rows)

    plot_decomposition(group_rows, args.output_dir)
    plot_scaling(group_rows, args.output_dir)
    plot_time_series(group_rows, args.output_dir)
    plot_throughput(group_rows, args.output_dir)
    print(json.dumps(summary["gap"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
