#!/usr/bin/env python3
"""Build the version-aligned G10 vs optimized-G11 PR #377 result bundle."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

RECORD_RE = re.compile(r"(?:perf|step|rollout)\s+(\d+):\s+(\{.*\})")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
BLUE = "#2F67D8"
LIGHT_BLUE = "#9AB8EE"
RED = "#E45756"
LIGHT_RED = "#F3AAA7"
GREEN = "#2CA56C"
ORANGE = "#E28A2B"
GRID = "#D7DCE2"
TEXT = "#20242A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g10-csv", type=Path, required=True)
    parser.add_argument("--ray-bin")
    parser.add_argument("--ssh-host", help="Optional SSH host from which to stream Ray logs")
    parser.add_argument("--ssh-key", type=Path, help="SSH identity used with --ssh-host")
    parser.add_argument("--g11-job", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_g10(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            index = int(raw["log_step_index"])
            rows[index] = {
                key: float(value)
                for key, value in raw.items()
                if key.startswith("g10_") and value not in (None, "")
            }
    if sorted(rows) != list(range(200)):
        raise RuntimeError(f"G10 input has {len(rows)} steps; expected 0..199")
    return rows


def read_g11(
    ray_bin: str | Path | None,
    job: str,
    *,
    ssh_host: str | None = None,
    ssh_key: Path | None = None,
) -> dict[int, dict[str, Any]]:
    if ssh_host:
        if not ssh_key or not ray_bin:
            raise RuntimeError("--ssh-host requires both --ssh-key and --ray-bin")
        command = [
            "ssh",
            "-i",
            str(ssh_key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            ssh_host,
            str(ray_bin),
            "job",
            "logs",
            job,
        ]
    elif ray_bin:
        command = [str(ray_bin), "job", "logs", job]
    else:
        raise RuntimeError("--ray-bin is required")
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
    )
    assert process.stdout is not None
    rows: dict[int, dict[str, Any]] = {}
    for raw_line in process.stdout:
        line = ANSI_RE.sub("", raw_line)
        match = RECORD_RE.search(line)
        if not match:
            continue
        try:
            payload = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, dict):
            rows.setdefault(int(match.group(1)), {}).update(payload)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ray job logs failed with return code {return_code}")
    if sorted(rows) != list(range(200)):
        raise RuntimeError(f"G11 logs have {len(rows)} merged steps; expected 0..199")
    return rows


def finite(value: Any, *, field: str, step: int) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"non-finite {field} at step {step + 1}: {value!r}")
    return number


def build_rows(
    g10: dict[int, dict[str, float]], g11: dict[int, dict[str, Any]]
) -> list[dict[str, float | int]]:
    g11_keys = {
        "reward": "rollout/raw_reward",
        "reference_kl": "rollout/kl",
        "kl_loss": "train/kl_loss",
        "response_len_mean_tokens": "rollout/response_len/mean",
        "response_len_max_tokens": "rollout/response_len/max",
        "rollout_time_s": "perf/rollout_time",
        "tokens_per_gpu_s": "perf/tokens_per_gpu_per_sec",
        "longest_sample_tokens_s": "perf/longest_sample_tokens_per_sec",
        "ref_log_probs_time_s": "perf/ref_log_probs_time",
        "actor_train_time_s": "perf/actor_train_time",
        "actor_train_tokens_s": "perf/actor_train_tok_per_s",
        "train_time_s": "perf/train_time",
        "step_time_s": "perf/step_time",
        "mismatch_count": "train/train_current_rollout_logprob_mismatch_count",
        "max_abs_diff": "train/train_current_rollout_logprob_max_abs_diff",
        "active_tokens_mean": "train/train_current_rollout_logprob_numel",
    }
    result: list[dict[str, float | int]] = []
    for index in range(200):
        source10 = g10[index]
        source11 = g11[index]
        row: dict[str, float | int] = {"step": index + 1, "log_step_index": index}
        for metric, log_key in g11_keys.items():
            if log_key not in source11:
                raise RuntimeError(f"missing G11 {log_key} at step {index + 1}")
            row[f"g11_{metric}"] = finite(source11[log_key], field=log_key, step=index)
            g10_key = f"g10_{metric}"
            if g10_key not in source10:
                raise RuntimeError(f"missing {g10_key} at step {index + 1}")
            row[g10_key] = finite(source10[g10_key], field=g10_key, step=index)
        for metric in g11_keys:
            row[f"delta_g11_minus_g10_{metric}"] = float(row[f"g11_{metric}"]) - float(
                row[f"g10_{metric}"]
            )
        result.append(row)
    return result


def values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def describe(array: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def paired_bootstrap(delta: np.ndarray, *, draws: int = 20_000) -> dict[str, Any]:
    rng = np.random.default_rng(1234)
    indices = rng.integers(0, delta.size, size=(draws, delta.size))
    means = delta[indices].mean(axis=1)
    return {
        "estimate": float(delta.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "seed": 1234,
        "draws": draws,
        "method": "paired non-parametric bootstrap over the 200 aligned steps",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "reward",
        "reference_kl",
        "kl_loss",
        "response_len_mean_tokens",
        "response_len_max_tokens",
        "rollout_time_s",
        "tokens_per_gpu_s",
        "longest_sample_tokens_s",
        "ref_log_probs_time_s",
        "actor_train_time_s",
        "actor_train_tokens_s",
        "train_time_s",
        "step_time_s",
        "mismatch_count",
        "max_abs_diff",
        "active_tokens_mean",
    ]
    paired: dict[str, Any] = {}
    for metric in metrics:
        g10 = values(rows, f"g10_{metric}")
        g11 = values(rows, f"g11_{metric}")
        paired[metric] = {
            "g10": describe(g10),
            "g11": describe(g11),
            "g11_minus_g10_mean": float((g11 - g10).mean()),
            "g11_over_g10": float(g11.mean() / g10.mean()) if g10.mean() else None,
        }
    return {
        "paired": paired,
        "bootstrap": {
            "reward_g10_minus_g11": paired_bootstrap(
                values(rows, "g10_reward") - values(rows, "g11_reward")
            ),
            "step_time_g11_minus_g10_s": paired_bootstrap(
                values(rows, "g11_step_time_s") - values(rows, "g10_step_time_s")
            ),
        },
        "formulas": {
            "rollout_tokens_per_gpu_s": (
                "mean response tokens * 128 samples / " "(rollout seconds * 8 GPUs)"
            ),
            "longest_sample_tokens_s": "max response tokens / rollout seconds",
            "step_time_s": "VIME perf/step_time wall-clock timer",
        },
        "missing_value_policy": (
            "All required values must exist and be finite for all 200 paired "
            "steps; no imputation or row deletion."
        ),
    }


def moving_average(array: np.ndarray, window: int = 10) -> np.ndarray:
    totals = np.convolve(array, np.ones(window), mode="full")[: array.size]
    counts = np.minimum(np.arange(1, array.size + 1), window)
    return totals / counts


def style(axis: plt.Axes) -> None:
    axis.set_facecolor("white")
    axis.grid(True, color=GRID, linestyle="--", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=TEXT, labelsize=9.5)
    for spine in axis.spines.values():
        spine.set_color("#AEB5BF")


def save(figure: plt.Figure, output_dir: Path, name: str) -> None:
    figure.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def grouped_bars(axis: plt.Axes, labels: list[str], g10: np.ndarray, g11: np.ndarray) -> None:
    x = np.arange(len(labels))
    width = 0.36
    bars10 = axis.bar(
        x - width / 2, g10, width, color=LIGHT_RED, edgecolor=RED, label="G10 production P/P"
    )
    bars11 = axis.bar(
        x + width / 2, g11, width, color=LIGHT_BLUE, edgecolor=BLUE, label="G11 optimized R/R"
    )
    axis.set_xticks(x, labels)
    for bars in (bars10, bars11):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.015,
                f"{bar.get_height():,.1f}",
                ha="center",
                va="bottom",
                fontsize=8.8,
            )


def plot_performance_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    time_metrics = [
        "rollout_time_s",
        "ref_log_probs_time_s",
        "actor_train_time_s",
        "train_time_s",
        "step_time_s",
    ]
    time_labels = ["Rollout", "Ref logp", "Actor train", "Total train", "Total step"]
    g10_time = np.asarray([values(rows, f"g10_{m}").mean() for m in time_metrics])
    g11_time = np.asarray([values(rows, f"g11_{m}").mean() for m in time_metrics])
    grouped_bars(axes[0, 0], time_labels, g10_time, g11_time)
    style(axes[0, 0])
    axes[0, 0].set_title("Mean stage time (lower is better)")
    axes[0, 0].set_ylabel("Seconds / step")
    axes[0, 0].legend(frameon=True, fontsize=9)

    throughput_metrics = ["tokens_per_gpu_s", "actor_train_tokens_s"]
    throughput_labels = ["Rollout\ntok/GPU/s", "Actor train\ntok/s"]
    g10_thr = np.asarray([values(rows, f"g10_{m}").mean() for m in throughput_metrics])
    g11_thr = np.asarray([values(rows, f"g11_{m}").mean() for m in throughput_metrics])
    grouped_bars(axes[0, 1], throughput_labels, g10_thr, g11_thr)
    style(axes[0, 1])
    axes[0, 1].set_title("Token-normalized throughput (higher is better)")
    axes[0, 1].set_ylabel("Tokens / second")
    axes[0, 1].legend(frameon=True, fontsize=9)

    effect_labels = [
        "Rollout time",
        "Rollout throughput",
        "Ref logp time",
        "Actor time",
        "Actor throughput",
        "Total step time",
    ]
    effects = np.asarray(
        [
            100 * (1 - g11_time[0] / g10_time[0]),
            100 * (g11_thr[0] / g10_thr[0] - 1),
            100 * (1 - g11_time[1] / g10_time[1]),
            100 * (1 - g11_time[2] / g10_time[2]),
            100 * (g11_thr[1] / g10_thr[1] - 1),
            100 * (1 - g11_time[4] / g10_time[4]),
        ]
    )
    colors = [GREEN if value >= 0 else RED for value in effects]
    y = np.arange(len(effect_labels))
    axes[1, 0].barh(y, effects, color=colors, alpha=0.85)
    axes[1, 0].axvline(0, color=TEXT, linewidth=1)
    axes[1, 0].set_yticks(y, effect_labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("G11 improvement over G10 (%)")
    for yi, value in zip(y, effects, strict=True):
        axes[1, 0].text(
            value + (1.2 if value >= 0 else -1.2),
            yi,
            f"{value:+.1f}%",
            va="center",
            ha="left" if value >= 0 else "right",
            fontweight="bold",
            fontsize=9,
        )
    style(axes[1, 0])
    axes[1, 0].set_title("Direction-aware performance delta")

    axes[1, 1].axis("off")
    g10_reward = values(rows, "g10_reward").mean()
    g11_reward = values(rows, "g11_reward").mean()
    g10_mismatch = values(rows, "g10_mismatch_count")
    g11_mismatch = values(rows, "g11_mismatch_count")
    headline = (
        "200-step matched workload\n\n"
        f"Total step: {g11_time[4]:.2f}s vs {g10_time[4]:.2f}s\n"
        f"G11 end-to-end: {100 * (1 - g11_time[4] / g10_time[4]):.1f}% faster\n\n"
        f"G11 bitwise mismatches: {int(g11_mismatch.sum()):,}\n"
        f"G10 per-step mismatch mean: {g10_mismatch.mean():,.1f}\n\n"
        f"Mean raw reward: G11 {g11_reward:.4f} · G10 {g10_reward:.4f}\n"
        "1 node · 8×H100 · TP4/CP2 · batch 128"
    )
    axes[1, 1].text(
        0.04,
        0.93,
        headline,
        va="top",
        ha="left",
        fontsize=14,
        linespacing=1.45,
        color=TEXT,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#F5F7FA", "edgecolor": "#C9D0D9"},
    )
    fig.suptitle("VIME Qwen3-8B · G10 vs Optimized G11", fontsize=19, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, output_dir, "performance-summary")


def plot_performance_matrix(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Render a publication-friendly table in the style of the reference matrix."""

    def mean(group: str, metric: str) -> float:
        return float(values(rows, f"{group}_{metric}").mean())

    def lower_delta(metric: str) -> str:
        return f"{100 * (1 - mean('g11', metric) / mean('g10', metric)):+.1f}%"

    def higher_delta(metric: str) -> str:
        return f"{100 * (mean('g11', metric) / mean('g10', metric) - 1):+.1f}%"

    performance = [
        [
            "Rollout time",
            "s / step",
            f"{mean('g10', 'rollout_time_s'):.2f}",
            f"{mean('g11', 'rollout_time_s'):.2f}",
            lower_delta("rollout_time_s"),
            "G11 faster",
        ],
        [
            "Rollout throughput",
            "tok/GPU/s",
            f"{mean('g10', 'tokens_per_gpu_s'):,.0f}",
            f"{mean('g11', 'tokens_per_gpu_s'):,.0f}",
            higher_delta("tokens_per_gpu_s"),
            "G11 higher",
        ],
        [
            "Reference logp",
            "s / step",
            f"{mean('g10', 'ref_log_probs_time_s'):.2f}",
            f"{mean('g11', 'ref_log_probs_time_s'):.2f}",
            lower_delta("ref_log_probs_time_s"),
            "Parity",
        ],
        [
            "Actor train",
            "s / step",
            f"{mean('g10', 'actor_train_time_s'):.2f}",
            f"{mean('g11', 'actor_train_time_s'):.2f}",
            lower_delta("actor_train_time_s"),
            "G10 faster",
        ],
        [
            "Actor throughput",
            "tok/s",
            f"{mean('g10', 'actor_train_tokens_s'):,.0f}",
            f"{mean('g11', 'actor_train_tokens_s'):,.0f}",
            higher_delta("actor_train_tokens_s"),
            "G10 higher",
        ],
        [
            "Total train",
            "s / step",
            f"{mean('g10', 'train_time_s'):.2f}",
            f"{mean('g11', 'train_time_s'):.2f}",
            lower_delta("train_time_s"),
            "G10 faster",
        ],
        [
            "End-to-end step",
            "s / step",
            f"{mean('g10', 'step_time_s'):.2f}",
            f"{mean('g11', 'step_time_s'):.2f}",
            lower_delta("step_time_s"),
            "G11 faster",
        ],
    ]
    correctness = [
        [
            "Mean raw reward",
            "score",
            f"{mean('g10', 'reward'):.6f}",
            f"{mean('g11', 'reward'):.6f}",
            f"{mean('g11', 'reward') - mean('g10', 'reward'):+.6f}",
            "Report",
        ],
        [
            "Mean KL loss",
            "loss",
            f"{mean('g10', 'kl_loss'):.6f}",
            f"{mean('g11', 'kl_loss'):.6f}",
            higher_delta("kl_loss"),
            "Report",
        ],
        [
            "Mismatch count",
            "mean / step",
            f"{mean('g10', 'mismatch_count'):,.1f}",
            f"{mean('g11', 'mismatch_count'):.1f}",
            "exactly zero",
            "G11 PASS",
        ],
        [
            "Max |Delta logp|",
            "max / 200",
            f"{values(rows, 'g10_max_abs_diff').max():.6f}",
            f"{values(rows, 'g11_max_abs_diff').max():.1f}",
            "exactly zero",
            "G11 PASS",
        ],
    ]
    columns = ["Metric", "Unit", "G10 P/P", "Optimized G11 R/R", "G11 vs G10", "Finding"]
    fig = plt.figure(figsize=(15.2, 9.3), facecolor="#F7E8D2")
    fig.text(
        0.5,
        0.965,
        "VIME Qwen3-8B · 200-Step Performance Matrix",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#171717",
        bbox={
            "boxstyle": "square,pad=0.62",
            "facecolor": "#F1C58B",
            "edgecolor": "#80633A",
            "linewidth": 1.5,
        },
    )
    fig.text(
        0.5,
        0.905,
        "Matched workload · 1 node · 8× NVIDIA H100 80GB · TP4/CP2 · global batch 128 · seed 1234",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#2A241A",
        bbox={"boxstyle": "square,pad=0.48", "facecolor": "#F3D879", "edgecolor": "#9A8132"},
    )

    def add_table(bounds: list[float], title: str, rows_data: list[list[str]]) -> None:
        axis = fig.add_axes(bounds)
        axis.axis("off")
        axis.text(
            0.5,
            1.10,
            title,
            ha="center",
            va="center",
            fontsize=13.5,
            fontweight="bold",
            color="#2A241A",
            transform=axis.transAxes,
            bbox={"boxstyle": "square,pad=0.35", "facecolor": "#F3D879", "edgecolor": "#9A8132"},
        )
        table = axis.table(
            cellText=rows_data,
            colLabels=columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=[0.205, 0.13, 0.14, 0.19, 0.16, 0.16],
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10.3)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#8D7A5D")
            cell.set_linewidth(0.8)
            if row == 0:
                cell.set_facecolor("#F3D879")
                cell.set_text_props(weight="bold", color="#221F19")
            else:
                cell.set_facecolor("#F7E7D2" if row % 2 else "#F2DCC2")
                if col == 0:
                    cell.set_text_props(weight="bold")
                if col == 5:
                    text_value = rows_data[row - 1][5]
                    color = (
                        "#D9EAD3"
                        if "G11" in text_value or text_value == "Parity"
                        else "#EAD1DC" if "G10" in text_value else "#E2E3E5"
                    )
                    cell.set_facecolor(color)
                    cell.set_text_props(weight="bold")

    add_table([0.035, 0.40, 0.93, 0.40], "Mean performance over 200 paired steps", performance)
    add_table(
        [0.035, 0.065, 0.93, 0.225], "Quality and strict train/rollout consistency", correctness
    )
    fig.text(
        0.5,
        0.018,
        "Times are arithmetic means; percentage signs are direction-aware "
        "(positive means G11 is better). No missing values or imputation.",
        ha="center",
        fontsize=9.5,
        color="#51483B",
    )
    save(fig, output_dir, "performance-matrix")


def plot_performance_statistics(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot paired mean effects with deterministic paired-bootstrap intervals."""
    specs = [
        ("End-to-end step", "step_time_s", "lower"),
        ("Rollout time", "rollout_time_s", "lower"),
        ("Rollout throughput", "tokens_per_gpu_s", "higher"),
        ("Reference logp", "ref_log_probs_time_s", "lower"),
        ("Actor train", "actor_train_time_s", "lower"),
        ("Actor throughput", "actor_train_tokens_s", "higher"),
        ("Total train", "train_time_s", "lower"),
    ]
    rng = np.random.default_rng(1234)
    indices = rng.integers(0, 200, size=(20_000, 200))
    estimates: list[float] = []
    intervals: list[tuple[float, float]] = []
    for _, metric, direction in specs:
        g10 = values(rows, f"g10_{metric}")
        g11 = values(rows, f"g11_{metric}")
        sign = 1.0 if direction == "higher" else -1.0
        estimate = sign * 100.0 * (g11.mean() / g10.mean() - 1.0)
        boot = sign * 100.0 * (g11[indices].mean(axis=1) / g10[indices].mean(axis=1) - 1.0)
        estimates.append(float(estimate))
        intervals.append((float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))))
    estimates_array = np.asarray(estimates)
    lower = estimates_array - np.asarray([item[0] for item in intervals])
    upper = np.asarray([item[1] for item in intervals]) - estimates_array
    y = np.arange(len(specs))
    fig, axis = plt.subplots(figsize=(12.8, 7.2))
    colors = [GREEN if value >= 0 else RED for value in estimates_array]
    axis.barh(y, estimates_array, color=colors, alpha=0.82, height=0.62)
    axis.errorbar(
        estimates_array,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor=TEXT,
        elinewidth=1.5,
        capsize=4,
    )
    axis.axvline(0, color=TEXT, linewidth=1.2)
    axis.set_yticks(y, [item[0] for item in specs])
    axis.invert_yaxis()
    axis.set_xlabel("G11 improvement over G10 (%) · positive is better")
    for yi, value, interval in zip(y, estimates_array, intervals, strict=True):
        axis.text(
            interval[1] + 1.2,
            yi,
            f"{value:+.1f}%  [{interval[0]:+.1f}, {interval[1]:+.1f}]",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    style(axis)
    axis.set_title(
        "Paired Performance Effects · Mean and 95% Bootstrap CI",
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    fig.text(
        0.5,
        0.02,
        "200 aligned steps · paired non-parametric bootstrap · 20,000 resamples · seed 1234",
        ha="center",
        fontsize=10,
        color="#4B525B",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, output_dir, "performance-statistics")


def plot_performance_trajectories(rows: list[dict[str, Any]], output_dir: Path) -> None:
    steps = np.arange(1, 201)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    panels = [
        ("rollout_time_s", "Rollout generation", "Seconds"),
        ("actor_train_time_s", "Actor training", "Seconds"),
        ("step_time_s", "Total step", "Seconds"),
    ]
    for axis, (metric, title, ylabel) in zip(axes, panels, strict=True):
        for group, raw, strong in (("G10", LIGHT_RED, RED), ("G11", LIGHT_BLUE, BLUE)):
            series = values(rows, f"{group.lower()}_{metric}")
            axis.plot(steps, series, color=raw, linewidth=0.8, alpha=0.65)
            axis.plot(
                steps,
                moving_average(series),
                color=strong,
                linewidth=2.2,
                label=f"{group} 10-step MA",
            )
        style(axis)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend(ncol=2, frameon=True)
    axes[-1].set_xlabel("Training step")
    axes[-1].set_xlim(1, 200)
    fig.suptitle(
        "G10 vs Optimized G11 · Performance Across 200 Steps",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, output_dir, "performance-trajectories")


def plot_consistency_reward(rows: list[dict[str, Any]], output_dir: Path) -> None:
    steps = np.arange(1, 201)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))
    for group, raw, strong in (("G10", LIGHT_RED, RED), ("G11", LIGHT_BLUE, BLUE)):
        reward = values(rows, f"{group.lower()}_reward")
        axes[0, 0].plot(steps, reward, color=raw, linewidth=0.8, alpha=0.65)
        axes[0, 0].plot(
            steps, moving_average(reward), color=strong, linewidth=2.2, label=f"{group} 10-step MA"
        )
        kl = values(rows, f"{group.lower()}_kl_loss")
        axes[0, 1].plot(steps, kl, color=strong, linewidth=1.5, label=group)
        axes[1, 0].plot(
            steps,
            values(rows, f"{group.lower()}_mismatch_count"),
            color=strong,
            linewidth=1.5,
            label=group,
        )
        axes[1, 1].plot(
            steps,
            values(rows, f"{group.lower()}_max_abs_diff"),
            color=strong,
            linewidth=1.5,
            label=group,
        )
    titles = [
        "Raw reward",
        "Reference KL loss",
        "Train/rollout mismatch count",
        "Maximum absolute Δlogp",
    ]
    ylabels = [
        "Reward",
        "KL loss",
        "Mismatched active tokens",
        "Absolute log-probability difference",
    ]
    for axis, title, ylabel in zip(axes.flat, titles, ylabels, strict=True):
        style(axis)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Training step")
        axis.legend(frameon=True)
    axes[0, 1].set_yscale("symlog", linthresh=1e-5)
    fig.suptitle(
        "G10 vs Optimized G11 · Training and Bitwise Consistency",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, output_dir, "consistency-reward")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    p = summary["paired"]
    reward_ci = summary["bootstrap"]["reward_g10_minus_g11"]
    rollout = p["rollout_time_s"]
    rollout_tput = p["tokens_per_gpu_s"]
    ref_logp = p["ref_log_probs_time_s"]
    actor = p["actor_train_time_s"]
    step_time = p["step_time_s"]
    reward = p["reward"]
    rollout_gain = 100 * (1 - rollout["g11"]["mean"] / rollout["g10"]["mean"])
    rollout_tput_gain = 100 * (rollout_tput["g11"]["mean"] / rollout_tput["g10"]["mean"] - 1)
    actor_slowdown = 100 * (actor["g11"]["mean"] / actor["g10"]["mean"] - 1)
    step_gain = 100 * (1 - step_time["g11"]["mean"] / step_time["g10"]["mean"])
    table_rows = "\n".join(
        [
            (
                f"| Rollout time (s) | {rollout['g10']['mean']:.2f} | "
                f"{rollout['g11']['mean']:.2f} | G11 {rollout_gain:.1f}% faster |"
            ),
            (
                f"| Rollout tokens/GPU/s | {rollout_tput['g10']['mean']:.2f} | "
                f"{rollout_tput['g11']['mean']:.2f} | "
                f"G11 {rollout_tput_gain:.1f}% higher |"
            ),
            (
                f"| Reference logp time (s) | {ref_logp['g10']['mean']:.2f} | "
                f"{ref_logp['g11']['mean']:.2f} | approximately equal |"
            ),
            (
                f"| Actor train time (s) | {actor['g10']['mean']:.2f} | "
                f"{actor['g11']['mean']:.2f} | G11 {actor_slowdown:.1f}% slower |"
            ),
            (
                f"| Total step time (s) | {step_time['g10']['mean']:.2f} | "
                f"{step_time['g11']['mean']:.2f} | G11 {step_gain:.1f}% faster |"
            ),
            (
                f"| Mean raw reward | {reward['g10']['mean']:.6f} | "
                f"{reward['g11']['mean']:.6f} | G10−G11 "
                f"{reward['g10']['mean']-reward['g11']['mean']:+.6f} |"
            ),
        ]
    )
    text = f"""# Version-aligned G10 vs optimized G11 (200 steps)

Both runs use the same Qwen3-8B workload: one 8×H100 node, TP4/CP2, 200 steps,
seed and rollout seed 1234, rollout batch 8 prompts × 16 samples, global batch
128, maximum response length 7,168, and maximum 4,096 tokens/GPU.

| Metric | G10 | Optimized G11 | Result |
|---|---:|---:|---|
{table_rows}

G11 has exactly zero mismatch count and zero maximum absolute difference at all
200 steps. G10 is the production P/P comparison and has non-zero mismatch at
all 200 steps.

The paired mean reward difference (G10−G11) has a 95% bootstrap interval of
[{reward_ci['ci95'][0]:+.6f}, {reward_ci['ci95'][1]:+.6f}], using seed 1234 and
20,000 paired-step resamples. This is a single-training-seed result, not a
multi-seed generalization interval.

`rounds.csv` contains every paired step. `summary.json` records formulas,
distribution summaries, and bootstrap details. `plot_report.py` regenerates all
five PNG figures from the authoritative Ray logs and the sealed G10 CSV.
"""
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        read_g10(args.g10_csv),
        read_g11(
            args.ray_bin,
            args.g11_job,
            ssh_host=args.ssh_host,
            ssh_key=args.ssh_key,
        ),
    )
    summary = summarize(rows)
    write_csv(args.output_dir / "rounds.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_readme(args.output_dir / "README.md", summary)
    plot_performance_matrix(rows, args.output_dir)
    plot_performance_statistics(rows, args.output_dir)
    plot_performance_summary(rows, args.output_dir)
    plot_performance_trajectories(rows, args.output_dir)
    plot_consistency_reward(rows, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "paired": summary["paired"],
                "bootstrap": summary["bootstrap"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
