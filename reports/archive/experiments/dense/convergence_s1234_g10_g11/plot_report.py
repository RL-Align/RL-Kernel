# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot the sealed G10/G11 consistency report from collected round records."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

BLUE = "#2F67D8"
LIGHT_BLUE = "#93B4F4"
RED = "#E53935"
LIGHT_RED = "#F4A3A0"
BAR_BLUE = "#A9C7ED"
BAR_BLUE_EDGE = "#356AC3"
BAR_ORANGE = "#F2B566"
BAR_ORANGE_EDGE = "#D66A27"
GRID = "#D7DCE2"
TEXT = "#20242A"
GROUP_ORDER = ("G11", "G10")
LABELS = {"G11": "G11 strict RL-Kernel", "G10": "G10 VIME-native baseline"}


def _parse(value: str) -> Any:
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def load_rounds(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {key: _parse(value) for key, value in raw.items()}
            group = str(row["group"])
            if group in GROUP_ORDER:
                grouped[group].append(row)
    for group in GROUP_ORDER:
        grouped[group].sort(key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in grouped[group]]
        if steps != list(range(200)):
            raise ValueError(f"{group}: expected steps 0..199, found {steps[:2]}..{steps[-2:]}")
    return grouped


def series(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = [row.get(metric) for row in rows]
    if any(not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"metric {metric!r} is incomplete")
    return [float(value) for value in values]


def trailing_ma(values: list[float], window: int = 10) -> list[float]:
    return [mean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    location = (len(ordered) - 1) * quantile
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def style_axis(axis) -> None:
    axis.set_facecolor("white")
    axis.grid(True, color=GRID, linestyle="--", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(TEXT)
        spine.set_linewidth(1.0)
    axis.tick_params(colors=TEXT, labelsize=10)


def save(figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def add_report_header(figure, title: str, subtitle: str) -> None:
    """Place a non-overlapping title/subtitle pair at the top of a report figure."""
    figure.suptitle(title, fontsize=19, fontweight="bold", y=0.99)
    figure.text(
        0.5,
        0.945,
        subtitle,
        ha="center",
        va="top",
        fontsize=11.5,
        color="#4A515A",
    )


def plot_pair(axis, steps, strict_values, native_values) -> None:
    axis.plot(steps, strict_values, color=LIGHT_BLUE, linewidth=1.0, alpha=0.78)
    axis.plot(steps, trailing_ma(strict_values), color=BLUE, linewidth=2.35)
    axis.plot(steps, native_values, color=LIGHT_RED, linewidth=1.0, alpha=0.72)
    axis.plot(steps, trailing_ma(native_values), color=RED, linewidth=2.35)


def consistency_trajectories(rows, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    steps = list(range(200))
    fig, axes = plt.subplots(3, 1, figsize=(14.5, 10.8), sharex=True)
    add_report_header(
        fig,
        "VIME · Qwen3-8B · 200-step GRPO (8×H100, TP4/CP2)",
        "Consistency dynamics: G11 strict RL-Kernel vs G10 VIME-native baseline",
    )
    panels = (
        ("mean_abs_dlogp", r"Mean absolute $\Delta\log p$", "Mean |Δ logp|"),
        ("max_abs_dlogp", r"Maximum absolute $\Delta\log p$", "Max |Δ logp|"),
    )
    for axis, (metric, title, ylabel) in zip(axes[:2], panels, strict=True):
        style_axis(axis)
        strict_values = series(rows["G11"], metric)
        native_values = series(rows["G10"], metric)
        plot_pair(axis, steps, strict_values, native_values)
        axis.set_title(title, fontsize=12.5, pad=8)
        axis.set_ylabel(ylabel, fontsize=11.5)
        upper = max(native_values)
        axis.set_ylim(-0.025 * upper, 1.08 * upper)
    mismatch = {group: series(rows[group], "mismatch_count") for group in GROUP_ORDER}
    style_axis(axes[2])
    plot_pair(axes[2], steps, mismatch["G11"], mismatch["G10"])
    axes[2].set_title("Bitwise log-probability mismatch count per step", fontsize=12.5, pad=8)
    axes[2].set_ylabel("Mismatch count", fontsize=11.5)
    axes[2].set_xlabel("Training Step", fontsize=12)
    mismatch_upper = max(mismatch["G10"])
    axes[2].set_ylim(-0.025 * mismatch_upper, 1.08 * mismatch_upper)
    axes[2].set_xlim(0, 199)
    axes[2].set_xticks(range(0, 200, 25))
    fig.legend(
        handles=[
            Line2D([0], [0], color=LIGHT_RED, label="G10 per-step"),
            Line2D([0], [0], color=RED, linewidth=2.4, label="G10 10-step MA"),
            Line2D([0], [0], color=BLUE, linewidth=1.9, label="G11 strict (exact zero)"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.095, 0.905),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#C9CDD2",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.025, 0.03, 0.99, 0.845), h_pad=1.15)
    save(fig, output_dir, "consistency-trajectories")
    plt.close(fig)


def consistency_summary(rows, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    summaries = {}
    for group in GROUP_ORDER:
        mismatches = series(rows[group], "mismatch_count")
        tokens = series(rows[group], "active_token_count")
        summaries[group] = {
            "agreement": 100.0 * (1.0 - sum(mismatches) / sum(tokens)),
            "mismatch_count": sum(mismatches),
            "mean_abs": mean(series(rows[group], "mean_abs_dlogp")),
            "p95_max": percentile(series(rows[group], "max_abs_dlogp"), 0.95),
        }
    panels: tuple[tuple[str, str, str, Callable[[float], str]], ...] = (
        ("agreement", "Bitwise agreement", "%", lambda value: f"{value:.2f}%"),
        (
            "mean_abs",
            r"Unweighted mean per-step $|\Delta\log p|$",
            "Mean per-step |Δ logp|",
            lambda value: "0" if value == 0 else f"{value:.5f}",
        ),
        (
            "p95_max",
            r"P95 maximum $|\Delta\log p|$",
            "P95 max |Δ logp|",
            lambda value: "0" if value == 0 else f"{value:.4f}",
        ),
        (
            "mismatch_count",
            "Total bitwise mismatches",
            "Mismatch count",
            lambda value: f"{value:,.0f}",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.7))
    add_report_header(
        fig,
        "VIME Consistency Summary · Qwen3-8B · 200-step GRPO",
        "G11 strict RL-Kernel vs G10 VIME-native baseline",
    )
    for axis, (key, title, ylabel, formatter) in zip(axes.flat, panels, strict=True):
        style_axis(axis)
        values = [summaries[group][key] for group in GROUP_ORDER]
        bars = axis.bar(
            [0, 1],
            values,
            width=0.46,
            color=[BAR_BLUE, BAR_ORANGE],
            edgecolor=[BAR_BLUE_EDGE, BAR_ORANGE_EDGE],
            linewidth=1.4,
        )
        axis.set_title(title, fontsize=12.5, pad=8)
        axis.set_ylabel(ylabel, fontsize=11)
        axis.set_xticks([0, 1], ["G11 strict", "G10 native"])
        axis.grid(False, axis="x")
        if key == "mismatch_count":
            from matplotlib.ticker import StrMethodFormatter

            axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        upper = max(values)
        axis.set_ylim(0, 1.18 * upper if upper else 1)
        for bar, value, edge in zip(bars, values, [BAR_BLUE_EDGE, BAR_ORANGE_EDGE], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025 * (upper or 1),
                formatter(value),
                ha="center",
                fontsize=10.5,
                color=edge,
                fontweight="bold",
            )
    fig.legend(
        handles=[
            Patch(facecolor=BAR_BLUE, edgecolor=BAR_BLUE_EDGE, label=LABELS["G11"]),
            Patch(facecolor=BAR_ORANGE, edgecolor=BAR_ORANGE_EDGE, label=LABELS["G10"]),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#C9CDD2",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        0.022,
        "Agreement = 1 − total bitwise mismatches / total active tokens; "
        "mismatch-count bars span all 200 training steps.",
        ha="center",
        fontsize=9.5,
        color="#606872",
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.99, 0.825), h_pad=2.1, w_pad=2.0)
    save(fig, output_dir, "consistency-summary")
    plt.close(fig)


def training_dynamics(rows, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter

    steps = list(range(200))
    panels = (
        ("raw_reward", "Rollout raw reward", "Raw reward", False),
        ("train_loss", "Training loss", "Loss", True),
        ("ppo_kl", "Train PPO KL", "PPO KL", True),
        ("pg_clipfrac", "PPO policy-ratio clip fraction", "Clip fraction", True),
    )
    fig, axes = plt.subplots(4, 1, figsize=(14.5, 14.0), sharex=True)
    add_report_header(
        fig,
        "VIME · Qwen3-8B · 200-step GRPO (8×H100, TP4/CP2)",
        "Training dynamics: G11 strict RL-Kernel vs G10 VIME-native baseline",
    )
    for axis, (metric, title, ylabel, zero_line) in zip(axes, panels, strict=True):
        style_axis(axis)
        strict_values = series(rows["G11"], metric)
        native_values = series(rows["G10"], metric)
        plot_pair(axis, steps, strict_values, native_values)
        if zero_line:
            axis.axhline(0, color="#8B939D", linewidth=0.9, linestyle="--")
        axis.set_title(title, fontsize=12.5, pad=8)
        axis.set_ylabel(ylabel, fontsize=11.5)
        lower = min(min(strict_values), min(native_values), 0.0 if zero_line else math.inf)
        upper = max(max(strict_values), max(native_values), 0.0 if zero_line else -math.inf)
        span = upper - lower or max(abs(upper), 1e-8)
        axis.set_ylim(lower - 0.07 * span, upper + 0.07 * span)
        if metric in {"train_loss", "ppo_kl"}:
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))
            axis.yaxis.set_major_formatter(formatter)
    axes[-1].set_xlabel("Training Step", fontsize=12)
    axes[-1].set_xlim(0, 199)
    axes[-1].set_xticks(range(0, 200, 25))
    fig.legend(
        handles=[
            Line2D([0], [0], color=LIGHT_BLUE, label="G11 per-step"),
            Line2D([0], [0], color=BLUE, linewidth=2.5, label="G11 10-step MA"),
            Line2D([0], [0], color=LIGHT_RED, label="G10 per-step"),
            Line2D([0], [0], color=RED, linewidth=2.5, label="G10 10-step MA"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.095, 0.905),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#C9CDD2",
        fontsize=10.5,
    )
    fig.text(
        0.986, 0.015, "Ratio metric: train/pg_clipfrac", ha="right", fontsize=9.3, color="#68717C"
    )
    fig.tight_layout(rect=(0.025, 0.028, 0.99, 0.85), h_pad=1.1)
    save(fig, output_dir, "training-dynamics")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds-csv", type=Path, default=Path("rounds.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10.5, "figure.facecolor": "white"}
    )
    rows = load_rounds(args.rounds_csv)
    consistency_trajectories(rows, args.output_dir)
    consistency_summary(rows, args.output_dir)
    training_dynamics(rows, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
