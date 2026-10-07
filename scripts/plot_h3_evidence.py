#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render an RFC #420 operator report (``scripts/h3_evidence.py``) as a PNG.

Needs only the report JSON and matplotlib (not torch or a GPU), so it can run
anywhere:

    python scripts/plot_h3_evidence.py docs/usage/evidence/h3-timestep-sinusoid-b200/report.json

writes ``figure.png`` next to the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed series identity across every H3 figure (categorical slots 1-3).
RL_KERNEL = "#2a78d6"
PROVIDER = "#eb6834"
THIRD = "#1baf7a"
INK = "#2b2b29"
MUTED = "#6f6e69"
GRID = "#e4e3dd"
SURFACE = "#fcfcfb"


def _style(ax, title: str, ylabel: str) -> None:
    """Apply the shared report axis styling, title, and vertical-axis label."""

    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _bars(ax, labels: list[str], series: list[tuple[str, str, list[float]]], fmt: str) -> None:
    """Draw grouped series on the axis with formatted value labels and a legend."""

    n = len(series)
    width = 0.8 / n
    for i, (name, color, values) in enumerate(series):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(labels))]
        bars = ax.bar(xs, values, width * 0.92, color=color, label=name, zorder=2)
        for bar, value in zip(bars, values, strict=True):
            ax.annotate(
                fmt.format(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=INK,
            )
    ax.set_xticks(range(len(labels)), labels)
    top = max(v for _, _, values in series for v in values)
    ax.set_ylim(0, top * 1.35)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", ncol=n)


# Exact zeros cannot sit on a log axis; draw them on this floor and label them.
LOG_FLOOR = 1e-9


def _log_values(values: list[float]) -> list[float]:
    """Replace nonpositive values with the plotting floor for logarithmic axes."""

    return [v if v > 0 else LOG_FLOOR for v in values]


def _figure(report: dict[str, Any], suptitle: str):
    """Create two report axes with a title and GPU, software, and revision provenance."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    env = report["environment"]
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(
        0.01,
        0.01,
        f"{env['gpu']} · torch {env['torch']} · CUDA {env['cuda']} · "
        f"commit {report['rl_kernel_commit'][:7]} · MiniMax-H3@{report['model_revision'][:7]}",
        fontsize=7,
        color=MUTED,
    )
    return fig, axes


def plot_sinusoid(report: dict[str, Any]):
    """Return a sinusoid report figure showing latency and errors against FP64."""

    fig, (left, right) = _figure(report, "timestep_sinusoid_h3")
    perf = report["perf"]
    _style(left, "Latency per call (lower is better)", "µs")
    _bars(
        left,
        [row["case"] for row in perf],
        [
            ("RL-Kernel CUDA", RL_KERNEL, [row["candidate_us"] for row in perf]),
            ("CUDA + range check", THIRD, [row["candidate_checked_us"] for row in perf]),
            ("diffusers", PROVIDER, [row["provider_us"] for row in perf]),
        ],
        "{:.0f}",
    )
    cases = report["accuracy"]["cases"]
    xs = [c["num_timesteps"] for c in cases]
    _style(right, "Max |error| vs FP64 golden (log)", "max abs error")
    cuda = [c["max_abs_vs_fp64"] for c in cases]
    provider = [c["provider_max_abs_vs_fp64"] for c in cases]
    right.plot(xs, _log_values(cuda), "o-", color=RL_KERNEL, lw=2, ms=8, label="RL-Kernel CUDA")
    right.plot(xs, _log_values(provider), "s--", color=PROVIDER, lw=2, ms=5, label="diffusers")
    atol = report["accuracy"]["contract_atol"]
    right.axhline(atol, color=MUTED, lw=1, ls=":")
    right.annotate(
        "contract atol 1e-5",
        (xs[-1], atol),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=7,
        color=MUTED,
        ha="right",
    )
    for x, value in zip(xs, cuda, strict=True):
        if value == 0:
            right.annotate(
                "exact",
                (x, LOG_FLOOR),
                xytext=(0, 8),
                textcoords="offset points",
                fontsize=7,
                color=INK,
                ha="center",
            )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_ylim(LOG_FLOOR / 3, atol * 30)
    right.set_xlabel("number of timesteps T", color=MUTED, fontsize=9)
    equal = sum(c["bitwise_equal_to_provider"] for c in cases)
    right.legend(
        frameon=False,
        fontsize=8,
        labelcolor=INK,
        loc="upper left",
        ncol=2,
        title=f"bitwise equal to diffusers at {equal}/{len(cases)} sizes",
        title_fontsize=8,
        alignment="left",
    )
    return fig


def _latency_panel(ax, perf: list[dict[str, Any]], provider_name: str) -> None:
    """Plot candidate and provider latency summaries in microseconds for each case."""

    _style(ax, "Latency per call (lower is better)", "µs")
    _bars(
        ax,
        [row["case"] for row in perf],
        [
            ("RL-Kernel CUDA", RL_KERNEL, [row["candidate_us"] for row in perf]),
            (provider_name, PROVIDER, [row["provider_us"] for row in perf]),
        ],
        "{:.0f}",
    )


def _error_boxes(ax, title: str, series: list[tuple[str, str, list[float]]], contract: float):
    """Per-draw max |error| as boxes with every draw overlaid, log scale."""

    _style(ax, title, "max abs error per draw")
    for i, (_name, color, values) in enumerate(series):
        ax.boxplot(
            [values],
            positions=[i],
            widths=0.45,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.25, "edgecolor": color},
            medianprops={"color": color, "linewidth": 2},
            whiskerprops={"color": color},
            capprops={"color": color},
        )
        jitter = [i + ((k * 0.6180339) % 1 - 0.5) * 0.3 for k in range(len(values))]
        ax.scatter(jitter, values, s=8, color=color, alpha=0.6, linewidths=0, zorder=3)
        median = sorted(values)[len(values) // 2]
        ax.annotate(
            f"median {median:.1e}",
            (i + 0.28, median),
            fontsize=8,
            color=INK,
            va="center",
        )
    ax.axhline(contract, color=MUTED, lw=1, ls=":")
    ax.annotate(
        f"contract atol {contract:g}",
        (len(series) - 0.5, contract),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=7,
        color=MUTED,
        ha="right",
    )
    ax.set_yscale("log")
    ax.set_xticks(range(len(series)), [name for name, _, _ in series])
    ax.set_xlim(-0.6, len(series) - 0.1)


def plot_mlp(report: dict[str, Any]):
    """Return an MLP report figure showing latency and per-draw FP64 error distributions."""

    fig, (left, right) = _figure(report, "timestep_mlp_fp32  (pinned time_embedder weights)")
    _latency_panel(left, report["perf"], "diffusers (cuBLAS)")
    acc = report["accuracy"]
    _error_boxes(
        right,
        f"Error vs FP64 golden, {acc['draws']} draws of T={acc['num_timesteps']}",
        [
            ("RL-Kernel CUDA", RL_KERNEL, acc["cuda_max_abs_vs_fp64"]),
            ("diffusers (cuBLAS)", PROVIDER, acc["provider_max_abs_vs_fp64"]),
        ],
        acc["contract_atol"],
    )
    return fig


def plot_projection(report: dict[str, Any]):
    """Return an AdaLN figure with latency, BF16 rounding accuracy, and the early-cast probe."""

    fig, (left, right) = _figure(report, "adaln_projection_3mod  (pinned block-0 AdaLN weights)")
    _latency_panel(left, report["perf"], "diffusers (cuBLAS)")
    acc = report["accuracy"]
    _style(
        right,
        f"Outputs equal to the correctly rounded FP64 golden, {acc['draws']} draws",
        "% of BF16 outputs",
    )
    series = [
        ("RL-Kernel CUDA", RL_KERNEL, [100 * v for v in acc["cuda_correctly_rounded"]]),
        ("diffusers (cuBLAS)", PROVIDER, [100 * v for v in acc["provider_correctly_rounded"]]),
    ]
    for i, (_name, color, values) in enumerate(series):
        jitter = [i + ((k * 0.6180339) % 1 - 0.5) * 0.3 for k in range(len(values))]
        ax_vals = sorted(values)
        median = ax_vals[len(ax_vals) // 2]
        right.scatter(jitter, values, s=36, color=color, alpha=0.8, linewidths=0, zorder=3)
        right.hlines(median, i - 0.25, i + 0.25, color=color, lw=2.5, zorder=4)
        right.annotate(
            f"median {median:.3f}%", (i + 0.28, median), fontsize=8, color=INK, va="center"
        )
    right.set_xticks(range(len(series)), [name for name, _, _ in series])
    right.set_xlim(-0.6, len(series) - 0.1)
    early = sorted(acc["early_cast_golden_match"])[len(acc["early_cast_golden_match"]) // 2]
    right.text(
        0.98,
        0.62,
        f"an early BF16 cast (probe H7)\nwould match only {100 * early:.0f}%",
        transform=right.transAxes,
        fontsize=8,
        color=MUTED,
        ha="right",
    )
    return fig


PLOTS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "timestep_sinusoid_h3": plot_sinusoid,
    "timestep_mlp_fp32": plot_mlp,
    "adaln_projection_3mod": plot_projection,
}


def main() -> None:
    """Read an operator report and save its figure as a PNG beside it or at ``--out``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    fig = PLOTS[report["op"]](report)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = args.out or args.report.with_name("figure.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
