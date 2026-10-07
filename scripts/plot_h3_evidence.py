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
    """Grouped bars with a 2px-style gap and value labels on top."""

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
    return [v if v > 0 else LOG_FLOOR for v in values]


def _figure(report: dict[str, Any], suptitle: str):
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


PLOTS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "timestep_sinusoid_h3": plot_sinusoid,
}


def main() -> None:
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
