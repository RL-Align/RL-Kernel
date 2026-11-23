#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render the Qwen3-Next end-to-end speed test as a PNG.

    python benchmarks/models/qwen3_next/plot_qwen3_next_e2e_bench.py <dir>/report.json [<more>]

writes ``figure.png`` next to the last report: one row per phase (rollout, replay, train) and
one column per workload. Each later report replaces the phases it measured (for example a
rollout re-measured at another commit), and the title then names each phase's commit. Each
panel shows the median seconds per shape for every arm (log scale, lower is better). The
ratios are in the report's ``strict_over_native``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.models.qwen3_next.e2e_bench_cases import (  # noqa: E402
    combine_reports,
    commit_line,
    headline,
)

INK = "#2b2b29"
MUTED = "#6f6e69"
SURFACE = "#fcfcfb"
COLORS = {
    "strict": "#2a78d6",
    "native": "#7b4fc4",
    "native_graph": "#eb6834",
    "native_eager": "#f29a62",
    "native_bi": "#a5461b",
}
LABELS = {
    "strict": "RL-Kernel strict",
    "native": "stock VIME",
    "native_graph": "stock vLLM (CUDA graphs)",
    "native_eager": "stock vLLM (eager)",
    "native_bi": "stock vLLM, VLLM_BATCH_INVARIANT=1",
}
TITLES = {
    "rollout": "Rollout: vLLM generate (prefill + decode)",
    "replay": "Replay: forward + selected-token logprob",
    "train": "Training: one train_one_step",
}


def panel(axis, phase, results, workload):
    arms = [arm for arm in COLORS if workload in results.get(arm, {})]
    shapes = []
    for arm in arms:
        for shape in results[arm][workload]:
            if shape not in shapes:
                shapes.append(shape)
    if not shapes:
        axis.set_axis_off()
        return
    width = 0.8 / len(arms)
    x = np.arange(len(shapes))
    for index, arm in enumerate(arms):
        values = [
            (
                headline(phase, results[arm][workload][shape])
                if shape in results[arm][workload]
                else np.nan
            )
            for shape in shapes
        ]
        axis.bar(
            x + (index - (len(arms) - 1) / 2) * width,
            values,
            width,
            color=COLORS[arm],
            label=LABELS[arm],
        )
    axis.set_yscale("log")
    axis.set_xticks(x, shapes, rotation=20, fontsize=8, color=INK)
    axis.set_ylabel("seconds (median)", color=MUTED)
    axis.set_title(f"{TITLES[phase]}: {workload}", fontsize=10, color=INK, loc="left")
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def layout_line(report):
    """Each arm's parallel layout, from the replay phase's ``layouts`` when the report has it."""
    layouts = report["phases"].get("replay", {}).get("layouts")
    if not layouts:
        return f"TP4 EP{report['topology']['ep']}"

    def text(layout):
        sp = " + SP" if layout.get("sp") else ""
        return f"TP{layout['tp']}{sp}, EP{layout['ep']}, ETP{layout['etp']}, DP{layout['dp']}"

    return (
        f"RL-Kernel strict: {text(layouts['strict'])}; "
        f"stock VIME replay and training: {text(layouts['native'])}"
    )


def render(report_paths):
    report_paths = [Path(path) for path in report_paths]
    report, commits = combine_reports([json.loads(path.read_text()) for path in report_paths])
    phases = [phase for phase in ("rollout", "replay", "train") if phase in report["phases"]]
    workloads = list(report["settings"]["workloads"])
    figure, axes = plt.subplots(
        len(phases),
        len(workloads),
        figsize=(6.5 * len(workloads), 3.6 * len(phases)),
        squeeze=False,
        facecolor=SURFACE,
    )
    for row, phase in enumerate(phases):
        for column, workload in enumerate(workloads):
            panel(axes[row][column], phase, report["phases"][phase]["results"], workload)
    handles, labels = {}, {}
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            handles.setdefault(label, handle)
            labels[label] = label
    refused = {
        LABELS[row["arm"]]
        for phase in report["phases"].values()
        for row in phase.get("unsupported", [])
    }
    names = [f"{label} (refused, not measured)" if label in refused else label for label in labels]
    figure.legend(handles.values(), names, loc="lower center", ncol=3, frameon=False)
    gpus = ", ".join(report["versions"].get("gpus", []))
    figure.suptitle(
        f"Qwen3-Next-80B-A3B end-to-end on {gpus}, {commit_line(commits)}\n"
        f"{layout_line(report)}",
        color=INK,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.93))
    output = report_paths[-1].with_name("figure.png")
    figure.savefig(output, dpi=150)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    print(render(parser.parse_args().reports))


if __name__ == "__main__":
    main()
