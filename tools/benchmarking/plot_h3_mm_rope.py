#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot all H3 RoPE benchmark cases from schema-v2 results.json (requires matplotlib).

Usage: python tools/benchmarking/plot_h3_mm_rope.py results.json --out figures
Writes latency, baseline speedup and incremental peak-allocation PNG/SVG figures.
No missing case is filled with historical data. Latency whiskers are sample P10/P90,
not confidence intervals; small eager calls can have substantial host-dispatch noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROVIDERS = {"provider": "Original eager H3", "native": "Native", "triton": "Triton"}
COLORS = {"provider": "#697586", "native": "#D18B32", "triton": "#137F78"}
MODES = {"forward": "Forward", "backward": "Backward", "forward_backward": "Forward + backward"}


def plot(report: dict, output: Path) -> None:
    """Render the complete measured case plan, with explicit units and baseline."""
    if report.get("schema_version") != 2 or not report["complete"]:
        raise ValueError("requires a complete schema-v2 benchmark with model-boundary acceptance")
    rows = {(r["dtype"], tuple(r["shape"]), r["mode"]): r for r in report["results"]}
    shapes = sorted({tuple(case["shape"]) for case in report["case_plan"]})
    labels = [f"{s[1]:,}\n{s[2]} heads" for s in shapes]
    xs = np.arange(len(shapes))
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for metric in ("latency", "speedup", "memory"):
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        for row, dtype in enumerate(("bf16", "fp32")):
            for column, (mode, mode_label) in enumerate(MODES.items()):
                ax = axes[row, column]
                cases = [rows[dtype, shape, mode] for shape in shapes]
                for key, label in PROVIDERS.items():
                    values = [case["timings"][key] for case in cases]
                    if metric == "latency":
                        median = np.array([v["median_us"] for v in values]) / 1000
                        bounds = (
                            np.array([np.percentile(v["samples_us"], [10, 90]) for v in values])
                            / 1000
                        )
                        ax.errorbar(
                            xs,
                            median,
                            yerr=[median - bounds[:, 0], bounds[:, 1] - median],
                            marker="o",
                            markersize=4,
                            capsize=2,
                            linewidth=1.6,
                            color=COLORS[key],
                            label=label,
                        )
                    elif metric == "memory":
                        ax.plot(
                            xs,
                            [v["extra_peak_mib"] for v in values],
                            "o-",
                            markersize=4,
                            color=COLORS[key],
                            label=label,
                        )
                    elif key != "triton":
                        ax.plot(
                            xs,
                            [
                                v["median_us"] / case["timings"]["triton"]["median_us"]
                                for v, case in zip(values, cases, strict=True)
                            ],
                            "o-",
                            markersize=4,
                            color=COLORS[key],
                            label=f"{label} / Triton",
                        )
                ax.set_title(f"{dtype.upper()} · {mode_label}", weight="bold")
                ax.set_xticks(xs, labels, fontsize=8)
                ax.set_xlabel("Packed sequence length S / head count H (B=1, D=128)")
                ax.grid(axis="y", alpha=0.2)
                if metric == "latency":
                    ax.set_yscale("log")
                    ax.set_ylabel("Latency (ms, log scale)")
                elif metric == "memory":
                    ax.set_yscale("log")
                    ax.set_ylabel("Incremental peak allocation (MiB, log scale)")
                else:
                    ax.axhline(1, color="#137F78", linestyle="--", linewidth=1)
                    ax.set_ylim(bottom=0)
                    ax.set_ylabel("Baseline latency / Triton latency (×)")
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=3,
            frameon=False,
        )
        title = {
            "latency": "Eager latency · median and P10–P90 samples",
            "speedup": "Speedup against both eager baselines · higher is faster",
            "memory": "Extra peak memory · inputs and prebuilt graphs excluded",
        }[metric]
        env = report["environment"]
        fig.suptitle(
            f"H3 partial MM-RoPE | {env['gpu']} | {title}", fontsize=16, weight="bold", y=0.985
        )
        note = (
            f"Torch {env['torch']} · Triton {env['triton']} · {report['warmup']} warmups / "
            f"{report['repeat']} samples · no compile/graph timing · 1 CPU thread\n"
            "Model-boundary acceptance passed. Unrounded FP32 error remains a separate diagnostic. "
            "Original BF16 backward has different product rounding."
        )
        fig.text(0.5, 0.015, note, ha="center", fontsize=10, color="#48505B")
        fig.tight_layout(rect=(0.015, 0.075, 0.995, 0.91), h_pad=2.5, w_pad=2)
        for extension in ("png", "svg"):
            fig.savefig(output / f"h3-mm-rope-{metric}.{extension}", dpi=180, facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.results.read_text()), args.out)


if __name__ == "__main__":
    main()
