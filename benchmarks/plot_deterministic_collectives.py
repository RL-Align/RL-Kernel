# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot before/after reports of ``benchmark_deterministic_collectives.py``.

    python benchmarks/plot_deterministic_collectives.py \
      --before before_w2.json before_w8.json --after after_w2.json after_w8.json \
      --operation all_gather --output figure.png

One panel per world size: deterministic time before and after, with NCCL
(from the "after" run) as a reference. Needs only matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BEFORE = "#eb6834"
AFTER = "#2a78d6"
REFERENCE = "#6f6e69"
INK = "#2b2b29"
GRID = "#e4e3dd"
SURFACE = "#fcfcfb"


def _rows(path: Path, operation: str) -> tuple[dict, list[dict]]:
    """Load a report and sort the selected operation's rows by input bytes per rank."""
    report = json.loads(path.read_text())
    rows = [row for row in report["rows"] if row["operation"] == operation]
    return report, sorted(rows, key=lambda row: row["input_bytes_per_rank"])


def main() -> None:
    """Validate paired benchmark reports and save their latency comparison plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, nargs="+", required=True)
    parser.add_argument("--after", type=Path, nargs="+", required=True)
    parser.add_argument("--operation", default="all_gather")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.before) != len(args.after):
        raise SystemExit("--before and --after need one report per world size each")

    reports = []
    for before_path, after_path in zip(args.before, args.after, strict=True):
        before_report, before = _rows(before_path, args.operation)
        after_report, after = _rows(after_path, args.operation)
        if before_report["world_size"] != after_report["world_size"]:
            raise SystemExit(
                f"{before_path} and {after_path} have different world sizes: "
                f"{before_report['world_size']} != {after_report['world_size']}"
            )
        before_sizes = [row["input_bytes_per_rank"] for row in before]
        after_sizes = [row["input_bytes_per_rank"] for row in after]
        if before_sizes != after_sizes:
            raise SystemExit(
                f"{before_path} and {after_path} have different input sizes for "
                f"{args.operation}: {before_sizes} != {after_sizes}"
            )
        reports.append((before, after_report, after))

    fig, axes = plt.subplots(
        1, len(args.before), figsize=(5.5 * len(args.before), 3.8), facecolor=SURFACE, squeeze=False
    )
    for ax, (before, after_report, after) in zip(axes[0], reports, strict=True):
        sizes = [row["input_bytes_per_rank"] / 1024 for row in after]
        series = (
            ("before", BEFORE, "-", [row["deterministic_us"] for row in before]),
            ("after", AFTER, "-", [row["deterministic_us"] for row in after]),
            ("NCCL (reference)", REFERENCE, "--", [row["nccl_us"] for row in after]),
        )
        for label, color, style, values in series:
            ax.plot(
                sizes,
                values,
                style,
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                label=label,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=REFERENCE, labelsize=8)
        ax.set_xlabel("input per rank (KiB)", color=REFERENCE, fontsize=9)
        ax.set_ylabel("µs (slowest rank, lower is better)", color=REFERENCE, fontsize=9)
        world = after_report["world_size"]
        ax.set_title(
            f"{args.operation}, {world} x {after_report['gpu']}", loc="left", fontsize=11, color=INK
        )
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    first = reports[0][1]
    fig.text(
        0.01,
        0.01,
        f"torch {first['torch']} · CUDA {first['cuda']} · {first['dtype']}",
        fontsize=7,
        color=REFERENCE,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
