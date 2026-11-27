#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot the SGLang rollout report and write the tables behind each chart.

    python tools/validation/models/plot_h3_sglang_rollout.py report.json --out-dir DIR

Writes two charts and one Markdown file into ``DIR``:

- ``h3-sglang-native-vs-training.png``: the max |Δ| of SGLang's own output against the
  training operator, per output and TP size. The bound outputs are bitwise equal (0) at
  every TP size, so they have no bar.
- ``h3-sglang-row-invariance.png``: the max |Δ| between a 37-row prefix run alone and
  the same rows inside the full batch, for SGLang's own outputs.
- ``h3-sglang-tables.md``: the numbers both charts plot, one table per chart, taken from
  the report the same way the bars are.

Each value is the largest over the TP ranks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = {
    "qkv": "qkv_proj",
    "out": "out_proj",
    "fc1": "fc1",
    "fc2": "fc2",
    "adaln_table": "block AdaLN table",
    "norm1": "block norm1 (modulated)",
    "norm2": "block norm2 (modulated)",
    "block_out": "block output",
    "final_table": "final AdaLN table",
    "final_out": "final layer output",
    "refiner_norm1": "refiner norm1",
    "refiner_norm2": "refiner norm2",
    "refiner_final_norm": "refiner final norm",
}
SERIES = "#2a78d6"  # one series: SGLang's own output
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _worst(entry: dict, field: str, bound_field: str) -> tuple[float | None, bool | None]:
    """Largest native diff over the ranks and whether every rank's bound output matched."""
    ranks = [v for k, v in entry.items() if k.startswith("rank")]
    if not ranks or field not in ranks[0]:
        return None, None
    return max(r[field] for r in ranks), all(r[bound_field] for r in ranks)


def collect(report: dict, field: str, bound_field: str) -> tuple[list[int], dict]:
    sizes = sorted(int(t) for t in report["tp"])
    table: dict[str, dict[int, tuple[float | None, bool | None]]] = {}
    for key in LABELS:
        table[key] = {
            tp: _worst(report["tp"][str(tp)]["rows"][key], field, bound_field) for tp in sizes
        }
    return sizes, table


def fmt(value: float | None) -> str:
    if value is None:
        return "–"
    return "0" if value == 0 else f"{value:g}"


def measured(sizes, table) -> list[str]:
    """The outputs with rows at some TP size (the AdaLN tables have none in the prefix run)."""
    return [k for k in LABELS if any(table[k][tp][0] is not None for tp in sizes)]


def bound_line(sizes, table) -> str:
    """The chart's line for the bound output, from the same per-rank results as the table."""
    keys = measured(sizes, table)
    if not keys:
        return "RL-Kernel bound output: not measured"
    failed = [LABELS[k] for k in keys if any(table[k][tp][1] is False for tp in sizes)]
    if failed:
        return "RL-Kernel bound output: NOT EQUAL for " + ", ".join(failed)
    if any(table[k][tp][1] is None for k in keys for tp in sizes):
        return "RL-Kernel bound output: 0 (bitwise) where measured; some TP sizes have no rows"
    return "RL-Kernel bound output: 0 (bitwise) for every output at every TP size"


def plot(sizes, table, *, title: str, xlabel: str, out: Path) -> None:
    keys = measured(sizes, table)
    plotted = [table[k][tp][0] for k in keys for tp in sizes]
    nonzero = [v for v in plotted if v]
    # All-zero or absent values draw no bar; any positive range holds their notes.
    lo, hi = (min(nonzero) / 2, max(nonzero) * 2) if nonzero else (2.0**-24, 1.0)
    fig, axes = plt.subplots(
        1,
        len(sizes),
        figsize=(3.2 * len(sizes) + 2.2, 0.36 * len(keys) + 1.9),
        sharey=True,
        layout="constrained",
        squeeze=False,  # one TP size still gives a row of axes
    )
    axes = axes[0]
    fig.set_facecolor("white")
    fig.suptitle(title, fontsize=12, color=INK, x=0.0, ha="left")
    y = list(range(len(keys)))
    for ax, tp in zip(axes, sizes):
        ax.set_facecolor("white")
        ax.set_xscale("log", base=2)
        ax.set_xlim(lo, hi)
        values = [table[k][tp][0] for k in keys]
        ax.barh(
            [i for i, v in zip(y, values) if v],
            [v - lo for v in values if v],  # bars start at the axis floor and end at the value
            height=0.55,
            color=SERIES,
            left=lo,
        )
        for i, v in zip(y, values):
            note = {None: "no rows", 0: "= 0, bitwise"}.get(v)
            ax.annotate(
                note or fmt(v),
                (lo if note else v, i),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=MUTED if note else INK,
            )
        ax.set_title(f"TP{tp}", fontsize=10, color=INK, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
        if not keys:
            ax.text(0.5, 0.5, "no measurements", transform=ax.transAxes, ha="center", color=MUTED)
    axes[0].set_yticks(y, [LABELS[k] for k in keys], fontsize=9, color=INK)
    axes[0].invert_yaxis()
    fig.savefig(out, dpi=150, facecolor="white", metadata={"Software": None})
    plt.close(fig)


def markdown(sizes, table, *, value_header: str, bound_header: str) -> str:
    keys = measured(sizes, table)
    if not keys:
        return "No measurements in this report."
    head = (
        ["Output"]
        + [f"{value_header}, TP{tp}".replace("|", "\\|") for tp in sizes]
        + [bound_header.replace("|", "\\|")]
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k in keys:
        cells = [fmt(table[k][tp][0]) for tp in sizes]
        status = [table[k][tp][1] for tp in sizes]
        if False in status:
            bound = "NOT EQUAL"
        elif None in status:
            bound = "0 (bitwise) where measured"
        else:
            bound = "0 (bitwise) at every TP"
        lines.append(
            "| "
            + " | ".join(
                [f"`{LABELS[k]}`" if k in ("qkv", "out", "fc1", "fc2") else LABELS[k]]
                + cells
                + [bound]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gpu = report["environment"]["gpu"]
    sglang = report["environment"]["sglang"]

    sizes, native = collect(report, "native_max_abs_diff", "bound_equals_canonical")
    plot(
        sizes,
        native,
        title=f"SGLang {sglang} native output vs the training operator, "
        f"max |Δ| per TP size ({gpu})\n" + bound_line(sizes, native),
        xlabel="max |Δ| vs training (log2)",
        out=args.out_dir / "h3-sglang-native-vs-training.png",
    )
    _, prefix = collect(report, "native_prefix_max_abs_diff", "bound_prefix_equals_rows")
    plot(
        sizes,
        prefix,
        title=f"Row invariance, SGLang {sglang} native ({gpu}): "
        f"a {report['prefix_rows']}-row prefix alone "
        f"vs the same rows inside {report['rows']} rows\n" + bound_line(sizes, prefix),
        xlabel="max |Δ| prefix vs full batch (log2)",
        out=args.out_dir / "h3-sglang-row-invariance.png",
    )
    tables = [
        "<!-- generated by tools/validation/models/plot_h3_sglang_rollout.py; "
        "largest value over the TP ranks -->",
        "",
        "Native vs training (`h3-sglang-native-vs-training.png`):",
        "",
        markdown(sizes, native, value_header="SGLang native max |Δ|", bound_header="Bound max |Δ|"),
        "",
        "Row invariance (`h3-sglang-row-invariance.png`):",
        "",
        markdown(sizes, prefix, value_header="SGLang native max |Δ|", bound_header="Bound max |Δ|"),
        "",
    ]
    (args.out_dir / "h3-sglang-tables.md").write_text("\n".join(tables))


if __name__ == "__main__":
    main()
