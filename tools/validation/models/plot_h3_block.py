#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render the RFC #420 one-block replay (``tools/validation/models/h3_block_replay.py``) as a PNG.

Needs only the report JSON and matplotlib:

    python tools/validation/models/plot_h3_block.py \
        docs/usage/evidence/h3-one-block-b200/report.json

writes ``figure.png`` next to the report. Panels: the per-node invariance and
provider checks over every layout; the per-node error against the FP32 golden
for the candidate and diffusers; the same for every backward leaf.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

RL_KERNEL = "#2a78d6"
PROVIDER = "#eb6834"
PASS = "#1baf7a"
FAIL = "#d64545"
NA = "#e4e3dd"
INK = "#2b2b29"
MUTED = "#6f6e69"
GRID = "#e4e3dd"
SURFACE = "#fcfcfb"
STATUS_MARK = {"row": "", "interim": "*", "reference": "†"}


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _checks_panel(ax, report: dict[str, Any]) -> None:
    """Rows: nodes. Columns: forward/backward checks, ANDed over every layout."""

    nodes = [n["node"] for n in report["nodes"]]
    status = {n["node"]: n["status"] for n in report["nodes"]}
    cols = [
        ("repeat", "forward", "repeat_bitwise_equal"),
        ("batch row", "forward", "batch_row_bitwise_equal"),
        ("= diffusers\n(isolated)", "forward", None),
        ("grad\nrepeat", "backward", "grad_repeat_bitwise_equal"),
        ("grad\nbatch row", "backward", "grad_batch_row_bitwise_equal"),
    ]
    grid = []
    for node in nodes:
        row = []
        for _label, kind, key in cols:
            cases = report["forward_cases" if kind == "forward" else "backward_cases"]
            entries = [e for case in cases for e in case["nodes"] if e["node"] == node]
            if key is None:
                promised = entries[0]["provider_bitwise_isolated_promised"]
                ok = all(e["isolated_vs_provider"]["bitwise_equal"] for e in entries)
                row.append(2 if not promised and not ok else (1 if ok else 0))
            else:
                values = [e[key] for e in entries]
                row.append(2 if not values or None in values else (1 if all(values) else 0))
        grid.append(row)
    ax.imshow(grid, cmap=ListedColormap([FAIL, PASS, NA]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(cols)), [c[0] for c in cols], fontsize=7.5, color=INK)
    ax.set_yticks(
        range(len(nodes)), [f"{n}{STATUS_MARK[status[n]]}" for n in nodes], fontsize=7.5, color=INK
    )
    ax.xaxis.tick_top()
    _style(ax, "Bitwise checks, all layouts")
    ax.set_xlabel(
        "green = holds   grey = not promised (reduction differs from diffusers)\n"
        "or not applicable (AdaLN tables have no batch axis)\n"
        "* interim operator   † provider replay (no kernel yet)",
        fontsize=7,
        color=MUTED,
    )


def _err_panel(ax, labels, cand, prov, title: str) -> None:
    xs = range(len(labels))
    ax.plot(
        xs, [math.log10(v) for v in cand], "o-", color=RL_KERNEL, lw=1.8, ms=5, label="RL-Kernel"
    )
    ax.plot(
        xs, [math.log10(v) for v in prov], "s--", color=PROVIDER, lw=1.4, ms=4, label="diffusers"
    )
    ax.set_xticks(list(xs), labels, rotation=55, ha="right", fontsize=7, color=INK)
    lo = math.floor(min(math.log10(v) for v in cand + prov))
    hi = math.ceil(max(math.log10(v) for v in cand + prov))
    ax.set_yticks(range(lo, hi + 1), [f"1e{v}" for v in range(lo, hi + 1)])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylabel("max|err| / max|golden|", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    _style(ax, title)


def plot(report: dict[str, Any]):
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 5.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.0, 1.6, 1.3]}
    )
    case = max(report["forward_cases"], key=lambda c: c["seq_len"])
    back = (
        max(report["backward_cases"], key=lambda c: c["seq_len"])
        if report["backward_cases"]
        else None
    )
    _checks_panel(axes[0], report)
    _err_panel(
        axes[1],
        [e["node"] for e in case["nodes"]],
        [e["candidate_rel_err_vs_golden"] for e in case["nodes"]],
        [e["provider_rel_err_vs_golden"] for e in case["nodes"]],
        f"Forward error vs FP32 golden, S = {case['seq_len']}",
    )
    if back is not None:
        leaves = list(back["leaves"])
        _err_panel(
            axes[2],
            [f"d {name}" for name in leaves],
            [back["leaves"][n]["candidate_rel_err_vs_golden"] for n in leaves],
            [back["leaves"][n]["provider_rel_err_vs_golden"] for n in leaves],
            f"Gradient error vs FP32 golden, S = {back['seq_len']}",
        )
    env = report["environment"]
    sizes = ", ".join(str(c["seq_len"]) for c in report["forward_cases"])
    fig.suptitle(
        "MiniMax-H3 block 0, forward and backward (RFC #420 ws1_one_h3_block)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.905,
        f"{env['gpu']} | torch {env['torch']} | layouts S = {sizes} | "
        f"first drift vs diffusers: {case['first_drift']} | "
        f"rl-kernel {report['rl_kernel_commit'][:7]}",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    out = args.report.with_name("figure.png")
    plot(report).savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
