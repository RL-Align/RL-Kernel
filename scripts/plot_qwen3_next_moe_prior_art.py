#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render the routed-MoE prior-art report (``scripts/qwen3_next_moe_prior_art.py``) as a PNG.

Needs only the report JSON and matplotlib, so it runs anywhere:

    python scripts/plot_qwen3_next_moe_prior_art.py \\
        docs/usage/evidence/qwen3-next-moe-route-b200/report.json

writes ``figure.png`` next to the report. Three panels: which batch-invariance
checks each implementation passes, accuracy against an FP64 evaluation with
FP64 routing, and forward latency. RL-Kernel is blue, existing libraries orange.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

RL_KERNEL = "#2a78d6"
INK = "#2b2b29"
MUTED = "#6f6e69"
GRID = "#e4e3dd"
SURFACE = "#fcfcfb"
PASS = "#1baf7a"
FAIL = "#d64545"
NA = "#e4e3dd"
EXISTING = ("#eb6834", "#b8501f", "#f29a62", "#8a3a15")

LABELS = {
    "rl_kernel_cuda": "RL-Kernel shared_moe",
    "hf_transformers": "HF Qwen3NextExperts",
    "vllm_bi0": "vLLM fused_moe (BI=0)",
    "vllm_bi1": "vLLM fused_moe (BI=1)",
    "flashinfer_cutlass": "FlashInfer cutlass_fused_moe",
}
CHECKS = (
    ("route_rows_bitwise", "routes"),
    ("output_rows_bitwise", "output"),
    ("dx_rows_bitwise", "dx"),
    ("dweight_zero_rows_bitwise", "dW"),
)


def _colors(keys):
    existing = iter(EXISTING)
    return {k: RL_KERNEL if k == "rl_kernel_cuda" else next(existing) for k in keys}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)


def plot(report: dict, out: Path) -> None:
    cands = [c for c in report["candidates"] if "unavailable" not in c]
    keys = [c["key"] for c in cands]
    colors = _colors(keys)
    sizes = list(next(iter(cands))["output_rows_bitwise"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.5, 1, 1.4]})
    fig.patch.set_facecolor("white")

    # Panel 1: batch-invariance matrix; 1 = pass, 0 = fail, 0.5 = no backward.
    ax = axes[0]
    columns = [f"{label} {size}" for _, label in CHECKS for size in sizes]
    grid = []
    for cand in cands:
        row = []
        for field, _ in CHECKS:
            values = cand.get(field)
            row.extend(0.5 if values is None else float(values[s]) for s in sizes)
        grid.append(row)
    ax.imshow(grid, cmap=ListedColormap([FAIL, NA, PASS]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, fontsize=7, rotation=90)
    ax.set_yticks(range(len(cands)), [LABELS.get(k, k) for k in keys], fontsize=8)
    for i, cand in enumerate(cands):
        mark = "BI" if cand.get("batch_invariant") else "not BI"
        ax.text(len(columns) - 0.4, i, f"  {mark}", va="center", fontsize=8, color=INK)
    ax.set_title(
        f"Batch invariance: {report['probe_rows']} probe tokens alone vs. in a batch\n"
        "(green bitwise equal, red differs, grey: no backward)",
        fontsize=9,
        color=INK,
    )
    ax.tick_params(length=0)

    # Panel 2: accuracy.
    ax = axes[1]
    _style(ax)
    rel = [c["accuracy"]["rel_l2_vs_fp64"] if "accuracy" in c else float("nan") for c in cands]
    ax.barh(range(len(cands)), rel, color=[colors[k] for k in keys])
    for i, cand in enumerate(cands):
        acc = cand.get("accuracy")
        if acc:
            ax.text(
                rel[i],
                i,
                f"  {rel[i]:.2e}  ({acc['tokens_with_different_expert_set']}/{acc['tokens']}"
                " tokens route differently)",
                va="center",
                fontsize=7,
                color=MUTED,
            )
    ax.set_yticks(range(len(cands)), [LABELS.get(k, k) for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.2 * max(r for r in rel if r == r))
    ax.set_xlabel("relative L2 error vs. FP64 HF formula with FP64 routing", fontsize=8)
    ax.set_title("Accuracy (256 tokens)", fontsize=9, color=INK)

    # Panel 3: forward latency.
    ax = axes[2]
    _style(ax)
    forward = report["latency"]["forward_us"]
    tokens = [int(t) for t in forward]
    for key in keys:
        ys = [forward[str(t)].get(key, float("nan")) / 1000.0 for t in tokens]
        ax.plot(tokens, ys, marker="o", ms=3.5, color=colors[key], label=LABELS.get(key, key))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("tokens", fontsize=8)
    ax.set_ylabel("forward latency (ms, median)", fontsize=8)
    ax.set_title("Forward latency\n(H=2048, 512 experts, top-10, width 512)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    env = report["environment"]
    fig.suptitle(
        f"Qwen3-Next routed MoE prior art - {env['gpu']}, torch {env['torch']}, "
        f"commit {report['rl_kernel_commit'][:7]}",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    out = args.report.with_name("figure.png")
    plot(report, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
