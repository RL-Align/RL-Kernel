#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render the TP4 attention prior-art report (``scripts/qwen3_next_attention_prior_art.py``).

    python scripts/plot_qwen3_next_attention_prior_art.py \\
        docs/usage/evidence/qwen3-next-tp4-mixers-b200/attention/report.json

writes ``figure.png`` next to the report: which invariance checks each
implementation passes, accuracy against FP64, and prefill/decode latency.
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
EXISTING = ("#eb6834", "#b8501f", "#f29a62", "#8a3a15", "#f6bd94", "#d9773f")

LABELS = {
    "rl_kernel_cuda": "RL-Kernel deterministic",
    "torch_sdpa": "torch SDPA",
    "vllm_fa2_auto": "vLLM FA2 (num_splits auto)",
    "vllm_fa2_split1": "vLLM FA2 (num_splits=1)",
    "vllm_triton_2d": "vLLM Triton unified (2D)",
    "flashinfer": "FlashInfer ragged prefill",
}
COLUMNS = (
    ("batch_bitwise", "first", "batch: first"),
    ("batch_bitwise", "last", "batch: last"),
    ("prefill_decode_bitwise", "last_64", "chunk: last 64"),
    ("prefill_decode_bitwise", "last_1", "decode: last 1"),
    ("backward_repeat_bitwise", "dq", "bwd dq"),
    ("backward_repeat_bitwise", "dk", "bwd dk"),
    ("backward_repeat_bitwise", "dv", "bwd dv"),
)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)


def plot(report: dict, out: Path) -> None:
    cands = [c for c in report["candidates"] if "unavailable" not in c and "failed" not in c]
    keys = [c["key"] for c in cands]
    existing = iter(EXISTING)
    colors = {k: RL_KERNEL if k == "rl_kernel_cuda" else next(existing) for k in keys}
    labels = [LABELS.get(k, k) for k in keys]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.3, 1, 1.4]})

    ax = axes[0]
    grid = [
        [0.5 if c.get(field) is None else float(c[field][item]) for field, item, _ in COLUMNS]
        for c in cands
    ]
    ax.imshow(grid, cmap=ListedColormap([FAIL, NA, PASS]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(COLUMNS)), [c[2] for c in COLUMNS], fontsize=7, rotation=90)
    ax.set_yticks(range(len(cands)), labels, fontsize=8)
    for i, cand in enumerate(cands):
        mark = "BI" if cand.get("batch_invariant") else "not BI"
        ax.text(len(COLUMNS) - 0.4, i, f"  {mark}", va="center", fontsize=8, color=INK)
    ax.set_title(
        f"Invariance: {report['target_tokens']}-token target alone vs. batched / chunked\n"
        "(green bitwise equal, red differs, grey: no backward)",
        fontsize=9,
        color=INK,
    )
    ax.tick_params(length=0)

    ax = axes[1]
    _style(ax)
    rel = [c["accuracy"]["rel_l2_vs_fp64"] for c in cands]
    ax.barh(range(len(cands)), rel, color=[colors[k] for k in keys])
    for i, value in enumerate(rel):
        ax.text(value, i, f"  {value:.2e}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(cands)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.35 * max(rel))
    ax.set_xlabel("relative L2 error vs. FP64", fontsize=8)
    ax.set_title("Accuracy", fontsize=9, color=INK)

    ax = axes[2]
    _style(ax)
    prefill = report["latency"]["prefill_us"]
    tokens = [int(t) for t in prefill]
    for key in keys:
        ys = [prefill[str(t)].get(key, float("nan")) / 1000.0 for t in tokens]
        ax.plot(tokens, ys, marker="o", ms=3.5, color=colors[key], label=LABELS.get(key, key))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("prefill tokens", fontsize=8)
    ax.set_ylabel("forward latency (ms, median)", fontsize=8)
    ax.set_title("Prefill forward latency\n(4 Q heads, 1 KV head, D=256, causal)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    env = report["environment"]
    fig.suptitle(
        f"Qwen3-Next TP4 attention prior art - {env['gpu']}, torch {env['torch']}, "
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
