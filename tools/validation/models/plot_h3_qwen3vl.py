# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Charts for ``h3_qwen3vl_evidence.py`` output: batch-invariance accuracy and latency.

Usage::

    python tools/validation/models/plot_h3_qwen3vl.py evidence.json --out-dir charts/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
FLOOR = 1e-4  # where a bitwise-equal (zero) difference is drawn on the log axis


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_accuracy(evidence: dict[str, Any], out: Path, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batching = evidence["prior_art_batching"]
    names = list(batching["hf_padded_right_all6"])
    methods = [
        (
            "RL-Kernel contract (per row, 50 layers)",
            {n: 0.0 if evidence["contract"][n]["equals_diffusers"] else None for n in names},
        ),
        (
            "HF padded batch, right pad",
            {n: batching["hf_padded_right_all6"][n]["max_abs"] for n in names},
        ),
        (
            "HF padded batch, left pad",
            {n: batching["hf_padded_left_all6"][n]["max_abs"] for n in names},
        ),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style(ax)
    height = 0.8 / len(methods)
    for m, (title, values) in enumerate(methods):
        ys = [i + (m - (len(methods) - 1) / 2) * height for i in range(len(names))]
        xs = [FLOOR if (values[n] in (0.0, None)) else values[n] for n in names]
        ax.barh(
            ys,
            xs,
            height=height * 0.85,
            color=SERIES[m],
            label=title,
            edgecolor=SURFACE,
            linewidth=1,
        )
        for y, n in zip(ys, names, strict=True):
            if values[n] == 0.0:
                ax.text(FLOOR * 1.3, y, "bitwise equal", va="center", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR, 10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [f"{n} ({evidence['reference'][n]['num_tokens']} tok)" for n in names], color=INK
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "max |feature - Diffusers hidden_states[50]| (BF16, log scale)", color=INK_2, fontsize=9
    )
    ax.set_title(
        f"Qwen3-VL layer-50 feature vs Diffusers, alone vs batched - {label}",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_latency(evidence: dict[str, Any], out: Path, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latency = evidence["latency"]
    names = sorted(latency, key=lambda n: latency[n]["num_tokens"])
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style(ax)
    height = 0.38
    rows = [
        (
            "Diffusers get_qwen3vl_prompt_embeds (64 layers + all hidden states)",
            "diffusers_median_ms",
            SERIES[1],
        ),
        ("RL-Kernel contract (stop after layer 49)", "contract_median_ms", SERIES[0]),
    ]
    for m, (title, key, color) in enumerate(rows):
        ys = [i + (m - 0.5) * height for i in range(len(names))]
        xs = [latency[n][key] for n in names]
        ax.barh(
            ys, xs, height=height * 0.9, color=color, label=title, edgecolor=SURFACE, linewidth=1
        )
        if m == 1:
            for y, n in zip(ys, names, strict=True):
                ratio = latency[n]["diffusers_median_ms"] / latency[n]["contract_median_ms"]
                ax.text(
                    latency[n][key],
                    y,
                    f"  {latency[n][key]:.1f} ms ({ratio:.2f}x)",
                    va="center",
                    fontsize=8,
                    color=INK_2,
                )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n} ({latency[n]['num_tokens']} tok)" for n in names], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("median latency per presentation, ms (lower is better)", color=INK_2, fontsize=9)
    memory = evidence["memory"]
    ax.set_title(
        f"Text-encoder latency - {label}; weights "
        f"{memory['full_model_param_gib']:.1f} -> {memory['layer50_param_gib']:.1f} GiB",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="B200")
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.label.lower()
    plot_accuracy(evidence, args.out_dir / f"h3-qwen3vl-batch-invariance-{tag}.png", args.label)
    plot_latency(evidence, args.out_dir / f"h3-qwen3vl-latency-{tag}.png", args.label)


if __name__ == "__main__":
    main()
