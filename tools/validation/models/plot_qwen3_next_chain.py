#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render a published Qwen3-Next full-model chain as a PNG.

Needs only the published evidence directory and matplotlib:

    python tools/validation/models/plot_qwen3_next_chain.py \\
        docs/usage/evidence/qwen3-next-chain-b200

writes ``figure.png`` into that directory. Left: for each L2 case and weight
version, whether the vLLM rollout and the VIME recompute agree bit for bit on
response tokens, logits, logprobs and the per-layer digests. Right: for each
decoder layer, the share of cases whose layer digest changes from one weight
version to the next, which shows the updates reach every layer and the reload
reproduces update 2 exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

INK = "#2b2b29"
MUTED = "#6f6e69"
SURFACE = "#fcfcfb"
PASS = "#1baf7a"
FAIL = "#d64545"
CHANGE = ("#e4e3dd", "#2a78d6")

STAGES = (("l2", "baseline"), ("train1", "update 1"), ("train2", "update 2"), ("train", "reload"))
COMPARISONS = (
    ("response_tokens", "tokens"),
    ("logits", "logits"),
    ("logprobs", "logprobs"),
    ("layer_states", "layer digests"),
)
STEPS = (("baseline", "update1"), ("update1", "update2"), ("update2", "reload"))


def _node_order(name):
    return (0, 0) if name == "embedding" else (1, int(name.removeprefix("layer")))


def plot(evidence: Path, out: Path) -> None:
    matrices = {
        label: json.loads((evidence / stage / "l2_matrix.json").read_text())
        for stage, label in STAGES
    }
    cases = matrices["baseline"]["case_ids"]
    grid = np.array(
        [
            [
                bool(matrices[label]["cases"][case]["comparisons"][key])
                for _, label in STAGES
                for key, _ in COMPARISONS
            ]
            for case in cases
        ]
    )
    report = json.loads((evidence / "chain-report.json").read_text())
    cells = report["cells"]
    nodes = sorted(cells[f"baseline/{cases[0]}"]["node_digests"], key=_node_order)
    change = np.array(
        [
            [
                np.mean(
                    [
                        cells[f"{a}/{case}"]["node_digests"][node]
                        != cells[f"{b}/{case}"]["node_digests"][node]
                        for case in cases
                    ]
                )
                for node in nodes
            ]
            for a, b in STEPS
        ]
    )

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(16, 6.2), gridspec_kw={"width_ratios": [1, 1.55]}, facecolor=SURFACE
    )
    left.imshow(grid, cmap=ListedColormap([FAIL, PASS]), vmin=0, vmax=1, aspect="auto")
    left.set_yticks(range(len(cases)), cases, fontsize=8, color=INK)
    names = [name for _, name in COMPARISONS]
    left.set_xticks(range(grid.shape[1]), names * len(STAGES), rotation=90, fontsize=8)
    for i, (_, label) in enumerate(STAGES):
        left.text(i * 4 + 1.5, -1.2, label, ha="center", fontsize=9, color=INK)
        if i:
            left.axvline(i * 4 - 0.5, color=SURFACE, linewidth=3)
    left.set_title(
        f"vLLM rollout vs VIME recompute, raw bits\n{int(grid.sum())}/{grid.size} equal",
        fontsize=10,
        color=INK,
        pad=24,
    )

    right.imshow(change, cmap=ListedColormap(CHANGE), vmin=0, vmax=1, aspect="auto")
    right.set_yticks(range(len(STEPS)), [f"{a} → {b}" for a, b in STEPS], fontsize=9, color=INK)
    labels = ["emb" if n == "embedding" else n.removeprefix("layer") for n in nodes]
    right.set_xticks(range(len(nodes)), labels, fontsize=6, color=MUTED)
    right.set_xlabel("embedding and decoder layers 0-47", fontsize=9, color=MUTED)
    for row in range(len(STEPS)):
        right.text(
            len(nodes) + 0.6,
            row,
            f"{change[row].mean():.0%} of digests change",
            va="center",
            fontsize=8,
            color=INK,
        )
    right.set_xlim(-0.5, len(nodes) + 9)
    right.set_title(
        "Per-layer digest change between weight versions (share of the 16 cases)",
        fontsize=10,
        color=INK,
    )
    for axis in (left, right):
        for spine in axis.spines.values():
            spine.set_visible(False)
    commit = report["measured_commit"][:7]
    gpu = f" on {report['gpu_name']}" if report.get("gpu_name") else ""
    fig.suptitle(
        f"Qwen3-Next-80B-A3B full-model chain, TP4{gpu}, commit {commit}: "
        f"passed={report['passed']}, first drift={report['first_drift']}",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    out = args.evidence / "figure.png"
    plot(args.evidence, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
