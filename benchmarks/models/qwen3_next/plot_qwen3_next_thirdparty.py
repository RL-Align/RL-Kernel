#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render the Miles / verl comparison (logprob forward and backward, Miles' MoE backward).

    python benchmarks/models/qwen3_next/plot_qwen3_next_thirdparty.py \\
        artifacts/qwen3-next-thirdparty-comparison-b200

reads ``logprob-report.json``, ``logprob/<setting>/rank-0.json``,
``logprob-backward/report.json`` and ``miles-moe-backward/report.json`` and
writes ``figure.png``.
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

INK, MUTED, SURFACE = "#2b2b29", "#6f6e69", "#fcfcfb"
RL_KERNEL, OTHER = "#2a78d6", "#eb6834"
STATUS = ListedColormap(["#d64545", "#1baf7a", "#d9d8d3"])
LABELS = {
    "rl_kernel_ws2": "RL-Kernel WS2",
    "vllm_gather": "vLLM gather + log_softmax",
    "megatron_ce": "Megatron-core CE",
    "vime_default": "VIME default",
    "miles_fused": "Miles fused_log_probs",
    "verl_fused": "verl linear_cross_entropy",
}


def _setting(name):
    if name == "tp1":
        return "TP1\n(1 GPU)"
    name = name.removeprefix("tp4-").replace("allreduce-tree", "Tree")
    return name.replace("-chdefault", "").replace("-ch1", "\n1 ch").replace("-", "\n")


def plot(root: Path, out: Path) -> None:
    runs = sorted(
        (r for r in (root / "logprob").iterdir() if r.is_dir()),
        key=lambda p: (p.name == "tp1", p.name != "tp4-default", p.name),
    )
    digests = {r.name: json.loads((r / "rank-0.json").read_text())["candidates"] for r in runs}
    report = json.loads((root / "logprob-report.json").read_text())
    keys = [k for k in LABELS if k in report["candidates"]]
    backward = json.loads((root / "logprob-backward" / "report.json").read_text())["candidates"]
    moe = json.loads((root / "miles-moe-backward" / "report.json").read_text())["candidates"]
    env = report["environment"]

    fig = plt.figure(figsize=(19, 15), facecolor=SURFACE)
    spec = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1.25, 1])

    top = fig.add_subplot(spec[0, :])
    grid = np.array(
        [
            [
                digests[r.name][k]["selected_sha256"]
                == digests["tp4-default"][k]["selected_sha256"]
                for r in runs
            ]
            for k in keys
        ],
        dtype=float,
    )
    top.imshow(grid, cmap=STATUS, vmin=0, vmax=2, aspect="auto")
    top.set_yticks(range(len(keys)), [LABELS[k] for k in keys], fontsize=8, color=INK)
    top.set_xticks(range(len(runs)), [_setting(r.name) for r in runs], fontsize=7, color=INK)
    top.axvline(len(runs) - 1.5, color=SURFACE, linewidth=4)
    top.set_title(
        "Selected-token logprob, real LM head, 1024 tokens: output under each NCCL setting (and TP1) "
        "equals TP4 / NCCL defaults (green) or not (red)",
        fontsize=10,
        color=INK,
    )

    fwd = fig.add_subplot(spec[1, 0])
    columns = (
        ("tp4_equals_tp1", "TP4 == TP1"),
        ("nccl_setting_invariant", "NCCL setting\ninvariant"),
        ("batch_bitwise", "batch\ninvariant"),
    )
    table = [[float(bool(report["candidates"][k][c])) for c, _ in columns] for k in keys]
    fwd.imshow(np.array(table), cmap=STATUS, vmin=0, vmax=2, aspect="auto")
    fwd.set_yticks(range(len(keys)), [LABELS[k] for k in keys], fontsize=8, color=INK)
    fwd.set_xticks(range(len(columns)), [label for _, label in columns], fontsize=8, color=INK)
    fwd.set_title("Forward invariance (bitwise)", fontsize=10, color=INK)

    lat = fig.add_subplot(spec[1, 1])
    values = [report["candidates"][k]["latency_us_tp4"] / 1000 for k in keys]
    lat.barh(
        range(len(keys)), values, color=[RL_KERNEL if k == "rl_kernel_ws2" else OTHER for k in keys]
    )
    lat.set_yticks(range(len(keys)), [LABELS[k] for k in keys], fontsize=8, color=INK)
    lat.invert_yaxis()
    for i, v in enumerate(values):
        lat.text(v, i, f" {v:.2f} ms", va="center", fontsize=8, color=MUTED)
    lat.set_xlabel("TP4 latency, 1024 tokens (ms, median)", fontsize=9, color=MUTED)
    lat.set_title("Forward latency", fontsize=10, color=INK)

    bwd = fig.add_subplot(spec[2, 0])
    rows, labels, colors = [], [], []
    for name, label in (("miles_moe", "Miles MoE backward"), ("rl_moe", "RL-Kernel MoE backward")):
        source = moe["miles" if name == "miles_moe" else "rl_kernel"]["repeat"]
        for tensor, short in (("dx", "dx"), ("dw_gate_up", "dW gate_up"), ("dw_down", "dW down")):
            rows.append(source[tensor]["repeats_equal_to_first"] / source[tensor]["repeats"])
            labels.append(f"{label}: {short}")
            colors.append(OTHER if name == "miles_moe" else RL_KERNEL)
    for name, label in (
        ("verl_fused", "verl logprob backward"),
        ("miles_fused", "Miles logprob backward"),
        ("rl_kernel_ws2", "RL-Kernel logprob backward"),
    ):
        for tensor, source in backward[name]["gradients"].items():
            rows.append(source["repeats_equal_to_first"] / source["repeats"])
            labels.append(f"{label}: {tensor}")
            colors.append(RL_KERNEL if name == "rl_kernel_ws2" else OTHER)
    bwd.barh(range(len(rows)), rows, color=colors)
    bwd.set_yticks(range(len(rows)), labels, fontsize=7, color=INK)
    bwd.invert_yaxis()
    bwd.set_xlim(0, 1.05)
    for i, v in enumerate(rows):
        bwd.text(v, i, f" {v * 20:.0f}/20", va="center", fontsize=7, color=MUTED)
    bwd.set_xlabel(
        "share of 20 identical launches equal to the first, bit for bit", fontsize=9, color=MUTED
    )
    bwd.set_title(
        "Backward reproducibility (identical inputs, 20 launches)", fontsize=10, color=INK
    )

    inv = fig.add_subplot(spec[2, 1])
    names = [
        ("Miles MoE dx", moe["miles"]["dx_batch_grid"]),
        ("RL-Kernel MoE dx", moe["rl_kernel"]["dx_batch_grid"]),
    ]
    for name, label in (
        ("verl_fused", "verl d_hidden"),
        ("miles_fused", "Miles d_logits"),
        ("rl_kernel_ws2", "RL-Kernel d_logits"),
    ):
        names.append((label, backward[name]["batch_grid"]))
    cells = sorted(names[0][1], key=lambda c: (int(c.split("@")[0]), int(c.split("@")[1])))
    matrix = np.array([[float(bool(g[c])) for c in cells] for _, g in names])
    inv.imshow(matrix, cmap=STATUS, vmin=0, vmax=2, aspect="auto")
    inv.set_yticks(range(len(names)), [n for n, _ in names], fontsize=8, color=INK)
    inv.set_xticks(range(len(cells)), cells, fontsize=6, rotation=60, color=INK)
    inv.set_title(
        "Row gradient: 8 probe rows alone == inside a batch (size@first-row position)",
        fontsize=10,
        color=INK,
    )

    for axis in fig.axes:
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle(
        f"Miles and verl vs RL-Kernel - {env['gpu']}, NCCL {env['nccl']}, torch {env['torch']}",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=140)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    out = args.evidence / "figure.png"
    plot(args.evidence, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
