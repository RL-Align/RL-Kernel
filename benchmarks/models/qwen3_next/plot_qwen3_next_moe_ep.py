#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render the Qwen3-Next expert-parallel MoE prior art as a PNG.

    python benchmarks/models/qwen3_next/plot_qwen3_next_moe_ep.py \\
        artifacts/qwen3-next-moe-ep-comparison-b200

reads ``runs/<launch>/rank-0.json`` and ``report.json`` and writes
``figure.png``. Top: for every candidate, whether its rank-0 output under each
EP4 NCCL setting, and at EP1, equals its output at EP4 with NCCL defaults.
Bottom: invariance checks, the largest difference between implementations and
latency.
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
RL_KERNEL = "#2a78d6"
EXISTING = ("#eb6834", "#b8501f", "#f29a62", "#c7783f", "#d9894f", "#a5461b")
STATUS = ListedColormap(["#d64545", "#1baf7a"])
LABELS = {
    "rl_kernel_ep": "RL-Kernel EP4\n(both engines here)",
    "vllm_fused_ep": "vLLM fused_experts\n+ NCCL all_reduce",
    "vllm_fused_ep_bi": "vLLM, VLLM_BATCH_\nINVARIANT=1",
    "megatron_alltoall": "Megatron MoELayer\nalltoall dispatcher",
    "megatron_allgather": "Megatron MoELayer\nallgather dispatcher",
    "megatron_alltoall_det": "Megatron alltoall,\ndeterministic algorithms",
    "megatron_allgather_det": "Megatron allgather,\ndeterministic algorithms",
}
SHORT = {
    "rl_kernel_ep": "RL-Kernel EP4",
    "vllm_fused_ep": "vLLM",
    "vllm_fused_ep_bi": "vLLM (BI mode)",
    "megatron_alltoall": "Megatron alltoall",
    "megatron_allgather": "Megatron allgather",
    "megatron_alltoall_det": "Megatron alltoall (det.)",
    "megatron_allgather_det": "Megatron allgather (det.)",
}
CHECKS = (
    ("ep4_equals_ep1", "EP4 == EP1"),
    ("nccl_setting_invariant", "NCCL setting\ninvariant"),
    ("repeat_bitwise", "repeat\nbitwise"),
    ("batch_bitwise", "batch\ninvariant"),
    ("ranks_agree", "ranks\nagree"),
)


def _launch_label(name):
    if name == "ep1":
        return "EP1\n(1 GPU)"
    name = name.removeprefix("ep4-").replace("allreduce-tree", "Tree")
    return name.replace("-chdefault", "").replace("-ch1", "\n1 ch").replace("-", "\n")


def plot(evidence: Path, out: Path) -> None:
    runs = sorted(
        (evidence / "runs").iterdir(),
        key=lambda p: (p.name == "ep1", p.name != "ep4-default", p.name),
    )
    runs = [run for run in runs if run.is_dir()]
    outputs = {
        run.name: json.loads((run / "rank-0.json").read_text())["candidates"] for run in runs
    }
    report = json.loads((evidence / "report.json").read_text())
    prior = report["candidates"]
    keys = [k for k in LABELS if k in prior and "unavailable" not in prior[k]]
    reference = outputs["ep4-default"]
    grid = np.array(
        [
            [outputs[run.name][k]["output_sha256"] == reference[k]["output_sha256"] for run in runs]
            for k in keys
        ]
    )

    fig = plt.figure(figsize=(18, 13), facecolor=SURFACE)
    spec = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], width_ratios=[1.2, 1, 1])
    top = fig.add_subplot(spec[0, :])
    top.imshow(grid, cmap=STATUS, vmin=0, vmax=1, aspect="auto")
    top.set_yticks(range(len(keys)), [LABELS[k] for k in keys], fontsize=8, color=INK)
    top.set_xticks(range(len(runs)), [_launch_label(r.name) for r in runs], fontsize=7, color=INK)
    top.axvline(len(runs) - 1.5, color=SURFACE, linewidth=4)
    top.set_title(
        "Layer-0 routed MoE on real weights, 1024 tokens: output equals the EP4 / NCCL-defaults "
        "output (green) or not (red)",
        fontsize=10,
        color=INK,
    )

    checks = fig.add_subplot(spec[1, 0])
    table = np.array([[bool(prior[k][c]) for c, _ in CHECKS] for k in keys])
    checks.imshow(table, cmap=STATUS, vmin=0, vmax=1, aspect="auto")
    checks.set_yticks(range(len(keys)), [LABELS[k] for k in keys], fontsize=8, color=INK)
    checks.set_xticks(range(len(CHECKS)), [label for _, label in CHECKS], fontsize=8, color=INK)
    for i, k in enumerate(keys):
        count = prior[k]["distinct_ep4_results"]
        checks.text(
            1,
            i,
            f"{count} result{'s' if count > 1 else ''}",
            ha="center",
            va="center",
            fontsize=7,
            color=SURFACE,
        )
    checks.set_title("Invariance (bitwise)", fontsize=10, color=INK)

    pairs = fig.add_subplot(spec[1, 1])
    delta = np.array(
        [[0.0 if a == b else prior[a]["max_abs_vs"].get(b, np.nan) for b in keys] for a in keys]
    )
    pairs.imshow(delta > 0, cmap=ListedColormap(["#1baf7a", "#f29a62"]), vmin=0, vmax=1)
    for i in range(len(keys)):
        for j in range(len(keys)):
            text = "0" if delta[i, j] == 0 else f"{delta[i, j]:.1e}"
            pairs.text(j, i, text, ha="center", va="center", fontsize=8, color=INK)
    short = [SHORT[k] for k in keys]
    pairs.set_xticks(range(len(keys)), short, fontsize=7, rotation=20, color=INK)
    pairs.set_yticks(range(len(keys)), short, fontsize=7, color=INK)
    pairs.set_title("Max |Δ output| between implementations (EP4)", fontsize=10, color=INK)

    latency = fig.add_subplot(spec[1, 2])
    rows = np.arange(len(keys))
    ep4 = [prior[k]["latency_us_ep4"] / 1000.0 for k in keys]
    ep1 = [prior[k]["latency_us_ep1"] / 1000.0 for k in keys]
    colors = [
        RL_KERNEL if k == "rl_kernel_ep" else EXISTING[i % len(EXISTING)]
        for i, k in enumerate(keys)
    ]
    latency.barh(rows - 0.2, ep4, height=0.38, color=colors)
    latency.barh(rows + 0.2, ep1, height=0.38, color=colors, alpha=0.4)
    latency.set_xscale("log")
    latency.set_yticks(rows, short, fontsize=8, color=INK)
    latency.invert_yaxis()
    for i, (a, b) in enumerate(zip(ep4, ep1)):
        latency.text(a, i - 0.2, f" EP4 {a:.2f} ms", va="center", fontsize=7, color=MUTED)
        latency.text(b, i + 0.2, f" EP1 {b:.2f} ms", va="center", fontsize=7, color=MUTED)
    latency.set_xlabel("latency, 1024 tokens (ms, median; log scale)", fontsize=9, color=MUTED)
    latency.set_title("Latency (EP4 solid, EP1 faded)", fontsize=10, color=INK)

    for axis in fig.axes:
        for spine in axis.spines.values():
            spine.set_visible(False)
    env = report["environment"]
    fig.suptitle(
        f"Qwen3-Next expert-parallel MoE prior art - {env['gpu']}, NCCL {env['nccl']}, "
        f"torch {env['torch']}, vLLM {env['vllm']}",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    out = args.evidence / "figure.png"
    plot(args.evidence, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
