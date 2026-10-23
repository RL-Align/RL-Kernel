#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render the Qwen3-Next distributed logprob parity evidence as a PNG.

    python benchmarks/models/qwen3_next/plot_qwen3_next_distributed_logprob.py \\
        artifacts/qwen3-next-distributed-logprob-b200

reads ``parity/runs/<setting>/rank-0.json`` and ``prior-art/report.json`` and
writes ``figure.png``. Top: for every TP4 cross-rank point, whether its digest
under each NCCL setting equals the digest under NCCL defaults (the raw-NCCL
negative control is the last row). Bottom: the selected-token logprob prior art
-- invariance checks, the largest difference between implementations, latency.
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
EXISTING = ("#eb6834", "#b8501f", "#f29a62")
STATUS = ListedColormap(["#d64545", "#1baf7a"])
CONTROL = "o_proj_partials.nccl_all_reduce"
LABELS = {
    "rl_kernel_ws2": "RL-Kernel WS2\n(both engines here)",
    "vllm_gather": "vLLM sampler\n(gather + log_softmax)",
    "megatron_ce": "Megatron-core\nvocab_parallel_cross_entropy",
    "vime_default": "VIME default\nlogprob",
}
CHECKS = (
    ("tp4_equals_tp1", "TP4 == TP1"),
    ("nccl_setting_invariant", "NCCL setting\ninvariant"),
    ("batch_bitwise", "batch\ninvariant"),
)


def _setting_label(name):
    return name.replace("-cdefault", "").replace("-c1", "\n1 ch").replace("-", "\n")


def plot(evidence: Path, out: Path) -> None:
    runs = sorted(
        (evidence / "parity" / "runs").iterdir(), key=lambda p: (p.name != "default", p.name)
    )
    digests = {run.name: json.loads((run / "rank-0.json").read_text())["points"] for run in runs}
    points = [p for p, v in digests["default"].items() if isinstance(v, str) and p != CONTROL]
    points = sorted(points) + [CONTROL]
    grid = np.array(
        [[digests[run.name][p] == digests["default"][p] for run in runs] for p in points]
    )
    prior = json.loads((evidence / "prior-art" / "report.json").read_text())["candidates"]
    keys = [k for k in LABELS if k in prior and "unavailable" not in prior[k]]

    fig = plt.figure(figsize=(17, 11), facecolor=SURFACE)
    spec = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], width_ratios=[1.1, 1, 1])
    top = fig.add_subplot(spec[0, :])
    top.imshow(grid, cmap=STATUS, vmin=0, vmax=1, aspect="auto")
    top.set_yticks(
        range(len(points)),
        [
            p if p != CONTROL else "NEGATIVE CONTROL: o_proj partials, NCCL all_reduce"
            for p in points
        ],
        fontsize=8,
        color=INK,
    )
    top.set_xticks(range(len(runs)), [_setting_label(r.name) for r in runs], fontsize=7, color=INK)
    top.axhline(len(points) - 1.5, color=SURFACE, linewidth=4)
    top.set_title(
        "TP4 cross-rank points on real weights: digest under each NCCL setting == NCCL defaults "
        "(green) or not (red)",
        fontsize=10,
        color=INK,
    )

    checks = fig.add_subplot(spec[1, 0])
    table = np.array([[bool(prior[k][c]) for c, _ in CHECKS] for k in keys])
    checks.imshow(table, cmap=STATUS, vmin=0, vmax=1, aspect="auto")
    checks.set_yticks(range(len(keys)), [LABELS[k] for k in keys], fontsize=8, color=INK)
    checks.set_xticks(range(len(CHECKS)), [label for _, label in CHECKS], fontsize=8, color=INK)
    checks.set_title("Selected-token logprob: invariance (bitwise)", fontsize=10, color=INK)

    pairs = fig.add_subplot(spec[1, 1])
    delta = np.array(
        [[0.0 if a == b else prior[a]["max_abs_vs"].get(b, np.nan) for b in keys] for a in keys]
    )
    pairs.imshow(delta > 0, cmap=ListedColormap(["#1baf7a", "#f29a62"]), vmin=0, vmax=1)
    for i in range(len(keys)):
        for j in range(len(keys)):
            text = "0" if delta[i, j] == 0 else f"{delta[i, j]:.1e}"
            pairs.text(j, i, text, ha="center", va="center", fontsize=8, color=INK)
    short = [LABELS[k].split("\n")[0] for k in keys]
    pairs.set_xticks(range(len(keys)), short, fontsize=7, rotation=20, color=INK)
    pairs.set_yticks(range(len(keys)), short, fontsize=7, color=INK)
    pairs.set_title("Max |Δ logprob| between implementations", fontsize=10, color=INK)

    latency = fig.add_subplot(spec[1, 2])
    values = [prior[k]["latency_us_tp4"] / 1000.0 for k in keys]
    colors = [RL_KERNEL if k == "rl_kernel_ws2" else EXISTING[i % 3] for i, k in enumerate(keys)]
    latency.barh(range(len(keys)), values, color=colors)
    latency.set_yticks(range(len(keys)), short, fontsize=8, color=INK)
    latency.invert_yaxis()
    for i, v in enumerate(values):
        latency.text(v, i, f" {v:.2f} ms", va="center", fontsize=8, color=MUTED)
    latency.set_xlabel("TP4 latency, 1024 tokens (ms, median)", fontsize=9, color=MUTED)
    latency.set_title("Latency", fontsize=10, color=INK)

    for axis in fig.axes:
        for spine in axis.spines.values():
            spine.set_visible(False)
    env = json.loads((evidence / "prior-art" / "report.json").read_text())["environment"]
    fig.suptitle(
        f"Qwen3-Next TP4 distributed logprob parity - {env['gpu']}, NCCL {env['nccl']}, "
        f"torch {env['torch']}",
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
