#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render the Qwen3-Next expert-parallel MoE prior art as a PNG.

    python benchmarks/models/qwen3_next/plot_qwen3_next_moe_ep.py \\
        artifacts/qwen3-next-moe-ep-comparison-b200

reads ``runs/<launch>/rank-0.json`` and ``report.json`` (schema 2) and writes
``figure.png``:

* top: whether each candidate's rank-0 output under every EP4 NCCL setting,
  and at EP1, equals its output at EP4 with NCCL defaults;
* middle: forward and backward invariance checks (grey: no backward), and the
  error against FP64;
* bottom: forward latency over token counts at EP4, and backward latency.
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
EXISTING = ("#eb6834", "#b8501f", "#f29a62", "#c7783f", "#d9894f", "#a5461b", "#e8a07a", "#8c3a12")
STATUS = ListedColormap(["#d64545", "#1baf7a", "#d9d8d3"])
LABELS = {
    "rl_kernel_ep": "RL-Kernel EP4 (both engines here)",
    "vllm_ep": "vLLM fused_experts + vLLM all-reduce",
    "vllm_ep_bi": "vLLM, VLLM_BATCH_INVARIANT=1",
    "vllm_ep_nccl": "vLLM fused_experts + NCCL all_reduce",
    "flashinfer_cutlass_ep": "FlashInfer cutlass_fused_moe (CUTLASS)",
    "megatron_alltoall": "Megatron MoELayer, alltoall",
    "megatron_allgather": "Megatron MoELayer, allgather",
    "megatron_alltoall_det": "Megatron alltoall, deterministic algorithms",
    "megatron_allgather_det": "Megatron allgather, deterministic algorithms",
}
SHORT = {
    "rl_kernel_ep": "RL-Kernel EP4",
    "vllm_ep": "vLLM + vLLM all-reduce",
    "vllm_ep_bi": "vLLM, BI mode",
    "vllm_ep_nccl": "vLLM + NCCL all_reduce",
    "flashinfer_cutlass_ep": "FlashInfer (CUTLASS)",
    "megatron_alltoall": "Megatron alltoall",
    "megatron_allgather": "Megatron allgather",
    "megatron_alltoall_det": "Megatron alltoall, det.",
    "megatron_allgather_det": "Megatron allgather, det.",
}
FORWARD = (
    ("ep4_equals_ep1", "fwd\nEP4 == EP1"),
    ("nccl_setting_invariant", "fwd\nNCCL inv."),
    ("repeat_bitwise", "fwd\nrepeat"),
    ("batch_bitwise", "fwd\nbatch inv."),
    ("ranks_agree", "fwd\nranks agree"),
)
BACKWARD = (
    ("dx_ep4_equals_ep1", "bwd dx\nEP4 == EP1"),
    ("dw_ep4_equals_ep1", "bwd dW\nEP4 == EP1"),
    ("nccl_setting_invariant", "bwd\nNCCL inv."),
    ("repeat_bitwise", "bwd\nrepeat"),
    ("batch_bitwise", "bwd\nbatch inv."),
)


def _launch_label(name):
    if name == "ep1":
        return "EP1\n(1 GPU)"
    name = name.removeprefix("ep4-").replace("allreduce-tree", "Tree")
    return name.replace("-chdefault", "").replace("-ch1", "\n1 ch").replace("-", "\n")


def _color(key, index):
    return RL_KERNEL if key == "rl_kernel_ep" else EXISTING[index % len(EXISTING)]


def plot(evidence: Path, out: Path) -> None:
    runs = sorted(
        (run for run in (evidence / "runs").iterdir() if run.is_dir()),
        key=lambda p: (p.name == "ep1", p.name != "ep4-default", p.name),
    )
    outputs = {
        run.name: json.loads((run / "rank-0.json").read_text())["candidates"] for run in runs
    }
    report = json.loads((evidence / "report.json").read_text())
    prior = report["candidates"]
    keys = [k for k in LABELS if k in prior and "unavailable" not in prior[k]]
    labels = [LABELS[k] for k in keys]
    reference = outputs["ep4-default"]
    grid = np.array(
        [
            [outputs[r.name][k]["output_sha256"] == reference[k]["output_sha256"] for r in runs]
            for k in keys
        ],
        dtype=float,
    )

    fig = plt.figure(figsize=(19, 17), facecolor=SURFACE)
    spec = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], width_ratios=[1.35, 1])
    top = fig.add_subplot(spec[0, :])
    top.imshow(grid, cmap=STATUS, vmin=0, vmax=2, aspect="auto")
    top.set_yticks(range(len(keys)), labels, fontsize=8, color=INK)
    top.set_xticks(range(len(runs)), [_launch_label(r.name) for r in runs], fontsize=7, color=INK)
    top.axvline(len(runs) - 1.5, color=SURFACE, linewidth=4)
    top.set_title(
        "Layer-0 routed MoE, real weights, 1024 tokens: forward output equals the EP4 / "
        "NCCL-defaults output (green) or not (red)",
        fontsize=10,
        color=INK,
    )

    checks = fig.add_subplot(spec[1, 0])
    table = []
    for k in keys:
        row = [float(bool(prior[k][c])) for c, _ in FORWARD]
        back = prior[k].get("backward")
        row += [2.0 if back is None else float(bool(back[c])) for c, _ in BACKWARD]
        table.append(row)
    checks.imshow(np.array(table), cmap=STATUS, vmin=0, vmax=2, aspect="auto")
    checks.set_yticks(range(len(keys)), labels, fontsize=8, color=INK)
    checks.set_xticks(
        range(len(FORWARD) + len(BACKWARD)),
        [label for _, label in FORWARD + BACKWARD],
        fontsize=7,
        color=INK,
    )
    checks.axvline(len(FORWARD) - 0.5, color=SURFACE, linewidth=4)
    for i, k in enumerate(keys):
        count = prior[k]["distinct_ep4_results"]
        checks.text(1, i, f"{count}", ha="center", va="center", fontsize=8, color=SURFACE)
    checks.set_title(
        "Invariance, bitwise (number: distinct EP4 results over 13 NCCL settings; grey: "
        "inference only)",
        fontsize=10,
        color=INK,
    )

    accuracy = fig.add_subplot(spec[1, 1])
    rows = np.arange(len(keys))
    worst = [prior[k]["max_abs_vs_fp64"] for k in keys]
    mean = [prior[k]["mean_abs_vs_fp64"] for k in keys]
    colors = [_color(k, i) for i, k in enumerate(keys)]
    accuracy.barh(rows - 0.2, worst, height=0.38, color=colors)
    accuracy.barh(rows + 0.2, mean, height=0.38, color=colors, alpha=0.4)
    accuracy.set_xscale("log")
    accuracy.set_yticks(rows, [SHORT[k] for k in keys], fontsize=7, color=INK)
    accuracy.invert_yaxis()
    for i, (a, b) in enumerate(zip(worst, mean)):
        accuracy.text(a, i - 0.2, f" max {a:.1e}", va="center", fontsize=7, color=MUTED)
        accuracy.text(b, i + 0.2, f" mean {b:.1e}", va="center", fontsize=7, color=MUTED)
    accuracy.set_xlabel("|output - FP64 reference|, EP4, 1024 tokens", fontsize=9, color=MUTED)
    accuracy.set_title("Accuracy (max solid, mean faded)", fontsize=10, color=INK)

    forward = fig.add_subplot(spec[2, 0])
    tokens = [int(n) for n in report["latency_tokens"]]
    for i, k in enumerate(keys):
        values = [prior[k]["latency_us_ep4"][str(n)] / 1000.0 for n in tokens]
        forward.plot(
            tokens,
            values,
            marker="o",
            color=_color(k, i),
            label=LABELS[k],
            linewidth=2.5 if k == "rl_kernel_ep" else 1.2,
        )
    forward.set_xscale("log", base=2)
    forward.set_yscale("log")
    forward.set_xticks(tokens, [str(n) for n in tokens])
    forward.set_xlabel("tokens", fontsize=9, color=MUTED)
    forward.set_ylabel("forward latency, EP4 (ms, median, slowest rank)", fontsize=9, color=MUTED)
    forward.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    forward.set_title("Forward performance", fontsize=10, color=INK)

    backward = fig.add_subplot(spec[2, 1])
    trained = [k for k in keys if prior[k].get("backward")]
    rows = np.arange(len(trained))
    ep4 = [prior[k]["backward"]["latency_us_ep4"] / 1000.0 for k in trained]
    ep1 = [prior[k]["backward"]["latency_us_ep1"] / 1000.0 for k in trained]
    colors = [_color(k, keys.index(k)) for k in trained]
    backward.barh(rows - 0.2, ep4, height=0.38, color=colors)
    backward.barh(rows + 0.2, ep1, height=0.38, color=colors, alpha=0.4)
    backward.set_yticks(rows, [SHORT[k] for k in trained], fontsize=7, color=INK)
    backward.invert_yaxis()
    for i, (a, b) in enumerate(zip(ep4, ep1)):
        backward.text(a, i - 0.2, f" EP4 {a:.0f} ms", va="center", fontsize=7, color=MUTED)
        backward.text(b, i + 0.2, f" EP1 {b:.0f} ms", va="center", fontsize=7, color=MUTED)
    backward.set_xlabel("backward latency, 1024 tokens (ms, median)", fontsize=9, color=MUTED)
    backward.set_title("Backward performance (EP4 solid, EP1 faded)", fontsize=10, color=INK)

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
