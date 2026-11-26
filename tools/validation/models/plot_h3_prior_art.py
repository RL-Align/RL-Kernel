#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot H3 prior-art latency, accuracy, and batch invariance.

    python tools/validation/models/plot_h3_prior_art.py \\
        reports/experiments/h3-prior-art-b200/<op>.json

Writes ``<op>.png`` next to the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import transforms  # noqa: E402


def _entries(report):
    for mode, cands in report["results"].items():
        if "unavailable" in cands:
            continue
        for name, entry in cands.items():
            if "unavailable" not in entry:
                yield (name if name.endswith("]") or mode == "plain" else f"{name}[{mode}]"), entry


def _differ(bi) -> int:
    rows = sum(v["fwd_differ"] + v["rowgrad_differ"] for v in bi["all_rows"].values())
    sub = bi["full_vs_sub_batches"]
    return rows + sub["fwd_differ"] + sub["rowgrad_differ"] + bi["batch_size_sweep"]["differ"]


def main() -> None:
    path = Path(sys.argv[1])
    report = json.loads(path.read_text())
    entries = list(_entries(report))
    names = [n for n, _ in entries]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), layout="constrained")
    fig.suptitle(
        f"{report['op']} vs existing implementations — {report['environment']['gpu']}, "
        f"commit {report['rl_kernel_commit'][:7]}",
        fontsize=13,
    )

    ax = axes[0]
    for name, entry in entries:
        perf = entry["accuracy_latency"]
        key = "fwd_bwd_us" if entry["has_backward"] else "fwd_us"
        sizes = sorted(perf, key=int)
        ax.plot(
            [int(s) for s in sizes],
            [perf[s][key] for s in sizes],
            marker="o",
            ls="-" if entry["has_backward"] else "--",
            label=name + ("" if entry["has_backward"] else " (forward only)"),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("rows (timesteps or tokens)")
    ax.set_ylabel("µs, median")
    ax.set_title("forward + backward latency (dashed: forward only)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    first = entries[0][1]["accuracy_latency"]
    size = "3" if "3" in first else sorted(first, key=int)[-1]
    ys = range(len(entries))
    fwd = [e["accuracy_latency"][size]["fwd_rel_err"] for _, e in entries]
    grad = [
        max(e["accuracy_latency"][size].get("grad_rel_err", {}).values(), default=0)
        for _, e in entries
    ]
    ax.barh([y - 0.2 for y in ys], fwd, 0.4, label="forward")
    ax.barh([y + 0.2 for y in ys], [g or float("nan") for g in grad], 0.4, label="worst gradient")
    values = [v for v in fwd + grad if v]
    if values and max(values) / min(values) > 10:
        ax.set_xscale("log")
    inside = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for y, a, b in zip(ys, fwd, grad):
        ax.text(
            0.98,
            y,
            f"fwd {a:.1e} / grad {b:.1e}" if b else f"fwd {a:.1e}",
            transform=inside,
            ha="right",
            va="center",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1},
        )
    ax.set_yticks(list(ys), names, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(f"error vs FP64, max|err| / max|ref| (size {size})")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[2]
    for y, (_, e) in zip(ys, entries):
        bi = e["batch_invariance"]
        rep = bi["params_repeatable"]
        lines = ["batch-invariant" if bi["batch_invariant"] else "NOT batch-invariant"]
        lines.append(f"{_differ(bi)} row comparisons differ")
        if rep is not None:
            lines.append("parameter/table grads " + ("repeatable" if rep else "NOT repeatable"))
        ax.text(
            0.02,
            y,
            " | ".join(lines),
            va="center",
            fontsize=8,
            color="#3aa676" if bi["batch_invariant"] else "#d14b4b",
        )
    ax.set_ylim(len(entries) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.axis("off")
    ax.set_title("batch invariance (bitwise)")
    ax.text(
        0.02,
        len(entries) - 0.3,
        "every row alone vs full batches + full batch vs covering sub-batches + size sweep",
        fontsize=7,
        color="#555555",
    )

    out = path.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
