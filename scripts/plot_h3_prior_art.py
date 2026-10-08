#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot a ``scripts/h3_prior_art.py`` report: latency, accuracy and batch invariance.

    python scripts/plot_h3_prior_art.py docs/usage/evidence/h3-prior-art-b200/<op>.json

Writes ``<op>.png`` next to the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
    for y, a, b in zip(ys, fwd, grad):
        ax.text(
            max(a, b or 0) * 1.3,
            y,
            f"{a:.1e} / {b:.1e}" if b else f"{a:.1e}",
            va="center",
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_yticks(list(ys), names, fontsize=7)
    ax.invert_yaxis()
    ax.set_title(f"error vs FP64, max|err| / max|ref| (size {size})")
    ax.legend(fontsize=7)
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[2]
    diff = [_differ(e["batch_invariance"]) for _, e in entries]
    colors = [
        "#3aa676" if e["batch_invariance"]["batch_invariant"] else "#d14b4b" for _, e in entries
    ]
    ax.barh(list(ys), [max(d, 0.5) for d in diff], color=colors)
    for y, (_, e), d in zip(ys, entries, diff):
        bi = e["batch_invariance"]
        rep = bi["params_repeatable"]
        note = (
            ""
            if rep is None
            else (
                "; parameter/table grads repeatable"
                if rep
                else "; parameter/table grads NOT repeatable"
            )
        )
        verdict = "batch-invariant" if bi["batch_invariant"] else "NOT batch-invariant"
        ax.text(
            max(d, 0.5) * 1.3,
            y,
            f"{verdict}: {d} row comparisons differ{note}",
            va="center",
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_yticks(list(ys), names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0.4, max(max(diff), 1) * 1e3)
    ax.set_title("batch invariance (bitwise)")
    ax.set_xlabel(
        "every row alone vs full batches + full batch vs covering sub-batches + size sweep",
        fontsize=7,
    )
    ax.grid(True, axis="x", alpha=0.3)

    out = path.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
