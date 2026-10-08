#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot a report from benchmarks/fused_logp_backward_evidence.py (writes figure.png beside it).

    python benchmarks/plot_fused_logp_backward_evidence.py report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "torch log_softmax + gather": "#e07b39",
    "previous VJP (full FP32 softmax)": "#7a7a7a",
    "chunked fallback": "#3aa676",
    "fused kernel": "#2a6fdb",
}


def main() -> None:
    path = Path(sys.argv[1])
    report = json.loads(path.read_text())
    names = list(report["row_invariance"])
    perf = report["performance"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    fig.suptitle(
        f"Generic fused logp backward (#174) — {report['environment']['gpu']}, "
        f"V = {report['vocab']}, BF16 logits, commit {report['git_commit'][:7]}",
        fontsize=13,
    )

    for ax, key, label in (
        (axes[0, 0], "step_ms", "forward + backward time (median)"),
        (axes[0, 1], "peak_gib", "peak memory above the inputs (incl. returned grad)"),
    ):
        for name in names:
            rows = sorted(int(r) for r in perf[name])
            ax.plot(
                rows,
                [perf[name][str(r)][key] for r in rows],
                "o-",
                color=COLORS.get(name),
                label=name,
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("rows (tokens)")
        ax.set_ylabel("ms" if key == "step_ms" else "GiB")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    acc = report["accuracy"]
    vals = [acc[n]["grad_max_abs_over_absmax"] for n in names]
    ax.bar(range(len(names)), vals, color=[COLORS.get(n) for n in names])
    for i, n in enumerate(names):
        cr = acc[n]["grad_correctly_rounded_fraction"]
        ax.text(
            i,
            vals[i],
            f"{vals[i]:.2e}\n{cr:.4%} rounded\ncorrectly",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(top=max(vals) * 5)
    ax.set_title("dlogits error vs FP64: max |err| / max |dlogits| (257 rows)")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    bi = report["row_invariance"]
    checked = next(iter(bi.values()))["rows_checked"]
    ys = range(len(names))
    lp = [bi[n]["logp_rows_differing"] for n in names]
    gr = [bi[n]["grad_rows_differing"] for n in names]
    ax.barh([y - 0.2 for y in ys], lp, 0.4, color="#2a6fdb", label="logp")
    ax.barh([y + 0.2 for y in ys], gr, 0.4, color="#e07b39", label="dlogits")
    for y, a, b in zip(ys, lp, gr):
        ax.text(max(a, b) + 0.05, y, f"{a} / {b}", va="center", fontsize=8)
    ax.set_yticks(list(ys), [n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(3, max(lp + gr) * 1.4))
    ax.set_xlabel(f"rows differing (of {checked}; row alone vs inside a batch, bitwise)")
    ax.set_title("row invariance (0 = batch-invariant)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    out = path.parent / "figure.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
