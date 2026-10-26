#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot a report from tools/validation/models/qwen3_next_norm_evidence.py: one 2x2 figure per op.

python -m pip install matplotlib  # optional plotting dependency
python tools/validation/models/plot_qwen3_next_norm_evidence.py report.json

Writes figure[-<op>].png beside the report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2a6fdb", "#7a7a7a", "#e07b39", "#3aa676", "#9b59b6", "#c0392b"]


def _short(name: str) -> str:
    return name.replace(" (forward only)", "*").replace(" (cast-first)", "\n(cast-first)")


def plot_op(op: str, data: dict, title: str, out: Path) -> None:
    names = list(data["row_invariance"])
    color = {n: COLORS[i % len(COLORS)] for i, n in enumerate(names)}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    fig.suptitle(title, fontsize=13)

    for ax, key, label in (
        (axes[0, 0], "forward_us", "forward latency"),
        (axes[0, 1], "backward_us", "backward latency (training-capable only)"),
    ):
        for name in names:
            rows = sorted(int(r) for r in data["latency"][name])
            ys = [data["latency"][name][str(r)].get(key) for r in rows]
            if all(y is None for y in ys):
                continue
            ax.plot(rows, ys, "o-", color=color[name], label=_short(name))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("rows")
        ax.set_ylabel("µs (median)")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes[1, 0]
    acc = data["accuracy"][max(data["accuracy"], key=int)]
    metrics = [
        ("forward_max_abs", "forward\nmax |err|"),
        ("dx_max_abs_over_absmax", "dx\nmax |err| / max"),
        ("dweight_max_abs_over_absmax", "dweight\nmax |err| / max"),
        ("dgate_max_abs_over_absmax", "dgate\nmax |err| / max"),
    ]
    metrics = [m for m in metrics if any(acc[n].get(m[0]) is not None for n in names)]
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [acc[name].get(m) for m, _ in metrics]
        xs = [j + (i - (len(names) - 1) / 2) * width for j in range(len(metrics))]
        ax.bar(
            [x for x, v in zip(xs, vals) if v is not None],
            [v for v in vals if v is not None],
            width,
            color=color[name],
            label=_short(name),
        )
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics], fontsize=9)
    ax.set_yscale("log")
    ax.set_title(f"error vs FP64 golden ({max(data['accuracy'], key=int)} rows, BF16)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    bi = data["row_invariance"]
    checked = next(iter(bi.values()))["rows_checked"]
    ys = range(len(names))
    fwd = [bi[n]["forward_rows_differing"] for n in names]
    dx = [
        bi[n]["dx_rows_differing"] if bi[n]["dx_rows_differing"] is not None else 0 for n in names
    ]
    ax.barh([y - 0.2 for y in ys], fwd, 0.4, color="#2a6fdb", label="forward")
    ax.barh([y + 0.2 for y in ys], dx, 0.4, color="#e07b39", label="dx")
    for y, n, f, d in zip(ys, names, fwd, dx):
        no_bwd = bi[n]["dx_rows_differing"] is None
        ax.text(max(f, d) + 0.2, y, f"{f} / {'n/a' if no_bwd else d}", va="center", fontsize=8)
    ax.set_yticks(list(ys), [_short(n) for n in names], fontsize=8)
    ax.set_xlim(0, max(3, max(fwd + dx) * 1.4))
    ax.invert_yaxis()
    ax.set_xlabel(f"rows differing (of {checked}; row alone vs inside a batch, bitwise)")
    ax.set_title("row invariance (0 = batch-invariant)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main() -> None:
    path = Path(sys.argv[1])
    report = json.loads(path.read_text())
    env = report["environment"]
    ops = report["ops"]
    for op, data in ops.items():
        name = "figure.png" if len(ops) == 1 else f"figure-{op}.png"
        hidden = data.get("hidden", report.get("hidden"))
        title = (
            f"Qwen3-Next {op.replace('_', ' ')} — {env['gpu']}, hidden {hidden}, "
            f"BF16, commit {report['git_commit'][:7]}  (* forward only)"
        )
        plot_op(op, data, title, path.parent / name)


if __name__ == "__main__":
    main()
