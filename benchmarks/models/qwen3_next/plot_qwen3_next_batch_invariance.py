#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render a published Qwen3-Next batch-invariance matrix as a PNG.

Needs only the published evidence directory and matplotlib:

    python benchmarks/models/qwen3_next/plot_qwen3_next_batch_invariance.py \\
        artifacts/qwen3-next-batch-invariance-b200

writes ``figure.png`` into that directory: one panel per cell family. A cell is
green when the target request's response tokens, logits, logprobs and every
layer digest are bitwise equal to its reference, red otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

INK = "#2b2b29"
MUTED = "#6f6e69"
SURFACE = "#fcfcfb"
NA = "#e4e3dd"
COLORS = ListedColormap([NA, "#d64545", "#1baf7a"])  # 0 absent, 1 differs, 2 bitwise equal


def _value(cell):
    return 2 if cell["passed"] and all(cell["comparisons"].values()) else 1


def _grid(axis, values, rows, cols, title, xlabel=None, ylabel=None):
    axis.imshow(values, cmap=COLORS, vmin=0, vmax=2, aspect="auto")
    axis.set_yticks(range(len(rows)), rows, fontsize=8, color=INK)
    axis.set_xticks(range(len(cols)), cols, fontsize=8, color=INK, rotation=0)
    axis.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=2)
    axis.tick_params(which="minor", length=0)
    passed = int((values == 2).sum())
    total = int((values > 0).sum())
    axis.set_title(f"{title}: {passed}/{total} bitwise", fontsize=10, color=INK)
    if xlabel:
        axis.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9, color=MUTED)
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot(evidence: Path, out: Path) -> None:
    matrix = json.loads((evidence / "bi" / "bi_matrix.json").read_text())
    cells = matrix["cells"]

    conc = {}
    for cid, cell in cells.items():
        m = re.fullmatch(r"concurrency-n(\d+)-(first|last)", cid)
        if m:
            conc[(int(m.group(1)), m.group(2))] = _value(cell)
    ns = sorted({n for n, _ in conc})
    conc_grid = np.array([[conc.get((n, p), 0) for n in ns] for p in ("first", "last")])

    chunk = {}
    for cid, cell in cells.items():
        m = re.fullmatch(r"chunk-p(\d+)-c(\d+)-f(\d+)", cid)
        if m:
            chunk[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = _value(cell)
    prompts = sorted({p for p, _, _ in chunk})
    plans = sorted({(c, f) for _, c, f in chunk})
    chunk_grid = np.array([[chunk.get((p, c, f), 0) for c, f in plans] for p in prompts])

    def listed(prefix):
        ids = [cid for cid in matrix["cell_ids"] if cid.startswith(prefix)]
        return ids, np.array([[_value(cells[cid])] for cid in ids])

    inter_ids, inter_grid = listed("interference-")
    pack_ids, pack_grid = listed("packing-")

    fig = plt.figure(figsize=(16, 8.5), facecolor=SURFACE)
    spec = fig.add_gridspec(2, 3, width_ratios=[1.5, 0.9, 0.9], height_ratios=[1, 2.6])
    _grid(
        fig.add_subplot(spec[0, 0]),
        conc_grid,
        ["target first", "target last"],
        [str(n) for n in ns],
        "Concurrency (N requests, vs. the target alone)",
        xlabel="N",
    )
    _grid(
        fig.add_subplot(spec[1, 0]),
        chunk_grid,
        [str(p) for p in prompts],
        [f"c{c}\nfirst {f}" for c, f in plans],
        "Chunk boundaries (vs. the replay engine)",
        xlabel="chunk size / first chunk",
        ylabel="prompt tokens",
    )
    _grid(
        fig.add_subplot(spec[:, 1]),
        inter_grid,
        [cid.removeprefix("interference-") for cid in inter_ids],
        [""],
        "Interference",
    )
    _grid(
        fig.add_subplot(spec[:, 2]),
        pack_grid,
        [cid.removeprefix("packing-") for cid in pack_ids],
        [""],
        "Packing",
    )
    total = len(cells)
    passed = sum(_value(cell) == 2 for cell in cells.values())
    fig.suptitle(
        f"Qwen3-Next-80B-A3B batch invariance, TP4, suite {matrix.get('suite', '')}: "
        f"{passed}/{total} cells bitwise (tokens, logits, logprobs, layer digests)",
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
