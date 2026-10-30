# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Chart of the WS2 evidence: bitwise agreement and per-rank forward time per parallel layout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(evidence: Path, out: Path) -> None:
    reports = [json.loads(p.read_text()) for p in sorted(evidence.glob("h3_ws2_*.json"))]
    labels = []
    times = []
    colors = []
    for r in reports:
        labels.append(f"TP{r['tp']}xCP{r['cp']}{'+SP' if r.get('sp') else ''}")
        ranks = r["forward_seconds_per_rank"]
        times.append(sum(x["forward_s"] for x in ranks) / len(ranks))
        colors.append("tab:green" if r["all_bitwise"] else "tab:red")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, times, color=colors)
    ax.set(
        ylabel="mean forward s per rank (H100)", title="WS2 layouts: green = bitwise == single GPU"
    )
    for i, r in enumerate(reports):
        ax.text(
            i, times[i], f"{r['world']} GPU\n{len(set(r['hosts']))} node", ha="center", va="bottom"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
