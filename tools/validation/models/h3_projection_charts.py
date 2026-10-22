# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Accuracy and performance charts of the H3 projection rows from ``h3_projection_bench`` output."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(report: Path, out: Path) -> None:
    rows = json.loads(report.read_text())["rows"]
    names = sorted({r["row"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name in names:
        sel = [r for r in rows if r["row"] == name]
        ax[0].plot([r["tokens"] for r in sel], [r["candidate_ms"] for r in sel], "o-", label=name)
        ax[0].plot([r["tokens"] for r in sel], [r["provider_ms"] for r in sel], "x--", label=f"{name} provider")
        ax[1].plot([r["tokens"] for r in sel], [r["rel_err_candidate"] for r in sel], "o-", label=name)
        ax[1].plot([r["tokens"] for r in sel], [r["rel_err_provider"] for r in sel], "x--", label=f"{name} provider")
    ax[0].set(xscale="log", yscale="log", xlabel="tokens", ylabel="forward ms (H100)", title="Performance")
    ax[1].set(xscale="log", yscale="log", xlabel="tokens", ylabel="relative error vs FP64", title="Accuracy")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
