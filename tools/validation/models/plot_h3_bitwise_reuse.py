#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot ``h3_bitwise_reuse.py`` reports: share of forward outputs that are not bitwise equal.

One heatmap per report. Rows are implementations, columns are the cases compared against
diffusers' path and against the rl-kernel op; 0 means bitwise identical.

    python tools/validation/models/plot_h3_bitwise_reuse.py \\
        reports/experiments/h3-bitwise-reuse-b200/*.json --out figure.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REFS = ("diffusers[plain]", "diffusers", "rl_kernel")


def _cell(entry: dict | None) -> float | None:
    if not entry or "out" not in entry:  # no entry, or a case the reference never saved
        return None
    outs = [o for o in entry["out"] if o is not None]
    if not outs:
        return None
    if any("differ" not in o for o in outs):  # shape or dtype mismatch
        return 1.0
    numel = sum(o["numel"] for o in outs)
    return sum(o["differ"] for o in outs) / numel if numel else None


def _text(frac: float | None) -> str:
    if frac is None:
        return "—"
    if frac == 0:
        return "0"
    return f"{frac:.1%}" if frac >= 0.001 else f"{frac:.2%}"


def _panel(ax, report: dict) -> None:
    results = {k: v for k, v in report["results"].items() if "unavailable" not in v}
    cases = report["cases"]
    diffusers = next((r for r in _REFS[:2] if r in results), None)
    refs = [r for r in (diffusers, "rl_kernel") if r]
    rows = ["rl_kernel"] + sorted(r for r in results if r not in _REFS)
    cols = [(ref, case) for ref in refs for case in cases]
    grid = np.full((len(rows), len(cols)), np.nan)
    for i, row in enumerate(rows):
        for j, (ref, case) in enumerate(cols):
            frac = _cell(results.get(row, {}).get(f"vs {ref}", {}).get(str(case)))
            if frac is not None:
                grid[i, j] = frac
            ax.text(
                j,
                i,
                "—" if row == ref else ("n/a" if frac is None else _text(frac)),
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if frac is not None and frac > 0.3 else "black",
                fontweight="bold" if frac == 0 else None,
            )
    # A comparison that was not made is grey, so it cannot be read as the white of "bitwise equal".
    cmap = plt.get_cmap("Reds").with_extremes(bad="0.82")
    ax.imshow(np.ma.masked_invalid(np.sqrt(grid)), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_xticks(range(len(cols)), [str(c) for _, c in cols], fontsize=7, rotation=45)
    transform = ax.get_xaxis_transform()
    for k, ref in enumerate(refs):
        if k:
            ax.axvline(k * len(cases) - 0.5, color="k", lw=1.5)
        label = "vs rl-kernel" if ref == "rl_kernel" else "vs diffusers"
        ax.text(
            k * len(cases) + (len(cases) - 1) / 2,
            1.01,
            label,
            ha="center",
            va="bottom",
            fontsize=8.5,
            transform=transform,
        )
    ax.set_title(f"{report['op']} @ {report['rl_kernel_commit'][:8]}", fontsize=10, pad=18)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in args.reports]
    fig, axes = plt.subplots(1, len(reports), figsize=(6.4 * len(reports), 5.8), squeeze=False)
    for ax, report in zip(axes[0], reports):
        _panel(ax, report)
    gpu = reports[0]["environment"].get("gpu", "")
    fig.suptitle(
        "Forward outputs that are not bitwise equal\n"
        f"(share of elements; 0 = bitwise identical), {gpu}",
        fontsize=10,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=125)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
