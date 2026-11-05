# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot measured benchmark JSON; never substitute estimates for missing runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    results = report["results"]
    if not results:
        raise ValueError("report has no measured results")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({(r["shape"][2], r["duplicate_fraction_requested"]) for r in results})
    fig, axes = plt.subplots(len(groups), 3, figsize=(13, 3.2 * len(groups)), squeeze=False)
    for row, (width, fraction) in enumerate(groups):
        for col, phase in enumerate(("forward", "backward", "forward_backward")):
            axis = axes[row, col]
            for name in ("torch", "torch_deterministic", "triton"):
                series = sorted(
                    (
                        r
                        for r in results
                        if r["implementation"] == name
                        and r["shape"][2] == width
                        and r["duplicate_fraction_requested"] == fraction
                    ),
                    key=lambda r: r["shape"][1],
                )
                axis.plot(
                    [r["shape"][1] for r in series],
                    [r["timings"][phase]["median_wall_ms"] for r in series],
                    marker="o",
                    label=name,
                )
            axis.set(
                xscale="log",
                xlabel="Mixed sequence rows",
                ylabel="Median wall time (ms)",
                title=f"D={width}, repeats={fraction:.0%}: {phase.replace('_', '+')}",
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    env = report["environment"]
    fig.suptitle(f"{env['gpu']} | {env['commit'][:12]} | eager complete calls; warmup/JIT excluded")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
