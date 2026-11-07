# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render latency and extra peak allocation from a complete packed SwiGLU report.

Latency has separate eager and CUDA Graph figures. Memory uses the eager-call
allocation records only: inputs and prebuilt graphs are excluded, and no CUDA
Graph pool or reserved-memory estimate is presented as operator workspace.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if not report["complete"]:
        raise ValueError("refusing to plot an incomplete benchmark")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = args.output_dir or args.report.parent
    output.mkdir(parents=True, exist_ok=True)
    rows = report["results"]
    dtypes = list(dict.fromkeys(r["dtype"] for r in rows))
    modes = ["forward", "backward", "forward_backward"]
    labels = {
        "forward": "Forward",
        "backward": "Backward",
        "forward_backward": "Forward + backward",
    }
    styles = {
        "pytorch": ("PyTorch (uncompiled)", "#A56717", "s", "--", 11, "none"),
        "adapter": ("Triton (two-input adapter)", "#687554", "D", "-.", 8, "none"),
        "packed": ("Triton (packed input)", "#006EAF", "o", "-", 4, "#006EAF"),
    }
    views = list(dict.fromkeys(r["timing"] for r in rows))
    if "eager" in views:
        views.append("memory")
    for view in views:
        memory = view == "memory"
        timing = "eager" if memory else view
        field = "extra_peak_mib" if memory else "median_ms"
        scale = 1 if memory else 1000
        fig, axes = plt.subplots(3, len(dtypes), figsize=(15, 11), squeeze=False, sharex=True)
        for i, mode in enumerate(modes):
            for j, dtype in enumerate(dtypes):
                ax = axes[i, j]
                selected = [
                    r
                    for r in rows
                    if r["timing"] == timing and r["mode"] == mode and r["dtype"] == dtype
                ]
                sizes = sorted({r["rows"] for r in selected})
                for provider, (
                    label,
                    color,
                    marker,
                    linestyle,
                    marker_size,
                    fill,
                ) in styles.items():
                    samples = [
                        [
                            r[field] * scale
                            for r in selected
                            if r["rows"] == size and r["provider"] == provider
                        ]
                        for size in sizes
                    ]
                    medians = [statistics.median(values) for values in samples]
                    ax.plot(
                        sizes,
                        medians,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.6,
                        zorder=2,
                    )
                    # Separate marker artists remain visible above every line and
                    # band. Nested sizes preserve coincident points without jitter.
                    ax.plot(
                        sizes,
                        medians,
                        color=color,
                        marker=marker,
                        linestyle="none",
                        markersize=marker_size,
                        markerfacecolor=fill,
                        markeredgecolor=color,
                        markeredgewidth=1.6,
                        zorder=5,
                    )
                    if not memory:
                        ax.fill_between(
                            sizes,
                            [min(v) for v in samples],
                            [max(v) for v in samples],
                            color=color,
                            alpha=0.12,
                            zorder=1,
                        )
                ax.set_xscale("log", base=2)
                ax.set_yscale("log")
                ax.set_title(f"{dtype.upper()} · {labels[mode]}", fontsize=12)
                ax.grid(True, which="major", alpha=0.2)
                ax.spines[["top", "right"]].set_visible(False)
                if j == 0:
                    ax.set_ylabel(
                        "Extra peak allocation\n(MiB, log scale)"
                        if memory
                        else "Latency (µs, log scale)"
                    )
                if i == 2:
                    ax.set_xlabel("Flattened rows M (log scale)")
        # All panels in a figure share limits and units for direct comparison.
        limits = [ax.get_ylim() for ax in axes.flat]
        for ax in axes.flat:
            ax.set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
        handles = [
            Line2D(
                [],
                [],
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                marker=marker,
                markersize=marker_size,
                markerfacecolor=fill,
                markeredgewidth=1.6,
                label=label,
            )
            for label, color, marker, linestyle, marker_size, fill in styles.values()
        ]
        if memory:
            title = "Extra peak GPU memory"
            footer = (
                "Eager-call peak allocated memory above the starting allocation · "
                "inputs and prebuilt graphs excluded\n"
                "MiB = 2^20 bytes; not reserved memory or total model VRAM."
            )
        else:
            title = "CUDA Graph replay" if timing == "graph" else "Eager launch"
            footer = (
                "CUDA events · median across rounds · shading = observed round range, "
                "not a confidence interval\nFP32 intermediates; no torch.compile. "
                "JIT/capture excluded. Eager measurements include host launch gaps."
            )
        fig.suptitle(f"Packed SwiGLU — {title}", fontsize=19, y=0.99)
        fig.text(
            0.5,
            0.95,
            f"{report['hardware']}  |  input [1, M, {2 * report['width']}]  |  "
            f"output [1, M, {report['width']}]  |  "
            f"order: {report['settings']['packed_order']}",
            ha="center",
            fontsize=11,
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=3,
            frameon=False,
            fontsize=10,
            handlelength=2.4,
        )
        fig.text(
            0.5,
            0.885,
            "Overlapping points use nested markers: large open square / open diamond / "
            "small filled circle. No offsets applied.",
            ha="center",
            fontsize=10,
            color="#425466",
        )
        fig.text(
            0.5,
            0.015,
            footer,
            ha="center",
            fontsize=10,
            color="#425466",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.87))
        path = output / f"swiglu-packed-{view}.png"
        fig.savefig(path, dpi=160, facecolor="white")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
