#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot the ``visual_vae_contract`` evidence (``h3_visual_vae_evidence.py`` output).

    python tools/validation/models/plot_h3_visual_vae.py evidence.json [--out-dir DIR]

Writes ``h3-visual-vae-accuracy.png`` (max |difference| per reproducibility property, contract
vs the Diffusers library path) and ``h3-visual-vae-latency.png`` next to the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONTRACT = "#2a78d6"  # categorical slot 1
DIFFUSERS = "#eb6834"  # categorical slot 2
UNTILED = "#1baf7a"  # categorical slot 3
PINNED = "#eda100"  # categorical slot 4
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
FLOOR = 1e-9  # where an exact zero is drawn on the log axis


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _accuracy_rows(report: dict) -> list[tuple[str, float, float | None]]:
    checks = report["checks"]
    run = checks["run_to_run"]
    parity = checks["diffusers_parity"]
    bi = checks["batch_invariance"]
    batched = bi["diffusers_batched_pinned_flags"]
    tiling = checks["tiling"]

    def zero_if(flag: bool) -> float:
        return 0.0 if flag else float("nan")

    rows: list[tuple[str, float, float | None]] = []
    for key, label in (
        ("encode_keyframe", "keyframe encode"),
        ("encode_video", "22-frame encode"),
        ("decode_video", "22-frame decode"),
    ):
        rows.append((f"run-to-run, {label}", zero_if(run[key]["bitwise_equal"]), None))
    for key, label in (
        ("encode_keyframe", "keyframe encode"),
        ("encode_video", "22-frame encode"),
        ("decode_video", "22-frame decode"),
    ):
        rows.append(
            (f"vs Diffusers recipe, {label}", parity["diffusers_pinned"][key]["max_abs"], None)
        )
    for key, label in (
        ("encode_keyframe", "keyframe encode"),
        ("encode_video", "22-frame encode"),
        ("decode_video", "22-frame decode"),
    ):
        rows.append(
            (
                f"batch row vs alone, {label}",
                zero_if(bi["contract_per_row"][key]),
                batched[key]["max_abs"],
            )
        )
    rows.append(
        (
            "tiled vs untiled, keyframe encode",
            tiling["tiled_vs_untiled_encode_keyframe"]["max_abs"],
            None,
        )
    )
    decode_untiled = tiling["tiled_vs_untiled_decode_video"]["max_abs"]
    rows.append(("tiled vs untiled, 22-frame decode", decode_untiled, None))
    return rows


def plot_accuracy(report: dict, out: Path) -> None:
    rows = _accuracy_rows(report)
    fig, ax = plt.subplots(figsize=(11, 7.0))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.86, bottom=0.13)
    _style(ax)
    height = 0.36
    for i, (_label, ours, theirs) in enumerate(rows):
        y = len(rows) - 1 - i
        offset = height / 2 if theirs is not None else 0.0
        value = max(ours, FLOOR) if ours == ours else FLOOR
        ax.barh(y + offset, value, height=height, color=CONTRACT, left=0)
        text = "bitwise equal" if ours == 0.0 else f"{ours:.3g}"
        ax.text(value * 1.4, y + offset, text, va="center", fontsize=9, color=INK)
        if theirs is not None:
            tval = max(theirs, FLOOR)
            ax.barh(y - offset, tval, height=height, color=DIFFUSERS)
            ttext = "bitwise equal" if theirs == 0.0 else f"{theirs:.3g}"
            ax.text(tval * 1.4, y - offset, ttext, va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR / 2, 10)
    ax.set_xlabel("max |difference| (log scale; exact zero drawn at 1e-9)", color=INK)
    env = report["environment"]
    ax.set_title(
        f"H3 VisualVAE, released weights — {env.get('device_name')}, "
        f"torch {env['torch']}, {report['rl_kernel_commit'][:8]}",
        fontsize=11,
        color=INK,
        loc="left",
        pad=26,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CONTRACT),
        plt.Rectangle((0, 0), 1, 1, color=DIFFUSERS),
    ]
    ax.legend(
        handles,
        ["RL-Kernel contract (per-row, pinned flags, LUT preprocessing)", "Diffusers batched path"],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    pre = report["checks"]["preprocessing"]
    fig.text(
        0.01,
        0.015,
        f"uint8/255 on CPU vs CUDA (Diffusers op): {pre['div255_cpu_vs_cuda_mismatches']}/256 "
        f"values differ; contract LUT equals the CUDA release path: "
        f"{pre['lut_equals_cuda_release']}.\nTiled vs untiled is not a defect: tiling is part "
        "of the release identity (informational row).",
        fontsize=8.5,
        color=MUTED,
    )
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_latency(report: dict, out: Path) -> None:
    timing = report["timing"]
    ops = [
        ("encode_keyframe", "keyframe encode"),
        ("encode_video", "22-frame encode"),
        ("decode_video", "22-frame decode"),
    ]
    series = [
        ("contract", "RL-Kernel contract (tiled)", CONTRACT),
        ("diffusers_default_flags", "Diffusers recipe (tiled, default flags)", DIFFUSERS),
        ("diffusers_pinned_flags", "Diffusers recipe (tiled, pinned flags)", PINNED),
        ("contract_untiled", "contract, tiling off (not the release)", UNTILED),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.4), layout="constrained")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    width = 0.2
    for s, (key, label, color) in enumerate(series):
        xs, ys = [], []
        for i, (op, _) in enumerate(ops):
            entry = timing.get(key, {}).get(op)
            if entry is None:
                continue
            xs.append(i + (s - 1.5) * (width + 0.015))
            ys.append(entry["median_ms"])
        bars = ax.bar(xs, ys, width=width, color=color, label=label)
        for bar, y in zip(bars, ys, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{y:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(range(len(ops)))
    ax.set_xticklabels([o[1] for o in ops])
    ax.set_ylabel("median latency per batch row (ms)", color=INK)
    h, w = report["inputs"]["video"][3:5]
    ax.set_title(
        f"VisualVAE latency at {h}x{w}, batch 1 — {report['environment'].get('device_name')}",
        fontsize=11,
        color=INK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    out_dir = args.out_dir or args.report.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(report, out_dir / "h3-visual-vae-accuracy.png")
    plot_latency(report, out_dir / "h3-visual-vae-latency.png")


if __name__ == "__main__":
    main()
