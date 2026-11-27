#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot the ``visual_vae_contract`` evidence (``h3_visual_vae_evidence.py`` output).

    python tools/validation/models/plot_h3_visual_vae.py REPORT.json [REPORT2.json] [--out-dir DIR]

For every report it writes three charts, tagged with the GPU (``b200``, ``h100``, ...):

- ``h3-visual-vae-accuracy-<gpu>.png``: max |difference| of each contract property;
- ``h3-visual-vae-batch-invariance-<gpu>.png``: batch row vs the same row alone, per case,
  for the contract and for the Diffusers batched path under default and pinned flags;
- ``h3-visual-vae-latency-<gpu>.png``: median latency per step.

With two reports it also writes ``h3-visual-vae-batch-invariance-<gpu1>-vs-<gpu2>.png``, the
Diffusers batched path per case on both GPUs. Next to every chart it writes a Markdown table
(same stem, ``.md``) holding exactly the plotted values, with the same labels and rounding
(max |difference| to 3 significant digits, latency to 0.1 ms), and a source line.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-4 of the reference palette, in order (validated: adjacent pairs pass).
CONTRACT = "#2a78d6"  # slot 1, blue
DIFFUSERS = "#eb6834"  # slot 2, orange
PINNED = "#1baf7a"  # slot 3, aqua
UNTILED = "#eda100"  # slot 4, yellow
# Two GPUs in the cross-GPU chart (violet, green: a validated pair, distinct from the above).
GPU_COLORS = ("#6250d6", "#008300")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
ZERO = "= 0, bitwise"
DIFFERS = "differs (no max recorded)"

OPS = (
    ("encode_keyframe", "keyframe encode"),
    ("encode_video", "22-frame encode"),
    ("decode_video", "22-frame decode"),
)


# --- formatting shared by charts and tables -------------------------------------------------


def fmt_delta(value: float) -> str:
    if value != value:  # NaN: not equal, but the report has no max for it
        return DIFFERS
    return ZERO if value == 0.0 else f"{value:.3g}"


def fmt_ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def gpu_tag(report: dict) -> str:
    """File-name tag from the whole device name, without the vendor prefix:
    ``NVIDIA B200`` -> ``b200``, ``NVIDIA H100 80GB HBM3`` -> ``h100-80gb-hbm3``."""

    name = str(report["environment"].get("device_name") or "gpu").lower()
    name = re.sub(r"^nvidia\s+", "", name)
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-") or "gpu"


def gpu_name(report: dict) -> str:
    return str(report["environment"].get("device_name") or "unknown GPU")


def commit(report: dict) -> str:
    return str(report["rl_kernel_commit"])[:8]


def source_line(reports: list[tuple[str, dict]], field: str) -> str:
    parts = [f"`{name}` (`{field}`), commit `{commit(r)}`, {gpu_name(r)}" for name, r in reports]
    return "Source: " + "; ".join(parts) + "."


def write_table(path: Path, header: list[str], rows: list[list[str]], source: str) -> None:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    path.write_text("\n".join(lines) + "\n\n" + source + "\n", encoding="utf-8")


# --- data extraction --------------------------------------------------------------------------


def max_abs(entry: dict) -> float:
    """``max_abs`` of a comparison; NaN ("differs") when the report has none (``null``, or
    ``NaN`` in reports written before the evidence tool switched to strict JSON), and when
    the values are equal but the bytes are not (``-0.0`` vs ``0.0``, another dtype), so a
    zero is charted as bitwise only when it is."""

    value = entry["max_abs"]
    if value is None or (value == 0.0 and entry.get("bitwise_equal") is False):
        return math.nan
    return float(value)


def accuracy_rows(report: dict) -> list[tuple[str, float]]:
    """Contract properties as (label, max |difference|); NaN = not equal, no max recorded."""

    checks = report["checks"]
    run, parity, per_row = (
        checks["run_to_run"],
        checks["diffusers_parity"],
        checks["batch_invariance"]["contract_per_row"],
    )
    tiling = checks["tiling"]

    def zero_if(flag: bool) -> float:
        return 0.0 if flag else math.nan

    rows: list[tuple[str, float]] = []
    for key, label in OPS:
        repeats = len(run[key]["digests"])
        rows.append(
            (f"run to run ({repeats} repeats), {label}", zero_if(run[key]["bitwise_equal"]))
        )
    for key, label in OPS:
        rows.append(
            (
                f"vs Diffusers recipe, pinned flags, {label}",
                max_abs(parity["diffusers_pinned"][key]),
            )
        )
    for key, label in OPS:
        value = max_abs(parity["diffusers_default_flags"][key])
        rows.append((f"vs Diffusers recipe, default flags, {label}", value))
    for key, label in OPS:
        rows.append((f"batch row vs same row alone, {label}", zero_if(per_row[key])))
    rows.append(
        (
            "tiled vs untiled, keyframe encode*",
            max_abs(tiling["tiled_vs_untiled_encode_keyframe"]),
        )
    )
    rows.append(
        ("tiled vs untiled, 22-frame decode*", max_abs(tiling["tiled_vs_untiled_decode_video"]))
    )
    return rows


def batch_cases(report: dict) -> list[tuple[str, float, float, float]]:
    """Per case: (label, contract, Diffusers batched default flags, Diffusers batched pinned)."""

    bi = report["checks"]["batch_invariance"]
    default, pinned = bi["diffusers_batched_default_flags"], bi["diffusers_batched_pinned_flags"]
    contract = bi.get("contract_cases")
    rows = []
    for key, label in OPS:
        pinned_cases = {(c["batch"], c["row"]): max_abs(c) for c in pinned[key]["cases"]}
        if contract is not None:
            ours = {(c["batch"], c["row"]): max_abs(c) for c in contract[key]["cases"]}
        else:
            # Reports without per-case contract results carry one verdict per step (every
            # row of the full batch vs the same row alone); it is shown on each case.
            verdict = 0.0 if bi["contract_per_row"][key] else math.nan
            ours = {(c["batch"], c["row"]): verdict for c in default[key]["cases"]}
        for case in default[key]["cases"]:
            b, r = case["batch"], case["row"]
            rows.append(
                (f"{label}, batch {b} row {r}", ours[(b, r)], max_abs(case), pinned_cases[(b, r)])
            )
    return rows


# --- chart helpers ----------------------------------------------------------------------------


def _style(ax: Any, axis: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _floor(values: list[float]) -> float:
    """Left edge of a log axis: one decade below the smallest nonzero value (at most 1e-2)."""

    nonzero = [v for v in values if v == v and v > 0.0]
    if not nonzero:
        return 1e-2
    return min(1e-2, 10.0 ** (math.floor(math.log10(min(nonzero))) - 1))


def _log_bar(ax: Any, y: float, value: float, color: str, height: float, floor: float) -> None:
    """One horizontal log-scale bar; an exact zero becomes a marker at the floor plus a label."""

    if value == 0.0 or value != value:
        ax.plot(
            [floor],
            [y],
            marker="D" if value == 0.0 else "x",
            color=color,
            markersize=6,
            clip_on=False,
        )
        ax.text(floor * 2.2, y, fmt_delta(value), va="center", fontsize=8.5, color=INK)
        return
    ax.barh(
        y, value - floor, height=height, color=color, left=floor, edgecolor="white", linewidth=2
    )
    ax.text(value * 1.35, y, fmt_delta(value), va="center", fontsize=8.5, color=INK)


def _log_axis(ax: Any, floor: float, values: list[float]) -> None:
    top = max([v for v in values if v == v] + [floor])
    ax.set_xscale("log")
    ax.set_xlim(floor / 2, 10.0 ** (math.ceil(math.log10(top)) + 0.3))
    ax.set_xlabel(
        "max |difference| (log scale; an exact zero is marked at the left edge, '= 0, bitwise')",
        color=INK,
    )


def _title(ax: Any, text: str) -> None:
    ax.set_title(text, fontsize=11, color=INK, loc="left", pad=28)


def _legend(ax: Any, items: list[tuple[str, str]]) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in items]
    ax.legend(
        handles,
        [label for label, _ in items],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=len(items),
        frameon=False,
        fontsize=9,
    )


def _save(fig: Any, out: Path) -> None:
    fig.savefig(out, dpi=150, metadata={"Software": None})
    plt.close(fig)


# --- charts -----------------------------------------------------------------------------------


def plot_accuracy(name: str, report: dict, out: Path) -> None:
    rows = accuracy_rows(report)
    fig, ax = plt.subplots(figsize=(11, 7.4))
    fig.subplots_adjust(left=0.33, right=0.97, top=0.88, bottom=0.14)
    _style(ax, "x")
    floor = _floor([v for _, v in rows])
    for i, (_, value) in enumerate(rows):
        _log_bar(ax, len(rows) - 1 - i, value, CONTRACT, 0.55, floor)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in reversed(rows)], fontsize=9.5)
    _log_axis(ax, floor, [v for _, v in rows])
    _title(ax, f"RL-Kernel VisualVAE contract: reproducibility properties — {gpu_name(report)}")
    _legend(ax, [("RL-Kernel contract (per row, pinned flags, LUT preprocessing)", CONTRACT)])
    pre = report["checks"]["preprocessing"]
    fig.text(
        0.01,
        0.015,
        f"* informational: tiling is part of the release identity, so tiled != untiled is "
        f"expected. uint8/255 differs between CPU and CUDA for "
        f"{pre['div255_cpu_vs_cuda_mismatches']}/256 values;\nthe contract LUT equals the CUDA "
        f"release path: {pre['lut_equals_cuda_release']}. Released weights, "
        f"commit {commit(report)}.",
        fontsize=8.5,
        color=MUTED,
    )
    _save(fig, out)
    write_table(
        out.with_suffix(".md"),
        ["Property (RL-Kernel contract)", "max \\|Δ\\|"],
        [
            [label.rstrip("*") + (" (informational)" if label.endswith("*") else ""), fmt_delta(v)]
            for label, v in rows
        ],
        source_line([(name, report)], "checks"),
    )


def plot_batch_invariance(name: str, report: dict, out: Path) -> None:
    rows = batch_cases(report)
    series = [
        ("RL-Kernel contract (per row)", CONTRACT),
        ("Diffusers batched, default flags", DIFFUSERS),
        ("Diffusers batched, pinned flags", PINNED),
    ]
    fig, ax = plt.subplots(figsize=(11, 9.0))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.9, bottom=0.08)
    _style(ax, "x")
    height = 0.28
    floor = _floor([v for row in rows for v in row[1:]])
    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        for s, (_, color) in enumerate(series):
            _log_bar(ax, y + (1 - s) * height, row[1 + s], color, height, floor)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9.5)
    _log_axis(ax, floor, [v for row in rows for v in row[1:]])
    _title(ax, f"Batch row vs the same row alone, per case — {gpu_name(report)}")
    _legend(ax, series)
    _save(fig, out)
    write_table(
        out.with_suffix(".md"),
        ["Case", *[label for label, _ in series]],
        [[r[0], *[fmt_delta(v) for v in r[1:]]] for r in rows],
        source_line([(name, report)], "checks.batch_invariance"),
    )


def plot_batch_invariance_gpus(reports: list[tuple[str, dict]], out: Path) -> None:
    per_gpu = [batch_cases(r) for _, r in reports]
    labels = [row[0] for row in per_gpu[0]]
    if any([row[0] for row in rows] != labels for rows in per_gpu):
        raise ValueError("the reports have different batch-invariance cases")
    names = [gpu_name(r) for _, r in reports]
    fig, ax = plt.subplots(figsize=(11, 7.6))
    fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.12)
    _style(ax, "x")
    height = 0.38
    floor = _floor([rows[i][2] for rows in per_gpu for i in range(len(labels))])
    for i in range(len(labels)):
        y = len(labels) - 1 - i
        for g, rows in enumerate(per_gpu):
            _log_bar(ax, y + (0.5 - g) * height, rows[i][2], GPU_COLORS[g], height, floor)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(list(reversed(labels)), fontsize=9.5)
    _log_axis(ax, floor, [rows[i][2] for rows in per_gpu for i in range(len(labels))])
    _title(
        ax,
        "Diffusers batched path (default flags), batch row vs the same row alone:\n"
        + " vs ".join(names),
    )
    _legend(ax, [(n, GPU_COLORS[g]) for g, n in enumerate(names)])
    contract_zero = all(row[1] == 0.0 for rows in per_gpu for row in rows)
    fig.text(
        0.01,
        0.015,
        (
            "The RL-Kernel contract is = 0, bitwise in every case on both GPUs."
            if contract_zero
            else "The RL-Kernel contract is not bitwise equal in every case; see per-GPU charts."
        ),
        fontsize=8.5,
        color=MUTED,
    )
    _save(fig, out)
    write_table(
        out.with_suffix(".md"),
        ["Case", *[f"Diffusers batched, default flags, {n}" for n in names]],
        [[label, *[fmt_delta(rows[i][2]) for rows in per_gpu]] for i, label in enumerate(labels)],
        source_line(reports, "checks.batch_invariance.diffusers_batched_default_flags"),
    )


def plot_latency(name: str, report: dict, out: Path) -> None:
    timing = report["timing"]
    series = [
        ("contract", "RL-Kernel contract (tiled)", CONTRACT),
        ("diffusers_default_flags", "Diffusers recipe (tiled, default flags)", DIFFUSERS),
        ("diffusers_pinned_flags", "Diffusers recipe (tiled, pinned flags)", PINNED),
        ("contract_untiled", "contract, tiling off (not the release)", UNTILED),
    ]

    def median(key: str, op: str) -> float | None:
        entry = timing.get(key, {}).get(op)
        return None if entry is None else float(entry["median_ms"])

    fig, ax = plt.subplots(figsize=(11, 5.6), layout="constrained")
    _style(ax, "y")
    width = 0.2
    for s, (key, label, color) in enumerate(series):
        xs, ys = [], []
        for i, (op, _) in enumerate(OPS):
            value = median(key, op)
            if value is not None:
                xs.append(i + (s - 1.5) * width)
                ys.append(value)
        bars = ax.bar(xs, ys, width=width, color=color, label=label, edgecolor="white", linewidth=2)
        for bar, y in zip(bars, ys, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                fmt_ms(y),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK,
            )
    ax.set_xticks(range(len(OPS)))
    ax.set_xticklabels([label for _, label in OPS])
    ax.set_ylabel("median latency per batch row (ms)", color=INK)
    h, w = report["inputs"]["video"][3:5]
    repeats = timing["contract"]["decode_video"]["repeats"]
    ax.set_title(
        f"VisualVAE latency at {h}x{w}, batch 1, median of {repeats} — {gpu_name(report)} "
        f"(commit {commit(report)})",
        fontsize=11,
        color=INK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out)
    write_table(
        out.with_suffix(".md"),
        ["Step (median ms)", *[label for _, label, _ in series]],
        [[label, *[fmt_ms(median(key, op)) for key, _, _ in series]] for op, label in OPS],
        source_line([(name, report)], "timing"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help="one or two evidence JSON files")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--no-latency",
        action="append",
        default=[],
        metavar="GPU",
        help="skip the latency chart for this GPU tag (e.g. timings from a shared GPU)",
    )
    args = parser.parse_args()
    if len(args.reports) > 2:
        parser.error("at most two reports")
    loaded = [(p.name, json.loads(p.read_text(encoding="utf-8"))) for p in args.reports]
    tags = [gpu_tag(r) for _, r in loaded]
    if len(set(tags)) != len(tags):
        parser.error(f"both reports are from the same GPU ({tags[0]}); their charts would collide")
    out_dir = args.out_dir or args.reports[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, report in loaded:
        tag = gpu_tag(report)
        plot_accuracy(name, report, out_dir / f"h3-visual-vae-accuracy-{tag}.png")
        plot_batch_invariance(name, report, out_dir / f"h3-visual-vae-batch-invariance-{tag}.png")
        if tag not in args.no_latency:
            plot_latency(name, report, out_dir / f"h3-visual-vae-latency-{tag}.png")
    if len(loaded) == 2:
        tags = "-vs-".join(gpu_tag(r) for _, r in loaded)
        plot_batch_invariance_gpus(loaded, out_dir / f"h3-visual-vae-batch-invariance-{tags}.png")


if __name__ == "__main__":
    main()
