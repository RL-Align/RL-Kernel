#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plot the image VAE evidence (``h3_image_vae_evidence.py`` output).

    python tools/validation/models/plot_h3_image_vae.py REPORT.json [--out-dir DIR]

Writes two charts, tagged with the GPU (``b200``, ``h100``, ...):

- ``h3-image-vae-accuracy-<gpu>.png``: max |difference| of every contract property (all
  expected bitwise zero) and of every ablation probe (I11, I15-I18: expected non-zero, i.e.
  the wrong variant is detectable);
- ``h3-image-vae-latency-<gpu>.png``: median latency of the edit-reference encode and the
  decode, for the contract and the Diffusers release blocks, tiled and untiled.

Next to every chart a Markdown table (same stem, ``.md``) holds exactly the plotted values,
with the same labels and rounding, and a source line. Styling and formatting are shared with
``plot_h3_visual_vae.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.validation.models import plot_h3_visual_vae as base  # noqa: E402

plt = base.plt

BAD = base.DIFFUSERS  # ablation bars: the wrong variant


def _worst(entries: list[dict[str, Any]]) -> float:
    values = [base.max_abs(e) for e in entries]
    if any(v != v for v in values):
        return math.nan
    return max(values)


def property_rows(report: dict) -> list[tuple[str, float]]:
    checks = report["checks"]
    run = checks["run_to_run"]
    pinned = checks["diffusers_parity"]["pinned_flags"]
    default = checks["diffusers_parity"]["default_flags"]
    bi = checks["batch_invariance"]["contract"]
    norm = checks["normalization"]

    def eq(flag: bool) -> float:
        return 0.0 if flag else math.nan

    return [
        ("run-to-run: reference encode", eq(run["encode_references"]["bitwise_equal"])),
        ("run-to-run: decode uint8 (I-C10)", eq(run["decode_uint8"]["bitwise_equal"])),
        (
            "vs Diffusers encode step, CPU generator",
            _worst(pinned["encode_references_cpu_generator"]),
        ),
        (
            "vs Diffusers encode step, CUDA generator",
            _worst(pinned["encode_references_cuda_generator"]),
        ),
        ("vs Diffusers decode step, uint8", base.max_abs(pinned["decode_uint8"])),
        (
            "vs Diffusers, default flags (all three)",
            _worst(
                [
                    *default["encode_references_cpu_generator"],
                    *default["encode_references_cuda_generator"],
                    default["decode_uint8"],
                ]
            ),
        ),
        ("batch row vs alone: encode (mode)", base.max_abs(bi["encode_mode"])),
        (
            "batch row vs alone: encode (shared generator)",
            eq(bi["encode_sample_shared_generator"]["bitwise_equal"]),
        ),
        ("batch row vs alone: decode uint8", base.max_abs(bi["decode_uint8"])),
        ("latent normalize CPU vs CUDA", base.max_abs(norm["normalize_cpu_vs_cuda"])),
        ("latent denormalize CPU vs CUDA", base.max_abs(norm["denormalize_cpu_vs_cuda"])),
        ("pixel denormalize CPU vs CUDA", base.max_abs(norm["denormalize_pixels_cpu_vs_cuda"])),
    ]


ABLATIONS = (
    ("I11_reference_fp32_sample_vs_fp16_round_trip", "I11 keep FP32 sample (latent)"),
    ("I15_scalar_normalization_vs_per_channel", "I15 scalar latent normalization"),
    ("I15_no_normalization_vs_per_channel", "I15 no latent normalization"),
    ("I16_untiled_vs_tiled_decode_uint8", "I16 untiled decode (uint8)"),
    ("I16_overlap96_vs_overlap64_decode_uint8", "I16 overlap 96 instead of 64 (uint8)"),
    ("I17_pm1_pixels_vs_imagenet_latent_mode", "I17 [-1, 1] pixels (latent)"),
    ("I18_round_vs_truncate_uint8", "I18 round instead of truncate (uint8)"),
)


def ablation_rows(report: dict) -> list[tuple[str, float]]:
    probes = report["checks"]["ablations"]
    return [(label, base.max_abs(probes[key])) for key, label in ABLATIONS]


def plot_accuracy(name: str, report: dict, out: Path) -> None:
    props, probes = property_rows(report), ablation_rows(report)
    rows = [(label, value, base.CONTRACT) for label, value in props]
    rows += [(label, value, BAD) for label, value in probes]
    fig, ax = plt.subplots(figsize=(11, 9.2))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.9, bottom=0.1)
    base._style(ax, "x")
    floor = base._floor([v for _, v, _ in rows])
    for i, (_, value, color) in enumerate(rows):
        base._log_bar(ax, len(rows) - 1 - i, value, color, 0.55, floor)
    ax.axhline(len(probes) - 0.5, color=base.MUTED, linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _, _ in reversed(rows)], fontsize=9.5)
    base._log_axis(ax, floor, [v for _, v, _ in rows])
    canvas = "x".join(str(v) for v in report["inputs"]["canvas"])
    base._title(
        ax,
        f"RL-Kernel image VAE contract at {canvas}: properties and probes — "
        f"{base.gpu_name(report)}",
    )
    base._legend(
        ax,
        [
            ("contract property (expect = 0)", base.CONTRACT),
            ("ablation probe, wrong variant vs contract (expect > 0)", BAD),
        ],
    )
    pre = report["checks"]["preprocessing"]
    fig.text(
        0.01,
        0.015,
        f"Pinned flags. Release pixel normalization differs between CPU and CUDA for "
        f"{pre['release_cpu_vs_cuda_mismatches']}/768 table entries; the contract table equals "
        f"the CUDA release on both devices: {pre['lut_cpu_equals_cuda_release']}.\n"
        f"Released weights, commit {base.commit(report)}.",
        fontsize=8.5,
        color=base.MUTED,
    )
    base._save(fig, out)
    base.write_table(
        out.with_suffix(".md"),
        ["Row", "Kind", "max \\|Δ\\|"],
        [[label, "property", base.fmt_delta(v)] for label, v in props]
        + [[label, "ablation probe", base.fmt_delta(v)] for label, v in probes],
        base.source_line([(name, report)], "checks"),
    )


OPS = (("encode_reference", "reference encode"), ("decode", "decode"))


def plot_latency(name: str, report: dict, out: Path) -> None:
    timing = report["timing"]
    series = [
        ("contract", "RL-Kernel contract (tiled)", base.CONTRACT),
        ("diffusers_default_flags", "Diffusers blocks (tiled, default flags)", base.DIFFUSERS),
        ("diffusers_pinned_flags", "Diffusers blocks (tiled, pinned flags)", base.PINNED),
        ("contract_untiled", "contract, tiling off (not the release)", base.UNTILED),
    ]

    def median(key: str, op: str) -> float | None:
        entry = timing.get(key, {}).get(op)
        return None if entry is None else float(entry["median_ms"])

    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    base._style(ax, "y")
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
                base.fmt_ms(y),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=base.INK,
            )
    ax.set_xticks(range(len(OPS)))
    ax.set_xticklabels([label for _, label in OPS])
    ax.set_ylabel("median latency, batch 1 (ms)", color=base.INK)
    ref = "x".join(str(v) for v in report["inputs"]["references"][0][-2:])
    canvas = "x".join(str(v) for v in report["inputs"]["canvas"])
    repeats = timing["contract"]["decode"]["repeats"]
    ax.set_title(
        f"Image VAE latency: {ref} reference, {canvas} decode, median of {repeats} — "
        f"{base.gpu_name(report)} (commit {base.commit(report)})",
        fontsize=10.5,
        color=base.INK,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9)
    base._save(fig, out)
    base.write_table(
        out.with_suffix(".md"),
        ["Step (median ms)", *[label for _, label, _ in series]],
        [[label, *[base.fmt_ms(median(key, op)) for key, _, _ in series]] for op, label in OPS],
        base.source_line([(name, report)], "timing"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    out_dir = args.out_dir or args.report.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = base.gpu_tag(report)
    plot_accuracy(args.report.name, report, out_dir / f"h3-image-vae-accuracy-{tag}.png")
    plot_latency(args.report.name, report, out_dir / f"h3-image-vae-latency-{tag}.png")


if __name__ == "__main__":
    main()
