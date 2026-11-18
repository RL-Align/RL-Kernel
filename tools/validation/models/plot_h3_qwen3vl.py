# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Charts and matching tables for ``h3_qwen3vl_evidence.py`` output.

Every chart is drawn from a :class:`Figure` built from the evidence JSON, and ``--tables``
prints the same :class:`Figure` as a Markdown table (same series, labels, units and rounding),
so a chart and its table can never disagree.

Usage::

    python tools/validation/models/plot_h3_qwen3vl.py evidence.json --out-dir charts/ --tables
    python tools/validation/models/plot_h3_qwen3vl.py b200.json --compare h100.json \
        --label B200 --compare-label H100 --out-dir charts/ --tables
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
FLOOR = 1e-4  # where an exact zero (bitwise equal) is drawn on a log axis
ZERO = "= 0, bitwise"


@dataclass
class Figure:
    """One chart: categories (rows) x series, values in one unit."""

    title: str
    xlabel: str
    unit: str
    log: bool
    categories: list[str]
    series: list[str]
    values: dict[tuple[str, str], float | None]  # None: not measured / unsupported
    fmt: str = ".3g"
    notes: dict[tuple[str, str], str] = field(default_factory=dict)

    def cell(self, category: str, series: str) -> str:
        value = self.values.get((category, series))
        if value is None:
            return self.notes.get((category, series), "n/a")
        if value == 0.0 and self.log:
            return ZERO
        return format(value, self.fmt)


def _case_label(evidence: dict[str, Any], name: str) -> str:
    return f"{name} ({evidence['reference'][name]['num_tokens']} tok)"


def batch_invariance(evidence: dict[str, Any], label: str) -> Figure:
    batching = evidence["prior_art_batching"]
    names = list(batching["hf_padded_right_all6"])
    series = [
        "RL-Kernel contract (per row, 50 layers)",
        "HF padded batch, right pad",
        "HF padded batch, left pad",
    ]
    values: dict[tuple[str, str], float | None] = {}
    notes = {}
    for name in names:
        cat = _case_label(evidence, name)
        values[(cat, series[0])] = 0.0 if evidence["contract"][name]["equals_diffusers"] else None
        if values[(cat, series[0])] is None:
            notes[(cat, series[0])] = "differs (see JSON)"
        values[(cat, series[1])] = batching["hf_padded_right_all6"][name]["max_abs"]
        values[(cat, series[2])] = batching["hf_padded_left_all6"][name]["max_abs"]
    return Figure(
        title=f"Layer-50 feature vs Diffusers, alone vs in a 6-prompt batch ({label})",
        xlabel="max |feature - Diffusers hidden_states[50]|, BF16 (log scale)",
        unit="max abs diff",
        log=True,
        categories=[_case_label(evidence, n) for n in names],
        series=series,
        values=values,
        notes=notes,
    )


def sdpa_backends(evidence: dict[str, Any], label: str) -> Figure:
    backends = evidence["sdpa_backends_vs_default"]
    names = list(next(iter(backends.values())))
    values: dict[tuple[str, str], float | None] = {}
    notes = {}
    for backend, entry in backends.items():
        for name in names:
            cat = _case_label(evidence, name)
            result = entry[name]
            if "error" in result:
                values[(cat, backend)] = None
                notes[(cat, backend)] = "unsupported"
            else:
                values[(cat, backend)] = 0.0 if result["bitwise_equal"] else result["max_abs"]
    return Figure(
        title=f"Forced SDPA backend vs the default selection, Diffusers call ({label})",
        xlabel="max |forced backend - default|, BF16 (log scale)",
        unit="max abs diff",
        log=True,
        categories=[_case_label(evidence, n) for n in names],
        series=list(backends),
        values=values,
        notes=notes,
    )


def other_invariance(evidence: dict[str, Any], label: str) -> Figure:
    series = ["Diffusers call / vision tower, alone vs variant"]
    values: dict[tuple[str, str], float | None] = {}
    categories = []
    for key, result in evidence["prefix_invariance"].items():
        cat = f"prefix rows {key} vs full prompt"
        categories.append(cat)
        values[(cat, series[0])] = 0.0 if result["bitwise_equal"] else result["max_abs"]
    for key, result in evidence["vision_tower_batch_composition"].items():
        cat = f"vision tower {key}"
        categories.append(cat)
        values[(cat, series[0])] = 0.0 if result["bitwise_equal"] else result["max_abs"]
    return Figure(
        title=f"Prefix rows and vision-tower batching vs the reference ({label})",
        xlabel="max abs diff, BF16 (log scale)",
        unit="max abs diff",
        log=True,
        categories=categories,
        series=series,
        values=values,
    )


def latency(evidence: dict[str, Any], label: str) -> Figure:
    lat = evidence["latency"]
    names = sorted(lat, key=lambda n: lat[n]["num_tokens"])
    series = ["Diffusers get_qwen3vl_prompt_embeds (64 layers)", "RL-Kernel contract (50 layers)"]
    values: dict[tuple[str, str], float | None] = {}
    for name in names:
        cat = f"{name} ({lat[name]['num_tokens']} tok)"
        values[(cat, series[0])] = lat[name]["diffusers_median_ms"]
        values[(cat, series[1])] = lat[name]["contract_median_ms"]
    return Figure(
        title=f"Text-encoder median latency per presentation ({label})",
        xlabel="median latency, ms (lower is better)",
        unit="ms",
        log=False,
        categories=[f"{n} ({lat[n]['num_tokens']} tok)" for n in names],
        series=series,
        values=values,
        fmt=".1f",
    )


def compare_latency(a: dict[str, Any], b: dict[str, Any], label_a: str, label_b: str) -> Figure:
    names = sorted(a["latency"], key=lambda n: a["latency"][n]["num_tokens"])
    series = [
        f"contract, {label_a}",
        f"contract, {label_b}",
        f"Diffusers, {label_a}",
        f"Diffusers, {label_b}",
    ]
    values: dict[tuple[str, str], float | None] = {}
    for name in names:
        cat = f"{name} ({a['latency'][name]['num_tokens']} tok)"
        for evidence, gpu in ((a, label_a), (b, label_b)):
            row = evidence["latency"].get(name)
            values[(cat, f"contract, {gpu}")] = row["contract_median_ms"] if row else None
            values[(cat, f"Diffusers, {gpu}")] = row["diffusers_median_ms"] if row else None
    return Figure(
        title=f"Text-encoder median latency, {label_a} vs {label_b}",
        xlabel="median latency, ms (lower is better)",
        unit="ms",
        log=False,
        categories=[f"{n} ({a['latency'][n]['num_tokens']} tok)" for n in names],
        series=series,
        values=values,
        fmt=".1f",
    )


def draw(fig_spec: Figure, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cat, n_ser = len(fig_spec.categories), len(fig_spec.series)
    fig, ax = plt.subplots(figsize=(9.5, 1.1 + 0.42 * n_cat * max(n_ser, 2) / 2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    height = 0.8 / n_ser
    top = max([v for v in fig_spec.values.values() if v] or [1.0])
    for m, series in enumerate(fig_spec.series):
        ys = [i + (m - (n_ser - 1) / 2) * height for i in range(n_cat)]
        xs = []
        for cat in fig_spec.categories:
            value = fig_spec.values.get((cat, series))
            xs.append(FLOOR if (fig_spec.log and not value) else (value or 0.0))
        ax.barh(ys, xs, height=height * 0.85, color=SERIES[m % len(SERIES)], label=series)
        for y, cat in zip(ys, fig_spec.categories, strict=True):
            value = fig_spec.values.get((cat, series))
            if fig_spec.log and value == 0.0:
                ax.plot([FLOOR * 1.12], [y], "o", ms=6, color=SERIES[m % len(SERIES)])
            if value is None or (fig_spec.log and value == 0.0):
                text = fig_spec.cell(cat, series)
                ax.text(
                    FLOOR * 1.35 if fig_spec.log else 0,
                    y,
                    f" {text}",
                    va="center",
                    fontsize=7.5,
                    color=INK_2,
                )
            elif not fig_spec.log:
                ax.text(
                    value,
                    y,
                    f" {fig_spec.cell(cat, series)}",
                    va="center",
                    fontsize=7.5,
                    color=INK_2,
                )
    if fig_spec.log:
        ax.set_xscale("log")
        ax.set_xlim(FLOOR, max(top * 4, 1e-2))
    else:
        ax.set_xlim(0, top * 1.2)
    ax.set_yticks(range(n_cat))
    ax.set_yticklabels(fig_spec.categories, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel(fig_spec.xlabel, color=INK_2, fontsize=9)
    ax.set_title(fig_spec.title, color=INK, fontsize=11, loc="left")
    bottom = 0.0
    if n_ser > 1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
        bottom = (0.25 * ((n_ser + 1) // 2) + 0.15) / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)


def table(fig_spec: Figure) -> str:
    """The chart's numbers as a Markdown table (same labels, units and rounding)."""

    head = "| Case | " + " | ".join(f"{s} ({fig_spec.unit})" for s in fig_spec.series) + " |"
    rule = "|---|" + "---|" * len(fig_spec.series)
    rows = [
        "| " + cat + " | " + " | ".join(fig_spec.cell(cat, s) for s in fig_spec.series) + " |"
        for cat in fig_spec.categories
    ]
    return "\n".join([head, rule, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="B200")
    parser.add_argument("--compare", type=Path, help="second evidence JSON (another GPU)")
    parser.add_argument("--compare-label", default="H100")
    parser.add_argument("--tables", action="store_true", help="print each chart's table")
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.label.lower()
    figures = [
        (batch_invariance(evidence, args.label), f"batch-invariance-{tag}"),
        (sdpa_backends(evidence, args.label), f"sdpa-backends-{tag}"),
        (other_invariance(evidence, args.label), f"prefix-vision-{tag}"),
        (latency(evidence, args.label), f"latency-{tag}"),
    ]
    if args.compare is not None:
        other = json.loads(args.compare.read_text())
        ctag = args.compare_label.lower()
        figures.append((batch_invariance(other, args.compare_label), f"batch-invariance-{ctag}"))
        figures.append(
            (
                compare_latency(evidence, other, args.label, args.compare_label),
                f"latency-{tag}-vs-{ctag}",
            )
        )
    for spec, name in figures:
        path = args.out_dir / f"h3-qwen3vl-{name}.png"
        draw(spec, path)
        if args.tables:
            print(f"### {path.name}\n\n{table(spec)}\n")


if __name__ == "__main__":
    main()
