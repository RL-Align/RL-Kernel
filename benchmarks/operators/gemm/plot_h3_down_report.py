#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Produce a readable report and static charts from measured H3 evidence only."""

import argparse
import csv
import json
import math
from pathlib import Path


def benchmark_prefix(benchmarks):
    """Identify a single timing series while preserving the report's original keys."""
    native = any(key.startswith("native_") for key in benchmarks)
    candidate = any(key.startswith("candidate_") for key in benchmarks)
    if native and candidate:
        raise ValueError("report case mixes native and candidate benchmark results")
    prefix = "native" if native else "candidate"
    required = (
        f"{prefix}_forward",
        "torch_bf16_forward",
        f"{prefix}_forward_backward",
        "torch_bf16_forward_backward",
    )
    missing = [key for key in required if key not in benchmarks]
    if missing:
        raise ValueError(f"report case lacks benchmark results: {', '.join(missing)}")
    return prefix


def plot_reference_accuracy(axis, cases):
    """Keep failed or unavailable accuracy checks visible without inventing errors."""
    counts = [case["rows"] for case in cases]
    for tensor in ("y", "dx", "dw"):
        values, flagged = [], []
        for case in cases:
            checks = [
                row
                for row in case["checks"]
                if row["tensor"] == tensor and row["transform"] == "fp32_reference"
            ]
            ratios = [row.get("max_error_over_tolerance") for row in checks]
            available = bool(ratios) and all(
                value is not None and math.isfinite(value) for value in ratios
            )
            values.append(max(ratios) if available else math.nan)
            if not available or not all(row["passed"] for row in checks):
                flagged.append(case["rows"])
        (line,) = axis.plot(counts, values, marker="o", label=tensor)
        if flagged:
            # These markers use axes height, not the numerical error scale.
            axis.scatter(
                flagged,
                [1.02] * len(flagged),
                transform=axis.get_xaxis_transform(),
                clip_on=False,
                marker="x",
                color=line.get_color(),
                label=f"{tensor}: failed / unavailable",
            )
    axis.axhline(1, color="red", linestyle="--", label="shared tolerance boundary")
    axis.set(
        xlabel="Logical rows M",
        ylabel="Max error / allowed error",
        xscale="log",
        title="Accuracy against independent FP32 reference",
    )
    axis.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = report["cases"]
    prefixes = {benchmark_prefix(case["benchmarks"]) for case in cases}
    if len(prefixes) > 1:
        raise ValueError("report mixes native and candidate benchmark cases")
    prefix = next(iter(prefixes), "candidate")
    backend = report.get("backend")
    # New validator reports use candidate keys even for the explicit native comparator.
    if prefix == "native" or backend == "rlkernel.h3_ffn_down.sm90.midtree.v1":
        backend_label = "Legacy native comparator"
    else:
        backend_label = "Candidate"
    if backend:
        backend_label += f" ({backend})"
    forward_key = f"{prefix}_forward"
    lines = [
        "# H3 down projection GPU evidence",
        "",
        f"Run status: **{report['status']}**.",
        "",
        "Operator-only evidence; production promotion requires review of the matrix, "
        "real checkpoint fixtures and the arithmetic contract.",
        "",
    ]
    if report.get("error"):
        lines += ["```text", report["error"], "```", ""]
    if report.get("fixture"):
        metadata = report["fixture"]["metadata"]
        lines += [f"Checkpoint fixture activation kind: **{metadata['activation_kind']}**.", ""]
    lines += [
        f"| Input | M | Checks passed | Total | {backend_label} fwd ms | torch BF16 fwd ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        bench = case["benchmarks"]
        lines.append(
            f"| {case['input_family']} | {case['rows']} | "
            f"{sum(row['passed'] for row in case['checks'])} | {len(case['checks'])} | "
            f"{bench[forward_key]['median_ms']:.4f} | "
            f"{bench['torch_bf16_forward']['median_ms']:.4f} |"
        )
    lines += ["", "Remaining acceptance:", ""]
    lines += [f"- {value}" for value in report.get("remaining_acceptance", [])]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    with (output / "checks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "input",
                "M",
                "judgment",
                "tensor",
                "transform",
                "passed",
                "max_abs_error",
                "max_error_over_tolerance",
                "mismatched_bytes",
            ]
        )
        for case in cases:
            for row in case["checks"]:
                writer.writerow(
                    [
                        case["input_family"],
                        case["rows"],
                        row["judgment"],
                        row["tensor"],
                        row["transform"],
                        row["passed"],
                        row.get("max_abs_error"),
                        row.get("max_error_over_tolerance"),
                        row.get("mismatched_bytes"),
                    ]
                )
    if not cases:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for family in sorted({case["input_family"] for case in cases}):
        subset = sorted(
            (case for case in cases if case["input_family"] == family),
            key=lambda case: case["rows"],
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        counts = [case["rows"] for case in subset]
        for key, label in (
            (forward_key, f"{backend_label} forward"),
            ("torch_bf16_forward", "torch BF16 forward"),
            (f"{prefix}_forward_backward", f"{backend_label} forward+backward"),
            ("torch_bf16_forward_backward", "torch BF16 forward+backward"),
        ):
            axes[0].plot(
                counts,
                [case["benchmarks"][key]["median_ms"] for case in subset],
                marker="o",
                label=label,
            )
        axes[0].set(
            xlabel="Logical rows M",
            ylabel="Synchronized wall time (ms)",
            xscale="log",
            yscale="log",
            title=family,
        )
        axes[0].legend(fontsize=8)
        plot_reference_accuracy(axes[1], subset)
        fig.savefig(output / f"{family}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
