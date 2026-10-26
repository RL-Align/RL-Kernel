#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Produce a readable report and static charts from measured H3 evidence only."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = report["cases"]
    # Render earlier native-midtree reports alongside the versioned FP32 candidate.
    for case in cases:
        case["benchmarks"] = {
            key.replace("native_", "candidate_", 1): value
            for key, value in case["benchmarks"].items()
        }
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
        "| Input | M | Checks passed | Total | Candidate fwd ms | torch BF16 fwd ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        bench = case["benchmarks"]
        lines.append(
            f"| {case['input_family']} | {case['rows']} | "
            f"{sum(row['passed'] for row in case['checks'])} | {len(case['checks'])} | "
            f"{bench['candidate_forward']['median_ms']:.4f} | "
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
            ("candidate_forward", "Candidate forward"),
            ("torch_bf16_forward", "torch BF16 forward"),
            ("candidate_forward_backward", "Candidate forward+backward"),
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
        for tensor in ("y", "dx", "dw"):
            values = [
                max(
                    row.get("max_error_over_tolerance", 0)
                    for row in case["checks"]
                    if row["tensor"] == tensor and row["transform"] == "fp32_reference"
                )
                for case in subset
            ]
            axes[1].plot(counts, values, marker="o", label=tensor)
        axes[1].axhline(1, color="red", linestyle="--", label="shared tolerance boundary")
        axes[1].set(
            xlabel="Logical rows M",
            ylabel="Max error / allowed error",
            xscale="log",
            title="Accuracy against independent FP32 reference",
        )
        axes[1].legend(fontsize=8)
        fig.savefig(output / f"{family}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
