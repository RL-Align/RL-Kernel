#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render retained measurements without changing the evidence bundle."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "h3-projections"})
    raw = args.evidence.read_bytes()
    bundle = json.loads(raw)
    unique_cases = {}
    for name in ("full.json", "tasks.json"):
        if name in bundle["artifacts"]:
            for case in bundle["artifacts"][name]["content"]["cases"]:
                key = (case["semantic_op"], case["input_family"], case["rows"])
                unique_cases.setdefault(key, case)
    report = {"cases": list(unique_cases.values())}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, (kind, title) in enumerate(
        [
            ("h3_ffn_gate_up_gemm", "FFN gate/up"),
            ("h3_attention_o_gemm", "Attention output"),
        ]
    ):
        cases = [c for c in report["cases"] if c["semantic_op"] == kind]
        sizes = sorted({c["rows"] for c in cases if c["input_family"] != "checkpoint_fixture"})
        for tensor in ("y", "dx", "dw"):
            values = [
                max(
                    check["max_error_over_tolerance"]
                    for c in cases
                    if c["rows"] == size
                    for check in c["checks"]
                    if check["tensor"] == tensor and check["transform"] == "fp32_reference"
                )
                for size in sizes
            ]
            axes[row, 0].plot(sizes, values, marker=".", label=tensor)
        axes[row, 0].axhline(1, color="black", linestyle="--", label="Failure threshold")
        axes[row, 0].set(
            title=f"{title}: worst accuracy ratio",
            ylabel="Error / tolerance",
            ylim=(0, 1.05),
        )
        random_cases = sorted(
            [c for c in cases if c["input_family"] == "random"], key=lambda c: c["rows"]
        )
        for col, phase in enumerate(("forward", "forward_backward"), 1):
            for prefix, label in (
                ("candidate", "Candidate"),
                ("torch_bf16", "PyTorch BF16"),
            ):
                axes[row, col].plot(
                    [c["rows"] for c in random_cases],
                    [c["benchmarks"][f"{prefix}_{phase}"]["median_ms"] for c in random_cases],
                    marker=".",
                    label=label,
                )
            axes[row, col].set(
                title=f"{title}: {phase.replace('_', ' + ')}",
                ylabel="Median wall time (ms)",
                yscale="log",
            )
        for ax in axes[row]:
            ax.set(xscale="log", xlabel="Rows (log scale)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle(
        f"H100 / SM90 measurements at {bundle['tested_source_revision'][:7]}\n"
        "Accuracy: available families/fixtures; timing: random BF16, seed 420, "
        "2 warmups / 5 samples\n"
        "Candidate includes validation/copies; PyTorch baseline is not BI-qualified",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        metadata={
            "Date": None,
            "Description": "Derived from h100.json SHA256 "
            + hashlib.sha256(raw).hexdigest()
            + "; no measurements modified; full report preferred for duplicate rows.",
        },
    )
    if args.output.suffix.lower() == ".svg":
        args.output.write_text(
            "\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n"
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
