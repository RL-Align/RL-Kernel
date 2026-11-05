#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render retained measurements without changing the evidence bundle."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def _load_cases(bundle):
    """Keep every accuracy case and prefer the full report for duplicate timings."""
    cases = []
    random_timing = {}
    for name in ("full.json", "tasks.json"):
        if name in bundle["artifacts"]:
            for case in bundle["artifacts"][name]["content"]["cases"]:
                cases.append(case)
                if case["input_family"] == "random":
                    timings = random_timing.setdefault(case["semantic_op"], {})
                    timings.setdefault(case["rows"], case)
    return cases, random_timing


def _accuracy_series(cases, tensor):
    """Collect measured maxima and retain rows with unavailable accuracy checks."""
    by_size = {}
    unavailable = set()
    for case in cases:
        size = case["rows"]
        ratios = [
            check.get("max_error_over_tolerance")
            for check in case["checks"]
            if check["tensor"] == tensor and check["transform"] == "fp32_reference"
        ]
        measured = [ratio for ratio in ratios if ratio is not None and math.isfinite(ratio)]
        by_size.setdefault(size, []).extend(measured)
        if not ratios or len(measured) != len(ratios):
            unavailable.add(size)
    sizes = sorted(by_size)
    return sizes, [max(by_size[size], default=math.nan) for size in sizes], unavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.evidence.read_bytes()
    bundle = json.loads(raw)
    all_cases, random_timing = _load_cases(bundle)
    required_ops = {"h3_ffn_gate_up_gemm", "h3_attention_o_gemm"}
    missing_ops = required_ops - random_timing.keys()
    if missing_ops:
        raise ValueError(f"evidence has no random cases for: {', '.join(sorted(missing_ops))}")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "h3-projections"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, (kind, title) in enumerate(
        [
            ("h3_ffn_gate_up_gemm", "FFN gate/up"),
            ("h3_attention_o_gemm", "Attention output"),
        ]
    ):
        cases = [c for c in all_cases if c["semantic_op"] == kind]
        unavailable = set()
        for tensor in ("y", "dx", "dw"):
            sizes, values, missing = _accuracy_series(cases, tensor)
            unavailable.update(missing)
            axes[row, 0].plot(sizes, values, marker=".", label=tensor)
        if unavailable:
            axes[row, 0].plot(
                sorted(unavailable),
                [0.04] * len(unavailable),
                "kx",
                transform=axes[row, 0].get_xaxis_transform(),
                label="Unavailable accuracy ratio",
            )
        axes[row, 0].axhline(1, color="black", linestyle="--", label="Failure threshold")
        axes[row, 0].set(
            title=f"{title}: worst accuracy ratio",
            ylabel="Error / tolerance",
        )
        axes[row, 0].set_ylim(bottom=0)
        timings = random_timing[kind]
        random_cases = [timings[size] for size in sorted(timings)]
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
            + "; all accuracy cases retained; full report preferred for duplicate timings.",
        },
    )
    if args.output.suffix.lower() == ".svg":
        args.output.write_text(
            "\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n"
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
