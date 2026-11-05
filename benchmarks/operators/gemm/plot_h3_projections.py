#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render retained H3 projection reports and historical evidence bundles."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def _load_reports(evidence):
    """Accept a validator report or a historical full/tasks evidence bundle."""
    if "cases" in evidence:
        return [evidence]
    artifacts = evidence.get("artifacts", {})
    return [artifacts[name]["content"] for name in ("full.json", "tasks.json") if name in artifacts]


def _load_cases(reports):
    """Keep all accuracy cases and prefer the first report for duplicate timings."""
    cases = []
    random_timing = {}
    for report in reports:
        for case in report["cases"]:
            cases.append(case)
            if case["input_family"] == "random":
                timings = random_timing.setdefault(case["semantic_op"], {})
                timings.setdefault(case["rows"], case)
    return cases, random_timing


def _accuracy_series(cases, tensor):
    """Collect measured maxima and distinguish failed and unavailable ratios."""
    by_size = {}
    failed = set()
    unavailable = set()
    for case in cases:
        size = case["rows"]
        measured = by_size.setdefault(size, [])
        checks = [
            check
            for check in case["checks"]
            if check["tensor"] == tensor and check["transform"] == "fp32_reference"
        ]
        if not checks:
            unavailable.add(size)
        for check in checks:
            ratio = check.get("max_error_over_tolerance")
            if ratio is not None and math.isfinite(ratio):
                measured.append(ratio)
            elif check.get("passed") is False:
                failed.add(size)
            else:
                unavailable.add(size)
    sizes = sorted(by_size)
    return sizes, [max(by_size[size], default=math.nan) for size in sizes], failed, unavailable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        required=True,
        help="Report or bundle to plot. Repeat for multiple reports, with full.json first.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    sources = []
    for path in args.evidence:
        raw = path.read_bytes()
        loaded = _load_reports(json.loads(raw))
        if not loaded:
            parser.error(
                f"{path.name}: expected a validator report or a full/tasks evidence bundle"
            )
        reports.extend(loaded)
        sources.append(f"{path.name} (SHA256 {hashlib.sha256(raw).hexdigest()})")
    all_cases, random_timing = _load_cases(reports)
    if not all_cases:
        parser.error("evidence has no completed cases")

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
        failed = set()
        unavailable = set()
        accuracy = axes[row, 0]
        if not cases:
            accuracy.text(0.5, 0.5, "No accuracy cases", ha="center", transform=accuracy.transAxes)
        for tensor in ("y", "dx", "dw"):
            sizes, values, failed_sizes, missing_sizes = _accuracy_series(cases, tensor)
            failed.update(failed_sizes)
            unavailable.update(missing_sizes)
            if sizes:
                accuracy.plot(sizes, values, marker=".", label=tensor)
        for sizes, height, marker, label in (
            (failed, 0.12, "rv", "Failed accuracy check without a finite ratio"),
            (unavailable, 0.04, "kx", "Unavailable accuracy ratio"),
        ):
            if sizes:
                accuracy.plot(
                    sorted(sizes),
                    [height] * len(sizes),
                    marker,
                    transform=accuracy.get_xaxis_transform(),
                    label=label,
                )
        accuracy.axhline(1, color="black", linestyle="--", label="Failure threshold")
        accuracy.set(title=f"{title}: worst accuracy ratio", ylabel="Error / tolerance")
        accuracy.set_ylim(bottom=0)
        timings = random_timing.get(kind, {})
        random_cases = [timings[size] for size in sorted(timings)]
        for col, phase in enumerate(("forward", "forward_backward"), 1):
            axis = axes[row, col]
            for prefix, label in (
                ("candidate", "Candidate"),
                ("torch_bf16", "PyTorch BF16"),
            ):
                series = f"{prefix}_{phase}"
                samples = [
                    (case["rows"], case["benchmarks"][series]["median_ms"])
                    for case in random_cases
                    if series in case.get("benchmarks", {})
                ]
                if samples:
                    rows, values = zip(*samples)
                    axis.plot(rows, values, marker=".", label=label)
            if not axis.has_data():
                axis.text(0.5, 0.5, "No random timings", ha="center", transform=axis.transAxes)
            axis.set(
                title=f"{title}: {phase.replace('_', ' + ')}",
                ylabel="Median wall time (ms)",
                yscale="log",
            )
        for ax in axes[row]:
            ax.set(xscale="log", xlabel="Rows (log scale)")
            ax.grid(alpha=0.2)
            if ax.has_data():
                ax.legend(fontsize=8)
    title = (
        "H3 projection measurements\n"
        "Accuracy covers the supplied families and fixtures. Timings use random inputs.\n"
        "Candidate timings include validation and copies. PyTorch BF16 is a timing comparator."
    )
    if any(report.get("status") in {"failed", "error", "running"} for report in reports):
        title += "\nSome reports are failed or incomplete."
    fig.suptitle(title, fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        metadata={
            "Date": None,
            "Description": "Input evidence: "
            + ", ".join(sources)
            + ". All accuracy cases retained. First report preferred for duplicate timings.",
        },
    )
    if args.output.suffix.lower() == ".svg":
        args.output.write_text(
            "\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n"
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
