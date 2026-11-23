#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render retained H3 projection reports and historical evidence bundles."""
import argparse
import hashlib
import json
import math
from pathlib import Path

_TIMING_PHASES = ("forward", "forward_backward")
_TIMING_BACKENDS = (("candidate", "Candidate"), ("torch_bf16", "PyTorch BF16"))
_TIMING_SERIES = {f"{prefix}_{phase}" for prefix, _ in _TIMING_BACKENDS for phase in _TIMING_PHASES}


def _load_reports(evidence):
    """Accept a validator report or a historical full/tasks evidence bundle."""
    if not isinstance(evidence, dict):
        raise ValueError("expected a validator report or a full/tasks evidence bundle")
    if "cases" in evidence:
        reports = [evidence]
    else:
        artifacts = evidence.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("expected a validator report or a full/tasks evidence bundle")
        reports = []
        for name in ("full.json", "tasks.json"):
            if name in artifacts:
                artifact = artifacts[name]
                if not isinstance(artifact, dict) or not isinstance(artifact.get("content"), dict):
                    raise ValueError(f"{name} artifact must contain a report object")
                reports.append(artifact["content"])
        if not reports:
            raise ValueError("bundle contains no full.json or tasks.json reports")
    for report in reports:
        cases = report.get("cases")
        if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
            raise ValueError("report cases must be an array of objects")
    return reports


def _recorded_provenance(report, *fields):
    """Return comparable source, software, hardware and report-specific fields."""
    try:
        provenance = report["provenance"]
        if provenance["working_tree_clean"] is not True or not provenance["source_hashes"]:
            return None
        source = tuple(
            provenance[field]
            for field in (
                "source_revision",
                "source_hashes",
                "python",
                "torch",
                "cuda",
                "seed",
                *fields,
            )
        )
        device = tuple(
            report["hardware"][field]
            for field in ("gpu_model", "architecture", "device_count_used", "memory_bytes")
        )
    except (KeyError, TypeError):
        return None
    if any(value is None or value == "" for value in source + device):
        return None
    return source, device


def _timing_provenance(report, timed_cases):
    """Return the recorded timing fields, including the executed Triton version."""
    recorded = _recorded_provenance(report, "repeats")
    try:
        versions = {
            trace["forward_launch"]["triton_version"]
            for case in timed_cases
            for trace in case["execution_traces"]
        }
    except (KeyError, TypeError):
        return None
    if (
        recorded is None
        or not all(case["execution_traces"] for case in timed_cases)
        or len(versions) != 1
        or not all(versions)
    ):
        return None
    return recorded, versions


def _accuracy_notes(reports):
    """Identify missing or mixed provenance and tolerances across accuracy reports."""
    sources = [report for report in reports if report["cases"]]
    if len(sources) < 2:
        return []
    signatures = [
        _recorded_provenance(report, "checkpoint_revision", "contract_sha256", "method")
        for report in sources
    ]
    known = [signature for signature in signatures if signature is not None]
    incomplete = len(known) != len(signatures)
    tolerances = {}
    for report in sources:
        for case in report["cases"]:
            for check in case["checks"]:
                if check["transform"] == "fp32_reference":
                    tolerance = check.get("tolerance")
                    if not isinstance(tolerance, dict) or not tolerance:
                        incomplete = True
                    else:
                        tolerances.setdefault(check["tensor"], set()).add(
                            json.dumps(tolerance, sort_keys=True)
                        )
    notes = []
    if incomplete:
        notes.append("Accuracy provenance or tolerance records are incomplete.")
    if any(signature != known[0] for signature in known[1:]) or any(
        len(values) > 1 for values in tolerances.values()
    ):
        notes.append("Accuracy provenance or tolerances differ across the plotted cases.")
    return notes


def _load_cases(reports):
    """Keep all accuracy cases and prefer the first report for duplicate timings."""
    cases = []
    random_timing = {}
    contributors = []
    for report in reports:
        selected = []
        for case in report["cases"]:
            cases.append(case)
            if case["input_family"] == "random" and _TIMING_SERIES.intersection(
                case.get("benchmarks", {})
            ):
                timings = random_timing.setdefault(case["semantic_op"], {})
                if case["rows"] not in timings:
                    timings[case["rows"]] = case
                    selected.append(case)
        if selected:
            contributors.append((report, selected))
    signatures = [_timing_provenance(report, selected) for report, selected in contributors]
    timing_unverified = any(signature is None for signature in signatures)
    if len(signatures) > 1:
        if timing_unverified:
            raise ValueError(
                "combined timings lack complete provenance. Render reports separately."
            )
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(
                "combined timings have different provenance. Render reports separately."
            )
    return cases, random_timing, timing_unverified


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
        try:
            raw = path.read_bytes()
            reports.extend(_load_reports(json.loads(raw)))
        except (OSError, ValueError) as error:
            parser.error(f"{path.name}: {error}")
        sources.append(f"{path.name} (SHA256 {hashlib.sha256(raw).hexdigest()})")
    try:
        all_cases, random_timing, timing_unverified = _load_cases(reports)
    except ValueError as error:
        parser.error(str(error))
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
            (unavailable, 0.04, "kx", "Some cases lack an accuracy ratio"),
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
        for col, phase in enumerate(_TIMING_PHASES, 1):
            axis = axes[row, col]
            for prefix, label in _TIMING_BACKENDS:
                series = f"{prefix}_{phase}"
                samples = [
                    (case["rows"], case["benchmarks"][series]["median_ms"])
                    for case in random_cases
                    if series in case.get("benchmarks", {})
                ]
                if samples:
                    rows, values = zip(*samples, strict=True)
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
    notes = _accuracy_notes(reports)
    if timing_unverified:
        notes.append("Timing provenance is incomplete. These timings are unverified.")
    if any(report.get("status") in {"failed", "error", "running"} for report in reports):
        notes.append("Some reports are failed or incomplete.")
    fig.suptitle(
        "\n".join(
            (
                "H3 projection measurements",
                "Accuracy ratios use each report's tolerance. Timings use random inputs.",
                "Candidate includes validation and copies. PyTorch BF16 is a timing comparator.",
                *notes,
            )
        ),
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        metadata={
            "Date": None,
            "Description": "Input evidence: "
            + ", ".join(sources)
            + ". Accuracy ratios use each report's tolerance."
            + " First report preferred for duplicate timings."
            + "".join(f" {note}" for note in notes),
        },
    )
    if args.output.suffix.lower() == ".svg":
        args.output.write_text(
            "\n".join(line.rstrip() for line in args.output.read_text().splitlines()) + "\n"
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
