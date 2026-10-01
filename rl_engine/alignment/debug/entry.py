# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


from .evidence import analyze_bundle, diagnosis


def analyze(directory: Path, *, output: Path | None = None) -> dict[str, Any]:
    """Analyze saved evidence without executing a model."""
    directory = directory.resolve()
    matrix = directory / "matrix-report.json"
    if matrix.is_file():
        from .session import apply_evidence_gates, finish_summary
        from .ux import compact_summary

        report = json.loads(matrix.read_text(encoding="utf-8"))
        for group in report["groups"]:
            group["report"] = analyze(
                Path(group["run_dir"]), output=directory / f"{group['group']}.report.json"
            )
            baseline = report["groups"][0]["report"] if group is not report["groups"][0] else None
            apply_evidence_gates(group, baseline)
            save_report(directory / f"{group['group']}.report.json", group["report"])
        finish_summary(directory, report)
        compact_summary(directory, report)
        return report
    if (directory / "capture-manifest.json").is_file():
        report = analyze_bundle(directory)
        save_report(output or directory / "diagnostic-report.json", report)
        return report
    paths = list(directory.rglob("diagnostics/layers/*-layer*-call*.pt"))
    if directory.name == "diagnostics":
        paths = list((directory / "layers").glob("*-layer*-call*.pt"))
    if paths:
        from .qwen3_report import analyze_replay

        report = analyze_replay(directory, paths, output=output)
        report["diagnosis"] = diagnosis(report)
        save_report(output or directory / "diagnostic-report.json", report)
        return report
    report = {
        "status": "not_comparable",
        "first_divergence": None,
        "errors": ["No paired snapshots. Run rlk debug on the saved rollout to capture them."],
    }
    report["diagnosis"] = diagnosis(report)
    save_report(output or directory / "diagnostic-report.json", report)
    return report


def report_lines(report: dict[str, Any], label: str = "debug") -> list[str]:
    lines = [f"[{label}] {report['status']}"]
    if "groups" in report:
        lines.append(f"[result] {report['conclusion']}")
        for group in report["groups"]:
            lines.extend(report_lines(group["report"], group["group"]))
        return lines
    first = report.get("first_divergence")
    if first:
        label = "candidate boundary" if report["status"] == "not_comparable" else "first observed"
        lines.append(
            f"[{label}] module={first.get('module', 'unknown')} "
            f"layer={first.get('layer')} stage={first.get('stage')} "
            f"token={first.get('position')}"
        )
        lines.append(f"[input evidence] {first.get('input_evidence', 'unknown')}")
    decision = report.get("diagnosis", diagnosis(report))
    lines.append(f"[diagnosis] {decision['level']}: {decision['reason']}")
    for check in decision.get("next_checks", []):
        lines.append(f"  next: {check}")
    if report.get("execution", {}).get("log"):
        lines.append(f"[job log] {report['execution']['log']}")
    replacement = report.get("replacement")
    if isinstance(replacement, dict):
        lines.append(f"[replacement] {replacement['status']}: {replacement['reason']}")
    lines.append(f"[scope] {report.get('scope', 'captured evidence only')}")
    return lines


def print_report(report: dict[str, Any], label: str = "debug") -> None:
    print("\n".join(report_lines(report, label)), flush=True)


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    path.with_suffix(".txt").write_text("\n".join(report_lines(report)) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Locate the first observed train/rollout divergence."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = analyze(args.run_dir, output=args.output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"debug: {exc}\n")
    print_report(report)
    filename = "matrix-report.md" if "groups" in report else "diagnostic-report.json"
    print(f"[report] {args.output or args.run_dir / filename}")
    return {"equal": 0, "diverged": 1, "not_comparable": 2}[report["status"]]
