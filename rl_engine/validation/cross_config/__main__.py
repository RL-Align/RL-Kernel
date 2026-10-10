# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Command-line interface for cross-configuration experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from rl_engine.validation.cross_config.artifacts import ArtifactStore
from rl_engine.validation.cross_config.config import ExperimentConfig, load_config
from rl_engine.validation.cross_config.execution_plan import build_execution_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="validate and persist a plan without execution")
    _add_common_arguments(plan)

    run = commands.add_parser("run", help="execute a plan with an explicit runtime adapter")
    _add_common_arguments(run)
    run.add_argument(
        "--runtime",
        required=True,
        choices=("cpu-smoke",),
        help="Runtime adapter; only the temporary CPU smoke adapter ships in V1",
    )
    run.add_argument(
        "--allow-smoke-operators",
        action="store_true",
        help="Explicitly authorize temporary smoke-only operator backends",
    )
    run.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Per paired-scoring attempt deadline",
    )
    run.add_argument(
        "--no-resume",
        action="store_true",
        help="Create new attempts even when matching COMPLETE artifacts exist",
    )

    report = commands.add_parser(
        "report",
        help="render a sealed attempt as an offline drift report",
    )
    report.add_argument("attempt_dir", type=Path, help="Completed cross-config attempt directory")
    report.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output .rlk-drift bundle, .png/.jpg image, or .json trace",
    )
    report.add_argument("--title", default=None, help="Optional report title")
    report.add_argument(
        "--no-preview", action="store_true", help="Omit PNG from a .rlk-drift bundle"
    )
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", type=Path, help="Versioned experiment JSON")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Append-only artifact root (default: runs)",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "report":
            summary = _report(args)
        else:
            config = load_config(args.config)
            if args.command == "plan":
                summary = record_plan(config, args.output_root)
            else:
                summary = _run(config, args)
    except Exception as exc:
        summary = {
            "schema_version": "cross_config.cli_summary.v1",
            "status": "error",
            "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "error": str(exc),
        }
        print(json.dumps(summary, sort_keys=True))
        print(f"cross-configuration error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, sort_keys=True))
    if args.command == "plan":
        print(
            f"planned {summary['planned_case_count']} cases; no runtime was created",
            file=sys.stderr,
        )
        return 0
    if args.command == "report":
        print(
            f"drift report: {summary['status']} ({summary['output']})",
            file=sys.stderr,
        )
        return 0
    print(
        f"CPU smoke: {summary['status']} ({len(summary['cases'])} cases)",
        file=sys.stderr,
    )
    for case in summary["cases"]:
        print(
            f"  {case['case_id']}: {case['status']}; actual backends "
            f"rollout={case['rollout_backend']}, training={case['training_backend']}; "
            f"worst sample/token={case['worst_token_index']}; "
            f"mismatches={case['mismatch_count']}; resumed={case['resumed']}",
            file=sys.stderr,
        )
    return 0 if summary["status"] == "pass" else 1


def record_plan(config: ExperimentConfig, output_root: Path) -> dict[str, Any]:
    plan = build_execution_plan(config)
    store = ArtifactStore(output_root)
    experiment_dir = store.initialize_experiment(
        config.definition.experiment_id,
        experiment=plan.experiment,
        plan=plan.rows(),
    )
    return {
        "schema_version": "cross_config.cli_summary.v1",
        "status": "planned",
        "experiment_id": config.definition.experiment_id,
        "scenario_id": config.definition.scenario_id,
        "planned_case_count": len(plan.entries),
        "planning_issues": [issue.to_dict() for issue in plan.issues],
        "artifact_dir": str(experiment_dir),
    }


def _run(config: ExperimentConfig, args: argparse.Namespace) -> dict[str, Any]:
    if args.runtime != "cpu-smoke":  # pragma: no cover - argparse owns choices
        raise ValueError(f"unsupported runtime {args.runtime!r}")
    from rl_engine.validation.fixtures.cpu_cross_config import run_cpu_experiment

    return run_cpu_experiment(
        config,
        output_root=args.output_root,
        allow_smoke_operators=args.allow_smoke_operators,
        timeout_seconds=args.timeout_seconds,
        resume=not args.no_resume,
    )


def _report(args: argparse.Namespace) -> dict[str, Any]:
    from rl_engine.validation.reports.drift_report import (
        build_cross_config_attempt_report,
        write_drift_bundle,
        write_drift_report_image,
        write_drift_trace,
    )

    report = build_cross_config_attempt_report(args.attempt_dir, title=args.title)
    suffix = args.output.suffix.lower()
    if suffix == ".rlk-drift":
        output = write_drift_bundle(report, args.output, include_preview=not args.no_preview)
        artifact_type = "bundle"
    elif suffix in {".png", ".jpg", ".jpeg"}:
        output = write_drift_report_image(report, args.output)
        artifact_type = "image"
    elif suffix == ".json":
        output = write_drift_trace(report, args.output)
        artifact_type = "trace"
    else:
        raise ValueError("report output must use .rlk-drift, .png/.jpg, or .json")
    return {
        "schema_version": "cross_config.drift_report_summary.v1",
        "status": report["status"],
        "artifact_type": artifact_type,
        "output": str(output),
        "case_id": report["axes"].get("case_id"),
        "attempt_id": report["axes"].get("attempt_id"),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
