#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Run declared operator acceptance or explicitly separate diagnostic cases."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from rl_engine.testing.qwen3_next_cases import case_digest, execution_verdict, load_cases

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "rl_engine/testing/qwen3_next_operator_cases.json"


def run_suite(out: Path, suite: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(MANIFEST, suite)
    diagnostic = suite == "diagnostics"
    case_path = out / "case-results.json"
    case_path.unlink(missing_ok=True)
    env = dict(os.environ)
    env["QWEN3_NEXT_ROUTING_EVIDENCE"] = str(out / "gdn-step-routing.json")
    env["QWEN3_NEXT_FLASHINFER_EVIDENCE"] = str(out)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "rl_engine.testing.pytest_case_manifest",
            "--case-manifest",
            str(MANIFEST),
            "--case-suite",
            suite,
            "--case-report",
            str(case_path),
            "--junitxml",
            str(out / "providers.xml"),
            *cases,
        ],
        cwd=ROOT,
        env=env,
        check=False,
    )
    try:
        case_report = json.loads(case_path.read_text()) if case_path.exists() else {}
    except json.JSONDecodeError:
        case_report = {}
    verified = execution_verdict(
        cases, case_report.get("collected_case_ids", []), case_report.get("outcomes", {})
    )
    passed = proc.returncode == 0 and case_report.get("passed") is True
    passed = passed and verified["passed"] and case_report.get("expected_case_ids") == cases
    passed = passed and case_report.get("expected_case_ids_sha256") == case_digest(cases)
    reports = {}
    if suite in ("norm", "all"):
        for op in ("qwen3_next_rms_norm", "rms_norm_gated"):
            for gate in ("forward", "gradient"):
                name = f"{op}-{gate}.json"
                with (out / name).open("w") as stream:
                    norm = subprocess.run(
                        [
                            sys.executable,
                            f"scripts/check_{gate}_invariance.py",
                            "--manifest",
                            "rl_engine/testing/qwen3_next_norm_manifest.json",
                            "--op",
                            op,
                            "--candidate",
                            "cuda",
                            "--backend-profile",
                            "cuda_bf16",
                            "--hidden",
                            "2048",
                            "--head-dim",
                            "128",
                            "--json",
                        ],
                        cwd=ROOT,
                        env=env,
                        stdout=stream,
                        check=False,
                    )
                try:
                    report = json.loads((out / name).read_text())
                except json.JSONDecodeError:
                    report = {}
                valid = all(
                    (
                        norm.returncode == 0,
                        report.get("passed") is True,
                        report.get("workload", {}).get("scope") == "qwen3_next_norm_operators",
                    )
                )
                reports[name] = {
                    "passed": valid,
                    "exit_code": norm.returncode,
                    "workload": report.get("workload"),
                }
                passed = passed and valid
    status = "passed" if passed else "failed"
    if diagnostic:
        status = "diagnostic_completed" if passed else "diagnostic_failed"
    return {
        "scope": "qwen3_next_operator_diagnostics" if diagnostic else "qwen3_next_operators_only",
        "status": status,
        "suite": suite,
        "diagnostic_only": diagnostic,
        "full_model_l2": "not_evaluated",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "vllm", "transformers", "triton")
        },
        "mandatory_case_ids": [] if diagnostic else cases,
        "case_ids_sha256": case_digest(cases),
        "pytest_exit_code": proc.returncode,
        "provider_tests": verified["counts"],
        "norm_gates": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", choices=("norm", "core", "all", "diagnostics"), default="all")
    args = parser.parse_args()
    result = run_suite(args.out.resolve(), args.suite)
    partial = args.out / "summary.json.partial"
    partial.write_text(json.dumps(result, indent=2) + "\n")
    partial.replace(args.out / "summary.json")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in ("passed", "diagnostic_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
