# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Mandatory case completeness must not be inferred from an aggregate count."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rl_engine.testing.qwen3_next_cases import case_digest, execution_verdict, load_cases

CASE = "tests/example.py::test_required"
SUCCESS = {"setup": "passed", "call": "passed", "teardown": "passed"}


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "duplicate", "skipped", "teardown", "incomplete"]
)
def test_declared_case_must_complete_all_phases(fault):
    collected = [CASE]
    outcomes = {CASE: dict(SUCCESS)}
    if fault == "missing":
        collected = []
    elif fault == "extra":
        collected.append("tests/example.py::test_undeclared")
    elif fault == "duplicate":
        collected.append(CASE)
    elif fault == "skipped":
        outcomes[CASE]["call"] = "skipped"
    elif fault == "teardown":
        outcomes[CASE]["teardown"] = "failed"
    else:
        del outcomes[CASE]["call"]
    assert execution_verdict([CASE], collected, outcomes)["passed"] is False


def test_complete_report_records_exact_identity():
    result = execution_verdict([CASE], [CASE], {CASE: SUCCESS})
    assert result["passed"] is True
    assert result["counts"] == {"tests": 1, "failures": 0, "errors": 0, "skipped": 0}
    assert result["expected_case_ids_sha256"] == case_digest([CASE])


def test_diagnostics_cannot_enter_required_case_set(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suites": {
                    "norm": [CASE],
                    "core": ["tests/example.py::test_core"],
                    "diagnostics": ["tests/example.py::test_approximate"],
                },
            }
        )
    )
    assert load_cases(path, "all") == sorted([CASE, "tests/example.py::test_core"])


@pytest.mark.parametrize("outcome", ["pass", "skip", "xfail", "deselected"])
def test_plugin_enforces_real_pytest_outcomes(tmp_path, outcome):
    test_file = tmp_path / "test_acceptance.py"
    body = "pass"
    if outcome == "skip":
        body = "pytest.skip('missing CUDA')"
    elif outcome == "xfail":
        body = "pytest.xfail('unimplemented')"
    test_file.write_text(f"import pytest\ndef test_required():\n    {body}\n")
    case = "test_acceptance.py::test_required"
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"schema_version": 1, "suites": {"core": [case]}}))
    report_path = tmp_path / "report.json"
    args = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "rl_engine.testing.pytest_case_manifest",
        "--case-manifest",
        str(manifest),
        "--case-suite",
        "core",
        "--case-report",
        str(report_path),
        str(test_file),
    ]
    if outcome == "deselected":
        args.extend(["-k", "not required"])
    # Use this source tree even when a different RL-Kernel checkout is installed.
    import os

    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run(args, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == (
        0 if outcome == "pass" else (5 if outcome == "deselected" else 1)
    ), (result.stdout + result.stderr)
    report = json.loads(report_path.read_text())
    assert report["passed"] is (outcome == "pass")
    assert report["expected_case_ids"] == [case]
