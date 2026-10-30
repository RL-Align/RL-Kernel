# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from rl_engine.contracts.strict_provider import ProviderLock
from tests.helpers.strict_provider import Bundle

SCRIPT = Path(__file__).resolve().parents[2] / "tools/validation/strict_provider_manifest.py"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, timeout=30
    )


def test_collect_candidate_and_seal_roundtrip(tmp_path):
    bundle = Bundle(tmp_path / "evidence")
    supplied = tmp_path / "supplied.json"
    supplied.write_text(json.dumps(bundle.lock.to_dict()), encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    result = run_cli("collect", "--input", supplied, "--output", candidate)
    assert result.returncode == 0, result.stderr
    data = json.loads(candidate.read_text(encoding="utf-8"))
    assert data["status"] == "candidate"
    assert data["manifest_enforcement"] == "not_evaluated"
    assert data["hardware_validation"] == "not_evaluated"
    sealed = tmp_path / "lock.json"
    result = run_cli("seal", "--input", candidate, "--output", sealed)
    assert result.returncode == 0, result.stderr
    assert ProviderLock.from_dict(json.loads(sealed.read_text())).lock_id == bundle.lock.lock_id


def test_validate_subprocess_checks_identity_and_provenance(tmp_path):
    bundle = Bundle(tmp_path / "evidence")
    supplied = tmp_path / "input.json"
    supplied.write_text(json.dumps(bundle.cli_input()), encoding="utf-8")
    output = tmp_path / "result.json"
    result = run_cli(
        "validate",
        "--input",
        supplied,
        "--output",
        output,
        "--evidence-root",
        bundle.root,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data["manifest_enforcement"] == "passed"
    assert data["numerical_parity"] == "not_evaluated"
    assert data["validation_scope"] == "identity_configuration_provenance"
    assert "compared_elements" not in data
    assert data["hardware_validation"] == "not_evaluated"
    assert not list(tmp_path.glob(".result.json.*.tmp"))


@pytest.mark.parametrize(
    "failure",
    [
        "content_drift",
        "stale_attempt",
        "missing_input",
        "malformed_input",
        "enforce",
        "usage",
        "numerical_option",
    ],
)
def test_cli_failure_never_leaves_old_passed_result(tmp_path, failure):
    bundle = Bundle(tmp_path / "evidence")
    supplied = tmp_path / "input.json"
    payload = bundle.cli_input()
    if failure == "content_drift":
        bundle.original.write_bytes(b"changed mock provider")
    elif failure == "stale_attempt":
        payload["expectation"]["attempt"] = "attempt0"
    supplied.write_text(json.dumps(payload), encoding="utf-8")
    if failure == "missing_input":
        supplied.unlink()
    elif failure == "malformed_input":
        supplied.write_text("{", encoding="utf-8")
    output = tmp_path / "result.json"
    output.write_text('{"status":"completed","manifest_enforcement":"passed"}')
    operation = "enforce" if failure == "enforce" else "validate"
    args = [
        operation,
        "--input",
        supplied,
        "--output",
        output,
        "--evidence-root",
        bundle.root,
    ]
    if failure == "usage":
        args.append("--invalid-option")
    elif failure == "numerical_option":
        args.append("--require-numerical")
    result = run_cli(*args)
    assert result.returncode == (2 if failure in {"usage", "numerical_option"} else 1)
    data = json.loads(output.read_text())
    assert data["status"] == "failed"
    assert data["manifest_enforcement"] == "failed"
    assert data["numerical_parity"] == "not_evaluated"
    if failure == "enforce":
        assert data["error"]["code"] == "execution_adapter_required"
    assert not list(tmp_path.glob(".result.json.*.tmp"))


@pytest.mark.parametrize("outputs", [[], {"passed": True}])
def test_numeric_output_bundle_cannot_be_misreported_as_manifest_acceptance(tmp_path, outputs):
    bundle = Bundle(tmp_path / "evidence")
    payload = {**bundle.cli_input(), "output_artifacts": outputs}
    supplied = tmp_path / "input.json"
    supplied.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "result.json"
    output.write_text('{"manifest_enforcement":"passed","numerical_parity":"passed"}')
    result = run_cli(
        "validate", "--input", supplied, "--output", output, "--evidence-root", bundle.root
    )
    assert result.returncode == 1
    data = json.loads(output.read_text())
    assert data["status"] == "failed"
    assert data["error"]["code"] == "invalid_evidence"
    assert data["manifest_enforcement"] == "failed"
    assert data["numerical_parity"] == "not_evaluated"
    assert data["hardware_validation"] == "not_evaluated"


def test_zero_cases_cannot_pass_subprocess(tmp_path):
    bundle = Bundle(tmp_path / "evidence")
    payload = bundle.cli_input()
    payload["expectation"]["cases"] = []
    supplied = tmp_path / "input.json"
    supplied.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "result.json"
    result = run_cli(
        "validate", "--input", supplied, "--output", output, "--evidence-root", bundle.root
    )
    assert result.returncode == 1
    assert json.loads(output.read_text())["status"] == "failed"


def test_same_input_and_output_is_rejected_without_destroying_input(tmp_path):
    path = tmp_path / "input.json"
    original = '{"original":"input"}'
    path.write_text(original, encoding="utf-8")
    result = run_cli("seal", "--input", path, "--output", path)
    assert result.returncode == 2
    assert path.read_text(encoding="utf-8") == original
