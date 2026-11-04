# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rl_engine.contracts.strict_provider import ProviderLock
from tests.helpers.strict_provider import Bundle
from tools.validation import strict_provider_manifest as manifest_cli

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


@pytest.mark.parametrize(
    ("input_option", "output_option"),
    [("--input", "--output"), ("--in", "--output"), ("--input", "--out"), ("--in", "--out")],
)
@pytest.mark.parametrize("equals_form", [False, True])
def test_same_input_and_output_is_rejected_without_destroying_input(
    tmp_path, input_option, output_option, equals_form
):
    path = tmp_path / "input.json"
    original = b'{"original":"input"}\n'
    path.write_bytes(original)
    args = (
        [f"{input_option}={path}", f"{output_option}={path}"]
        if equals_form
        else [input_option, path, output_option, path]
    )
    result = run_cli("seal", *args)
    assert result.returncode == 2
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".input.json.*.tmp"))


@pytest.mark.parametrize("allow_abbrev", [False, True])
def test_input_abbreviation_is_safe_before_argument_error(tmp_path, monkeypatch, allow_abbrev):
    class Parser(manifest_cli._Parser):
        def __init__(self, *args, **kwargs):
            kwargs["allow_abbrev"] = allow_abbrev
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(manifest_cli, "_Parser", Parser)
    path = tmp_path / "evidence.json"
    original = b'{"original":"evidence"}\n'
    path.write_bytes(original)
    code = manifest_cli.main(["seal", "--in", str(path), "--output", str(path)])
    assert code == 2
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".evidence.json.*.tmp"))


def test_resolved_input_and_output_alias_is_rejected_without_writing(tmp_path):
    path = tmp_path / "evidence.json"
    original = b'{"original":"evidence"}\n'
    path.write_bytes(original)
    nested = tmp_path / "nested"
    nested.mkdir()
    alias = nested / ".." / path.name
    result = run_cli("seal", "--input", path, "--output", alias)
    assert result.returncode == 2
    assert path.read_bytes() == original


def test_hardlinked_input_and_output_are_rejected_without_writing(tmp_path):
    path = tmp_path / "evidence.json"
    alias = tmp_path / "result.json"
    original = b'{"original":"evidence"}\n'
    path.write_bytes(original)
    try:
        os.link(path, alias)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Requires filesystem hard-link support: {exc}")
    result = run_cli("seal", "--input", path, "--output", alias)
    assert result.returncode == 2
    assert path.read_bytes() == alias.read_bytes() == original
    assert path.samefile(alias)


def test_previous_result_is_invalidated_before_loading_evidence(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    original = b'{"original":"evidence"}\n'
    path.write_bytes(original)
    output = tmp_path / "result.json"
    output.write_text('{"manifest_enforcement":"passed"}', encoding="utf-8")

    def interrupt_read(input_path):
        assert input_path == path
        assert path.read_bytes() == original
        assert json.loads(output.read_text())["status"] == "incomplete"
        raise KeyboardInterrupt

    monkeypatch.setattr(manifest_cli, "read_json", interrupt_read)
    code = manifest_cli.main(["seal", "--input", str(path), "--output", str(output)])
    assert code == 1
    assert json.loads(output.read_text())["error"]["code"] == "cancelled"
