# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Runner failures cannot reuse stale evidence or promote diagnostics."""

import json
from types import SimpleNamespace

import pytest

from rl_engine.validation.models.qwen3_next_cases import execution_verdict
from tools.validation.models import qwen3_next_operator_gates as gates

CASE = "tests/example.py::test_required"


@pytest.mark.parametrize("mode", ["passed", "missing", "forged", "exit_failed", "diagnostics"])
def test_run_suite_requires_current_complete_evidence(tmp_path, monkeypatch, mode):
    suite = "diagnostics" if mode == "diagnostics" else "core"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "suites": {suite: [CASE]}}))
    monkeypatch.setattr(gates, "MANIFEST", manifest)
    monkeypatch.setattr(gates.importlib.metadata, "version", lambda _: "test-version")
    monkeypatch.setattr(gates.subprocess, "check_output", lambda *args, **kwargs: "")
    out = tmp_path / "output"
    out.mkdir()
    complete = execution_verdict(
        [CASE], [CASE], {CASE: {"setup": "passed", "call": "passed", "teardown": "passed"}}
    )
    # This must be removed before invoking pytest, even if pytest writes nothing.
    (out / "case-results.json").write_text(json.dumps(complete))

    def run(*args, **kwargs):
        if mode != "missing":
            report = dict(complete)
            if mode == "forged":
                report["outcomes"] = {}
            (out / "case-results.json").write_text(json.dumps(report))
        return SimpleNamespace(returncode=1 if mode == "exit_failed" else 0)

    monkeypatch.setattr(gates.subprocess, "run", run)
    result = gates.run_suite(out, suite)
    if mode == "diagnostics":
        assert result["status"] == "diagnostic_completed"
        assert result["diagnostic_only"] is True
        assert result["mandatory_case_ids"] == []
    else:
        assert result["status"] == ("passed" if mode == "passed" else "failed")
        assert result["mandatory_case_ids"] == [CASE]
    assert result["full_model_l2"] == "not_evaluated"
