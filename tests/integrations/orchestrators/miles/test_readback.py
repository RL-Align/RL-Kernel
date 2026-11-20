# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json

import pytest

from rl_engine.integrations.orchestrators.miles.readback import audit_readbacks, main

_MODULES = ("attention", "ffn", "logp")


def _readback(
    *,
    cases: dict[str, str] | None = None,
    hooks: dict[str, str] | None = None,
    operators: dict[str, dict] | None = None,
    fallbacks: list | None = None,
    target: str = "training",
) -> dict:
    resolved = cases or {module: "P/P" for module in _MODULES}
    return {
        "framework": "megatron",
        "target": target,
        "plan": {
            "schema_version": "rlkernel.debug_matrix.v1",
            "cases": {
                module: {"module": module, "case_id": case_id}
                for module, case_id in resolved.items()
            },
        },
        "installed_hooks": hooks or {},
        "operators": operators or {},
        "fallbacks": fallbacks or [],
    }


def _write(directory, pid: int, payload: dict, *, name: str | None = None):
    path = directory / (name or f"megatron-training-{pid}.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _healthy_logp_r() -> dict:
    return _readback(
        cases={"attention": "P/P", "ffn": "P/P", "logp": "R/P"},
        hooks={"attention": "A.forward", "ffn": "M.forward", "logp": "L.forward"},
        operators={
            "logp": {
                "implementation": "rl_kernel",
                "backend_id": "pytorch-vocab-parallel-logp",
                "call_count": 4,
            }
        },
    )


def test_healthy_workers_pass(tmp_path):
    _write(tmp_path, 101, _healthy_logp_r())
    _write(tmp_path, 102, _healthy_logp_r())

    report = audit_readbacks(tmp_path, expected_workers=2)

    assert report.healthy
    assert len(report.workers) == 2
    assert {worker.pid for worker in report.workers} == {101, 102}


def test_native_only_workers_need_no_operator(tmp_path):
    _write(tmp_path, 101, _readback(hooks={"attention": "A.forward"}))

    report = audit_readbacks(tmp_path, expected_workers=1)

    assert report.healthy


def test_missing_worker_file_is_reported(tmp_path):
    _write(tmp_path, 101, _healthy_logp_r())

    report = audit_readbacks(tmp_path, expected_workers=2)

    assert not report.healthy
    assert any("expected 2 worker" in problem for problem in report.problems)


def test_empty_directory_is_reported(tmp_path):
    report = audit_readbacks(tmp_path, expected_workers=1)

    assert not report.healthy
    assert any("no readback files" in problem for problem in report.problems)


def test_missing_directory_is_reported(tmp_path):
    report = audit_readbacks(tmp_path / "absent", expected_workers=1)

    assert not report.healthy
    assert any("does not exist" in problem for problem in report.problems)


def test_rl_kernel_route_without_hook_is_reported(tmp_path):
    payload = _healthy_logp_r()
    payload["installed_hooks"].pop("logp")
    _write(tmp_path, 101, payload)

    report = audit_readbacks(tmp_path)

    assert not report.healthy
    assert any("no installed hook" in problem for problem in report.workers[0].problems)


def test_rl_kernel_route_that_never_executed_is_reported(tmp_path):
    payload = _healthy_logp_r()
    payload["operators"] = {}
    _write(tmp_path, 101, payload)

    report = audit_readbacks(tmp_path, require_execution=True)

    assert not report.healthy
    assert any("never executed" in problem for problem in report.workers[0].problems)


def test_unexecuted_rl_kernel_route_is_allowed_by_default(tmp_path):
    payload = _healthy_logp_r()
    payload["operators"] = {}
    _write(tmp_path, 101, payload)

    assert audit_readbacks(tmp_path).healthy
    assert not audit_readbacks(tmp_path, require_execution=True).healthy


def test_rl_kernel_route_executed_as_production_is_reported(tmp_path):
    payload = _healthy_logp_r()
    payload["operators"]["logp"]["implementation"] = "production"
    _write(tmp_path, 101, payload)

    report = audit_readbacks(tmp_path, require_execution=True)

    assert not report.healthy
    assert any("not 'rl_kernel'" in problem for problem in report.workers[0].problems)


def test_rl_kernel_route_with_foreign_backend_is_reported(tmp_path):
    payload = _healthy_logp_r()
    payload["operators"]["logp"]["backend_id"] = "megatron.production.logp"
    _write(tmp_path, 101, payload)

    report = audit_readbacks(tmp_path, require_execution=True)

    assert not report.healthy
    assert any("not an RL-Kernel backend" in problem for problem in report.workers[0].problems)


def test_zero_call_count_is_reported(tmp_path):
    payload = _healthy_logp_r()
    payload["operators"]["logp"]["call_count"] = 0
    _write(tmp_path, 101, payload)

    report = audit_readbacks(tmp_path, require_execution=True)

    assert not report.healthy
    assert any("call_count 0" in problem for problem in report.workers[0].problems)


def test_fallback_is_reported(tmp_path):
    payload = _healthy_logp_r()
    payload["fallbacks"] = [{"module": "logp", "reason": "backend"}]
    _write(tmp_path, 101, payload)

    report = audit_readbacks(tmp_path)

    assert not report.healthy
    assert any("fallbacks recorded" in problem for problem in report.workers[0].problems)


def test_inconsistent_plans_are_reported(tmp_path):
    _write(tmp_path, 101, _healthy_logp_r())
    _write(
        tmp_path,
        102,
        _readback(hooks={"attention": "A.forward", "ffn": "M.forward"}),
    )

    report = audit_readbacks(tmp_path)

    assert not report.healthy
    assert any("disagree on the integration plan" in problem for problem in report.problems)


def test_unexpected_file_name_is_reported(tmp_path):
    _write(tmp_path, 101, _readback(), name="readback.json")

    report = audit_readbacks(tmp_path)

    assert not report.healthy
    assert any("unexpected readback file name" in problem for problem in report.workers[0].problems)


def test_unreadable_json_is_reported(tmp_path):
    (tmp_path / "megatron-training-101.json").write_text("{not json", encoding="utf-8")

    report = audit_readbacks(tmp_path)

    assert not report.healthy
    assert any("unreadable readback file" in problem for problem in report.workers[0].problems)


def test_wrong_target_is_reported(tmp_path):
    _write(tmp_path, 101, _readback(target="rollout"))

    report = audit_readbacks(tmp_path)

    assert not report.healthy
    assert any("rollout" in problem for problem in report.workers[0].problems)


def test_cli_exit_codes_and_json(tmp_path, capsys):
    _write(tmp_path, 101, _healthy_logp_r())

    assert main([str(tmp_path), "--expect-workers", "1"]) == 0
    assert "1/1 workers healthy" in capsys.readouterr().out

    assert main([str(tmp_path), "--expect-workers", "2"]) == 1
    capsys.readouterr()

    assert main([str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["healthy"] is True
    assert payload["workers"][0]["pid"] == 101


@pytest.mark.parametrize("case_id", ["R/P", "R/R"])
def test_training_side_routes_require_evidence(tmp_path, case_id):
    payload = _readback(
        cases={"attention": case_id, "ffn": "P/P", "logp": "P/P"},
        hooks={"attention": "A.forward"},
    )
    _write(tmp_path, 101, payload)

    report = audit_readbacks(tmp_path, require_execution=True)

    assert not report.healthy
    assert any("never executed" in problem for problem in report.workers[0].problems)
