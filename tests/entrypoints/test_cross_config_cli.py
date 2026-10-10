# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

import rl_engine.validation.cross_config.__main__ as cli_main
from rl_engine.validation.cross_config.artifacts import ArtifactStore

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _REPOSITORY_ROOT / "configs/experiments/cross_config"
_CPU_RUNTIME_MODULE = "rl_engine.validation.fixtures.cpu_cross_config"


def _summary(captured: str) -> dict:
    summaries = []
    for line in captured.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("schema_version") == "cross_config.cli_summary.v1":
            summaries.append(value)
    assert len(summaries) == 1
    return summaries[0]


def test_run_uses_only_cpu_and_resumes_when_cuda_is_available(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config_path = _EXAMPLES / "cross_config_s0_cpu_smoke.json"
    plan_argv = [
        "plan",
        str(config_path),
        "--output-root",
        str(tmp_path),
    ]
    argv = [
        "run",
        str(config_path),
        "--runtime",
        "cpu-smoke",
        "--allow-smoke-operators",
        "--output-root",
        str(tmp_path),
        "--timeout-seconds",
        "10",
    ]

    assert cli_main.main(plan_argv) == 0
    planned = _summary(capsys.readouterr().out)
    experiment_path = Path(planned["artifact_dir"]) / "experiment.json"
    plan_path = Path(planned["artifact_dir"]) / "plan.jsonl"
    planned_experiment = experiment_path.read_bytes()
    planned_cases = plan_path.read_bytes()
    stored_config = json.loads(planned_experiment)
    stored_row = json.loads(planned_cases)
    assert stored_config["schema_version"] == "cross_config.experiment_config.v1"
    assert stored_row["schema_version"] == "cross_config.execution_plan_entry.v1"
    assert stored_row["case"]["execution_binding"]["operators"] == stored_row["operators"]

    assert cli_main.main(argv) == 0
    captured = capsys.readouterr()
    first = _summary(captured.out)
    assert first["status"] == "pass"
    assert first["runtime"] == "cpu-smoke"
    assert "actual backends rollout=smoke_only.logp_reference" in captured.err
    assert "training=smoke_only.logp_reference" in captured.err
    assert "worst sample/token=[0, 3]" in captured.err
    assert first["cases"]
    assert all(case["status"] == "pass" for case in first["cases"])
    assert all(case["resumed"] is False for case in first["cases"])
    assert experiment_path.read_bytes() == planned_experiment
    assert plan_path.read_bytes() == planned_cases

    store = ArtifactStore(tmp_path)
    for case in first["cases"]:
        attempt_dir = Path(case["attempt_dir"])
        assert (attempt_dir / "COMPLETE").is_file()
        actual = json.loads((attempt_dir / "actual.json").read_text(encoding="utf-8"))
        assert actual["environment"]["execution_devices"] == {
            "rollout": "cpu",
            "training": "cpu",
        }
        for name in ("score_rollout.pt", "score_training.pt"):
            bundle = store.load_tensor_bundle(attempt_dir / name)
            assert bundle["tensors"]["selected_logprobs"].device.type == "cpu"

    assert cli_main.main(argv) == 0
    resumed = _summary(capsys.readouterr().out)
    assert resumed["status"] == "pass"
    assert [case["attempt_id"] for case in resumed["cases"]] == [
        case["attempt_id"] for case in first["cases"]
    ]
    assert all(case["resumed"] is True for case in resumed["cases"])


@pytest.mark.parametrize(
    ("filename", "expected_cases"),
    [
        ("cross_config_s1_distributed_smoke.json", 5),
        ("cross_config_s2_vllm_tp_vs_fsdp.json", 10),
        ("cross_config_s3_qwen3_8b_tp4_cp4_bf16.json", 11),
    ],
)
def test_plan_records_named_scenarios_without_loading_a_runtime(
    tmp_path,
    capsys,
    monkeypatch,
    filename,
    expected_cases,
):
    monkeypatch.delitem(sys.modules, _CPU_RUNTIME_MODULE, raising=False)

    def runtime_must_not_run(*args, **kwargs):
        raise AssertionError(f"plan unexpectedly invoked the CPU runtime: {args!r}, {kwargs!r}")

    monkeypatch.setattr(cli_main, "_run", runtime_must_not_run)
    assert (
        cli_main.main(
            [
                "plan",
                str(_EXAMPLES / filename),
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    summary = _summary(capsys.readouterr().out)
    artifact_dir = Path(summary["artifact_dir"])
    assert summary["status"] == "planned"
    assert summary["planned_case_count"] == expected_cases
    assert (artifact_dir / "experiment.json").is_file()
    assert len((artifact_dir / "plan.jsonl").read_text(encoding="utf-8").splitlines()) == (
        expected_cases
    )
    assert not list(artifact_dir.glob("cases/*/*"))
    assert _CPU_RUNTIME_MODULE not in sys.modules
