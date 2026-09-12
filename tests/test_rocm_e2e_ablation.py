# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import rl_engine.integrations.rocm_ablation as rocm_ablation
from rl_engine.integrations.ablation import Implementation
from rl_engine.integrations.rocm_ablation import (
    STRICT_ROCM_ATTENTION_CORE,
    STRICT_ROCM_ATTENTION_RUNTIME,
    STRICT_ROCM_ATTENTION_SCHEDULE,
    build_case_environment,
    replay_identity,
    rocm_attention_ablation_matrix,
    run_rocm_attention_ablation,
    validate_case_readbacks,
    validate_replay_environment,
)


def _environment() -> dict[str, str]:
    return {
        "MODEL_ROOT": "/models/Qwen3-8B",
        "TORCH_DIST_ROOT": "/models/Qwen3-8B_torch_dist",
        "VIME_CKPT": "/checkpoints/pre-update",
        "PROMPT_DATA": "/data/prompts.jsonl",
        "NUM_ROLLOUT": "1",
        "TRAIN_SEED": "1234",
        "ROLLOUT_SEED": "42",
    }


def _readback(case, *, side: str, calls: int = 1, strict: bool | None = None):
    if side == "training":
        framework, target = "megatron", "training"
    else:
        framework, target = "vllm", "rollout"
    implementation = case.implementation_for(target)
    is_strict = implementation is Implementation.RL_KERNEL if strict is None else strict
    backend = (
        "rlkernel.attention.deterministic.v1" if is_strict else f"{framework}.production.attention"
    )
    provenance = (
        {
            "execution": {
                "runtime_platform": "rocm",
                "operator": {
                    "actual_backend": STRICT_ROCM_ATTENTION_RUNTIME,
                    "strict_core_id": STRICT_ROCM_ATTENTION_CORE,
                    "strict_schedule": STRICT_ROCM_ATTENTION_SCHEDULE,
                    "fallback": False,
                    "triton_used": False,
                },
            }
        }
        if is_strict
        else {}
    )
    return {
        "framework": framework,
        "target": target,
        "plan": case.plan.to_dict(),
        "installed_hooks": {"attention": f"{framework}.attention"},
        "fallbacks": [],
        "operators": {
            "attention": {
                "implementation": implementation.value,
                "backend_id": backend,
                "call_count": calls,
                "provenance": provenance,
            }
        },
    }


def test_matrix_is_pr230_four_cell_attention_matrix():
    matrix = rocm_attention_ablation_matrix()

    assert tuple(case.case_id for case in matrix) == ("P/P", "P/R", "R/P", "R/R")
    assert [
        (
            case.implementation_for("training"),
            case.implementation_for("rollout"),
        )
        for case in matrix
    ] == [
        (Implementation.PRODUCTION, Implementation.PRODUCTION),
        (Implementation.PRODUCTION, Implementation.RL_KERNEL),
        (Implementation.RL_KERNEL, Implementation.PRODUCTION),
        (Implementation.RL_KERNEL, Implementation.RL_KERNEL),
    ]
    assert all(case.plan.cases["ffn"].case_id == "P/P" for case in matrix)
    assert all(case.plan.cases["logp"].case_id == "P/P" for case in matrix)


def test_matrix_rejects_unknown_and_duplicate_cells():
    with pytest.raises(ValueError, match="unknown"):
        rocm_attention_ablation_matrix(["R/X"])
    with pytest.raises(ValueError, match="unique"):
        rocm_attention_ablation_matrix(["R/R", "R/R"])


def test_replay_identity_changes_only_when_frozen_inputs_change():
    environment = _environment()
    first = replay_identity(["bash", "run.sh"], environment)
    environment["RL_KERNEL_ATTENTION_CASE"] = "R/R"
    second = replay_identity(["bash", "run.sh"], environment)
    environment["PROMPT_DATA"] = "/data/other.jsonl"
    changed = replay_identity(["bash", "run.sh"], environment)

    assert first == second
    assert changed["sha256"] != first["sha256"]


def test_executable_replay_requires_inputs_and_single_rollout():
    environment = _environment()
    validate_replay_environment(environment)
    environment.pop("PROMPT_DATA")
    with pytest.raises(RuntimeError, match="PROMPT_DATA"):
        validate_replay_environment(environment)
    environment = _environment()
    environment["NUM_ROLLOUT"] = "2"
    with pytest.raises(RuntimeError, match="NUM_ROLLOUT=1"):
        validate_replay_environment(environment)


def test_case_environment_propagates_plan_to_megatron_vllm_and_ray(tmp_path):
    case = rocm_attention_ablation_matrix(["P/R"])[0]
    environment = build_case_environment(_environment(), case, case_dir=tmp_path / "p-r")

    assert environment["RL_KERNEL_ATTENTION_CASE"] == "P/R"
    assert environment["RL_KERNEL_FFN_CASE"] == "P/P"
    assert environment["RL_KERNEL_LOGP_CASE"] == "P/P"
    assert environment["RL_KERNEL_VLLM_INTEGRATION"] == "1"
    assert environment["RL_KERNEL_PLATFORM"] == "rocm"
    assert environment["VLLM_ATTENTION_BACKEND"] == "ROCM_AITER_FA"
    assert environment["RL_KERNEL_READBACK_DIR"].endswith("p-r/readbacks")
    assert environment["NUM_ROLLOUT"] == "1"


@pytest.mark.parametrize("case_id", ["P/P", "P/R", "R/P", "R/R"])
def test_case_readbacks_validate_both_real_framework_sides(case_id):
    case = rocm_attention_ablation_matrix([case_id])[0]
    routes, errors = validate_case_readbacks(
        case,
        [
            _readback(case, side="training", calls=11),
            _readback(case, side="rollout", calls=17),
        ],
    )

    assert errors == ()
    assert set(routes) == {"training", "rollout"}
    assert routes["training"].call_count == 11
    assert routes["rollout"].call_count == 17


def test_strict_side_requires_rocm_runtime_identity():
    case = rocm_attention_ablation_matrix(["R/R"])[0]
    bad = _readback(case, side="rollout")
    bad["operators"]["attention"]["provenance"]["execution"]["operator"][
        "actual_backend"
    ] = "rlkernel.cuda.attention.fa4_ag_rs.v1"

    _routes, errors = validate_case_readbacks(
        case,
        [_readback(case, side="training"), bad],
    )

    assert any("strict AITER/CK runtime" in error for error in errors)


def test_strict_side_rejects_triton_and_missing_fixed_schedule():
    case = rocm_attention_ablation_matrix(["P/R"])[0]
    bad = _readback(case, side="rollout")
    operator = bad["operators"]["attention"]["provenance"]["execution"]["operator"]
    operator["triton_used"] = True
    operator["core_actual_backends"] = ["triton.attention"]
    operator.pop("strict_schedule")

    _routes, errors = validate_case_readbacks(
        case,
        [_readback(case, side="training"), bad],
    )

    assert any("fixed no-Split-KV schedule" in error for error in errors)
    assert any("used Triton" in error for error in errors)


def test_production_side_requires_framework_native_backend_identity():
    case = rocm_attention_ablation_matrix(["P/P"])[0]
    bad = _readback(case, side="training")
    bad["operators"]["attention"]["backend_id"] = "rlkernel.attention.deterministic.v1"

    _routes, errors = validate_case_readbacks(
        case,
        [bad, _readback(case, side="rollout")],
    )

    assert any("framework-native Attention" in error for error in errors)


def test_zero_exit_without_framework_evidence_is_a_failure():
    case = rocm_attention_ablation_matrix(["R/R"])[0]
    _routes, errors = validate_case_readbacks(case, [])

    assert errors == (
        "missing megatron/training runtime readback",
        "missing vllm/rollout runtime readback",
    )


def test_malformed_call_count_is_a_validation_error():
    case = rocm_attention_ablation_matrix(["P/P"])[0]
    bad = _readback(case, side="training")
    bad["operators"]["attention"]["call_count"] = "not-an-integer"

    _routes, errors = validate_case_readbacks(
        case,
        [bad, _readback(case, side="rollout")],
    )

    assert any("invalid call count" in error for error in errors)


def test_dry_run_writes_human_summary_without_fabricating_results(tmp_path):
    results = run_rocm_attention_ablation(
        ["bash", "run.sh"],
        output_dir=tmp_path,
        base_environment={},
        case_ids=["P/P", "R/R"],
        execute=False,
    )

    assert [result.status for result in results] == ["not_run", "not_run"]
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Executed: `no`" in summary
    assert "| P/P |" in summary and "| R/R |" in summary
    assert not list(tmp_path.glob("*.json"))


def test_execute_rejects_stale_case_evidence_before_gpu_probe(monkeypatch, tmp_path):
    (tmp_path / "r-r").mkdir()
    probed = False

    def probe():
        nonlocal probed
        probed = True

    monkeypatch.setattr(rocm_ablation, "validate_rocm_host", probe)

    with pytest.raises(FileExistsError, match="existing case directories"):
        run_rocm_attention_ablation(
            ["bash", "run.sh"],
            output_dir=tmp_path,
            base_environment=_environment(),
            case_ids=["R/R"],
            execute=True,
        )

    assert probed is False


def test_execute_runs_each_case_fresh_and_checks_emitted_readbacks(monkeypatch, tmp_path):
    invocations: list[str] = []
    monkeypatch.setattr(rocm_ablation, "validate_rocm_host", lambda: {"hip_runtime": "7.1"})

    def fake_run(command, *, env, stdout, stderr, check):
        del command, stdout, stderr, check
        case = rocm_attention_ablation_matrix([env["RL_KERNEL_ATTENTION_CASE"]])[0]
        readback_dir = Path(env["RL_KERNEL_READBACK_DIR"])
        invocations.append(case.case_id)
        for side in ("training", "rollout"):
            payload = _readback(case, side=side)
            (readback_dir / f"{side}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rocm_ablation.subprocess, "run", fake_run)
    results = run_rocm_attention_ablation(
        ["bash", "run.sh"],
        output_dir=tmp_path,
        base_environment=_environment(),
        case_ids=["P/R", "R/P"],
        execute=True,
    )

    assert invocations == ["P/R", "R/P"]
    assert [result.status for result in results] == ["passed", "passed"]
    assert all(result.log_path.is_file() for result in results)
    assert "Executed: `yes`" in (tmp_path / "summary.md").read_text(encoding="utf-8")
