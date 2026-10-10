# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from rl_engine.contracts.numerical import resolve_logprob_threshold
from rl_engine.validation.cross_config.comparison import (
    compare_score_artifacts,
    recompute_mismatch_mask,
)
from rl_engine.validation.cross_config.config import (
    CONFIG_SCHEMA_VERSION,
    bind_operator_selection,
    load_config,
)
from rl_engine.validation.cross_config.planner import MAX_PLAN_CASES, Planner, PlanningError
from rl_engine.validation.cross_config.schema import (
    AlignmentStatus,
    ExperimentDefinition,
    InterventionSpec,
    PlanningStrategy,
    RuntimeProvenance,
    ScoreArtifact,
    ScorerSpec,
    ScoreSide,
    SemanticIdentitySpec,
    TokenComparisonArtifact,
)


def _identity(
    *,
    checkpoint_id: str = "tiny-checkpoint",
    tokenizer_policy: str = "tokenizer-v1:right-padding",
    active_mask: tuple[tuple[bool, ...], ...] = ((True, False, True),),
) -> SemanticIdentitySpec:
    return SemanticIdentitySpec(
        checkpoint_id=checkpoint_id,
        model_version="weights-v7",
        tokenizer_id="tiny-tokenizer",
        tokenizer_policy=tokenizer_policy,
        token_ids=((11, 12, 13),),
        selected_token_ids=((12, 13, 14),),
        active_mask=active_mask,
        attention_mask=((True, True, True),),
        position_ids=((0, 1, 2),),
        pre_update_state="state-before-step-9",
        cache_metadata={"use_cache": False},
        packing_metadata={"packed": False},
    )


def _score(
    side: ScoreSide,
    values: torch.Tensor,
    *,
    identity: SemanticIdentitySpec | None = None,
    active_mask: torch.Tensor | None = None,
) -> ScoreArtifact:
    identity = identity or _identity()
    backend = f"test.{side.value}.selected_logprob"
    return ScoreArtifact(
        case_id="case-001",
        attempt_id="attempt-001",
        side=side,
        identity=identity,
        scorer=ScorerSpec(
            side=side,
            backend_id=f"{side.value}-scorer",
            dtype="float32",
            operator_overrides={"selected_logprob": backend},
        ),
        selected_logprobs=values,
        active_mask=(
            active_mask
            if active_mask is not None
            else torch.tensor(identity.active_mask, dtype=torch.bool)
        ),
        provenance=RuntimeProvenance(
            requested={"logp": {"backend": backend}},
            normalized={"logp": {"backend": backend}},
            materialized={"logp": {"backend": backend}},
            actual={"logp": {"backend": backend}},
            implementation_fingerprint=f"{side.value}-implementation-v1",
        ),
    )


def _tensor_from_payload(payload: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(payload["values"], dtype=getattr(torch, str(payload["dtype"]))).reshape(
        payload["shape"]
    )


def _baseline() -> dict[str, Any]:
    return {
        "batch": {"size": 8},
        "rollout": {
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "dtype": "float32",
            "enable_prefix_caching": False,
            "enforce_eager": True,
        },
        "training": {
            "sharding": "unsharded",
            "attention_backend": "eager",
            "compute_dtype": "float32",
        },
        "logp": {"backend": "rlkernel.reference_logp"},
    }


def _definition(
    *,
    strategy: PlanningStrategy = PlanningStrategy.ONE_AT_A_TIME,
    pairwise_paths: tuple[tuple[str, str], ...] = (),
) -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_id="planner-test",
        scenario_id="cpu-contract",
        scenario={"model": "synthetic", "device": "cpu"},
        identity=_identity(),
        baseline=_baseline(),
        interventions=(
            InterventionSpec("batch.size", (1, 4)),
            InterventionSpec("rollout.dtype", ("bfloat16",)),
            InterventionSpec("training.attention_backend", ("sdpa",)),
        ),
        strategy=strategy,
        pairwise_paths=pairwise_paths,
    )


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            result.update(_flatten(child, path))
        else:
            result[path] = child
    return result


def _config() -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "experiment_id": "cpu-config-test",
        "scenario_id": "cpu-smoke",
        "contract_source": "ws1",
        "contract_version": "current",
        "strategy": "one_at_a_time",
        "strict_fallback": True,
        "identity": {
            "checkpoint_id": "tiny",
            "model_version": "weights-v1",
            "tokenizer_policy": "synthetic-v1",
            "token_ids": [[1, 2, 3]],
            "selected_token_ids": [[0, 2, 3]],
            "active_mask": [[False, True, True]],
            "attention_mask": [[True, True, True]],
            "pre_update_state": "iteration-0",
        },
        "baseline": {
            **_baseline(),
            "batch": {"size": 1},
        },
        "interventions": [{"path": "batch.size", "values": [2]}],
        "operators": {
            "selected_logprob": {
                "rollout": "rlkernel.reference_logp",
                "training": {
                    "backend": "smoke_only.logp_offset",
                    "options": {"offset": 0.1},
                },
            }
        },
        "scenario": {"device": "cpu", "workload": "tiny"},
    }


def _write_config(tmp_path: Path, value: Mapping[str, Any], name: str = "config.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_fixed_threshold_uses_only_active_tokens_and_is_reproducible_offline():
    threshold = resolve_logprob_threshold("float32")
    rollout_values = torch.zeros((1, 3), dtype=torch.float32)
    training_values = torch.tensor(
        [[threshold * 2.0, 1_000.0, threshold * 0.5]],
        dtype=torch.float32,
    )

    result = compare_score_artifacts(
        _score(ScoreSide.ROLLOUT, rollout_values),
        _score(ScoreSide.TRAINING, training_values),
    )

    assert result.status is AlignmentStatus.FAIL
    assert result.active_token_count == 2
    assert result.mismatch_count == 1
    assert result.fixed_threshold == threshold
    assert result.token_artifact is not None
    assert result.token_artifact.mismatch_mask.tolist() == [[True, False, False]]

    payload = json.loads(json.dumps(result.token_artifact.to_dict()))
    offline = recompute_mismatch_mask(
        _tensor_from_payload(payload["rollout_logprobs"]),
        _tensor_from_payload(payload["training_logprobs"]),
        _tensor_from_payload(payload["active_mask"]),
        float(payload["fixed_threshold"]),
    )
    assert torch.equal(offline, result.token_artifact.mismatch_mask)
    assert not recompute_mismatch_mask(
        torch.zeros(1, dtype=torch.float64),
        torch.tensor([threshold], dtype=torch.float64),
        torch.tensor([True]),
        threshold,
    ).item()

    inactive_nonfinite = training_values.clone()
    inactive_nonfinite[0, 1] = float("nan")
    sanitized = compare_score_artifacts(
        _score(ScoreSide.ROLLOUT, rollout_values),
        _score(ScoreSide.TRAINING, inactive_nonfinite),
    )
    assert sanitized.status is result.status
    assert sanitized.mismatch_count == result.mismatch_count
    assert sanitized.token_artifact is not None
    assert sanitized.token_artifact.training_logprobs[0, 1].item() == 0.0
    json.dumps(sanitized.to_dict(), allow_nan=False)


def test_zero_tokens_identity_mismatch_and_invalid_scores_are_not_numerical_failures():
    empty_identity = _identity(active_mask=((False, False, False),))
    empty = compare_score_artifacts(
        _score(ScoreSide.ROLLOUT, torch.zeros((1, 3)), identity=empty_identity),
        _score(ScoreSide.TRAINING, torch.zeros((1, 3)), identity=empty_identity),
    )
    assert empty.status is AlignmentStatus.ZERO_ACTIVE_TOKENS
    assert empty.comparable is False
    assert empty.passed is False

    identity = _identity()
    changed_identity = replace(identity, tokenizer_policy="different-policy")
    mismatched = compare_score_artifacts(
        _score(ScoreSide.ROLLOUT, torch.zeros((1, 3)), identity=identity),
        _score(ScoreSide.TRAINING, torch.ones((1, 3)), identity=changed_identity),
    )
    assert mismatched.status is AlignmentStatus.INVALID_IDENTITY
    assert "tokenizer_policy" in mismatched.identity_errors

    invalid = compare_score_artifacts(
        _score(ScoreSide.ROLLOUT, torch.zeros((1, 3))),
        _score(ScoreSide.TRAINING, torch.tensor([[float("nan"), 0.0, 0.0]])),
    )
    assert invalid.status is AlignmentStatus.INVALID_ARTIFACT
    assert invalid.comparable is False

    with pytest.raises(ValueError, match="finite and non-negative"):
        TokenComparisonArtifact(
            rollout_logprobs=torch.zeros(1),
            training_logprobs=torch.zeros(1),
            active_mask=torch.ones(1, dtype=torch.bool),
            absolute_diff=torch.zeros(1),
            mismatch_mask=torch.zeros(1, dtype=torch.bool),
            fixed_threshold=float("nan"),
        )


def test_planner_emits_one_stable_baseline_and_one_change_per_oat_case():
    definition = _definition()
    plan = Planner().plan(definition)
    baseline = _flatten(plan.cases[0].requested)

    assert len(plan.cases) == 5
    assert sum(not case.changed_paths for case in plan.cases) == 1
    for case in plan.cases[1:]:
        requested = _flatten(case.requested)
        changed = {path for path, value in requested.items() if value != baseline[path]}
        assert changed == set(case.changed_paths)
        assert len(changed) == 1

    reordered = replace(
        definition,
        experiment_id="same-plan-from-another-run",
        baseline={
            "logp": {"backend": "reference"},
            "training": {
                "compute_dtype": "fp32",
                "attention_backend": "eager",
                "sharding": "unsharded",
            },
            "rollout": {
                "enforce_eager": True,
                "enable_prefix_caching": False,
                "dtype": "fp32",
                "context_parallel_size": 1,
                "tensor_parallel_size": 1,
            },
            "batch": {"size": 8},
        },
    )
    assert [case.case_id for case in plan.cases] == [
        case.case_id for case in Planner().plan(reordered).cases
    ]


def test_pairwise_is_explicit_and_planning_errors_remain_structured():
    pairwise = _definition(
        strategy=PlanningStrategy.PAIRWISE,
        pairwise_paths=(("batch.size", "rollout.dtype"),),
    )
    pairwise_cases = [
        case for case in Planner().plan(pairwise).cases if len(case.changed_paths) == 2
    ]
    assert len(pairwise_cases) == 2
    assert all(case.changed_paths == ("batch.size", "rollout.dtype") for case in pairwise_cases)

    with pytest.raises(PlanningError) as not_enabled:
        Planner().plan(
            replace(
                _definition(),
                pairwise_paths=(("batch.size", "rollout.dtype"),),
            )
        )
    assert {issue.code for issue in not_enabled.value.issues} == {"PAIRWISE_NOT_ENABLED"}

    invalid_requests = (
        ({"logp": {"tp_layout": "arbitrary"}}, "DERIVED_KNOB"),
        ({"batch": {"size": True}}, "UNSUPPORTED_VALUE"),
        ({"rollout": {"unknown": 1}}, "UNSUPPORTED_PATH"),
    )
    for requested, expected_code in invalid_requests:
        with pytest.raises(PlanningError) as invalid:
            Planner().normalize_requested(requested)
        assert invalid.value.issues[0].code == expected_code

    incomplete = replace(
        _definition(),
        baseline={key: value for key, value in _baseline().items() if key != "training"},
    )
    with pytest.raises(PlanningError) as missing:
        Planner().plan(incomplete)
    assert {issue.path for issue in missing.value.issues} == {
        "training.attention_backend",
        "training.compute_dtype",
        "training.sharding",
    }
    assert all(issue.code == "MISSING_BASELINE_VALUE" for issue in missing.value.issues)

    oversized = replace(
        _definition(),
        interventions=(InterventionSpec("batch.size", tuple(range(1, MAX_PLAN_CASES + 2))),),
    )
    with pytest.raises(PlanningError) as too_large:
        Planner().plan(oversized)
    assert too_large.value.issues[0].code == "PLAN_TOO_LARGE"


def test_versioned_config_loads_and_binds_target_specific_operators(tmp_path: Path):
    loaded = load_config(_write_config(tmp_path, _config()))
    base_case = loaded.plan().cases[0]
    selection = loaded.operators_for(base_case)
    bound = bind_operator_selection(base_case, selection)

    assert loaded.schema_version == CONFIG_SCHEMA_VERSION
    assert loaded.definition.strategy is PlanningStrategy.ONE_AT_A_TIME
    assert selection.rollout_backend == "rlkernel.reference_logp"
    assert selection.training_backend == "smoke_only.logp_offset"
    assert selection.training_options == {"offset": 0.1}
    assert bound == bind_operator_selection(base_case, selection)
    assert bound.case_id != base_case.case_id
    assert bound.requested == base_case.requested
    assert bound.execution_binding["operators"] == selection.to_dict()


def test_config_rejects_schema_escape_hatches_and_incomplete_operator_coverage(
    tmp_path: Path,
):
    wrong_schema = _config()
    wrong_schema["schema_version"] = "cross_config.experiment_config.v999"

    unknown_key = _config()
    unknown_key["strict_falback"] = True

    threshold_override = _config()
    threshold_override["scenario"]["nested"] = {"threshold": 999.0}

    scenario_policy = _config()
    scenario_policy["scenario"]["execution"] = "run"

    conflicting_axis = _config()
    conflicting_axis["interventions"].append(
        {"path": "logp.backend", "values": ["smoke_only.logp_offset"]}
    )

    incomplete_targets = _config()
    del incomplete_targets["operators"]["selected_logprob"]["training"]

    invalid_configs = (
        (wrong_schema, "unsupported cross-configuration config schema"),
        (unknown_key, "unknown config keys"),
        (threshold_override, "fixed numerical-contract threshold"),
        (scenario_policy, "scenario is metadata only"),
        (conflicting_axis, "cannot be combined with logp.backend interventions"),
        (incomplete_targets, "selected_logprob.training"),
    )
    for index, (value, message) in enumerate(invalid_configs):
        with pytest.raises(ValueError, match=message):
            load_config(_write_config(tmp_path, value, f"invalid-{index}.json"))

    duplicate_key = tmp_path / "duplicate.json"
    duplicate_key.write_text(
        '{"schema_version":"cross_config.experiment_config.v1",'
        '"schema_version":"cross_config.experiment_config.v1"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_config(duplicate_key)

    overflow = tmp_path / "overflow.json"
    overflow.write_text(
        json.dumps(_config()).replace('"size": 1', '"size": 1e400'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite JSON number"):
        load_config(overflow)
