# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from rl_engine.contracts.numerical import resolve_logprob_threshold
from rl_engine.runtime.operators import OperatorBridge, OperatorOverride
from rl_engine.runtime.semantic_registry import OperatorRequirements
from rl_engine.validation.cross_config.artifacts import ArtifactError, ArtifactStore
from rl_engine.validation.cross_config.comparison import recompute_mismatch_mask
from rl_engine.validation.cross_config.config import OperatorSelection, bind_operator_selection
from rl_engine.validation.cross_config.runner import (
    ChildScoringError,
    PairedRunner,
    RankCompletenessError,
    RankScore,
    ScoringTimeoutError,
)
from rl_engine.validation.cross_config.runtime import RuntimeTools
from rl_engine.validation.cross_config.schema import (
    CanonicalScoringBatch,
    ExperimentCase,
    ScorerSpec,
    ScoreSide,
    SemanticIdentitySpec,
)
from rl_engine.validation.fixtures.cpu_cross_config import CpuSmokeMaterializer, run_cpu_case
from rl_engine.validation.fixtures.smoke_ops import (
    SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID,
    SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID,
)
from rl_engine.validation.fixtures.smoke_ops.smoke_only_logp_reference import SmokeOnlyLogpReference

_JSON_ARTIFACTS = (
    "requested.json",
    "materialized.json",
    "actual.json",
    "identity.json",
    "comparison.json",
)
_TENSOR_ARTIFACTS = (
    "score_rollout.pt",
    "score_training.pt",
    "token_diffs.pt",
)


class FixedRankScorer:
    optimizer = None
    model_state_fingerprint = "fixed-rank-scorer-state-v1"

    def __init__(self, spec: ScorerSpec, ranks):
        self.spec = spec
        self.ranks = tuple(ranks)

    def score(self, batch, *, batch_size, operator):
        del batch_size, operator
        return tuple(
            RankScore(
                rank=rank,
                world_size=self.spec.world_size,
                selected_logprobs=torch.zeros_like(batch.input_ids, dtype=torch.float32),
            )
            for rank in self.ranks
        )


class FailingScorer(FixedRankScorer):
    def score(self, batch, *, batch_size, operator):
        del batch, batch_size, operator
        raise RuntimeError("intentional scorer failure")


class SlowScorer(FixedRankScorer):
    def score(self, batch, *, batch_size, operator):
        time.sleep(2.0)
        return super().score(batch, batch_size=batch_size, operator=operator)


def _identity() -> SemanticIdentitySpec:
    token_ids = (
        (1, 2, 3, 4),
        (2, 3, 4, 5),
        (3, 4, 5, 6),
    )
    selected = (
        (0, 2, 3, 4),
        (0, 3, 4, 5),
        (0, 4, 5, 6),
    )
    active = tuple((False, True, True, True) for _ in token_ids)
    attention = tuple((True, True, True, True) for _ in token_ids)
    return SemanticIdentitySpec(
        checkpoint_id="tiny-cpu-checkpoint",
        model_version="weights-v1",
        tokenizer_policy="synthetic-tokenizer-v1",
        token_ids=token_ids,
        selected_token_ids=selected,
        active_mask=active,
        attention_mask=attention,
        pre_update_state="iteration-0",
    )


def _batch() -> CanonicalScoringBatch:
    identity = _identity()
    return CanonicalScoringBatch(
        identity=identity,
        input_ids=torch.tensor(identity.token_ids, device="cpu"),
        selected_token_ids=torch.tensor(identity.selected_token_ids, device="cpu"),
        active_mask=torch.tensor(identity.active_mask, device="cpu"),
        attention_mask=torch.tensor(identity.attention_mask, device="cpu"),
        metadata={"source": "runner-test", "device": "cpu"},
    )


def _requested(*, backend: str = "rlkernel.reference_logp") -> dict[str, object]:
    return {
        "batch": {"size": 2},
        "rollout": {
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "dtype": "float32",
            "enable_prefix_caching": False,
            "enforce_eager": True,
        },
        "training": {
            "attention_backend": "eager",
            "compute_dtype": "float32",
            "sharding": "unsharded",
        },
        "logp": {"backend": backend},
    }


def _case(
    *,
    case_id: str = "case-runner",
    backend: str = "rlkernel.reference_logp",
) -> ExperimentCase:
    return ExperimentCase(
        case_id=case_id,
        experiment_id="runner-test",
        scenario_id="S0",
        identity=_identity(),
        requested=_requested(backend=backend),
        contract_fingerprint="contract-sha",
        scenario_fingerprint="scenario-sha",
    )


def _topology(side: ScoreSide) -> dict[str, object]:
    if side is ScoreSide.ROLLOUT:
        return {
            "world_size": 1,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        }
    return {"world_size": 1, "sharding": "unsharded"}


def _requirements(side: ScoreSide) -> OperatorRequirements:
    return OperatorRequirements(
        device="cpu",
        dtype="float32",
        topology=_topology(side),
        alignment_properties={"deterministic": True},
    )


def _operators():
    bridge = OperatorBridge()
    resolved = bridge.resolve_override(
        OperatorOverride.for_target(
            semantic_op="selected_logprob",
            backend_id="rlkernel.reference_logp",
            target="both",
        ),
        requirements={
            "rollout": _requirements(ScoreSide.ROLLOUT),
            "training": _requirements(ScoreSide.TRAINING),
        },
        strict=True,
    )
    instances = {
        target: bridge.instantiate(resolved, target=target) for target in ("rollout", "training")
    }
    provenance = {
        target: bridge.instance_provenance(
            resolved,
            target=target,
            instance=instances[target],
        )
        for target in ("rollout", "training")
    }
    return resolved, instances, provenance


def _materialization(case: ExperimentCase):
    backend = str(case.requested["logp"]["backend"])
    backends = {"rollout": backend, "training": backend}
    return RuntimeTools().materialize(
        case,
        CpuSmokeMaterializer(
            requested_operator_backends=backends,
            actual_operator_backends=backends,
        ),
    )


def _spec(side: ScoreSide) -> ScorerSpec:
    identity = _identity()
    return ScorerSpec(
        side=side,
        backend_id="fixed_cpu_teacher_forcing",
        dtype="float32",
        device="cpu",
        world_size=1,
        topology=_topology(side),
        construction_options={
            "checkpoint_id": identity.checkpoint_id,
            "model_version": identity.model_version,
            "pre_update_state": identity.pre_update_state,
            "teacher_forcing": True,
            "use_cache": False,
        },
        operator_overrides={"selected_logprob": "rlkernel.reference_logp"},
    )


def _bound_smoke_case(scenario: str) -> tuple[ExperimentCase, OperatorSelection]:
    threshold_offset = resolve_logprob_threshold("float32") * 4.0
    if scenario == "reference-reference":
        rollout_backend = SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID
        training_backend = SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID
        rollout_options = {}
        training_options = {}
    elif scenario == "reference-offset":
        rollout_backend = SMOKE_ONLY_LOGP_REFERENCE_BACKEND_ID
        training_backend = SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID
        rollout_options = {}
        training_options = {"offset": threshold_offset}
    elif scenario == "offset-offset":
        rollout_backend = SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID
        training_backend = SMOKE_ONLY_LOGP_OFFSET_BACKEND_ID
        rollout_options = {"offset": threshold_offset}
        training_options = {"offset": threshold_offset}
    else:  # pragma: no cover - test helper contract
        raise ValueError(f"unknown scenario: {scenario}")
    selection = OperatorSelection(
        rollout_backend=rollout_backend,
        training_backend=training_backend,
        rollout_options=rollout_options,
        training_options=training_options,
    )
    case = bind_operator_selection(
        _case(case_id=f"case-{scenario}", backend=rollout_backend),
        selection,
    )
    return case, selection


def _write_required_artifacts(
    store: ArtifactStore,
    attempt_dir: Path,
    *,
    case_id: str = "case-1",
    omit: frozenset[str] = frozenset(),
    rollout_logprobs: torch.Tensor | None = None,
    training_logprobs: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
    threshold: float = 0.05,
) -> None:
    attempt_id = attempt_dir.name
    json_values = {
        "requested.json": {
            "schema_version": "cross_config.requested.v1",
            "case_id": case_id,
            "attempt_id": attempt_id,
            "case": {"case_id": case_id},
        },
        "materialized.json": {
            "schema_version": "cross_config.materialized_envelope.v1",
            "case_id": case_id,
            "attempt_id": attempt_id,
            "materialized_case": {"case": {"case_id": case_id}},
        },
        "actual.json": {
            "schema_version": "cross_config.actual.v1",
            "case_id": case_id,
            "attempt_id": attempt_id,
            "rollout": {},
            "training": {},
        },
        "identity.json": {
            "schema_version": "cross_config.identity_envelope.v1",
            "case_id": case_id,
            "attempt_id": attempt_id,
            "identity": {"checkpoint_id": "tiny"},
        },
        "comparison.json": {
            "schema_version": "cross_config.alignment_result.v1",
            "case_id": case_id,
            "attempt_id": attempt_id,
            "status": "pass",
            "comparable": True,
            "passed": True,
        },
    }
    for name, value in json_values.items():
        if name not in omit:
            store.write_json(attempt_dir, name, value)

    rollout = rollout_logprobs if rollout_logprobs is not None else torch.tensor([-1.0, -2.0, -3.0])
    training = training_logprobs if training_logprobs is not None else rollout.clone()
    active = active_mask if active_mask is not None else torch.tensor([True, True, True])
    mismatch = recompute_mismatch_mask(rollout, training, active, threshold)
    tensor_values = {
        "score_rollout.pt": {
            "selected_logprobs": rollout,
            "active_mask": active,
        },
        "score_training.pt": {
            "selected_logprobs": training,
            "active_mask": active,
        },
        "token_diffs.pt": {
            "rollout_logprobs": rollout,
            "training_logprobs": training,
            "active_mask": active,
            "absolute_diff": torch.abs(training - rollout),
            "mismatch_mask": mismatch,
        },
    }
    for name, tensors in tensor_values.items():
        if name not in omit:
            store.write_tensor_bundle(
                attempt_dir,
                name,
                tensors,
                metadata={
                    "case_id": case_id,
                    "attempt_id": attempt_id,
                    "artifact": name,
                    "fixed_threshold": threshold,
                },
            )


def _complete_attempt(
    store: ArtifactStore,
    *,
    experiment_id: str = "experiment-1",
    case_id: str = "case-1",
    **artifact_options,
) -> Path:
    attempt_dir = store.create_attempt(experiment_id, case_id)
    _write_required_artifacts(store, attempt_dir, case_id=case_id, **artifact_options)
    store.complete_attempt(
        attempt_dir,
        summary={
            "schema_version": "cross_config.complete.v1",
            "case_id": case_id,
            "attempt_id": attempt_dir.name,
            "status": "pass",
        },
    )
    return attempt_dir


@pytest.mark.smoke_operator
def test_cpu_smoke_cases_preserve_read_only_scoring_and_exact_provenance(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    batch = _batch()
    inputs_before = batch.input_ids.clone()
    expected = {
        "reference-reference": (True, 0),
        "reference-offset": (False, int(batch.active_mask.sum().item())),
        "offset-offset": (True, 0),
    }

    for scenario, (expected_pass, expected_mismatches) in expected.items():
        case, selection = _bound_smoke_case(scenario)
        result = run_cpu_case(
            store,
            case,
            batch,
            selection,
            allow_smoke_operators=True,
            strict=True,
            timeout_seconds=5.0,
            resume=False,
        )

        assert result.resumed is False
        assert result.alignment is not None
        assert result.alignment.passed is expected_pass
        assert result.alignment.mismatch_count == expected_mismatches
        assert result.rollout_score is not None
        assert result.training_score is not None
        assert result.rollout_score.selected_logprobs.device.type == "cpu"
        assert result.training_score.selected_logprobs.device.type == "cpu"
        assert result.rollout_score.scorer.device == "cpu"
        assert result.training_score.scorer.device == "cpu"

        guard = result.training_score.provenance.evidence["scoring_guard"]
        assert guard == {
            "model_state_verified": True,
            "model_eval": True,
            "no_grad": True,
            "optimizer_step": False,
            "model_modes_restored": True,
            "model_state_unchanged": True,
        }
        assert result.rollout_score.provenance.evidence["scoring_guard"] == guard
        assert result.training_score.provenance.evidence["rank_metadata"][0]["batch_ranges"] == (
            (0, 2),
            (2, 3),
        )
        rollout_state = result.rollout_score.provenance.evidence["model_state_fingerprint"]
        training_state = result.training_score.provenance.evidence["model_state_fingerprint"]
        assert rollout_state == training_state

        actual = json.loads((result.attempt_dir / "actual.json").read_text(encoding="utf-8"))
        assert actual["operator_source"] == "exact_resolution_and_instance"
        for target, backend in (
            ("rollout", selection.rollout_backend),
            ("training", selection.training_backend),
        ):
            operator = actual[target]["actual"]["operators"]["selected_logprob"]
            assert operator["backend_id"] == backend
            assert operator["descriptor_fingerprint"]
            assert operator["implementation_fingerprint"]
            assert operator["instance_fingerprint"]
        complete = result.attempt_dir / "COMPLETE"
        assert complete.is_file()
        assert result.summary == json.loads(complete.read_text(encoding="utf-8"))

    assert torch.equal(batch.input_ids, inputs_before)


@pytest.mark.smoke_operator
def test_runner_resumes_valid_attempt_and_retries_after_identity_or_tensor_change(
    tmp_path: Path,
    monkeypatch,
):
    store = ArtifactStore(tmp_path)
    case, selection = _bound_smoke_case("reference-reference")
    batch = _batch()

    def run():
        return run_cpu_case(
            store,
            case,
            batch,
            selection,
            allow_smoke_operators=True,
            strict=True,
            timeout_seconds=5.0,
            resume=True,
        )

    first = run()
    resumed = run()
    assert first.attempt_id == "attempt-0001"
    assert resumed.resumed is True
    assert resumed.attempt_id == first.attempt_id
    assert resumed.rollout_score is None
    assert resumed.summary == first.summary

    token_path = first.attempt_dir / "token_diffs.pt"
    payload = torch.load(token_path, map_location="cpu", weights_only=True)
    payload["tensors"]["mismatch_mask"] = torch.ones_like(payload["tensors"]["mismatch_mask"])
    torch.save(payload, token_path)

    retried = run()
    assert retried.resumed is False
    assert retried.attempt_id == "attempt-0002"
    assert (retried.attempt_dir / "COMPLETE").is_file()

    original_apply_fp32 = SmokeOnlyLogpReference.apply_fp32

    def equivalent_apply_fp32(self, logits, token_ids, active_mask=None):
        return original_apply_fp32(self, logits, token_ids, active_mask=active_mask)

    monkeypatch.setattr(SmokeOnlyLogpReference, "apply_fp32", equivalent_apply_fp32)
    implementation_changed = run()
    assert implementation_changed.resumed is False
    assert implementation_changed.attempt_id == "attempt-0003"
    before = json.loads((retried.attempt_dir / "actual.json").read_text(encoding="utf-8"))
    after = json.loads(
        (implementation_changed.attempt_dir / "actual.json").read_text(encoding="utf-8")
    )
    assert (
        before["rollout"]["actual"]["operators"]["selected_logprob"]["implementation_fingerprint"]
        != after["rollout"]["actual"]["operators"]["selected_logprob"]["implementation_fingerprint"]
    )
    attempts = sorted(path.name for path in retried.attempt_dir.parent.iterdir())
    assert attempts == ["attempt-0001", "attempt-0002", "attempt-0003"]


@pytest.mark.parametrize(
    ("mode", "error_type", "message"),
    [
        ("failure", ChildScoringError, "intentional scorer failure"),
        ("timeout", ScoringTimeoutError, "stopped children"),
        ("missing-rank", RankCompletenessError, r"missing=\[0\]"),
        ("duplicate-rank", RankCompletenessError, "duplicate ranks"),
    ],
)
def test_runner_supervision_fails_closed_and_cleans_children(
    tmp_path: Path,
    mode: str,
    error_type: type[Exception],
    message: str,
):
    case = _case(case_id=f"case-{mode}")
    resolved, instances, provenance = _operators()
    if mode == "failure":
        rollout = FailingScorer(_spec(ScoreSide.ROLLOUT), ())
        training = FixedRankScorer(_spec(ScoreSide.TRAINING), (0,))
        timeout = 5.0
    elif mode == "timeout":
        rollout = SlowScorer(_spec(ScoreSide.ROLLOUT), (0,))
        training = SlowScorer(_spec(ScoreSide.TRAINING), (0,))
        timeout = 0.1
    elif mode == "missing-rank":
        rollout = FixedRankScorer(_spec(ScoreSide.ROLLOUT), ())
        training = FixedRankScorer(_spec(ScoreSide.TRAINING), (0,))
        timeout = 5.0
    else:
        rollout = FixedRankScorer(_spec(ScoreSide.ROLLOUT), (0, 0))
        training = FixedRankScorer(_spec(ScoreSide.TRAINING), (0,))
        timeout = 5.0

    runner = PairedRunner(ArtifactStore(tmp_path), timeout_seconds=timeout)
    with pytest.raises(error_type, match=message):
        runner.run(
            case,
            _materialization(case),
            _batch(),
            rollout,
            training,
            resolved,
            instances,
            provenance,
            timeout_seconds=timeout,
        )

    assert runner.active_child_pids == ()
    attempt_dir = tmp_path / case.experiment_id / "cases" / case.case_id / "attempt-0001"
    assert attempt_dir.is_dir()
    assert not (attempt_dir / "COMPLETE").exists()
    assert not list(attempt_dir.glob(".paired-runner-*"))


def test_artifacts_are_append_only_and_complete_marker_is_published_last(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    attempt_dir = store.create_attempt("experiment-1", "case-1")
    _write_required_artifacts(
        store,
        attempt_dir,
        omit=frozenset({"token_diffs.pt"}),
    )
    requested_path = attempt_dir / "requested.json"
    rollout_path = attempt_dir / "score_rollout.pt"
    requested_before = requested_path.read_bytes()
    rollout_before = rollout_path.read_bytes()

    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        store.write_json(attempt_dir, "requested", {"case_id": "changed"})
    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        store.write_tensor_bundle(
            attempt_dir,
            "score_rollout",
            {"selected_logprobs": torch.tensor([0.0])},
        )
    assert requested_path.read_bytes() == requested_before
    assert rollout_path.read_bytes() == rollout_before

    summary = {
        "schema_version": "cross_config.complete.v1",
        "case_id": "case-1",
        "attempt_id": attempt_dir.name,
        "status": "pass",
    }
    with pytest.raises(ArtifactError, match=r"missing artifacts.*token_diffs\.pt"):
        store.complete_attempt(attempt_dir, summary=summary)
    assert not (attempt_dir / "COMPLETE").exists()

    _write_required_artifacts(
        store,
        attempt_dir,
        omit=frozenset(_JSON_ARTIFACTS + _TENSOR_ARTIFACTS[:-1]),
    )
    marker = store.complete_attempt(attempt_dir, summary=summary)
    store.validate_completed_attempt(attempt_dir, expected_case_id="case-1")
    payload_times = [
        (attempt_dir / name).stat().st_mtime_ns for name in _JSON_ARTIFACTS + _TENSOR_ARTIFACTS
    ]
    assert marker.stat().st_mtime_ns >= max(payload_times)
    marker_value = json.loads(marker.read_text(encoding="utf-8"))
    artifact_hashes = marker_value.pop("artifact_sha256")
    assert marker_value == summary
    assert set(artifact_hashes) == set(_JSON_ARTIFACTS + _TENSOR_ARTIFACTS)
    assert all(len(value) == 64 for value in artifact_hashes.values())
    assert not list(attempt_dir.glob(".COMPLETE.*"))
    with pytest.raises(ArtifactError, match="refusing to overwrite"):
        store.complete_attempt(attempt_dir, summary=summary)

    next_attempt = store.create_attempt("experiment-1", "case-1")
    assert next_attempt.name == "attempt-0002"


def test_resume_uses_newest_valid_attempt_and_tensors_support_offline_recompute(tmp_path: Path):
    store = ArtifactStore(tmp_path)
    rollout = torch.tensor([-1.0, -2.0, -3.0])
    training = torch.tensor([-1.01, -2.20, -2.50])
    active = torch.tensor([True, True, False])
    older = _complete_attempt(
        store,
        rollout_logprobs=rollout,
        training_logprobs=training,
        active_mask=active,
        threshold=0.05,
    )
    newer = _complete_attempt(store)
    partial = store.create_attempt("experiment-1", "case-1")
    store.write_json(partial, "requested", {"case_id": "case-1"})

    assert store.completed_attempt("experiment-1", "case-1") == newer
    (newer / "COMPLETE").write_text("{not-json", encoding="utf-8")
    assert store.completed_attempt("experiment-1", "case-1") == older

    token_payload = store.load_tensor_bundle(older / "token_diffs.pt")
    tensors = token_payload["tensors"]
    recomputed = recompute_mismatch_mask(
        tensors["rollout_logprobs"],
        tensors["training_logprobs"],
        tensors["active_mask"],
        token_payload["metadata"]["fixed_threshold"],
    )
    assert torch.equal(recomputed, tensors["mismatch_mask"])
    assert torch.equal(recomputed, torch.tensor([False, True, False]))
    assert all(tensor.device.type == "cpu" for tensor in tensors.values())
    assert partial.name == "attempt-0003"

    materialized_path = older / "materialized.json"
    materialized = json.loads(materialized_path.read_text(encoding="utf-8"))
    materialized["materialized_case"]["case"]["case_id"] = "tampered"
    materialized_path.write_text(json.dumps(materialized), encoding="utf-8")
    assert store.completed_attempt("experiment-1", "case-1") is None
