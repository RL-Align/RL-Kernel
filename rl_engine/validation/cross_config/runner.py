# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Control plane for paired read-only scoring runs."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from rl_engine.runtime.operators import ResolvedOperatorOverride
from rl_engine.runtime.semantic_registry import (
    OperatorInstanceProvenance,
    OperatorResolution,
    operator_implementation_fingerprint,
    operator_instance_fingerprint,
)
from rl_engine.validation.cross_config._execution import (
    ChildScoringError,
    ChildSupervisor,
    OperatorExecutionError,
    PairedRunnerError,
    PairedScorer,
    RankCompletenessError,
    RankScore,
    ScorerIdentityError,
    ScoringTimeoutError,
    canonical_fingerprint,
    device_type,
    json_safe,
    normalized_dtype,
    paired_model_state_fingerprints,
    scorer_implementation_fingerprint,
    scorer_spec,
    validate_rank_outputs,
    validate_scorer_identity,
)
from rl_engine.validation.cross_config._provenance import (
    PAIRED_RUNNER_IMPLEMENTATION_FINGERPRINT,
    concrete_scorer_spec,
    effective_runtime_status,
    execution_environment_provenance,
    execution_fingerprint,
    factory_options_fingerprint,
    mapping_target,
    runtime_adapter_fingerprint,
    score_metadata,
    side_provenance,
    target_factory_options,
)
from rl_engine.validation.cross_config._resume import completed_attempt_matches, read_json_object
from rl_engine.validation.cross_config.artifacts import ArtifactStore
from rl_engine.validation.cross_config.comparison import compare_score_artifacts
from rl_engine.validation.cross_config.runtime import (
    RuntimeMaterialization,
    RuntimeMaterializationError,
)
from rl_engine.validation.cross_config.schema import (
    AlignmentResult,
    CanonicalScoringBatch,
    ExperimentCase,
    MaterializationStatus,
    ScoreArtifact,
    ScorerSpec,
    ScoreSide,
)


@dataclass(frozen=True)
class PairedRunResult:
    """Completed attempt or a validated resume hit."""

    case_id: str
    attempt_id: str
    attempt_dir: Path
    resumed: bool
    rollout_score: Optional[ScoreArtifact] = None
    training_score: Optional[ScoreArtifact] = None
    alignment: Optional[AlignmentResult] = None
    summary: Mapping[str, Any] = field(default_factory=dict)


class PairedRunner:
    """Supervise paired scorers and publish one append-only attempt."""

    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        timeout_seconds: float = 30.0,
        start_method: Optional[str] = None,
    ):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and greater than zero")
        self.artifact_store = artifact_store
        self.timeout_seconds = float(timeout_seconds)
        self._child_supervisor = ChildSupervisor(start_method)
        self.start_method = self._child_supervisor.start_method

    @property
    def active_child_pids(self) -> tuple[int, ...]:
        return self._child_supervisor.active_child_pids

    def run(
        self,
        case: ExperimentCase,
        materialization: RuntimeMaterialization,
        batch: CanonicalScoringBatch,
        rollout_scorer: PairedScorer,
        training_scorer: PairedScorer,
        resolved_override: ResolvedOperatorOverride,
        operator_instances: Mapping[str | ScoreSide, Any],
        operator_instance_provenance: Mapping[
            str | ScoreSide,
            OperatorInstanceProvenance,
        ],
        *,
        operator_factory_options: Optional[Mapping[str | ScoreSide, Mapping[str, Any]]] = None,
        strict: bool = True,
        timeout_seconds: Optional[float] = None,
        resume: bool = True,
    ) -> PairedRunResult:
        """Run both sides against one canonical batch and persist the comparison."""

        deadline_seconds = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if not math.isfinite(deadline_seconds) or deadline_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and greater than zero")
        self._validate_case_inputs(case, materialization, batch)
        rollout_spec = scorer_spec(rollout_scorer, ScoreSide.ROLLOUT)
        training_spec = scorer_spec(training_scorer, ScoreSide.TRAINING)
        specs = {"rollout": rollout_spec, "training": training_spec}
        validate_scorer_identity(specs, batch)
        model_state_fingerprints = paired_model_state_fingerprints(
            rollout_scorer,
            training_scorer,
        )
        scorer_implementation_fingerprints = {
            "rollout": scorer_implementation_fingerprint(rollout_scorer),
            "training": scorer_implementation_fingerprint(training_scorer),
        }
        resolutions, instances, instance_provenance = _validate_exact_operators(
            materialization,
            resolved_override,
            operator_instances,
            operator_instance_provenance,
            operator_factory_options=operator_factory_options,
            specs=specs,
            strict=strict,
        )
        environment = execution_environment_provenance(
            specs,
            runtime_adapter_fingerprint=runtime_adapter_fingerprint(materialization),
            operator_implementation_fingerprints={
                target: instance_provenance[target].implementation_fingerprint
                for target in ("rollout", "training")
            },
        )
        environment_fingerprint = canonical_fingerprint(environment)
        _require_materialization_executable(materialization, strict=strict)
        current_execution_fingerprint = execution_fingerprint(
            materialization,
            specs=specs,
            instance_provenance=instance_provenance,
            operator_factory_options=operator_factory_options,
            model_state_fingerprints=model_state_fingerprints,
            scorer_implementation_fingerprints=scorer_implementation_fingerprints,
            environment=environment,
        )

        if resume:
            completed = self.artifact_store.completed_attempt(case.experiment_id, case.case_id)
            if completed is not None and completed_attempt_matches(
                completed,
                case,
                batch,
                materialization=materialization,
                specs=specs,
                instance_provenance=instance_provenance,
                operator_factory_options=operator_factory_options,
                model_state_fingerprints=model_state_fingerprints,
                scorer_implementation_fingerprints=scorer_implementation_fingerprints,
                environment=environment,
                execution_fingerprint=current_execution_fingerprint,
            ):
                summary = read_json_object(completed / "COMPLETE")
                return PairedRunResult(
                    case_id=case.case_id,
                    attempt_id=completed.name,
                    attempt_dir=completed,
                    resumed=True,
                    summary=summary,
                )

        attempt_dir = self.artifact_store.create_attempt(case.experiment_id, case.case_id)
        attempt_id = attempt_dir.name
        self._write_attempt_inputs(attempt_dir, attempt_id, case, materialization, batch)

        child_results = self._child_supervisor.run(
            attempt_dir,
            batch,
            batch_size=materialization.binding.batch_size,
            scorers={
                "rollout": rollout_scorer,
                "training": training_scorer,
            },
            specs=specs,
            instances=instances,
            timeout_seconds=float(deadline_seconds),
        )
        rollout_ranks = validate_rank_outputs(
            child_results["rollout"],
            rollout_spec,
            expected_shape=batch.input_ids.shape,
            target="rollout",
        )
        training_ranks = validate_rank_outputs(
            child_results["training"],
            training_spec,
            expected_shape=batch.input_ids.shape,
            target="training",
        )

        rollout_provenance = side_provenance(
            materialization.provenance,
            resolutions["rollout"],
            instance_provenance["rollout"],
            child_results["rollout"],
            rollout_spec,
            status=effective_runtime_status(materialization),
            factory_options=target_factory_options(operator_factory_options, "rollout"),
            model_state_fingerprint=model_state_fingerprints["rollout"],
            scorer_implementation_fingerprint=scorer_implementation_fingerprints["rollout"],
        )
        training_provenance = side_provenance(
            materialization.provenance,
            resolutions["training"],
            instance_provenance["training"],
            child_results["training"],
            training_spec,
            status=effective_runtime_status(materialization),
            factory_options=target_factory_options(operator_factory_options, "training"),
            model_state_fingerprint=model_state_fingerprints["training"],
            scorer_implementation_fingerprint=scorer_implementation_fingerprints["training"],
        )
        rollout_artifact = ScoreArtifact(
            case_id=case.case_id,
            attempt_id=attempt_id,
            side=ScoreSide.ROLLOUT,
            identity=batch.identity,
            scorer=concrete_scorer_spec(
                rollout_spec,
                instance_provenance["rollout"],
            ),
            selected_logprobs=rollout_ranks[0].selected_logprobs,
            active_mask=batch.active_mask,
            provenance=rollout_provenance,
        )
        training_artifact = ScoreArtifact(
            case_id=case.case_id,
            attempt_id=attempt_id,
            side=ScoreSide.TRAINING,
            identity=batch.identity,
            scorer=concrete_scorer_spec(
                training_spec,
                instance_provenance["training"],
            ),
            selected_logprobs=training_ranks[0].selected_logprobs,
            active_mask=batch.active_mask,
            provenance=training_provenance,
        )
        alignment = compare_score_artifacts(rollout_artifact, training_artifact)
        self._write_attempt_results(
            attempt_dir,
            rollout_artifact,
            training_artifact,
            alignment,
            execution_fingerprint=current_execution_fingerprint,
            environment=environment,
            environment_fingerprint=environment_fingerprint,
        )
        summary = {
            "schema_version": "cross_config.complete.v1",
            "case_id": case.case_id,
            "attempt_id": attempt_id,
            "status": alignment.status.value,
            "comparable": alignment.comparable,
            "passed": alignment.passed,
            "active_token_count": alignment.active_token_count,
            "mismatch_count": alignment.mismatch_count,
            "worst_token_index": alignment.diagnostics.get("worst_token_index"),
            "max_abs_diff": alignment.diagnostics.get("max_abs_diff"),
            "rollout_backend": instance_provenance["rollout"].backend_id,
            "training_backend": instance_provenance["training"].backend_id,
            "execution_fingerprint": current_execution_fingerprint,
            "environment_fingerprint": environment_fingerprint,
            "runner_implementation_fingerprint": PAIRED_RUNNER_IMPLEMENTATION_FINGERPRINT,
        }
        marker = self.artifact_store.complete_attempt(attempt_dir, summary=summary)
        summary = read_json_object(marker)
        return PairedRunResult(
            case_id=case.case_id,
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
            resumed=False,
            rollout_score=rollout_artifact,
            training_score=training_artifact,
            alignment=alignment,
            summary=summary,
        )

    @staticmethod
    def _validate_case_inputs(
        case: ExperimentCase,
        materialization: RuntimeMaterialization,
        batch: CanonicalScoringBatch,
    ) -> None:
        materialized_case = materialization.materialized_case.case
        if materialized_case != case:
            raise ValueError("materialization case does not exactly match the requested case")
        if batch.identity != case.identity:
            raise ValueError("canonical scoring batch identity does not match the case identity")
        if batch.input_ids.shape[0] < 1:
            raise ValueError("canonical scoring batch must contain at least one sequence")

    def _write_attempt_inputs(
        self,
        attempt_dir: Path,
        attempt_id: str,
        case: ExperimentCase,
        materialization: RuntimeMaterialization,
        batch: CanonicalScoringBatch,
    ) -> None:
        envelope = {"case_id": case.case_id, "attempt_id": attempt_id}
        self.artifact_store.write_json(
            attempt_dir,
            "requested",
            {**envelope, "schema_version": "cross_config.requested.v1", "case": case.to_dict()},
        )
        self.artifact_store.write_json(
            attempt_dir,
            "materialized",
            {
                **envelope,
                "schema_version": "cross_config.materialized_envelope.v1",
                "materialized_case": materialization.materialized_case.to_dict(),
            },
        )
        self.artifact_store.write_json(
            attempt_dir,
            "identity",
            {
                **envelope,
                "schema_version": "cross_config.identity_envelope.v1",
                "identity": batch.identity.to_dict(),
            },
        )

    def _write_attempt_results(
        self,
        attempt_dir: Path,
        rollout: ScoreArtifact,
        training: ScoreArtifact,
        alignment: AlignmentResult,
        *,
        execution_fingerprint: str,
        environment: Mapping[str, Any],
        environment_fingerprint: str,
    ) -> None:
        self.artifact_store.write_json(
            attempt_dir,
            "actual",
            {
                "case_id": rollout.case_id,
                "attempt_id": rollout.attempt_id,
                "schema_version": "cross_config.actual.v1",
                "execution_fingerprint": execution_fingerprint,
                "environment": environment,
                "environment_fingerprint": environment_fingerprint,
                "runner_implementation_fingerprint": PAIRED_RUNNER_IMPLEMENTATION_FINGERPRINT,
                "operator_source": "exact_resolution_and_instance",
                "rollout": rollout.provenance.to_dict(),
                "training": training.provenance.to_dict(),
            },
        )
        self.artifact_store.write_tensor_bundle(
            attempt_dir,
            "score_rollout",
            {
                "selected_logprobs": rollout.selected_logprobs,
                "active_mask": rollout.active_mask,
            },
            metadata=score_metadata(rollout),
        )
        self.artifact_store.write_tensor_bundle(
            attempt_dir,
            "score_training",
            {
                "selected_logprobs": training.selected_logprobs,
                "active_mask": training.active_mask,
            },
            metadata=score_metadata(training),
        )
        self.artifact_store.write_json(
            attempt_dir,
            "comparison",
            alignment.to_dict(),
        )
        token_artifact = alignment.token_artifact
        if token_artifact is None:
            empty = torch.empty((0,), dtype=torch.float32)
            token_tensors = {
                "rollout_logprobs": empty,
                "training_logprobs": empty,
                "active_mask": torch.empty((0,), dtype=torch.bool),
                "absolute_diff": empty,
                "mismatch_mask": torch.empty((0,), dtype=torch.bool),
            }
        else:
            token_tensors = {
                "rollout_logprobs": token_artifact.rollout_logprobs,
                "training_logprobs": token_artifact.training_logprobs,
                "active_mask": token_artifact.active_mask,
                "absolute_diff": token_artifact.absolute_diff,
                "mismatch_mask": token_artifact.mismatch_mask,
            }
        self.artifact_store.write_tensor_bundle(
            attempt_dir,
            "token_diffs",
            token_tensors,
            metadata={
                "case_id": alignment.case_id,
                "attempt_id": alignment.attempt_id,
                "status": alignment.status.value,
                "fixed_threshold": alignment.fixed_threshold,
            },
        )


def _validate_exact_operators(
    materialization: RuntimeMaterialization,
    resolved: ResolvedOperatorOverride,
    instances: Mapping[str | ScoreSide, Any],
    instance_provenance: Mapping[str | ScoreSide, OperatorInstanceProvenance],
    *,
    operator_factory_options: Optional[Mapping[str | ScoreSide, Mapping[str, Any]]],
    specs: Mapping[str, ScorerSpec],
    strict: bool,
) -> tuple[
    dict[str, OperatorResolution],
    dict[str, Any],
    dict[str, OperatorInstanceProvenance],
]:
    if resolved.semantic_op != "selected_logprob":
        raise OperatorExecutionError("PairedRunner V1 requires semantic_op='selected_logprob'")
    resolutions: dict[str, OperatorResolution] = {}
    concrete_instances: dict[str, Any] = {}
    provenance: dict[str, OperatorInstanceProvenance] = {}
    for target in ("rollout", "training"):
        resolution = resolved.for_target(target)  # type: ignore[arg-type]
        if resolution is None:
            raise OperatorExecutionError(f"missing exact {target} operator resolution")
        if resolution.target != target:
            raise OperatorExecutionError(
                f"{target} operator resolution reports target={resolution.target!r}"
            )
        if (
            resolution.descriptor.semantic_op != "selected_logprob"
            or resolution.trace.semantic_op != "selected_logprob"
        ):
            raise OperatorExecutionError(f"{target} resolution does not describe selected_logprob")
        if resolution.trace.status != "resolved" or resolution.trace.concrete_backend is None:
            raise OperatorExecutionError(
                f"{target} operator is not exactly observable: {resolution.trace.status}"
            )
        if resolution.trace.fallback_attempts:
            raise OperatorExecutionError(f"{target} operator resolution attempted fallback")
        if strict and not resolution.strict:
            raise OperatorExecutionError(f"{target} operator was not resolved in strict mode")
        if resolution.trace.concrete_backend != resolution.descriptor.backend_id:
            raise OperatorExecutionError(f"{target} resolution backend evidence is inconsistent")
        if resolution.trace.descriptor_fingerprint != resolution.descriptor.descriptor_fingerprint:
            raise OperatorExecutionError(
                f"{target} resolution descriptor fingerprint is inconsistent"
            )
        if device_type(resolution.requirements.device) != device_type(specs[target].device):
            raise OperatorExecutionError(
                f"{target} operator resolution device does not match scorer device"
            )
        if normalized_dtype(resolution.requirements.dtype) != normalized_dtype(specs[target].dtype):
            raise OperatorExecutionError(f"{target} resolution dtype does not match scorer dtype")
        _validate_exact_topology(
            materialization,
            resolution,
            specs[target],
            target=target,
        )
        instance = mapping_target(instances, target)
        if instance is None:
            raise OperatorExecutionError(f"missing instantiated {target} operator")
        _require_instance_matches_resolution(resolution, instance, target=target)
        instance_evidence = mapping_target(instance_provenance, target)
        if not isinstance(instance_evidence, OperatorInstanceProvenance):
            raise OperatorExecutionError(f"missing sealed {target} operator instance provenance")
        _validate_instance_provenance(
            resolution,
            instance,
            instance_evidence,
            factory_options=target_factory_options(operator_factory_options, target),
            target=target,
        )
        if instance_evidence.backend_id != resolution.trace.concrete_backend:
            raise OperatorExecutionError(f"{target} instance backend does not match resolution")
        declared = materialization.binding.operator_backends.get(target)
        if declared is not None and declared != instance_evidence.backend_id:
            raise OperatorExecutionError(
                f"{target} exact backend {instance_evidence.backend_id!r} does not match "
                f"declared override {declared!r}"
            )
        resolutions[target] = resolution
        concrete_instances[target] = instance
        provenance[target] = instance_evidence
    requested_logp = materialization.materialized_case.case.requested.get("logp")
    requested_backend = (
        requested_logp.get("backend") if isinstance(requested_logp, Mapping) else None
    )
    if requested_backend != provenance["rollout"].backend_id:
        raise OperatorExecutionError(
            "exact rollout operator does not match the public logp.backend request: "
            f"{provenance['rollout'].backend_id!r} != {requested_backend!r}"
        )
    return resolutions, concrete_instances, provenance


def _require_materialization_executable(
    materialization: RuntimeMaterialization,
    *,
    strict: bool,
) -> None:
    rejected = {
        MaterializationStatus.ERROR,
        MaterializationStatus.UNSUPPORTED,
        MaterializationStatus.UNOBSERVABLE,
    }
    if strict:
        rejected.add(MaterializationStatus.FALLBACK)
    if not materialization.applications and materialization.materialized_case.status in rejected:
        raise RuntimeMaterializationError(
            f"case {materialization.materialized_case.case.case_id} is not executable: "
            f"materialization={materialization.materialized_case.status.value}"
        )
    problems = []
    for application in materialization.applications:
        if (
            application.path == "logp.backend"
            and application.status is MaterializationStatus.UNOBSERVABLE
        ):
            continue
        if application.status in rejected:
            problems.append(
                f"{application.path}={application.status.value}: "
                f"{application.evidence.get('reason', 'no evidence')}"
            )
    if problems:
        raise RuntimeMaterializationError(
            f"case {materialization.materialized_case.case.case_id} is not executable: "
            + "; ".join(problems)
        )


def _validate_exact_topology(
    materialization: RuntimeMaterialization,
    resolution: OperatorResolution,
    spec: ScorerSpec,
    *,
    target: str,
) -> None:
    bound_topology = mapping_target(materialization.binding.topology, target)
    if not isinstance(bound_topology, Mapping):
        raise OperatorExecutionError(f"{target} materialized topology is missing")
    expected = dict(bound_topology)
    topology_paths = {
        "rollout": (
            ("rollout.tensor_parallel_size", "tensor_parallel_size"),
            ("rollout.context_parallel_size", "context_parallel_size"),
        ),
        "training": (("training.sharding", "sharding"),),
    }
    required_keys = {"world_size", *(key for _, key in topology_paths[target])}
    missing_keys = sorted(required_keys.difference(expected))
    if missing_keys:
        raise OperatorExecutionError(
            f"{target} materialized topology is missing required keys: {missing_keys!r}"
        )
    if expected.get("world_size") != spec.world_size:
        raise OperatorExecutionError(
            f"{target} scorer world_size does not match materialized topology"
        )
    if dict(resolution.requirements.topology) != expected:
        raise OperatorExecutionError(
            f"{target} resolution topology does not match materialized topology"
        )
    if dict(spec.topology) != expected:
        raise OperatorExecutionError(
            f"{target} scorer topology does not match materialized topology"
        )

    actual_by_path = {
        application.path: application.actual for application in materialization.applications
    }
    for path, key in topology_paths[target]:
        if actual_by_path.get(path) != expected[key]:
            raise OperatorExecutionError(
                f"{target} actual {path} does not match exact operator topology"
            )


def _require_instance_matches_resolution(
    resolution: OperatorResolution,
    instance: Any,
    *,
    target: str,
) -> None:
    implementation = resolution.descriptor.implementation_class_or_factory
    factory = implementation
    if isinstance(implementation, str):
        try:
            module_name, object_name = implementation.rsplit(".", 1)
            factory = getattr(importlib.import_module(module_name), object_name)
        except (ValueError, ImportError, AttributeError, ModuleNotFoundError) as exc:
            raise OperatorExecutionError(
                f"{target} exact operator factory cannot be verified: {exc}"
            ) from exc
    if isinstance(factory, type) and not isinstance(instance, factory):
        raise OperatorExecutionError(
            f"{target} operator instance type {type(instance).__qualname__!r} "
            f"does not match resolved factory {factory.__qualname__!r}"
        )
    if not callable(instance) and not callable(getattr(instance, "apply_fp32", None)):
        raise OperatorExecutionError(
            f"{target} selected-logprob operator instance is not executable"
        )


def _validate_instance_provenance(
    resolution: OperatorResolution,
    instance: Any,
    provenance: OperatorInstanceProvenance,
    *,
    factory_options: Mapping[str, Any],
    target: str,
) -> None:
    expected_concrete = f"{type(instance).__module__}.{type(instance).__qualname__}"
    expected_factory = resolution.descriptor.implementation_reference
    if expected_factory is None:
        raise OperatorExecutionError(f"{target} resolved operator has no factory reference")
    implementation = resolution.descriptor.implementation_class_or_factory
    if implementation is None:
        raise OperatorExecutionError(f"{target} resolved operator has no implementation")
    mismatches = []
    if provenance.semantic_op != resolution.descriptor.semantic_op:
        mismatches.append("semantic_op")
    if provenance.backend_id != resolution.descriptor.backend_id:
        mismatches.append("backend_id")
    if provenance.target != target:
        mismatches.append("target")
    if provenance.factory_reference != expected_factory:
        mismatches.append("factory_reference")
    if provenance.concrete_implementation != expected_concrete:
        mismatches.append("concrete_implementation")
    if provenance.descriptor_fingerprint != resolution.descriptor.descriptor_fingerprint:
        mismatches.append("descriptor_fingerprint")
    observed_implementation_fingerprint = operator_implementation_fingerprint(
        implementation,
        instance,
    )
    if provenance.implementation_fingerprint != observed_implementation_fingerprint:
        mismatches.append("implementation_fingerprint")

    if json_safe(provenance.factory_options) != json_safe(factory_options):
        mismatches.append("factory_options")
    if provenance.factory_options_fingerprint != factory_options_fingerprint(factory_options):
        mismatches.append("factory_options_fingerprint")
    expected_instance_fingerprint = operator_instance_fingerprint(
        descriptor_fingerprint=resolution.descriptor.descriptor_fingerprint,
        factory_reference=expected_factory,
        concrete_implementation=expected_concrete,
        implementation_fingerprint=observed_implementation_fingerprint,
        factory_options_fingerprint=factory_options_fingerprint(factory_options),
    )
    if provenance.instance_fingerprint != expected_instance_fingerprint:
        mismatches.append("instance_fingerprint")
    if mismatches:
        raise OperatorExecutionError(
            f"{target} operator instance provenance is inconsistent: " + ", ".join(mismatches)
        )


__all__ = [
    "ChildScoringError",
    "OperatorExecutionError",
    "PairedRunResult",
    "PairedRunner",
    "PairedRunnerError",
    "PairedScorer",
    "RankCompletenessError",
    "RankScore",
    "ScorerIdentityError",
    "ScoringTimeoutError",
]
