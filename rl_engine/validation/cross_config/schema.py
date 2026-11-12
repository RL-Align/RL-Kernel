# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Stable, versioned domain schema for cross-configuration alignment."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

import torch


class ScoreSide(str, Enum):
    ROLLOUT = "rollout"
    TRAINING = "training"


class IsolationScope(str, Enum):
    REQUEST = "request"
    ENGINE_CONSTRUCTION = "engine_construction"
    DISTRIBUTED_CONTEXT = "distributed_context"
    PROCESS = "process"


class PlanningStrategy(str, Enum):
    ONE_AT_A_TIME = "one_at_a_time"
    PAIRWISE = "pairwise"


class MaterializationStatus(str, Enum):
    UNSUPPORTED = "unsupported"
    APPLIED = "applied"
    FALLBACK = "fallback"
    UNOBSERVABLE = "unobservable"
    ERROR = "error"


class AlignmentStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INVALID_IDENTITY = "invalid_identity"
    INVALID_ARTIFACT = "invalid_artifact"
    ZERO_ACTIVE_TOKENS = "zero_active_tokens"


class SerializableModel:
    """Mixin providing a stable JSON-compatible representation."""

    def to_dict(self) -> dict[str, Any]:
        return {
            item.name: _serialize_value(getattr(self, item.name))
            for item in fields(self)  # type: ignore[arg-type]
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, SerializableModel):
        return value.to_dict()
    if isinstance(value, torch.Tensor):
        snapshot = value.detach().cpu()
        return {
            "dtype": str(snapshot.dtype).replace("torch.", ""),
            "shape": list(snapshot.shape),
            "values": snapshot.tolist(),
        }
    if isinstance(value, Mapping):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_serialize_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    return value


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze_value(value)


def _coerce_enum(value: Any, enum_type: type[Enum]) -> Enum:
    if isinstance(value, enum_type):
        return value
    return enum_type(value)


def _int_matrix(value: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for row in value:
        normalized: list[int] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("integer identity matrices accept JSON integers only")
            normalized.append(item)
        rows.append(tuple(normalized))
    return tuple(rows)


def _bool_matrix(value: Sequence[Sequence[bool]]) -> tuple[tuple[bool, ...], ...]:
    rows: list[tuple[bool, ...]] = []
    for row in value:
        normalized: list[bool] = []
        for item in row:
            if not isinstance(item, bool):
                raise ValueError("boolean identity matrices accept JSON booleans only")
            normalized.append(item)
        rows.append(tuple(normalized))
    return tuple(rows)


def _validate_rectangular(name: str, value: tuple[tuple[Any, ...], ...]) -> None:
    if not value:
        return
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValueError(f"{name} must be rectangular")


def _validate_same_matrix_shape(
    left_name: str,
    left: tuple[tuple[Any, ...], ...],
    right_name: str,
    right: tuple[tuple[Any, ...], ...],
) -> None:
    if left and right and (len(left), len(left[0])) != (len(right), len(right[0])):
        raise ValueError(f"{left_name} shape must match {right_name} shape")


def _snapshot_tensor(value: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(value)!r}")
    snapshot = value.detach().clone()
    return snapshot.to(dtype=dtype) if dtype is not None else snapshot


@dataclass(frozen=True)
class SemanticIdentitySpec(SerializableModel):
    """Logical inputs that must match before numerical comparison is meaningful."""

    checkpoint_id: str
    model_version: str
    tokenizer_policy: str
    token_ids: tuple[tuple[int, ...], ...]
    selected_token_ids: tuple[tuple[int, ...], ...]
    active_mask: tuple[tuple[bool, ...], ...]
    pre_update_state: str
    tokenizer_id: str = ""
    attention_mask: tuple[tuple[bool, ...], ...] = ()
    position_ids: tuple[tuple[int, ...], ...] = ()
    cache_metadata: Mapping[str, Any] = field(default_factory=dict)
    packing_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "cross_config.semantic_identity.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cross_config.semantic_identity.v1":
            raise ValueError("unsupported SemanticIdentitySpec schema_version")
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must not be empty")
        if not self.model_version:
            raise ValueError("model_version must not be empty")
        if not self.tokenizer_policy:
            raise ValueError("tokenizer_policy must not be empty")
        if not self.pre_update_state:
            raise ValueError("pre_update_state must not be empty")

        object.__setattr__(self, "token_ids", _int_matrix(self.token_ids))
        object.__setattr__(self, "selected_token_ids", _int_matrix(self.selected_token_ids))
        object.__setattr__(self, "active_mask", _bool_matrix(self.active_mask))
        object.__setattr__(self, "attention_mask", _bool_matrix(self.attention_mask))
        object.__setattr__(self, "position_ids", _int_matrix(self.position_ids))
        object.__setattr__(self, "cache_metadata", _freeze_mapping(self.cache_metadata))
        object.__setattr__(self, "packing_metadata", _freeze_mapping(self.packing_metadata))

        if not self.token_ids or not self.token_ids[0]:
            raise ValueError("token_ids must contain at least one token")
        if not self.selected_token_ids:
            raise ValueError("selected_token_ids must not be empty")
        if not self.active_mask:
            raise ValueError("active_mask must not be empty")
        if not self.attention_mask:
            raise ValueError("attention_mask must not be empty")
        for name in (
            "token_ids",
            "selected_token_ids",
            "active_mask",
            "attention_mask",
            "position_ids",
        ):
            _validate_rectangular(name, getattr(self, name))
        _validate_same_matrix_shape(
            "token_ids",
            self.token_ids,
            "selected_token_ids",
            self.selected_token_ids,
        )
        _validate_same_matrix_shape(
            "selected_token_ids", self.selected_token_ids, "active_mask", self.active_mask
        )
        _validate_same_matrix_shape(
            "token_ids", self.token_ids, "attention_mask", self.attention_mask
        )
        _validate_same_matrix_shape("token_ids", self.token_ids, "position_ids", self.position_ids)


@dataclass(frozen=True)
class ScorerSpec(SerializableModel):
    side: ScoreSide
    backend_id: str
    dtype: str
    device: str = "cpu"
    world_size: int = 1
    topology: Mapping[str, Any] = field(default_factory=dict)
    construction_options: Mapping[str, Any] = field(default_factory=dict)
    operator_overrides: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = "cross_config.scorer.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _coerce_enum(self.side, ScoreSide))
        if not self.backend_id:
            raise ValueError("backend_id must not be empty")
        if not self.dtype:
            raise ValueError("dtype must not be empty")
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        object.__setattr__(self, "topology", _freeze_mapping(self.topology))
        object.__setattr__(self, "construction_options", _freeze_mapping(self.construction_options))
        object.__setattr__(self, "operator_overrides", _freeze_mapping(self.operator_overrides))


@dataclass(frozen=True)
class KnobDescriptor(SerializableModel):
    path: str
    lifecycle: IsolationScope
    targets: tuple[str, ...]
    allowed_values: tuple[Any, ...] = ()
    derived: bool = False
    critical: bool = True
    schema_version: str = "cross_config.knob_descriptor.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "lifecycle", _coerce_enum(self.lifecycle, IsolationScope))
        object.__setattr__(self, "targets", tuple(str(target) for target in self.targets))
        object.__setattr__(self, "allowed_values", tuple(_freeze_value(self.allowed_values)))
        if not self.path:
            raise ValueError("path must not be empty")
        if not self.targets:
            raise ValueError("targets must not be empty")


@dataclass(frozen=True)
class InterventionSpec(SerializableModel):
    path: str
    values: tuple[Any, ...]
    schema_version: str = "cross_config.intervention.v1"

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path must not be empty")
        object.__setattr__(self, "values", tuple(_freeze_value(self.values)))
        if not self.values:
            raise ValueError("values must not be empty")


@dataclass(frozen=True)
class ExperimentDefinition(SerializableModel):
    experiment_id: str
    scenario_id: str
    identity: SemanticIdentitySpec
    baseline: Mapping[str, Any]
    interventions: tuple[InterventionSpec, ...] = ()
    scenario: Mapping[str, Any] = field(default_factory=dict)
    strategy: PlanningStrategy = PlanningStrategy.ONE_AT_A_TIME
    strict_fallback: bool = True
    pairwise_paths: tuple[tuple[str, str], ...] = ()
    contract_source: str = "ws1"
    contract_version: str = "current"
    schema_version: str = "cross_config.experiment_definition.v1"

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty")
        if self.contract_source != "ws1":
            raise ValueError("Cross-configuration alignment V1 requires contract_source='ws1'")
        if self.contract_version != "current":
            raise ValueError("Cross-configuration alignment V1 requires contract_version='current'")
        object.__setattr__(self, "baseline", _freeze_mapping(self.baseline))
        object.__setattr__(self, "scenario", _freeze_mapping(self.scenario))
        object.__setattr__(self, "interventions", tuple(self.interventions))
        object.__setattr__(self, "strategy", _coerce_enum(self.strategy, PlanningStrategy))
        normalized_pairs: list[tuple[str, str]] = []
        for pair in self.pairwise_paths:
            if len(pair) != 2:
                raise ValueError("each pairwise_paths entry must contain exactly two paths")
            normalized_pairs.append((str(pair[0]), str(pair[1])))
        object.__setattr__(self, "pairwise_paths", tuple(normalized_pairs))


@dataclass(frozen=True)
class ExperimentCase(SerializableModel):
    case_id: str
    experiment_id: str
    scenario_id: str
    identity: SemanticIdentitySpec
    requested: Mapping[str, Any]
    execution_binding: Mapping[str, Any] = field(default_factory=dict)
    changed_paths: tuple[str, ...] = ()
    contract_fingerprint: str = ""
    scenario_fingerprint: str = ""
    schema_version: str = "cross_config.experiment_case.v1"

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if not self.scenario_id:
            raise ValueError("scenario_id must not be empty")
        object.__setattr__(self, "requested", _freeze_mapping(self.requested))
        object.__setattr__(self, "execution_binding", _freeze_mapping(self.execution_binding))
        object.__setattr__(self, "changed_paths", tuple(str(path) for path in self.changed_paths))


@dataclass(frozen=True)
class MaterializedCase(SerializableModel):
    case: ExperimentCase
    normalized: Mapping[str, Any]
    materialized: Mapping[str, Any]
    isolation_scope: IsolationScope
    construction_fingerprint: str = ""
    distributed_context_fingerprint: str = ""
    process_fingerprint: str = ""
    status: MaterializationStatus = MaterializationStatus.APPLIED
    evidence: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "cross_config.materialized_case.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalized", _freeze_mapping(self.normalized))
        object.__setattr__(self, "materialized", _freeze_mapping(self.materialized))
        object.__setattr__(
            self,
            "isolation_scope",
            _coerce_enum(self.isolation_scope, IsolationScope),
        )
        object.__setattr__(self, "status", _coerce_enum(self.status, MaterializationStatus))
        object.__setattr__(self, "evidence", _freeze_mapping(self.evidence))


@dataclass(frozen=True)
class CanonicalScoringBatch(SerializableModel):
    identity: SemanticIdentitySpec
    input_ids: torch.Tensor
    selected_token_ids: torch.Tensor
    active_mask: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: Optional[torch.Tensor] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "cross_config.canonical_scoring_batch.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_ids", _snapshot_tensor(self.input_ids, dtype=torch.long))
        object.__setattr__(
            self, "selected_token_ids", _snapshot_tensor(self.selected_token_ids, dtype=torch.long)
        )
        object.__setattr__(
            self, "active_mask", _snapshot_tensor(self.active_mask, dtype=torch.bool)
        )
        object.__setattr__(
            self, "attention_mask", _snapshot_tensor(self.attention_mask, dtype=torch.bool)
        )
        if self.position_ids is not None:
            object.__setattr__(
                self, "position_ids", _snapshot_tensor(self.position_ids, dtype=torch.long)
            )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

        shape = self.input_ids.shape
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        for name in ("selected_token_ids", "active_mask", "attention_mask"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape must match input_ids shape")
        if self.position_ids is not None and self.position_ids.shape != shape:
            raise ValueError("position_ids shape must match input_ids shape")

        _require_tensor_matches_matrix("input_ids", self.input_ids, self.identity.token_ids)
        _require_tensor_matches_matrix(
            "selected_token_ids", self.selected_token_ids, self.identity.selected_token_ids
        )
        _require_tensor_matches_matrix("active_mask", self.active_mask, self.identity.active_mask)
        _require_tensor_matches_matrix(
            "attention_mask", self.attention_mask, self.identity.attention_mask
        )
        if self.identity.position_ids:
            if self.position_ids is None:
                raise ValueError("position_ids are required by the semantic identity")
            _require_tensor_matches_matrix(
                "position_ids", self.position_ids, self.identity.position_ids
            )
        elif self.position_ids is not None:
            raise ValueError("position_ids were supplied but are absent from semantic identity")


def _require_tensor_matches_matrix(
    name: str,
    tensor: torch.Tensor,
    matrix: tuple[tuple[Any, ...], ...],
) -> None:
    expected = torch.tensor(matrix, dtype=tensor.dtype, device=tensor.device)
    if expected.shape != tensor.shape or not torch.equal(tensor, expected):
        raise ValueError(f"{name} does not match the semantic identity")


@dataclass(frozen=True)
class RuntimeProvenance(SerializableModel):
    requested: Mapping[str, Any]
    normalized: Mapping[str, Any]
    materialized: Mapping[str, Any]
    actual: Mapping[str, Any]
    status: MaterializationStatus = MaterializationStatus.APPLIED
    construction_fingerprint: str = ""
    distributed_context_fingerprint: str = ""
    process_fingerprint: str = ""
    implementation_fingerprint: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    rank: int = 0
    world_size: int = 1
    schema_version: str = "cross_config.runtime_provenance.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested", _freeze_mapping(self.requested))
        object.__setattr__(self, "normalized", _freeze_mapping(self.normalized))
        object.__setattr__(self, "materialized", _freeze_mapping(self.materialized))
        object.__setattr__(self, "actual", _freeze_mapping(self.actual))
        object.__setattr__(self, "status", _coerce_enum(self.status, MaterializationStatus))
        object.__setattr__(self, "evidence", _freeze_mapping(self.evidence))
        if self.rank < 0:
            raise ValueError("rank must be >= 0")
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if self.rank >= self.world_size:
            raise ValueError("rank must be less than world_size")


@dataclass(frozen=True)
class ScoreArtifact(SerializableModel):
    case_id: str
    attempt_id: str
    side: ScoreSide
    identity: SemanticIdentitySpec
    scorer: ScorerSpec
    selected_logprobs: torch.Tensor
    active_mask: torch.Tensor
    provenance: RuntimeProvenance
    schema_version: str = "cross_config.score_artifact.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _coerce_enum(self.side, ScoreSide))
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not self.attempt_id:
            raise ValueError("attempt_id must not be empty")
        if self.scorer.side is not self.side:
            raise ValueError("scorer side must match score artifact side")
        object.__setattr__(self, "selected_logprobs", _snapshot_tensor(self.selected_logprobs))
        object.__setattr__(
            self, "active_mask", _snapshot_tensor(self.active_mask, dtype=torch.bool)
        )
        if self.selected_logprobs.shape != self.active_mask.shape:
            raise ValueError("selected_logprobs shape must match active_mask shape")


@dataclass(frozen=True)
class TokenComparisonArtifact(SerializableModel):
    rollout_logprobs: torch.Tensor
    training_logprobs: torch.Tensor
    active_mask: torch.Tensor
    absolute_diff: torch.Tensor
    mismatch_mask: torch.Tensor
    fixed_threshold: float
    schema_version: str = "cross_config.token_comparison.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rollout_logprobs", _snapshot_tensor(self.rollout_logprobs))
        object.__setattr__(self, "training_logprobs", _snapshot_tensor(self.training_logprobs))
        object.__setattr__(
            self, "active_mask", _snapshot_tensor(self.active_mask, dtype=torch.bool)
        )
        object.__setattr__(self, "absolute_diff", _snapshot_tensor(self.absolute_diff))
        object.__setattr__(
            self, "mismatch_mask", _snapshot_tensor(self.mismatch_mask, dtype=torch.bool)
        )
        shape = self.rollout_logprobs.shape
        for name in (
            "training_logprobs",
            "active_mask",
            "absolute_diff",
            "mismatch_mask",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape must match rollout_logprobs shape")
        if not math.isfinite(self.fixed_threshold) or self.fixed_threshold < 0.0:
            raise ValueError("fixed_threshold must be finite and non-negative")


@dataclass(frozen=True)
class AlignmentResult(SerializableModel):
    case_id: str
    attempt_id: str
    status: AlignmentStatus
    comparable: bool
    passed: bool
    active_token_count: int
    mismatch_count: int
    contract_fingerprint: str
    fixed_threshold: Optional[float] = None
    identity_errors: tuple[str, ...] = ()
    artifact_errors: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    token_artifact: Optional[TokenComparisonArtifact] = None
    schema_version: str = "cross_config.alignment_result.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _coerce_enum(self.status, AlignmentStatus))
        object.__setattr__(self, "identity_errors", tuple(self.identity_errors))
        object.__setattr__(self, "artifact_errors", tuple(self.artifact_errors))
        object.__setattr__(self, "diagnostics", _freeze_mapping(self.diagnostics))
        if self.active_token_count < 0:
            raise ValueError("active_token_count must be non-negative")
        if self.mismatch_count < 0:
            raise ValueError("mismatch_count must be non-negative")
        if self.mismatch_count > self.active_token_count:
            raise ValueError("mismatch_count cannot exceed active_token_count")
        if self.status is AlignmentStatus.PASS and not self.passed:
            raise ValueError("PASS result must set passed=True")
        if self.status is not AlignmentStatus.PASS and self.passed:
            raise ValueError("only PASS results may set passed=True")


__all__ = [
    "AlignmentResult",
    "AlignmentStatus",
    "CanonicalScoringBatch",
    "ExperimentCase",
    "ExperimentDefinition",
    "InterventionSpec",
    "IsolationScope",
    "KnobDescriptor",
    "MaterializationStatus",
    "MaterializedCase",
    "PlanningStrategy",
    "RuntimeProvenance",
    "ScoreArtifact",
    "ScoreSide",
    "ScorerSpec",
    "SemanticIdentitySpec",
    "SerializableModel",
    "TokenComparisonArtifact",
]
