# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Thin runtime materialization facade for the V1 allowlist."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence

from rl_engine.validation.cross_config.planner import V1_KNOBS
from rl_engine.validation.cross_config.schema import (
    ExperimentCase,
    IsolationScope,
    KnobDescriptor,
    MaterializationStatus,
    MaterializedCase,
    RuntimeProvenance,
    SerializableModel,
)


@dataclass(frozen=True)
class KnobApplication(SerializableModel):
    """One adapter's requested, materialized, and observed value."""

    path: str
    requested: Any
    materialized: Any
    actual: Any
    lifecycle: IsolationScope
    status: MaterializationStatus
    evidence: Mapping[str, Any] = field(default_factory=dict)
    critical: bool = True
    schema_version: str = "cross_config.knob_application.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "lifecycle", IsolationScope(self.lifecycle))
        object.__setattr__(self, "status", MaterializationStatus(self.status))
        for name in ("requested", "materialized", "actual"):
            object.__setattr__(self, name, _freeze_value(getattr(self, name)))
        object.__setattr__(self, "evidence", _freeze_mapping(self.evidence))


@dataclass(frozen=True)
class RuntimeBinding:
    """Small, backend-neutral handoff from materialization to execution.

    Runtime adapters may construct repository-specific objects internally, but
    the core runner sees only the values required to create scorers and validate
    lifecycle identity. New vLLM, FSDP, or other adapters therefore do not
    change the runner's type surface.
    """

    batch_size: int
    side_configs: Mapping[str, Mapping[str, Any]]
    topology: Mapping[str, Mapping[str, Any]]
    scorer: Mapping[str, Any]
    operator_backends: Mapping[str, str]
    runtime_kind: str

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an integer")
        if self.batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        if not isinstance(self.runtime_kind, str) or not self.runtime_kind.strip():
            raise ValueError("runtime_kind must be a non-empty string")
        for name, value in (
            ("side_configs", self.side_configs),
            ("topology", self.topology),
        ):
            for target in ("rollout", "training"):
                if not isinstance(value.get(target), Mapping):
                    raise ValueError(f"{name} must define a {target} mapping")
        for target in ("rollout", "training"):
            world_size = self.topology[target].get("world_size")
            if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
                raise ValueError(f"{target} topology must define a positive integer world_size")
            backend = self.operator_backends.get(target)
            if not isinstance(backend, str) or not backend.strip():
                raise ValueError(f"operator_backends must define a non-empty {target} backend")
        object.__setattr__(self, "side_configs", _freeze_mapping(self.side_configs))
        object.__setattr__(self, "topology", _freeze_mapping(self.topology))
        object.__setattr__(self, "scorer", _freeze_mapping(self.scorer))
        object.__setattr__(self, "operator_backends", _freeze_mapping(self.operator_backends))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cross_config.runtime_binding.v1",
            "runtime_kind": self.runtime_kind,
            "batch_size": self.batch_size,
            "side_configs": _plain_mapping(self.side_configs),
            "topology": _plain_mapping(self.topology),
            "scorer": _plain_mapping(self.scorer),
            "operators": dict(self.operator_backends),
        }


@dataclass(frozen=True)
class AdapterMaterialization:
    """Output of a typed runtime adapter before the facade adds fingerprints."""

    applications: tuple[KnobApplication, ...]
    binding: RuntimeBinding

    def __post_init__(self) -> None:
        object.__setattr__(self, "applications", tuple(self.applications))


class RuntimeMaterializer(Protocol):
    """Adapter boundary used by the small ``RuntimeTools`` facade.

    The declared implementation fingerprint must deterministically identify the
    executable materialization path and change when that implementation changes.
    """

    runtime_kind: str

    @property
    def implementation_fingerprint(self) -> str: ...

    def materialize(
        self,
        normalized: Mapping[str, Any],
        descriptors: Mapping[str, KnobDescriptor],
    ) -> AdapterMaterialization: ...


@dataclass(frozen=True)
class RuntimeMaterialization:
    materialized_case: MaterializedCase
    provenance: RuntimeProvenance
    applications: tuple[KnobApplication, ...]
    binding: RuntimeBinding

    @property
    def executable_in_strict_mode(self) -> bool:
        return self.materialized_case.status is MaterializationStatus.APPLIED


class RuntimeMaterializationError(RuntimeError):
    pass


class RuntimeTools:
    """Materialize cases and compute reuse fingerprints without owning execution."""

    def __init__(self, descriptors: Mapping[str, KnobDescriptor] = V1_KNOBS):
        self.descriptors = dict(descriptors)

    def materialize(
        self,
        case: ExperimentCase,
        adapter: RuntimeMaterializer,
    ) -> RuntimeMaterialization:
        runtime_kind = _adapter_identity(adapter, "runtime_kind")
        adapter_implementation_fingerprint = _adapter_identity(
            adapter,
            "implementation_fingerprint",
        )
        normalized = _plain_mapping(case.requested)
        adapter_result = adapter.materialize(normalized, self.descriptors)
        if not isinstance(adapter_result, AdapterMaterialization):
            raise RuntimeMaterializationError("runtime adapter must return AdapterMaterialization")
        if not isinstance(adapter_result.binding, RuntimeBinding):
            raise RuntimeMaterializationError("runtime adapter must return a RuntimeBinding")
        if adapter_result.binding.runtime_kind != runtime_kind:
            raise RuntimeMaterializationError(
                "runtime binding kind must match the materializer runtime_kind"
            )
        applications = tuple(adapter_result.applications)
        _validate_application_contract(normalized, applications, self.descriptors)
        materialized = _mapping_from_applications(applications, "materialized")
        actual = _mapping_from_applications(applications, "actual")
        status = _aggregate_status(application.status for application in applications)
        construction_fingerprint = _scope_fingerprint(
            runtime_kind,
            adapter_implementation_fingerprint,
            applications,
            scopes=(
                IsolationScope.ENGINE_CONSTRUCTION,
                IsolationScope.DISTRIBUTED_CONTEXT,
                IsolationScope.PROCESS,
            ),
        )
        distributed_fingerprint = _scope_fingerprint(
            runtime_kind,
            adapter_implementation_fingerprint,
            applications,
            scopes=(IsolationScope.DISTRIBUTED_CONTEXT, IsolationScope.PROCESS),
        )
        process_fingerprint = _scope_fingerprint(
            runtime_kind,
            adapter_implementation_fingerprint,
            applications,
            scopes=(IsolationScope.PROCESS,),
        )
        isolation_scope = _strongest_scope(
            [
                self.descriptors[path].lifecycle
                for path in (case.changed_paths or tuple(_flatten(normalized)))
            ]
        )
        evidence = {
            "runtime_kind": runtime_kind,
            "execution_binding": case.execution_binding,
            "adapter_implementation_fingerprint": adapter_implementation_fingerprint,
            "binding_fingerprint": _fingerprint(adapter_result.binding.to_dict()),
            "applications": {
                application.path: application.to_dict() for application in applications
            },
        }
        materialized_case = MaterializedCase(
            case=case,
            normalized=normalized,
            materialized=materialized,
            isolation_scope=isolation_scope,
            construction_fingerprint=construction_fingerprint,
            distributed_context_fingerprint=distributed_fingerprint,
            process_fingerprint=process_fingerprint,
            status=status,
            evidence=evidence,
        )
        provenance = RuntimeProvenance(
            requested=_plain_mapping(case.requested),
            normalized=normalized,
            materialized=materialized,
            actual=actual,
            status=status,
            construction_fingerprint=construction_fingerprint,
            distributed_context_fingerprint=distributed_fingerprint,
            process_fingerprint=process_fingerprint,
            implementation_fingerprint=adapter_implementation_fingerprint,
            evidence=evidence,
        )
        return RuntimeMaterialization(
            materialized_case=materialized_case,
            provenance=provenance,
            applications=applications,
            binding=adapter_result.binding,
        )

    @staticmethod
    def require_executable(
        materialization: RuntimeMaterialization,
        *,
        strict: bool,
    ) -> None:
        status = materialization.materialized_case.status
        rejected = {
            MaterializationStatus.ERROR,
            MaterializationStatus.UNSUPPORTED,
            MaterializationStatus.UNOBSERVABLE,
        }
        if strict:
            rejected.add(MaterializationStatus.FALLBACK)
        if status in rejected:
            problems = [
                f"{application.path}={application.status.value}: "
                f"{application.evidence.get('reason', 'no evidence')}"
                for application in materialization.applications
                if application.status is not MaterializationStatus.APPLIED
            ]
            raise RuntimeMaterializationError(
                f"case {materialization.materialized_case.case.case_id} is not executable: "
                + "; ".join(problems)
            )

    @staticmethod
    def can_reuse(previous: RuntimeMaterialization, current: RuntimeMaterialization) -> bool:
        """Reuse only exact semantic and implementation identities with matching state."""

        previous_case = previous.materialized_case
        current_case = current.materialized_case
        return (
            previous_case.status is MaterializationStatus.APPLIED
            and current_case.status is MaterializationStatus.APPLIED
            and previous_case.case.identity == current_case.case.identity
            and previous_case.case.execution_binding == current_case.case.execution_binding
            and previous.provenance.implementation_fingerprint
            == current.provenance.implementation_fingerprint
            and previous_case.process_fingerprint == current_case.process_fingerprint
            and previous_case.distributed_context_fingerprint
            == current_case.distributed_context_fingerprint
            and previous_case.construction_fingerprint == current_case.construction_fingerprint
        )


def _aggregate_status(statuses: Iterable[MaterializationStatus]) -> MaterializationStatus:
    priority = (
        MaterializationStatus.ERROR,
        MaterializationStatus.UNSUPPORTED,
        MaterializationStatus.UNOBSERVABLE,
        MaterializationStatus.FALLBACK,
        MaterializationStatus.APPLIED,
    )
    status_set = set(statuses)
    if not status_set:
        return MaterializationStatus.ERROR
    return next(status for status in priority if status in status_set)


def _validate_application_contract(
    normalized: Mapping[str, Any],
    applications: tuple[KnobApplication, ...],
    descriptors: Mapping[str, KnobDescriptor],
) -> None:
    expected = _flatten(normalized)
    observed_paths = [application.path for application in applications]
    duplicate_paths = sorted(path for path in set(observed_paths) if observed_paths.count(path) > 1)
    missing_paths = sorted(set(expected).difference(observed_paths))
    unknown_paths = sorted(set(observed_paths).difference(expected))
    missing_descriptors = sorted(set(expected).difference(descriptors))
    problems: list[str] = []
    if missing_paths:
        problems.append(f"missing paths={missing_paths!r}")
    if duplicate_paths:
        problems.append(f"duplicate paths={duplicate_paths!r}")
    if unknown_paths:
        problems.append(f"unknown paths={unknown_paths!r}")
    if missing_descriptors:
        problems.append(f"missing descriptors={missing_descriptors!r}")
    for application in applications:
        descriptor = descriptors.get(application.path)
        if descriptor is None or application.path not in expected:
            continue
        if _plain_value(application.requested) != _plain_value(expected[application.path]):
            problems.append(f"{application.path} requested value differs from normalized case")
        if application.lifecycle is not descriptor.lifecycle:
            problems.append(f"{application.path} lifecycle differs from descriptor")
        if application.critical is not descriptor.critical:
            problems.append(f"{application.path} critical flag differs from descriptor")
        if application.status is MaterializationStatus.APPLIED:
            if _plain_value(application.actual) != _plain_value(application.materialized):
                problems.append(f"{application.path} applied actual differs from materialized")
            if not descriptor.derived and _plain_value(application.materialized) != _plain_value(
                expected[application.path]
            ):
                problems.append(
                    f"{application.path} applied materialized value differs from normalized case"
                )
    if problems:
        raise RuntimeMaterializationError(
            "runtime adapter returned invalid V1 knob applications: " + "; ".join(problems)
        )


def _mapping_from_applications(
    applications: Sequence[KnobApplication], attribute: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for application in applications:
        _set_path(result, application.path, getattr(application, attribute))
    return result


def _scope_fingerprint(
    runtime_kind: str,
    implementation_fingerprint: str,
    applications: Sequence[KnobApplication],
    *,
    scopes: Sequence[IsolationScope],
) -> str:
    scope_set = set(scopes)
    values = {
        application.path: application.materialized
        for application in applications
        if application.lifecycle in scope_set
    }
    return _fingerprint(
        {
            "runtime_kind": runtime_kind,
            "implementation_fingerprint": implementation_fingerprint,
            "values": values,
        }
    )


def _strongest_scope(scopes: Sequence[IsolationScope]) -> IsolationScope:
    order = {
        IsolationScope.REQUEST: 0,
        IsolationScope.ENGINE_CONSTRUCTION: 1,
        IsolationScope.DISTRIBUTED_CONTEXT: 2,
        IsolationScope.PROCESS: 3,
    }
    return max(scopes, key=order.__getitem__, default=IsolationScope.REQUEST)


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            result.update(_flatten(child, path))
        else:
            result[path] = child
    return result


def _set_path(value: dict[str, Any], path: str, child: Any) -> None:
    current = value
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = child


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_plain_value(item) for item in sorted(value, key=repr)]
    return value


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze_value(value)


def _fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _plain_mapping(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _adapter_identity(adapter: RuntimeMaterializer, attribute: str) -> str:
    value = getattr(adapter, attribute, None)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeMaterializationError(f"runtime adapter {attribute} must be a non-empty string")
    return value.strip()


__all__ = [
    "AdapterMaterialization",
    "KnobApplication",
    "RuntimeBinding",
    "RuntimeMaterialization",
    "RuntimeMaterializationError",
    "RuntimeMaterializer",
    "RuntimeTools",
]
