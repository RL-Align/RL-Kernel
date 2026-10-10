# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict, dependency-free experiment configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

from rl_engine.validation.cross_config._json import strict_json_loads
from rl_engine.validation.cross_config.planner import normalize_backend_id
from rl_engine.validation.cross_config.schema import (
    ExperimentCase,
    ExperimentDefinition,
    InterventionSpec,
    PlanningStrategy,
    SemanticIdentitySpec,
)

if TYPE_CHECKING:
    from rl_engine.validation.cross_config.planner import ExperimentPlan


CONFIG_SCHEMA_VERSION = "cross_config.experiment_config.v1"
_FORBIDDEN_THRESHOLD_KEYS = frozenset({"threshold", "fixed_threshold", "tolerance", "atol", "rtol"})
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "scenario_id",
        "contract_source",
        "contract_version",
        "strategy",
        "strict_fallback",
        "identity",
        "baseline",
        "interventions",
        "pairwise_paths",
        "operators",
        "scenario",
    }
)
_IDENTITY_KEYS = frozenset(
    item.name for item in dataclass_fields(SemanticIdentitySpec) if item.name != "schema_version"
)
_INTERVENTION_KEYS = frozenset({"path", "values"})
_OPERATOR_NAMES = frozenset({"selected_logprob"})
_OPERATOR_TARGETS = frozenset({"rollout", "training"})
_OPERATOR_BINDING_KEYS = frozenset({"backend", "options"})


@dataclass(frozen=True)
class OperatorSelection:
    """Concrete selected-logprob implementation requested for each scorer side.

    ``logp.backend`` remains the concise both-sides shortcut. This explicit form
    is needed only when rollout and training intentionally use different
    implementations.
    """

    rollout_backend: str
    training_backend: str
    rollout_options: Mapping[str, Any] = field(default_factory=dict)
    training_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rollout_backend", normalize_backend_id(self.rollout_backend))
        object.__setattr__(
            self,
            "training_backend",
            normalize_backend_id(self.training_backend),
        )
        object.__setattr__(self, "rollout_options", _freeze_mapping(self.rollout_options))
        object.__setattr__(self, "training_options", _freeze_mapping(self.training_options))

    def backend_for(self, target: str) -> str:
        if target == "rollout":
            return self.rollout_backend
        if target == "training":
            return self.training_backend
        raise ValueError("operator target must be 'rollout' or 'training'")

    def options_for(self, target: str) -> Mapping[str, Any]:
        if target == "rollout":
            return self.rollout_options
        if target == "training":
            return self.training_options
        raise ValueError("operator target must be 'rollout' or 'training'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_logprob": {
                "rollout": {
                    "backend": self.rollout_backend,
                    "options": _plain_value(self.rollout_options),
                },
                "training": {
                    "backend": self.training_backend,
                    "options": _plain_value(self.training_options),
                },
            }
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Loaded experiment plus optional target-specific operator selection."""

    definition: ExperimentDefinition
    source_path: Path
    operators: OperatorSelection | None = None
    schema_version: str = CONFIG_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized, portable experiment-config representation."""

        payload = self.definition.to_dict()
        payload["schema_version"] = self.schema_version
        if self.operators is not None:
            payload["operators"] = self.operators.to_dict()
        return payload

    def plan(self) -> ExperimentPlan:
        """Build the deterministic plan without importing a runtime backend."""

        from rl_engine.validation.cross_config.planner import Planner

        return Planner().plan(self.definition)

    def operators_for(self, case: ExperimentCase) -> OperatorSelection:
        """Resolve the concise ``logp.backend`` shortcut for one planned case."""

        backend = _case_logp_backend(case)
        if self.operators is None:
            return OperatorSelection(backend, backend)
        if self.operators.rollout_backend != backend:
            raise ValueError(
                "operators.selected_logprob.rollout must match the planned "
                f"logp.backend: {self.operators.rollout_backend!r} != {backend!r}"
            )
        return self.operators


def load_config(path: str | Path) -> ExperimentConfig:
    """Load one versioned JSON experiment with no threshold override surface."""

    source = Path(path)
    try:
        raw = strict_json_loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load cross-configuration config {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("cross-configuration config must contain a JSON object")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "config")
    _reject_threshold_keys(raw)
    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported cross-configuration config schema {raw.get('schema_version')!r}; "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )

    identity_raw = _required_mapping(raw, "identity")
    _reject_unknown_keys(identity_raw, _IDENTITY_KEYS, "identity")
    scenario = _optional_mapping(raw, "scenario")
    _reject_scenario_controls(scenario)

    interventions_raw = raw.get("interventions", [])
    if not isinstance(interventions_raw, list):
        raise ValueError("interventions must be a list")
    interventions = tuple(_load_intervention(item) for item in interventions_raw)

    pairwise_raw = raw.get("pairwise_paths", [])
    if not isinstance(pairwise_raw, list):
        raise ValueError("pairwise_paths must be a list")
    pairwise_paths = tuple(_load_pair(item) for item in pairwise_raw)

    strict_fallback = raw.get("strict_fallback", True)
    if not isinstance(strict_fallback, bool):
        raise ValueError("strict_fallback must be a JSON boolean")

    definition = ExperimentDefinition(
        experiment_id=_required_string(raw, "experiment_id"),
        scenario_id=_required_string(raw, "scenario_id"),
        identity=SemanticIdentitySpec(**identity_raw),
        baseline=_required_mapping(raw, "baseline"),
        interventions=interventions,
        scenario=scenario,
        strategy=PlanningStrategy(raw.get("strategy", "one_at_a_time")),
        strict_fallback=strict_fallback,
        pairwise_paths=pairwise_paths,
        contract_source=raw.get("contract_source", "ws1"),
        contract_version=raw.get("contract_version", "current"),
    )
    operators = _load_operators(raw.get("operators"))
    if operators is not None:
        if any(item.path == "logp.backend" for item in interventions):
            raise ValueError(
                "explicit operators cannot be combined with logp.backend interventions; "
                "use the shortcut or one fixed target mapping"
            )
        baseline_backend = _definition_logp_backend(definition)
        if operators.rollout_backend != baseline_backend:
            raise ValueError(
                "operators.selected_logprob.rollout must match baseline logp.backend: "
                f"{operators.rollout_backend!r} != {baseline_backend!r}"
            )

    return ExperimentConfig(
        definition=definition,
        operators=operators,
        source_path=source,
    )


def bind_operator_selection(
    case: ExperimentCase,
    selection: OperatorSelection,
) -> ExperimentCase:
    """Bind target-specific operators into the execution identity.

    Planning remains semantic-operator agnostic; the immutable binding extends
    the case and resume key before any runtime is created.
    """

    requested_backend = _case_logp_backend(case)
    if selection.rollout_backend != requested_backend:
        raise ValueError(
            "rollout operator must match the planned logp.backend: "
            f"{selection.rollout_backend!r} != {requested_backend!r}"
        )
    binding = selection.to_dict()
    payload = {
        "base_case_id": case.case_id,
        "base_scenario_fingerprint": case.scenario_fingerprint,
        "operators": binding,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    operator_fingerprint = hashlib.sha256(serialized).hexdigest()
    case_hash = hashlib.sha256(
        json.dumps(
            {"base_case_id": case.case_id, "operator_fingerprint": operator_fingerprint},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    scenario_fingerprint = hashlib.sha256(
        f"{case.scenario_fingerprint}:{operator_fingerprint}".encode("utf-8")
    ).hexdigest()
    return ExperimentCase(
        case_id=f"cross-config-{case_hash}",
        experiment_id=case.experiment_id,
        scenario_id=case.scenario_id,
        identity=case.identity,
        requested=case.requested,
        execution_binding={"operators": binding},
        changed_paths=case.changed_paths,
        contract_fingerprint=case.contract_fingerprint,
        scenario_fingerprint=scenario_fingerprint,
    )


def _load_intervention(value: Any) -> InterventionSpec:
    if not isinstance(value, Mapping):
        raise ValueError("each intervention must be an object")
    _reject_unknown_keys(value, _INTERVENTION_KEYS, "intervention")
    values = value.get("values")
    if not isinstance(values, list):
        raise ValueError("intervention values must be a list")
    return InterventionSpec(path=_required_string(value, "path"), values=tuple(values))


def _load_pair(value: Any) -> tuple[str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError("each pairwise_paths entry must contain exactly two string paths")
    return value[0], value[1]


def _load_operators(value: Any) -> OperatorSelection | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("operators must be an object")
    _reject_unknown_keys(value, _OPERATOR_NAMES, "operators")
    selected = value.get("selected_logprob")
    if not isinstance(selected, Mapping):
        raise ValueError("operators.selected_logprob must be an object")
    _reject_unknown_keys(selected, _OPERATOR_TARGETS, "operators.selected_logprob")
    rollout_backend, rollout_options = _load_operator_binding(selected, "rollout")
    training_backend, training_options = _load_operator_binding(selected, "training")
    return OperatorSelection(
        rollout_backend=rollout_backend,
        training_backend=training_backend,
        rollout_options=rollout_options,
        training_options=training_options,
    )


def _load_operator_binding(
    value: Mapping[str, Any],
    target: str,
) -> tuple[str, Mapping[str, Any]]:
    binding = value.get(target)
    if isinstance(binding, str):
        if not binding.strip():
            raise ValueError(f"operators.selected_logprob.{target} must not be empty")
        return binding, {}
    if not isinstance(binding, Mapping):
        raise ValueError(f"operators.selected_logprob.{target} must be a backend string or object")
    _reject_unknown_keys(binding, _OPERATOR_BINDING_KEYS, f"{target} operator binding")
    return _required_string(binding, "backend"), _optional_mapping(binding, "options")


def _case_logp_backend(case: ExperimentCase) -> str:
    logp = case.requested.get("logp")
    backend = logp.get("backend") if isinstance(logp, Mapping) else None
    if not isinstance(backend, str) or not backend:
        raise ValueError("planned cases must contain a non-empty string logp.backend")
    return normalize_backend_id(backend)


def _definition_logp_backend(definition: ExperimentDefinition) -> str:
    logp = definition.baseline.get("logp")
    backend = logp.get("backend") if isinstance(logp, Mapping) else None
    if not isinstance(backend, str) or not backend:
        raise ValueError("baseline must contain a non-empty string logp.backend")
    return normalize_backend_id(backend)


def _reject_scenario_controls(scenario: Mapping[str, Any]) -> None:
    behavior_keys = sorted(
        set(scenario).intersection(
            {"execution", "plan_only", "operator_cases", "expected_status", "allow_smoke_operators"}
        )
    )
    if behavior_keys:
        raise ValueError(
            "scenario is metadata only; move execution and operator policy to the CLI/config: "
            f"{behavior_keys}"
        )


def _reject_threshold_keys(value: Any, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            path = f"{prefix}.{key}" if prefix else str(key)
            if normalized in _FORBIDDEN_THRESHOLD_KEYS:
                raise ValueError(
                    f"{path} is forbidden: the fixed numerical-contract threshold is imported"
                )
            _reject_threshold_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_threshold_keys(child, f"{prefix}[{index}]")


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"unknown {label} keys: {unknown}")


def _required_mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    child = value.get(key)
    if not isinstance(child, Mapping):
        raise ValueError(f"{key} must be an object")
    return dict(child)


def _optional_mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    child = value.get(key, {})
    if not isinstance(child, Mapping):
        raise ValueError(f"{key} must be an object")
    return dict(child)


def _required_string(value: Mapping[str, Any], key: str) -> str:
    child = value.get(key)
    if not isinstance(child, str) or not child.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return child.strip()


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(child) for key, child in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(child) for child in value)
    return value


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_value(child) for child in value]
    return value


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ExperimentConfig",
    "OperatorSelection",
    "bind_operator_selection",
    "load_config",
]
