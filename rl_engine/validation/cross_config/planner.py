# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Typed baseline, one-at-a-time, and explicitly bounded pairwise planning."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from rl_engine.contracts.numerical import tolerance_contract_fingerprint
from rl_engine.validation.cross_config.schema import (
    ExperimentCase,
    ExperimentDefinition,
    IsolationScope,
    KnobDescriptor,
    PlanningStrategy,
)

Normalizer = Callable[[Any], Any]
Constraint = Callable[[str, Any, Mapping[str, Any]], Optional["PlanningIssue"]]
MAX_PLAN_CASES = 256


@dataclass(frozen=True)
class PlanningIssue:
    """Structured planning rejection that callers can persist or display."""

    code: str
    reason: str
    path: Optional[str] = None
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "reason": self.reason,
            "path": self.path,
            "value": self.value,
        }


class PlanningError(ValueError):
    """Raised for an invalid experiment definition with structured issues."""

    def __init__(self, issues: Sequence[PlanningIssue]):
        self.issues = tuple(issues)
        message = "; ".join(
            f"{issue.code}{f'[{issue.path}]' if issue.path else ''}: {issue.reason}"
            for issue in self.issues
        )
        super().__init__(message)


@dataclass(frozen=True)
class ExperimentPlan:
    """A deterministic plan plus non-fatal capability findings."""

    definition: ExperimentDefinition
    cases: tuple[ExperimentCase, ...]
    issues: tuple[PlanningIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cross_config.experiment_plan.v1",
            "experiment_id": self.definition.experiment_id,
            "cases": [case.to_dict() for case in self.cases],
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _positive_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("must be a positive integer")
    return value


def _strict_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("must be a JSON boolean")
    return value


def _normalize_dtype(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("must be a dtype string")
    normalized = value.strip().lower().replace("torch.", "")
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float": "float32",
        "float32": "float32",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {value!r}") from exc


def _normalize_choice(*choices: str) -> Normalizer:
    allowed = frozenset(choices)

    def normalize(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("must be a string")
        normalized = value.strip().lower().replace("-", "_")
        if normalized not in allowed:
            raise ValueError(f"must be one of {sorted(allowed)}")
        return normalized

    return normalize


def normalize_backend_id(value: Any) -> str:
    """Normalize the public selected-logprob backend shortcut."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("must be a non-empty backend ID")
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "auto": "native",
        "default": "native",
        "pytorch": "rlkernel.reference_logp",
        "reference": "rlkernel.reference_logp",
    }
    return aliases.get(normalized, normalized)


V1_KNOB_DESCRIPTORS: tuple[KnobDescriptor, ...] = (
    KnobDescriptor("batch.size", IsolationScope.REQUEST, ("rollout", "training")),
    KnobDescriptor("rollout.tensor_parallel_size", IsolationScope.PROCESS, ("rollout",)),
    KnobDescriptor("rollout.context_parallel_size", IsolationScope.PROCESS, ("rollout",)),
    KnobDescriptor("rollout.dtype", IsolationScope.ENGINE_CONSTRUCTION, ("rollout",)),
    KnobDescriptor(
        "rollout.enable_prefix_caching",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("rollout",),
    ),
    KnobDescriptor("rollout.enforce_eager", IsolationScope.ENGINE_CONSTRUCTION, ("rollout",)),
    KnobDescriptor(
        "training.attention_backend",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("training",),
        allowed_values=("flash_attention_2", "sdpa", "eager", "model_default"),
    ),
    KnobDescriptor("training.compute_dtype", IsolationScope.ENGINE_CONSTRUCTION, ("training",)),
    KnobDescriptor(
        "logp.backend",
        IsolationScope.ENGINE_CONSTRUCTION,
        ("rollout", "training"),
        derived=True,
    ),
    KnobDescriptor(
        "training.sharding",
        IsolationScope.PROCESS,
        ("training",),
        allowed_values=("unsharded", "fsdp"),
    ),
)

V1_KNOBS: Mapping[str, KnobDescriptor] = {
    descriptor.path: descriptor for descriptor in V1_KNOB_DESCRIPTORS
}

_NORMALIZERS: Mapping[str, Normalizer] = {
    "batch.size": _positive_int,
    "rollout.tensor_parallel_size": _positive_int,
    "rollout.context_parallel_size": _positive_int,
    "rollout.dtype": _normalize_dtype,
    "rollout.enable_prefix_caching": _strict_bool,
    "rollout.enforce_eager": _strict_bool,
    "training.attention_backend": _normalize_choice(
        "flash_attention_2", "sdpa", "eager", "model_default"
    ),
    "training.compute_dtype": _normalize_dtype,
    "logp.backend": normalize_backend_id,
    "training.sharding": _normalize_choice("unsharded", "fsdp"),
}


class Planner:
    """Generate a bounded plan without importing runtime- or operator-specific branches."""

    def __init__(
        self,
        *,
        knobs: Mapping[str, KnobDescriptor] = V1_KNOBS,
        normalizers: Mapping[str, Normalizer] = _NORMALIZERS,
        constraints: Sequence[Constraint] = (),
    ):
        self.knobs = dict(knobs)
        self.normalizers = dict(normalizers)
        self.constraints = tuple(constraints)

    def plan(self, definition: ExperimentDefinition) -> ExperimentPlan:
        issues = self._validate_definition(definition)
        if issues:
            raise PlanningError(issues)

        baseline = self.normalize_requested(definition.baseline)
        requested_cases: list[tuple[dict[str, Any], tuple[str, ...]]] = [(baseline, ())]
        intervention_values: dict[str, tuple[Any, ...]] = {}

        def append_requested(requested: dict[str, Any], changed_paths: tuple[str, ...]) -> None:
            if len(requested_cases) >= MAX_PLAN_CASES:
                raise PlanningError(
                    (
                        PlanningIssue(
                            code="PLAN_TOO_LARGE",
                            reason=f"a plan may contain at most {MAX_PLAN_CASES} cases",
                            value=MAX_PLAN_CASES,
                        ),
                    )
                )
            requested_cases.append((requested, changed_paths))

        for intervention in definition.interventions:
            if len(intervention.values) > MAX_PLAN_CASES:
                raise PlanningError(
                    (
                        PlanningIssue(
                            code="PLAN_TOO_LARGE",
                            reason=(f"an intervention may contain at most {MAX_PLAN_CASES} values"),
                            path=intervention.path,
                            value=len(intervention.values),
                        ),
                    )
                )
            normalized_values = tuple(
                self._normalize_value(intervention.path, value) for value in intervention.values
            )
            intervention_values[intervention.path] = normalized_values
            baseline_value = _get_path(baseline, intervention.path)
            for value in normalized_values:
                if value == baseline_value:
                    continue
                requested = _deep_copy_mapping(baseline)
                _set_path(requested, intervention.path, value)
                append_requested(requested, (intervention.path,))

        if definition.strategy == PlanningStrategy.PAIRWISE:
            for first_path, second_path in definition.pairwise_paths:
                first_baseline = _get_path(baseline, first_path)
                second_baseline = _get_path(baseline, second_path)
                for first_value, second_value in itertools.product(
                    intervention_values[first_path], intervention_values[second_path]
                ):
                    if first_value == first_baseline or second_value == second_baseline:
                        continue
                    requested = _deep_copy_mapping(baseline)
                    _set_path(requested, first_path, first_value)
                    _set_path(requested, second_path, second_value)
                    append_requested(requested, tuple(sorted((first_path, second_path))))

        contract_fingerprint = tolerance_contract_fingerprint()
        scenario_fingerprint = _fingerprint(
            {"scenario_id": definition.scenario_id, "scenario": definition.scenario}
        )
        cases: list[ExperimentCase] = []
        seen_ids: set[str] = set()
        capability_issues: list[PlanningIssue] = []
        for requested, changed_paths in requested_cases:
            case_issues = self._apply_constraints(requested, changed_paths)
            capability_issues.extend(case_issues)
            case_id = self._case_id(
                definition,
                requested,
                contract_fingerprint=contract_fingerprint,
                scenario_fingerprint=scenario_fingerprint,
            )
            if case_id in seen_ids:
                continue
            seen_ids.add(case_id)
            cases.append(
                ExperimentCase(
                    case_id=case_id,
                    experiment_id=definition.experiment_id,
                    scenario_id=definition.scenario_id,
                    identity=definition.identity,
                    requested=requested,
                    changed_paths=changed_paths,
                    contract_fingerprint=contract_fingerprint,
                    scenario_fingerprint=scenario_fingerprint,
                )
            )

        return ExperimentPlan(
            definition=definition,
            cases=tuple(cases),
            issues=tuple(capability_issues),
        )

    def normalize_requested(self, requested: Mapping[str, Any]) -> dict[str, Any]:
        flattened = _flatten(requested)
        issues: list[PlanningIssue] = []
        normalized: dict[str, Any] = {}
        for path, value in flattened.items():
            if path not in self.knobs:
                code = "DERIVED_KNOB" if path == "logp.tp_layout" else "UNSUPPORTED_PATH"
                issues.append(
                    PlanningIssue(
                        code=code,
                        path=path,
                        value=value,
                        reason="path is not a user-settable V1 knob",
                    )
                )
                continue
            try:
                normalized[path] = self._normalize_value(path, value)
            except (TypeError, ValueError) as exc:
                issues.append(
                    PlanningIssue(
                        code="UNSUPPORTED_VALUE",
                        path=path,
                        value=value,
                        reason=str(exc),
                    )
                )
        if issues:
            raise PlanningError(issues)
        result: dict[str, Any] = {}
        for path, value in normalized.items():
            _set_path(result, path, value)
        return result

    def isolation_for(self, changed_paths: Sequence[str]) -> IsolationScope:
        if not changed_paths:
            return IsolationScope.REQUEST
        order = {
            IsolationScope.REQUEST: 0,
            IsolationScope.ENGINE_CONSTRUCTION: 1,
            IsolationScope.DISTRIBUTED_CONTEXT: 2,
            IsolationScope.PROCESS: 3,
        }
        return max((self.knobs[path].lifecycle for path in changed_paths), key=order.__getitem__)

    def _validate_definition(self, definition: ExperimentDefinition) -> list[PlanningIssue]:
        issues: list[PlanningIssue] = []
        try:
            baseline = self.normalize_requested(definition.baseline)
        except PlanningError as exc:
            return list(exc.issues)
        baseline_paths = set(_flatten(baseline))
        for path in sorted(set(self.knobs).difference(baseline_paths)):
            issues.append(
                PlanningIssue(
                    code="MISSING_BASELINE_VALUE",
                    path=path,
                    reason="strict baselines must declare every allowlisted knob",
                )
            )
        declared_paths: set[str] = set()
        for intervention in definition.interventions:
            path = intervention.path
            if path not in self.knobs:
                issues.append(
                    PlanningIssue(
                        code="UNSUPPORTED_PATH",
                        path=path,
                        reason="intervention path is not in the V1 allowlist",
                    )
                )
                continue
            if path in declared_paths:
                issues.append(
                    PlanningIssue(
                        code="DUPLICATE_INTERVENTION",
                        path=path,
                        reason="each intervention path must be declared once",
                    )
                )
            declared_paths.add(path)
            if not intervention.values:
                issues.append(
                    PlanningIssue(
                        code="EMPTY_INTERVENTION",
                        path=path,
                        reason="intervention values cannot be empty",
                    )
                )
            try:
                _get_path(baseline, path)
            except KeyError:
                issues.append(
                    PlanningIssue(
                        code="MISSING_BASELINE_VALUE",
                        path=path,
                        reason="every intervention path must exist in baseline",
                    )
                )
            for value in intervention.values:
                try:
                    self._normalize_value(path, value)
                except (TypeError, ValueError) as exc:
                    issues.append(
                        PlanningIssue(
                            code="UNSUPPORTED_VALUE",
                            path=path,
                            value=value,
                            reason=str(exc),
                        )
                    )

        if definition.strategy == PlanningStrategy.ONE_AT_A_TIME and definition.pairwise_paths:
            issues.append(
                PlanningIssue(
                    code="PAIRWISE_NOT_ENABLED",
                    reason="pairwise_paths require strategy='pairwise'",
                )
            )
        if definition.strategy == PlanningStrategy.PAIRWISE and not definition.pairwise_paths:
            issues.append(
                PlanningIssue(
                    code="PAIRWISE_PATHS_REQUIRED",
                    reason="pairwise strategy requires at least one explicit path pair",
                )
            )
        seen_pairs: set[tuple[str, str]] = set()
        for pair in definition.pairwise_paths:
            if len(pair) != 2:
                issues.append(
                    PlanningIssue(
                        code="INVALID_PAIR",
                        reason="each pairwise entry must contain exactly two paths",
                        value=pair,
                    )
                )
                continue
            first, second = pair
            canonical_pair = (first, second) if first < second else (second, first)
            if first == second:
                issues.append(
                    PlanningIssue(
                        code="INVALID_PAIR",
                        reason="pairwise paths must be distinct",
                        value=pair,
                    )
                )
            elif first not in declared_paths or second not in declared_paths:
                issues.append(
                    PlanningIssue(
                        code="UNDECLARED_PAIR_PATH",
                        reason="pairwise paths must both have declared interventions",
                        value=pair,
                    )
                )
            elif canonical_pair in seen_pairs:
                issues.append(
                    PlanningIssue(
                        code="DUPLICATE_PAIR",
                        reason="pairwise path pair is duplicated",
                        value=pair,
                    )
                )
            seen_pairs.add(canonical_pair)
        return issues

    def _normalize_value(self, path: str, value: Any) -> Any:
        try:
            normalizer = self.normalizers[path]
        except KeyError as exc:
            raise ValueError(f"no normalizer registered for {path}") from exc
        return normalizer(value)

    def _apply_constraints(
        self, requested: Mapping[str, Any], changed_paths: Sequence[str]
    ) -> list[PlanningIssue]:
        issues: list[PlanningIssue] = []
        paths = changed_paths or tuple(_flatten(requested))
        for path in paths:
            value = _get_path(requested, path)
            for constraint in self.constraints:
                issue = constraint(path, value, requested)
                if issue is not None:
                    issues.append(issue)
        return issues

    @staticmethod
    def _case_id(
        definition: ExperimentDefinition,
        requested: Mapping[str, Any],
        *,
        contract_fingerprint: str,
        scenario_fingerprint: str,
    ) -> str:
        payload = {
            "requested": requested,
            "identity": definition.identity.to_dict(),
            "contract": {
                "source": definition.contract_source,
                "version": definition.contract_version,
                "fingerprint": contract_fingerprint,
            },
            "scenario_id": definition.scenario_id,
            "scenario_fingerprint": scenario_fingerprint,
        }
        return f"cross-config-{_fingerprint(payload)[:20]}"


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise PlanningError(
                [PlanningIssue(code="INVALID_PATH", reason="configuration keys must be strings")]
            )
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            flattened.update(_flatten(child, path))
        else:
            flattened[path] = child
    return flattened


def _get_path(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def _set_path(value: dict[str, Any], path: str, child: Any) -> None:
    current = value
    parts = path.split(".")
    for part in parts[:-1]:
        existing = current.setdefault(part, {})
        if not isinstance(existing, dict):
            raise ValueError(f"configuration path collision at {path}")
        current = existing
    current[parts[-1]] = child


def _deep_copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value))


def _fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_plain(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_plain(child) for child in value]
    return value


__all__ = [
    "ExperimentPlan",
    "Planner",
    "PlanningError",
    "PlanningIssue",
    "V1_KNOBS",
    "V1_KNOB_DESCRIPTORS",
    "normalize_backend_id",
]
