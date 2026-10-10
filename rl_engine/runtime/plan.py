# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Executable production/RL-Kernel routes for the module debug matrix."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from rl_engine.contracts.diagnostics.modules import module_debug_matrix


class Implementation(str, Enum):
    PRODUCTION = "production"
    RL_KERNEL = "rl_kernel"


@dataclass(frozen=True)
class OperatorAblationCase:
    module: str
    case_id: str
    training: Implementation
    rollout: Implementation
    purpose: str
    diagnostic_axes: tuple[str, ...]

    def implementation_for(self, target: str) -> Implementation:
        normalized = target.strip().lower()
        if normalized == "training":
            return self.training
        if normalized == "rollout":
            return self.rollout
        raise ValueError("target must be 'training' or 'rollout'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "case_id": self.case_id,
            "training_implementation": self.training.value,
            "rollout_implementation": self.rollout.value,
            "purpose": self.purpose,
            "diagnostic_axes": list(self.diagnostic_axes),
        }


_CASE_DEFINITIONS = (
    ("P/P", Implementation.PRODUCTION, Implementation.PRODUCTION, "native baseline"),
    ("R/R", Implementation.RL_KERNEL, Implementation.RL_KERNEL, "RL-Kernel control"),
    (
        "P/R",
        Implementation.PRODUCTION,
        Implementation.RL_KERNEL,
        "rollout-only mismatch",
    ),
    (
        "R/P",
        Implementation.RL_KERNEL,
        Implementation.PRODUCTION,
        "training-only mismatch",
    ),
)


def operator_ablation_cases(module: str) -> tuple[OperatorAblationCase, ...]:
    normalized = module.strip().lower()
    matrix = module_debug_matrix()
    try:
        axes = tuple(str(axis["id"]) for axis in matrix["modules"][normalized]["axes"])
    except KeyError as exc:
        raise ValueError(f"unknown ablation module {module!r}") from exc
    return tuple(
        OperatorAblationCase(normalized, case_id, training, rollout, purpose, axes)
        for case_id, training, rollout, purpose in _CASE_DEFINITIONS
    )


def operator_ablation_case(module: str, case_id: str) -> OperatorAblationCase:
    normalized = case_id.strip().upper()
    for case in operator_ablation_cases(module):
        if case.case_id == normalized:
            return case
    raise ValueError(f"unknown ablation case {case_id!r}")


@dataclass(frozen=True)
class IntegrationPlan:
    """One independently selectable P/R case for Attention, FFN, and Logp."""

    cases: Mapping[str, OperatorAblationCase]

    def __post_init__(self) -> None:
        normalized = dict(self.cases)
        if set(normalized) != {"attention", "ffn", "logp"}:
            raise ValueError("integration plan must define attention, ffn, and logp")
        for module, case in normalized.items():
            if not isinstance(case, OperatorAblationCase) or case.module != module:
                raise ValueError(f"invalid integration case for {module!r}")
        object.__setattr__(self, "cases", MappingProxyType(normalized))

    @classmethod
    def from_case_ids(
        cls,
        *,
        attention: str = "P/P",
        ffn: str = "P/P",
        logp: str = "P/P",
    ) -> "IntegrationPlan":
        return cls(
            {
                "attention": operator_ablation_case("attention", attention),
                "ffn": operator_ablation_case("ffn", ffn),
                "logp": operator_ablation_case("logp", logp),
            }
        )

    def implementation_for(self, module: str, target: str) -> Implementation:
        try:
            case = self.cases[module.strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unknown integration module {module!r}") from exc
        return case.implementation_for(target)

    def to_dict(self) -> dict[str, Any]:
        matrix = module_debug_matrix()
        return {
            "schema_version": matrix["schema_version"],
            "cases": {module: case.to_dict() for module, case in self.cases.items()},
        }


def integration_plan_from_environment() -> IntegrationPlan:
    """Materialize the one plan inherited by framework worker processes."""

    return IntegrationPlan.from_case_ids(
        attention=os.getenv("RL_KERNEL_ATTENTION_CASE", "P/P"),
        ffn=os.getenv("RL_KERNEL_FFN_CASE", "P/P"),
        logp=os.getenv("RL_KERNEL_LOGP_CASE", "P/P"),
    )


def configure_integration_environment(
    plan: IntegrationPlan,
    *,
    readback_dir: str | None = None,
) -> None:
    """Export one plan for Megatron actors and vLLM subprocesses."""

    for module, variable in (
        ("attention", "RL_KERNEL_ATTENTION_CASE"),
        ("ffn", "RL_KERNEL_FFN_CASE"),
        ("logp", "RL_KERNEL_LOGP_CASE"),
    ):
        os.environ[variable] = plan.cases[module].case_id
    if readback_dir:
        os.environ["RL_KERNEL_READBACK_DIR"] = readback_dir


__all__ = [
    "Implementation",
    "IntegrationPlan",
    "OperatorAblationCase",
    "configure_integration_environment",
    "integration_plan_from_environment",
    "operator_ablation_case",
    "operator_ablation_cases",
]
