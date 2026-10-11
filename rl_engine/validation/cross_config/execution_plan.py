# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Canonical, runtime-independent execution-plan construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rl_engine.validation.cross_config.config import (
    ExperimentConfig,
    OperatorSelection,
    bind_operator_selection,
)
from rl_engine.validation.cross_config.planner import PlanningIssue
from rl_engine.validation.cross_config.schema import ExperimentCase


@dataclass(frozen=True)
class ExecutionPlanEntry:
    """One operator-bound case and its resolved operator selection."""

    case: ExperimentCase
    operators: OperatorSelection
    schema_version: str = "cross_config.execution_plan_entry.v1"

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical append-only plan row."""

        return {
            "schema_version": self.schema_version,
            "case": self.case.to_dict(),
            "operators": self.operators.to_dict(),
        }


@dataclass(frozen=True)
class ExecutionPlan:
    """Canonical metadata shared by planning and every runtime adapter."""

    experiment: Mapping[str, Any]
    entries: tuple[ExecutionPlanEntry, ...]
    issues: tuple[PlanningIssue, ...] = ()
    schema_version: str = "cross_config.execution_plan.v1"

    def rows(self) -> tuple[dict[str, Any], ...]:
        """Serialize all plan entries in deterministic execution order."""

        return tuple(entry.to_dict() for entry in self.entries)


def build_execution_plan(config: ExperimentConfig) -> ExecutionPlan:
    """Plan, resolve operators, and bind them into immutable case identities."""

    planned = config.plan()
    entries: list[ExecutionPlanEntry] = []
    for case in planned.cases:
        operators = config.operators_for(case)
        entries.append(
            ExecutionPlanEntry(
                case=bind_operator_selection(case, operators),
                operators=operators,
            )
        )
    return ExecutionPlan(
        experiment=config.to_dict(),
        entries=tuple(entries),
        issues=planned.issues,
    )


__all__ = [
    "ExecutionPlan",
    "ExecutionPlanEntry",
    "build_execution_plan",
]
