# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Plan and run cross-configuration alignment experiments.

The package root is intentionally small and lazily loads execution code. Extension
authors import adapter, artifact, operator, or schema details from their owning
submodule.
"""

from importlib import import_module
from typing import Any

from rl_engine.validation.cross_config.comparison import compare_score_artifacts
from rl_engine.validation.cross_config.config import ExperimentConfig, load_config
from rl_engine.validation.cross_config.execution_plan import ExecutionPlan, build_execution_plan
from rl_engine.validation.cross_config.planner import ExperimentPlan, Planner


def __getattr__(name: str) -> Any:
    if name in {"PairedRunResult", "PairedRunner"}:
        return getattr(import_module("rl_engine.validation.cross_config.runner"), name)
    if name in {"RuntimeMaterializer", "RuntimeTools"}:
        return getattr(import_module("rl_engine.validation.cross_config.runtime"), name)
    raise AttributeError(name)


__all__ = [
    "ExperimentConfig",
    "ExperimentPlan",
    "ExecutionPlan",
    "PairedRunResult",
    "PairedRunner",
    "Planner",
    "RuntimeMaterializer",
    "RuntimeTools",
    "build_execution_plan",
    "compare_score_artifacts",
    "load_config",
]
