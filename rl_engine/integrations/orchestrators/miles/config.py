# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Configuration for the miles orchestrator integration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from rl_engine.runtime.plan import IntegrationPlan, integration_plan_from_environment


@dataclass(frozen=True)
class MilesConfig:
    enabled: bool
    plan: IntegrationPlan
    readback_dir: str | None

    @classmethod
    def from_environment(cls) -> MilesConfig:
        value = os.getenv("RL_KERNEL_MILES_INTEGRATION", "0").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            enabled = True
        elif value in {"0", "false", "no", "off"}:
            enabled = False
        else:
            raise ValueError(
                "RL_KERNEL_MILES_INTEGRATION must be a boolean "
                "(1/0, true/false, yes/no, or on/off)"
            )
        return cls(
            enabled=enabled,
            plan=integration_plan_from_environment(),
            readback_dir=os.getenv("RL_KERNEL_READBACK_DIR") or None,
        )
