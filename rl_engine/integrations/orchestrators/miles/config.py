# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Configuration for the miles orchestrator integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from rl_engine.runtime.plan import IntegrationPlan, integration_plan_from_environment

CALLBACK_ENABLED_FIELD = "rl_kernel_miles_enabled"
CALLBACK_CASE_FIELDS = {
    "attention": "rl_kernel_attention_case",
    "ffn": "rl_kernel_ffn_case",
    "logp": "rl_kernel_logp_case",
}
CALLBACK_READBACK_FIELD = "rl_kernel_readback_dir"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def parse_flag(value: Any, *, source: str) -> bool:
    """Parse one documented boolean spelling; anything else is an error, not a guess."""
    text = str(value).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    raise ValueError(f"{source} must be a boolean (1/0, true/false, yes/no, or on/off)")


def _callback_value(args: Any, field: str) -> Any:
    return None if args is None else getattr(args, field, None)


@dataclass(frozen=True)
class MilesConfig:
    enabled: bool
    plan: IntegrationPlan
    readback_dir: str | None

    @classmethod
    def from_environment(cls) -> MilesConfig:
        return cls(
            enabled=parse_flag(
                os.getenv("RL_KERNEL_MILES_INTEGRATION", "0"),
                source="RL_KERNEL_MILES_INTEGRATION",
            ),
            plan=integration_plan_from_environment(),
            readback_dir=os.getenv("RL_KERNEL_READBACK_DIR", "").strip() or None,
        )

    @classmethod
    def resolve(cls, args: Any = None) -> MilesConfig:
        """Read the Miles callback config field by field, falling back to the environment.

        Miles merges ``custom_megatron_init_path.config`` into the callback args after
        validating it against the fields declared by ``add_arguments``, so those fields
        reach every worker through the orchestration config instead of process
        inheritance. A field left out of the callback config falls back to its
        environment variable, which keeps local debugging independent of Miles.
        """
        enabled_value = _callback_value(args, CALLBACK_ENABLED_FIELD)
        enabled = (
            parse_flag(enabled_value, source=CALLBACK_ENABLED_FIELD)
            if enabled_value is not None
            else parse_flag(
                os.getenv("RL_KERNEL_MILES_INTEGRATION", "0"),
                source="RL_KERNEL_MILES_INTEGRATION",
            )
        )
        readback_value = _callback_value(args, CALLBACK_READBACK_FIELD)
        if readback_value is not None:
            readback_dir = str(readback_value).strip() or None
        else:
            readback_dir = os.getenv("RL_KERNEL_READBACK_DIR", "").strip() or None
        if not enabled:
            # A disabled worker never installs, so an invalid case setting must not
            # keep it from starting.
            return cls(
                enabled=False,
                plan=IntegrationPlan.from_case_ids(),
                readback_dir=readback_dir,
            )
        cases: dict[str, str] = {}
        for module, field in CALLBACK_CASE_FIELDS.items():
            value = _callback_value(args, field)
            cases[module] = (
                str(value)
                if value is not None
                else os.getenv(f"RL_KERNEL_{module.upper()}_CASE", "P/P")
            )
        return cls(
            enabled=True,
            plan=IntegrationPlan.from_case_ids(
                attention=cases["attention"],
                ffn=cases["ffn"],
                logp=cases["logp"],
            ),
            readback_dir=readback_dir,
        )
