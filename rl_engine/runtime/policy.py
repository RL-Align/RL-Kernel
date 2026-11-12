# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""User-facing RL-Kernel runtime policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class RLKernelMode(str, Enum):
    STRICT = "strict"
    AUDIT = "audit"
    AUTO = "auto"
    OFF = "off"


@dataclass(frozen=True)
class RLKernelModePolicy:
    mode: RLKernelMode
    enabled: bool
    case_id: str
    provider_mode: str | None
    fail_on_fallback: bool
    report_only: bool


_POLICIES = {
    RLKernelMode.STRICT: RLKernelModePolicy(
        RLKernelMode.STRICT, True, "R/R", "strict", True, False
    ),
    RLKernelMode.AUDIT: RLKernelModePolicy(RLKernelMode.AUDIT, True, "R/R", "strict", False, True),
    RLKernelMode.AUTO: RLKernelModePolicy(RLKernelMode.AUTO, True, "P/P", "auto", False, False),
    RLKernelMode.OFF: RLKernelModePolicy(RLKernelMode.OFF, False, "P/P", None, False, False),
}


def rl_kernel_mode(value: str | RLKernelMode | None = None) -> RLKernelMode:
    raw = os.getenv("RL_KERNEL_MODE", "strict") if value is None else value
    if isinstance(raw, RLKernelMode):
        return raw
    normalized = str(raw).strip().lower()
    try:
        return RLKernelMode(normalized)
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in RLKernelMode)
        raise RuntimeError(f"RL_KERNEL_MODE must be one of {choices}; got {raw!r}") from exc


def rl_kernel_mode_policy(
    value: str | RLKernelMode | None = None,
) -> RLKernelModePolicy:
    return _POLICIES[rl_kernel_mode(value)]


def strict_contract_enabled() -> bool:
    configured = os.getenv("RL_KERNEL_MODE")
    if configured is not None:
        return rl_kernel_mode(configured) in {RLKernelMode.STRICT, RLKernelMode.AUDIT}
    legacy = os.getenv("VIME_RL_KERNEL_STRICT", "").strip().lower()
    return legacy in {"1", "true", "yes", "on"}


def route_report_enabled() -> bool:
    enabled = os.getenv("RL_KERNEL_ROUTE_REPORT", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return False
    all_ranks = os.getenv("RL_KERNEL_ROUTE_REPORT_ALL_RANKS", "0").strip().lower()
    if all_ranks in {"1", "true", "yes", "on"}:
        return True
    for name in ("RANK", "LOCAL_RANK"):
        value = os.getenv(name, "").strip()
        if value:
            try:
                return int(value) == 0
            except ValueError:
                continue
    return True


__all__ = [
    "RLKernelMode",
    "RLKernelModePolicy",
    "rl_kernel_mode",
    "rl_kernel_mode_policy",
    "route_report_enabled",
    "strict_contract_enabled",
]
