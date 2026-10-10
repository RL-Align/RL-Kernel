# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Process-local framework integration state.

Ray actors and vLLM workers are separate processes, so each process owns one
integration object and emits its own readback file.  Keeping the registry here
also lets the Vime logprob provider join the same Megatron call accounting as
the Attention and FFN hooks.
"""

from __future__ import annotations

from threading import Lock
from typing import Literal

from rl_engine.integrations.common.runtime import FrameworkOperatorIntegration

FrameworkName = Literal["megatron", "vllm"]

_ACTIVE: dict[FrameworkName, FrameworkOperatorIntegration] = {}
_LOCK = Lock()


def set_active_integration(
    framework: FrameworkName,
    integration: FrameworkOperatorIntegration,
) -> None:
    if integration.framework != framework:
        raise ValueError(
            f"integration framework {integration.framework!r} does not match {framework!r}"
        )
    with _LOCK:
        existing = _ACTIVE.get(framework)
        if existing is not None and existing is not integration:
            raise RuntimeError(f"{framework} integration is already installed in this process")
        _ACTIVE[framework] = integration


def get_active_integration(
    framework: FrameworkName,
) -> FrameworkOperatorIntegration | None:
    with _LOCK:
        return _ACTIVE.get(framework)


def clear_active_integration(framework: FrameworkName) -> None:
    with _LOCK:
        _ACTIVE.pop(framework, None)


__all__ = [
    "FrameworkName",
    "clear_active_integration",
    "get_active_integration",
    "set_active_integration",
]
