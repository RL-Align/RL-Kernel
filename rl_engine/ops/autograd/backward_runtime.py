# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Runtime record of the kernel that actually executed a candidate backward."""

from __future__ import annotations

from threading import Lock
from typing import Any

_LOCK = Lock()
_EVENTS: dict[str, dict[str, Any]] = {}


def record_backward(
    kind: str,
    *,
    kernel_id: str,
    impl: str,
    family: str,
) -> None:
    with _LOCK:
        previous = _EVENTS.get(kind)
        count = 1 if previous is None else int(previous["execution_count"]) + 1
        kernel_ids = tuple(part for part in kernel_id.split("+") if part)
        _EVENTS[kind] = {
            "kind": kind,
            "implementation_ids": list(kernel_ids),
            "kernel_ids": list(kernel_ids),
            "kernel_id": kernel_id,
            "impl": impl,
            "family": family,
            "execution_count": count,
        }


def snapshot_backward_runtime() -> dict[str, dict[str, Any]]:
    with _LOCK:
        return {key: dict(value) for key, value in _EVENTS.items()}


def reset_backward_runtime() -> None:
    with _LOCK:
        _EVENTS.clear()


__all__ = [
    "record_backward",
    "reset_backward_runtime",
    "snapshot_backward_runtime",
]
