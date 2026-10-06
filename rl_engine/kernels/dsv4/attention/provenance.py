# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Kernel and reduction metadata returned by the selected backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rl_engine.kernels.dsv4.attention.contract import (
    DOWNCAST_FINAL_WRITE,
    QK_LAUNCH,
    REDUCTION_TREE_SEQUENTIAL_D,
    REDUCTION_TREE_SEQUENTIAL_H,
    REDUCTION_TREE_SEQUENTIAL_J,
    SOFTMAX_CTA_NOTE,
)


@dataclass(frozen=True)
class ActualProvenance:
    backend: str
    kernel_id: str
    device: str
    dtype: str
    reduction_tree_qk: str = REDUCTION_TREE_SEQUENTIAL_D
    reduction_tree_softmax: str = REDUCTION_TREE_SEQUENTIAL_J
    reduction_tree_head: str = REDUCTION_TREE_SEQUENTIAL_H
    qk_launch: str = QK_LAUNCH
    softmax_schedule: str = SOFTMAX_CTA_NOTE
    downcast_point: str = DOWNCAST_FINAL_WRITE
    tile: str = "none_ws1_reference"
    unroll: str = "sequential"
    warp: str | None = None
    split_kv: str = "disabled"
    num_splits: int = 1
    debug: bool = False
    fallback: bool = False
    fallback_reason: str | None = None
    build_fingerprint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fallback and self.fallback_reason is None:
            raise ValueError("fallback requires fallback_reason; silent fallback is forbidden")
        if self.split_kv != "disabled" or self.num_splits != 1:
            raise ValueError("actual provenance cannot record Split-KV as enabled")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "backend": self.backend,
            "kernel_id": self.kernel_id,
            "device": self.device,
            "dtype": self.dtype,
            "reduction_tree_qk": self.reduction_tree_qk,
            "reduction_tree_softmax": self.reduction_tree_softmax,
            "reduction_tree_head": self.reduction_tree_head,
            "qk_launch": self.qk_launch,
            "softmax_schedule": self.softmax_schedule,
            "downcast_point": self.downcast_point,
            "tile": self.tile,
            "unroll": self.unroll,
            "warp": self.warp,
            "split_kv": self.split_kv,
            "num_splits": self.num_splits,
            "debug": self.debug,
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "build_fingerprint": self.build_fingerprint,
        }
        payload.update(self.extra)
        return payload
