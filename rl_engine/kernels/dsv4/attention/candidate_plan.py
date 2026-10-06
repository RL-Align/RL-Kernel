# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Canonical candidate plan: compressed prefix first, recent-128 second, sink not a column."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor

from rl_engine.kernels.dsv4.attention.contract import (
    CANDIDATE_ORDER_COMPRESSED_THEN_RECENT,
    HEAD_DIM,
    RECENT_WINDOW,
    SCHEMA_VERSION_CANDIDATE_PLAN,
    SOFTMAX_MODE_ONE_DENOMINATOR,
    SPLIT_KV_DISABLED,
    AttentionKind,
    LayerType,
    attention_kind_for_layer,
)
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status


def _as_layer_type(value: LayerType | str) -> LayerType:
    if isinstance(value, LayerType):
        return value
    try:
        return LayerType(value)
    except ValueError as exc:
        raise DSv4FailClosedError(
            DSv4Status.INVALID_LAYER_TYPE,
            f"layer_type must be C0|C4|C128, got {value!r}",
        ) from exc


@dataclass(frozen=True)
class CandidatePlan:
    """Candidate layout and reduction constraints for recorded attention."""

    layer_type: LayerType
    n_compressed: int
    n_recent: int
    valid: Tensor
    softmax_mode: str = SOFTMAX_MODE_ONE_DENOMINATOR
    candidate_order: str = CANDIDATE_ORDER_COMPRESSED_THEN_RECENT
    split_kv_mode: str = SPLIT_KV_DISABLED
    sink_has_v: bool = False
    global_visible: bool = True
    kind: Tensor | None = None
    dequant_manifest: Mapping[str, Any] | None = None
    num_splits: int = 1
    atomic_reduction: bool = False
    schema_version: str = SCHEMA_VERSION_CANDIDATE_PLAN

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_type", _as_layer_type(self.layer_type))
        if not isinstance(self.n_compressed, int) or isinstance(self.n_compressed, bool):
            raise DSv4FailClosedError(
                DSv4Status.SCHEMA_MISMATCH, f"n_compressed must be int, got {self.n_compressed!r}"
            )
        if not isinstance(self.n_recent, int) or isinstance(self.n_recent, bool):
            raise DSv4FailClosedError(
                DSv4Status.SCHEMA_MISMATCH, f"n_recent must be int, got {self.n_recent!r}"
            )
        if self.n_compressed < 0 or self.n_recent < 0:
            raise DSv4FailClosedError(
                DSv4Status.INVALID_CANDIDATE_ORDER,
                f"n_compressed={self.n_compressed} n_recent={self.n_recent} must be >= 0",
            )
        if self.valid.dtype != torch.bool or self.valid.dim() != 1:
            raise DSv4FailClosedError(
                DSv4Status.SCHEMA_MISMATCH,
                f"valid must be 1-D bool [N], got dtype={self.valid.dtype} "
                f"shape={tuple(self.valid.shape)}",
            )
        n = int(self.valid.shape[0])
        if n != self.n_candidates:
            raise DSv4FailClosedError(
                DSv4Status.INVALID_CANDIDATE_ORDER,
                f"valid length {n} != n_compressed+n_recent {self.n_candidates}",
            )
        if self.kind is not None:
            if self.kind.shape != self.valid.shape or self.kind.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                raise DSv4FailClosedError(
                    DSv4Status.SCHEMA_MISMATCH,
                    "kind must be integer [N] with 0=compressed, 1=recent",
                )
        if self.dequant_manifest is not None:
            raise DSv4FailClosedError(
                DSv4Status.UNSUPPORTED_CAPABILITY,
                "T06 does not apply FP8 dequant; dequant_manifest must be unset or consumed by T04",
            )

    @property
    def n_candidates(self) -> int:
        return self.n_compressed + self.n_recent

    @property
    def attention_kind(self) -> AttentionKind:
        return attention_kind_for_layer(self.layer_type)

    def validate_for_kv(self, k: Tensor, v: Tensor) -> None:
        if k.dim() != 2 or v.dim() != 2:
            raise DSv4FailClosedError(
                DSv4Status.SCHEMA_MISMATCH,
                f"K/V must be [N, {HEAD_DIM}], got k={tuple(k.shape)} v={tuple(v.shape)}",
            )
        if k.shape != v.shape:
            raise DSv4FailClosedError(
                DSv4Status.SCHEMA_MISMATCH,
                f"K/V shape mismatch k={tuple(k.shape)} v={tuple(v.shape)}",
            )
        if k.shape[-1] != HEAD_DIM:
            raise DSv4FailClosedError(
                DSv4Status.SCHEMA_MISMATCH,
                f"head dim must be {HEAD_DIM}, got {k.shape[-1]}",
            )
        if int(k.shape[0]) != self.n_candidates:
            raise DSv4FailClosedError(
                DSv4Status.INVALID_CANDIDATE_ORDER,
                f"K rows {k.shape[0]} != plan N {self.n_candidates}",
            )
        if self.softmax_mode != SOFTMAX_MODE_ONE_DENOMINATOR:
            raise DSv4FailClosedError(
                DSv4Status.MULTIPLE_SOFTMAX_DENOMINATORS,
                f"softmax_mode must be {SOFTMAX_MODE_ONE_DENOMINATOR!r}, got {self.softmax_mode!r}",
            )
        if self.candidate_order != CANDIDATE_ORDER_COMPRESSED_THEN_RECENT:
            raise DSv4FailClosedError(
                DSv4Status.INVALID_CANDIDATE_ORDER,
                f"candidate_order must be {CANDIDATE_ORDER_COMPRESSED_THEN_RECENT!r}",
            )
        if self.split_kv_mode != SPLIT_KV_DISABLED or self.num_splits != 1:
            raise DSv4FailClosedError(
                DSv4Status.FORBIDDEN_SPLIT_REDUCTION,
                f"Split-KV is forbidden: mode={self.split_kv_mode!r} num_splits={self.num_splits}",
            )
        if self.atomic_reduction:
            raise DSv4FailClosedError(
                DSv4Status.FORBIDDEN_ATOMIC_REDUCTION,
                "atomic partial accumulation is forbidden",
            )
        if self.sink_has_v:
            raise DSv4FailClosedError(
                DSv4Status.INVALID_SINK_SEMANTICS,
                "sink participates in the denominator only and must not have a V column",
            )
        if not self.global_visible:
            raise DSv4FailClosedError(
                DSv4Status.MISSING_GLOBAL_VISIBILITY,
                "candidate KV must be globally visible before attention",
            )
        if self.layer_type is LayerType.C0 and self.n_compressed != 0:
            raise DSv4FailClosedError(
                DSv4Status.INVALID_COMPRESSION_PLAN,
                "C0 must not create compressed rows (n_compressed must be 0)",
            )
        if self.n_recent > RECENT_WINDOW:
            raise DSv4FailClosedError(
                DSv4Status.INVALID_CANDIDATE_ORDER,
                f"n_recent={self.n_recent} exceeds recent window {RECENT_WINDOW}",
            )
        if self.kind is not None:
            expected = torch.empty(
                self.n_candidates, dtype=self.kind.dtype, device=self.kind.device
            )
            if self.n_compressed:
                expected[: self.n_compressed] = 0
            if self.n_recent:
                expected[self.n_compressed :] = 1
            if not torch.equal(self.kind, expected):
                raise DSv4FailClosedError(
                    DSv4Status.INVALID_CANDIDATE_ORDER,
                    "kind tags must be compressed(0) prefix then recent(1)",
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layer_type": self.layer_type.value,
            "attention_kind": self.attention_kind.value,
            "n_compressed": self.n_compressed,
            "n_recent": self.n_recent,
            "n_candidates": self.n_candidates,
            "n_valid": int(self.valid.sum().item()) if self.valid.numel() else 0,
            "softmax_mode": self.softmax_mode,
            "candidate_order": self.candidate_order,
            "split_kv_mode": self.split_kv_mode,
            "sink_has_v": self.sink_has_v,
            "global_visible": self.global_visible,
            "num_splits": self.num_splits,
            "atomic_reduction": self.atomic_reduction,
            "dequant_manifest": dict(self.dequant_manifest) if self.dequant_manifest else None,
        }
