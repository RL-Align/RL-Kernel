# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Minimal test doubles for the Foundation ABI fields T05 consumes.

Only the fields listed in p7-task-contract.v3 ("Foundation ABI consumption"
table) that T05 forwards are modelled. Values are stored verbatim; the doubles
never infer or default a missing field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

DOUBLE_OF = "T01 Foundation ABI v1: SemanticTensor and TraceEnvelope"

SEMANTIC_TENSOR_DOUBLE_SCHEMA = "test-double.semantic_tensor.v0"
TRACE_ENVELOPE_DOUBLE_SCHEMA = "test-double.trace_envelope.v0"

TRACE_ENVELOPE_REQUIRED = (
    "producer",
    "upstream_contract_id",
    "producer_schema_version",
    "run_id",
    "case_id",
    "absolute_layer",
    "pass_name",
    "global_token_ids",
    "requested_backend",
    "actual_backend",
    "kernel_build",
    "comparison_mode",
    "producer_verdict",
    "fixture_checksum",
)


@dataclass(frozen=True)
class SemanticTensorDouble:
    run_id: str
    checkpoint_id: str
    weight_id: str
    absolute_layer: int
    global_token_ids: tuple[int, ...]
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    boundary_hash: str
    schema_version: str = SEMANTIC_TENSOR_DOUBLE_SCHEMA


@dataclass(frozen=True)
class TraceEnvelopeDouble:
    """Producer evidence forwarded verbatim by T05; never recomputed or patched."""

    producer: str
    upstream_contract_id: str
    producer_schema_version: str
    run_id: str
    case_id: str
    absolute_layer: int
    pass_name: str
    global_token_ids: tuple[int, ...]
    requested_backend: str
    actual_backend: str
    kernel_build: str
    comparison_mode: str
    producer_verdict: str
    fixture_checksum: str
    extra: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = TRACE_ENVELOPE_DOUBLE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in TRACE_ENVELOPE_REQUIRED}
        payload["global_token_ids"] = list(self.global_token_ids)
        payload["extra"] = dict(self.extra)
        payload["schema_version"] = self.schema_version
        return payload


def envelope_missing_fields(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Fields required by T05 that a recorded envelope payload lacks."""

    return tuple(name for name in TRACE_ENVELOPE_REQUIRED if payload.get(name) in (None, ""))
