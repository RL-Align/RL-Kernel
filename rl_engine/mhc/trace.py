# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Boundary trace for P1 (first-divergence localization).

Every operator boundary records (name, dtype, shape, sha256 of raw bytes).
This is the P1-local stand-in for the Foundation ``TraceEnvelope``; the
``notes`` map carries the arithmetic provenance issue #2 requires a trace to
record -- reduction tree, rsqrt vs 1/sqrt, FMA policy, rounding points and
addition order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rl_engine.mhc.contract import tensor_sha256


@dataclass(frozen=True)
class BoundaryRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "sha256": self.sha256,
        }


@dataclass
class MHCTrace:
    """Ordered boundary hashes plus arithmetic provenance for one P1 run."""

    numeric_profile: str
    records: list[BoundaryRecord] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)

    def record(self, name: str, tensor: torch.Tensor) -> None:
        self.records.append(
            BoundaryRecord(
                name=name,
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=tuple(tensor.shape),
                sha256=tensor_sha256(tensor),
            )
        )

    def note(self, key: str, value: str) -> None:
        self.notes[key] = value

    def hashes(self) -> dict[str, str]:
        return {r.name: r.sha256 for r in self.records}

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_profile": self.numeric_profile,
            "records": [r.to_dict() for r in self.records],
            "notes": dict(self.notes),
        }


def first_divergence(a: MHCTrace, b: MHCTrace) -> str | None:
    """Name of the first boundary whose hash differs, or None if identical."""
    for ra, rb in zip(a.records, b.records, strict=False):
        if ra.name != rb.name:
            return ra.name
        if ra.sha256 != rb.sha256:
            return ra.name
    if len(a.records) != len(b.records):
        return "<record-count-mismatch>"
    return None


__all__ = ["BoundaryRecord", "MHCTrace", "first_divergence"]
