# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Require an upstream state verdict before attention comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl_engine.kernels.dsv4.attention.contract import SCHEMA_VERSION_STATE_GATE
from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status


@dataclass(frozen=True)
class StateGateVerdict:
    status: str
    source: str
    recent_checksum: str | None = None
    main_checksum: str | None = None
    index_checksum: str | None = None
    generation: int | None = None
    schema_version: str = SCHEMA_VERSION_STATE_GATE
    details: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "recent_checksum": self.recent_checksum,
            "main_checksum": self.main_checksum,
            "index_checksum": self.index_checksum,
            "generation": self.generation,
            "schema_version": self.schema_version,
            "details": self.details,
        }


def require_state_gate(verdict: StateGateVerdict | None) -> StateGateVerdict:
    """Reject missing or failed upstream state verdicts."""

    if verdict is None:
        raise DSv4FailClosedError(
            DSv4Status.STATE_BYTES_MISMATCH,
            "attention comparison requires an explicit state-byte verdict",
        )
    if verdict.status != DSv4Status.PASS.value:
        raise DSv4FailClosedError(
            DSv4Status.STATE_BYTES_MISMATCH,
            "state bytes failed; stopping attention/P3/P7 attribution",
            state_status=verdict.status,
            source=verdict.source,
            details=verdict.details,
        )
    return verdict


def synthetic_pass_verdict(*, tag: str) -> StateGateVerdict:
    """Build an explicit PASS verdict for synthetic fixtures."""

    return StateGateVerdict(
        status=DSv4Status.PASS.value,
        source="synthetic_recorded",
        recent_checksum=f"synthetic:{tag}:recent",
        main_checksum=f"synthetic:{tag}:main",
        index_checksum=f"synthetic:{tag}:index",
        generation=0,
        details="T06 synthetic recorded fixture; not live T03/T04/T05 state",
    )
