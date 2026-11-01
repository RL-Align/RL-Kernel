# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Versioned canonical sampling record for the T02 sampling-parameters gate.

The consistency contract requires both the training side and the rollout side
to score raw model log probabilities, i.e. identity sampling. Framework-raw
sampling parameters are normalized to a single canonical form *before* they are
compared (see ``p7-task-contract.v3`` §二 and T02 §3.3).

Canonical identity sentinels are ``temperature=1.0``, ``top_k=DISABLED``,
``top_p=IDENTITY`` and ``min_p=DISABLED``. The two word sentinels are encoded
as ``None`` internally so that an identity test is a plain ``is None`` check.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

from rl_engine.validation.cross_config.schema import SerializableModel

SAMPLING_RECORD_VERSION = "p7-sampling-record-v1"

TEMPERATURE_SENTINEL = 1.0
TOP_K_DISABLED: Optional[int] = None
TOP_P_IDENTITY: Optional[float] = None
MIN_P_DISABLED: Optional[float] = None


class SamplingLayerName(str, Enum):
    """The four sampling layers the gate checks (contract §3.3)."""

    REQUESTED = "requested"
    NORMALIZED = "normalized"
    MATERIALIZED = "materialized"
    RUNTIME_ACTUAL = "runtime_actual"

    @classmethod
    def ordered(cls) -> tuple["SamplingLayerName", ...]:
        return (cls.REQUESTED, cls.NORMALIZED, cls.MATERIALIZED, cls.RUNTIME_ACTUAL)


class SamplingGateStatus(str, Enum):
    """Fail-closed verdicts of the sampling-parameters gate."""

    PASS = "pass"
    SAMPLING_PARAMS_MISMATCH = "sampling_params_mismatch"
    MISSING_PROVENANCE = "missing_provenance"
    SILENT_FALLBACK = "silent_fallback"


class SamplingValueSource(str, Enum):
    """Where a layer value came from (provenance only, not part of the value)."""

    CONFIG = "config"
    API = "api"
    FRAMEWORK_DEFAULT = "framework_default"
    FALLBACK = "fallback"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CanonicalSamplingLayer(SerializableModel):
    """One sampling layer after framework-raw values are normalized.

    ``top_k is None`` means ``DISABLED``, ``top_p is None`` means ``IDENTITY``
    and ``min_p is None`` means ``DISABLED``.
    """

    temperature: float = TEMPERATURE_SENTINEL
    top_k: Optional[int] = TOP_K_DISABLED
    top_p: Optional[float] = TOP_P_IDENTITY
    min_p: Optional[float] = MIN_P_DISABLED
    source: Optional[str] = None
    raw: Optional[Any] = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise ValueError("temperature must be a real number")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int)
        ):
            raise ValueError("top_k must be an integer or None")
        for name in ("top_p", "min_p"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError(f"{name} must be a real number or None")

    def is_identity(self) -> bool:
        """True when the layer equals the canonical identity sentinel."""
        return (
            self.temperature == TEMPERATURE_SENTINEL
            and self.top_k == TOP_K_DISABLED
            and self.top_p == TOP_P_IDENTITY
            and self.min_p == MIN_P_DISABLED
        )

    def canonical_values(self) -> dict[str, Any]:
        """Value-only view with canonical numeric types, used for hashing."""
        return {
            "temperature": float(self.temperature),
            "top_k": None if self.top_k is None else int(self.top_k),
            "top_p": None if self.top_p is None else float(self.top_p),
            "min_p": None if self.min_p is None else float(self.min_p),
        }


@dataclass
class CanonicalSamplingRecord(SerializableModel):
    """The canonical sampling record shared by both sides of one case."""

    case_id: str
    engine: str
    requested: Optional[CanonicalSamplingLayer] = None
    normalized: Optional[CanonicalSamplingLayer] = None
    materialized: Optional[CanonicalSamplingLayer] = None
    runtime_actual: Optional[CanonicalSamplingLayer] = None
    record_version: str = SAMPLING_RECORD_VERSION
    provenance: Mapping[str, Any] = field(default_factory=dict)
    fallback_layers: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, AbcMapping):
            raise ValueError("provenance must be a mapping")
        if self.record_version != SAMPLING_RECORD_VERSION:
            raise ValueError(
                f"unsupported sampling record_version {self.record_version!r}, "
                f"expected {SAMPLING_RECORD_VERSION!r}"
            )

    def layer(self, name: SamplingLayerName) -> Optional[CanonicalSamplingLayer]:
        return getattr(self, name.value)

    def layers(self) -> dict[SamplingLayerName, Optional[CanonicalSamplingLayer]]:
        return {name: self.layer(name) for name in SamplingLayerName.ordered()}

    def canonical_fingerprint(self) -> Optional[str]:
        """Stable hash over the four canonical layers, or ``None`` if incomplete."""
        payload: dict[str, Any] = {"record_version": self.record_version, "layers": {}}
        for name in SamplingLayerName.ordered():
            layer = self.layer(name)
            if layer is None:
                return None
            payload["layers"][name.value] = layer.canonical_values()
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()
