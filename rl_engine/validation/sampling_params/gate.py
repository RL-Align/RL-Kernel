# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Framework-raw sampling normalization and the fail-closed sampling gate.

Normalization maps every framework's raw expression of "no sampling" onto the
single canonical sentinel before comparison. The gate then requires all four
canonical layers of a record to be identity and fails closed otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from rl_engine.validation.cross_config.schema import SerializableModel
from rl_engine.validation.sampling_params.schema import (
    MIN_P_DISABLED,
    TEMPERATURE_SENTINEL,
    TOP_K_DISABLED,
    TOP_P_IDENTITY,
    CanonicalSamplingLayer,
    CanonicalSamplingRecord,
    SamplingGateStatus,
)


class SamplingParamsError(ValueError):
    """Raised when framework-raw sampling parameters are structurally invalid."""


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SamplingParamsError(f"{field_name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SamplingParamsError(f"{field_name} must be finite, got {value!r}")
    return number


def _normalize_top_k(value: Any) -> Optional[int]:
    if value is None:
        return TOP_K_DISABLED
    if isinstance(value, bool):
        raise SamplingParamsError("top_k must be an integer")
    if isinstance(value, float):
        if not value.is_integer():
            raise SamplingParamsError(f"top_k must be an integer, got {value!r}")
        value = int(value)
    if not isinstance(value, int):
        raise SamplingParamsError(f"top_k must be an integer, got {value!r}")
    if value in (0, -1):
        return TOP_K_DISABLED
    if value > 0:
        return value
    raise SamplingParamsError(f"top_k must be 0, -1 or positive, got {value!r}")


def _normalize_top_p(value: Any) -> Optional[float]:
    if value is None:
        return TOP_P_IDENTITY
    number = _finite_float(value, "top_p")
    if number == 1.0:
        return TOP_P_IDENTITY
    if 0.0 < number < 1.0:
        return number
    raise SamplingParamsError(f"top_p must be in (0, 1], got {value!r}")


def _normalize_min_p(value: Any) -> Optional[float]:
    if value is None:
        return MIN_P_DISABLED
    number = _finite_float(value, "min_p")
    if number == 0.0:
        return MIN_P_DISABLED
    if 0.0 < number <= 1.0:
        return number
    raise SamplingParamsError(f"min_p must be in [0, 1], got {value!r}")


def normalize_sampling_layer(
    *,
    temperature: Any = None,
    top_k: Any = None,
    top_p: Any = None,
    min_p: Any = None,
) -> CanonicalSamplingLayer:
    """Normalize one layer of framework-raw sampling parameters.

    Structural errors raise :class:`SamplingParamsError`. The record builder must
    surface such an error as a fail-closed case; it must never substitute a
    sentinel and let an unnormalizable layer look like identity.
    """
    temp = (
        TEMPERATURE_SENTINEL if temperature is None else _finite_float(temperature, "temperature")
    )
    if temp <= 0.0:
        raise SamplingParamsError(f"temperature must be positive, got {temperature!r}")
    return CanonicalSamplingLayer(
        temperature=temp,
        top_k=_normalize_top_k(top_k),
        top_p=_normalize_top_p(top_p),
        min_p=_normalize_min_p(min_p),
    )


@dataclass(frozen=True)
class SamplingGateResult(SerializableModel):
    """Typed, fail-closed result of the sampling-parameters gate."""

    case_id: str
    status: SamplingGateStatus
    passed: bool
    failing_layer: Optional[str] = None
    failing_field: Optional[str] = None
    canonical_fingerprint: Optional[str] = None
    detail: str = ""


def _first_non_identity_field(layer: CanonicalSamplingLayer) -> Optional[str]:
    if layer.temperature != TEMPERATURE_SENTINEL:
        return "temperature"
    if layer.top_k is not None:
        return "top_k"
    if layer.top_p is not None:
        return "top_p"
    if layer.min_p is not None:
        return "min_p"
    return None


def _result(
    record: CanonicalSamplingRecord,
    status: SamplingGateStatus,
    *,
    failing_layer: Optional[str] = None,
    failing_field: Optional[str] = None,
    detail: str = "",
) -> SamplingGateResult:
    return SamplingGateResult(
        case_id=record.case_id,
        status=status,
        passed=status is SamplingGateStatus.PASS,
        failing_layer=failing_layer,
        failing_field=failing_field,
        canonical_fingerprint=record.canonical_fingerprint(),
        detail=detail,
    )


def evaluate_sampling_gate(record: CanonicalSamplingRecord) -> SamplingGateResult:
    """Evaluate one canonical sampling record; never let a non-identity pass.

    Order is fail-closed and diagnostic-first: a missing layer or a silent
    fallback is reported before the identity/drift mismatch it may also cause.
    """
    layers = record.layers()

    missing = [name for name, layer in layers.items() if layer is None]
    if missing:
        return _result(
            record,
            SamplingGateStatus.MISSING_PROVENANCE,
            failing_layer=missing[0].value,
            detail="missing sampling layer(s): " + ", ".join(name.value for name in missing),
        )

    fallback = tuple(record.provenance.get("fallback_layers") or ())
    if fallback:
        return _result(
            record,
            SamplingGateStatus.SILENT_FALLBACK,
            failing_layer=str(fallback[0]),
            detail="sampling layer(s) resolved through a fallback: "
            + ", ".join(str(name) for name in fallback),
        )

    for name, layer in layers.items():
        assert layer is not None  # narrow for the type checker; missing handled above
        field_name = _first_non_identity_field(layer)
        if field_name is not None:
            return _result(
                record,
                SamplingGateStatus.SAMPLING_PARAMS_MISMATCH,
                failing_layer=name.value,
                failing_field=field_name,
                detail=f"{name.value}.{field_name} is not the identity sentinel",
            )

    return _result(
        record,
        SamplingGateStatus.PASS,
        detail="all four canonical sampling layers are identity",
    )
