# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic first-error records shared by cross-configuration checks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

SCHEMA_VERSION = "cross_config.divergence.v1"
COMPARISON_SEMANTICS_VERSION = "cross_config.comparison_semantics.v2"


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


@dataclass(frozen=True)
class DivergenceRecord:
    """One lexicographically first mismatch plus the context needed to replay it."""

    tensor_name: str
    flat_index: int
    coordinates: tuple[int, ...]
    axis_names: tuple[str, ...]
    expected_value: Any
    actual_value: Any
    absolute_diff: float | None
    relative_diff: float | None
    expected_shape: tuple[int, ...]
    actual_shape: tuple[int, ...]
    expected_dtype: str
    actual_dtype: str
    reason: str = "value_mismatch"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.tensor_name:
            raise ValueError("tensor_name must not be empty")
        if self.flat_index < 0:
            raise ValueError("flat_index must be non-negative")
        if len(self.coordinates) != len(self.axis_names):
            raise ValueError("coordinates and axis_names must have equal length")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported divergence schema_version")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "tensor_name": self.tensor_name,
            "flat_index": self.flat_index,
            "coordinates": list(self.coordinates),
            "axis_names": list(self.axis_names),
            "expected_value": _json_value(self.expected_value),
            "actual_value": _json_value(self.actual_value),
            "absolute_diff": self.absolute_diff,
            "relative_diff": self.relative_diff,
            "expected_shape": list(self.expected_shape),
            "actual_shape": list(self.actual_shape),
            "expected_dtype": self.expected_dtype,
            "actual_dtype": self.actual_dtype,
            "reason": self.reason,
            "provenance": _json_value(self.provenance),
        }
        payload["coordinate_map"] = {
            name: coordinate for name, coordinate in zip(self.axis_names, self.coordinates)
        }
        return payload


@dataclass(frozen=True)
class DivergenceReport:
    """Summary of a comparison, retaining only the first mismatch in detail."""

    tensor_name: str
    mismatch_count: int
    compared_count: int
    passed: bool
    first: DivergenceRecord | None = None
    metadata_errors: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.mismatch_count < 0 or self.compared_count < 0:
            raise ValueError("divergence counts must be non-negative")
        if self.mismatch_count > self.compared_count:
            raise ValueError("mismatch_count cannot exceed compared_count")
        if self.passed != (self.mismatch_count == 0 and not self.metadata_errors):
            raise ValueError("passed must agree with mismatch and metadata status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tensor_name": self.tensor_name,
            "mismatch_count": self.mismatch_count,
            "compared_count": self.compared_count,
            "passed": self.passed,
            "metadata_errors": list(self.metadata_errors),
            "first": self.first.to_dict() if self.first is not None else None,
        }


def first_divergence(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    tensor_name: str = "tensor",
    active_mask: torch.Tensor | None = None,
    atol: float = 0.0,
    rtol: float = 0.0,
    axis_names: Sequence[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
    mismatch_mask: torch.Tensor | None = None,
    reason: str = "value_mismatch",
) -> DivergenceReport:
    """Compare tensors and return a stable, structured first-error record.

    The first mismatch is selected in row-major flattened order. Empty tensors and
    masks with no active elements are rejected so a vacuous sample cannot pass.
    """

    if expected.numel() == 0 or actual.numel() == 0:
        raise ValueError("divergence comparison requires non-empty tensors")
    if not reason:
        raise ValueError("reason must not be empty")
    axes = _axes(expected.ndim, axis_names)
    if expected.shape != actual.shape:
        record = DivergenceRecord(
            tensor_name=tensor_name,
            flat_index=0,
            coordinates=(),
            axis_names=(),
            expected_value=None,
            actual_value=None,
            absolute_diff=None,
            relative_diff=None,
            expected_shape=tuple(expected.shape),
            actual_shape=tuple(actual.shape),
            expected_dtype=str(expected.dtype).replace("torch.", ""),
            actual_dtype=str(actual.dtype).replace("torch.", ""),
            reason="shape_mismatch",
            provenance=provenance or {},
        )
        return DivergenceReport(
            tensor_name,
            0,
            0,
            False,
            record,
            metadata_errors=("shape_mismatch",),
        )
    if expected.dtype != actual.dtype:
        record = DivergenceRecord(
            tensor_name=tensor_name,
            flat_index=0,
            coordinates=tuple(0 for _ in expected.shape),
            axis_names=axes,
            expected_value=None,
            actual_value=None,
            absolute_diff=None,
            relative_diff=None,
            expected_shape=tuple(expected.shape),
            actual_shape=tuple(actual.shape),
            expected_dtype=str(expected.dtype).replace("torch.", ""),
            actual_dtype=str(actual.dtype).replace("torch.", ""),
            reason="dtype_mismatch",
            provenance=provenance or {},
        )
        return DivergenceReport(
            tensor_name,
            0,
            0,
            False,
            record,
            metadata_errors=("dtype_mismatch",),
        )
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be finite and non-negative")
    expected_cpu = expected.detach().cpu()
    actual_cpu = actual.detach().cpu()
    mask = torch.ones(expected_cpu.shape, dtype=torch.bool)
    if active_mask is not None:
        if active_mask.shape != expected.shape:
            raise ValueError("active_mask shape must match compared tensors")
        mask = active_mask.detach().cpu().to(dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError("divergence comparison requires at least one active element")
    if mismatch_mask is not None:
        if mismatch_mask.shape != expected.shape:
            raise ValueError("mismatch_mask shape must match compared tensors")
        supplied = mismatch_mask.detach().cpu().to(dtype=torch.bool) & mask
        computed = compute_mismatch_mask(expected_cpu, actual_cpu, mask, atol, rtol)
        if not torch.equal(supplied, computed):
            raise ValueError("mismatch_mask does not match tensor comparison semantics")
        mismatch = supplied
    else:
        mismatch = compute_mismatch_mask(expected_cpu, actual_cpu, mask, atol, rtol)
    compared_count = int(mask.sum().item())
    mismatch_count = int(mismatch.sum().item())
    if not mismatch_count:
        return DivergenceReport(tensor_name, 0, compared_count, True)
    flat_index = int(torch.nonzero(mismatch.reshape(-1), as_tuple=False)[0].item())
    coordinates = tuple(
        int(item) for item in torch.unravel_index(torch.tensor(flat_index), expected.shape)
    )
    expected_value = expected_cpu.reshape(-1)[flat_index].item()
    actual_value = actual_cpu.reshape(-1)[flat_index].item()
    finite_pair = bool(
        torch.isfinite(expected_cpu.reshape(-1)[flat_index])
        and torch.isfinite(actual_cpu.reshape(-1)[flat_index])
    )
    absolute = abs(float(actual_value) - float(expected_value)) if finite_pair else None
    record_reason = reason
    if not finite_pair:
        record_reason = "non_finite"
    elif (
        atol == 0.0
        and rtol == 0.0
        and expected_cpu.is_floating_point()
        and expected_value == 0
        and actual_value == 0
        and bool(torch.signbit(expected_cpu.reshape(-1)[flat_index]))
        != bool(torch.signbit(actual_cpu.reshape(-1)[flat_index]))
    ):
        record_reason = "signed_zero"
    denominator = max(abs(float(expected_value)), 1e-12) if absolute is not None else None
    return DivergenceReport(
        tensor_name=tensor_name,
        mismatch_count=mismatch_count,
        compared_count=compared_count,
        passed=False,
        first=DivergenceRecord(
            tensor_name=tensor_name,
            flat_index=flat_index,
            coordinates=coordinates,
            axis_names=axes,
            expected_value=expected_value,
            actual_value=actual_value,
            absolute_diff=absolute,
            relative_diff=(absolute / denominator if absolute is not None else None),
            expected_shape=tuple(expected.shape),
            actual_shape=tuple(actual.shape),
            expected_dtype=str(expected.dtype).replace("torch.", ""),
            actual_dtype=str(actual.dtype).replace("torch.", ""),
            reason=record_reason,
            provenance=provenance or {},
        ),
    )


def _axes(ndim: int, names: Sequence[str] | None) -> tuple[str, ...]:
    if names is None:
        return tuple(f"dim_{index}" for index in range(ndim))
    result = tuple(str(name) for name in names)
    if len(result) != ndim:
        raise ValueError("axis_names length must match tensor rank")
    return result


def compute_mismatch_mask(
    expected: torch.Tensor,
    actual: torch.Tensor,
    active_mask: torch.Tensor,
    fixed_threshold: float,
    relative_tolerance: float = 0.0,
) -> torch.Tensor:
    """Apply comparison_semantics.v2 rules shared by comparison and resume.

    Matching signed infinities are equal; NaN and other non-finite pairs are
    mismatches. Signed zeros differ when the absolute threshold is zero.
    This intentionally supersedes the old abs-difference-only predicate.
    Persisted diagnostics carry COMPARISON_SEMANTICS_VERSION, so attempts
    recorded with older semantics do not qualify for resume and are rerun.
    The live comparator separately rejects non-finite active score artifacts.
    """

    if expected.shape != actual.shape or active_mask.shape != expected.shape:
        raise ValueError("mismatch operands and active_mask must have matching shapes")
    if (
        not math.isfinite(fixed_threshold)
        or not math.isfinite(relative_tolerance)
        or fixed_threshold < 0.0
        or relative_tolerance < 0.0
    ):
        raise ValueError("tolerances must be finite and non-negative")
    active = active_mask.to(dtype=torch.bool)
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    same_infinity = torch.isinf(expected) & torch.isinf(actual) & (expected == actual)
    tolerance = fixed_threshold + relative_tolerance * torch.abs(expected)
    mismatch = active & (
        (~finite & ~same_infinity) | (finite & (torch.abs(actual - expected) > tolerance))
    )
    if fixed_threshold == 0.0 and expected.is_floating_point():
        mismatch |= (
            active
            & (expected == 0)
            & (actual == 0)
            & (torch.signbit(expected) != torch.signbit(actual))
        )
    return mismatch


def runtime_identity(
    provenance: Mapping[str, Any], *, model_version: str, checkpoint_id: str
) -> dict[str, Any]:
    """Normalize runtime provenance for comparison and resume using one schema."""

    required = ("requested", "actual", "implementation_fingerprint", "rank", "world_size")
    missing = [key for key in required if key not in provenance]
    if missing:
        raise ValueError("runtime provenance is missing required fields: " + ", ".join(missing))
    requested = provenance["requested"]
    actual = provenance["actual"]
    if not isinstance(requested, Mapping) or not isinstance(actual, Mapping):
        raise ValueError("runtime provenance requested and actual fields must be mappings")
    requested_logp = requested.get("logp")
    actual_logp = actual.get("logp")
    if not isinstance(requested_logp, Mapping) or not isinstance(actual_logp, Mapping):
        raise ValueError("runtime provenance must include requested and actual logp mappings")
    if "backend" not in requested_logp or "backend" not in actual_logp:
        raise ValueError("runtime provenance logp mappings must include backend")
    return {
        "requested_backend": requested_logp["backend"],
        "actual_backend": actual_logp["backend"],
        "implementation_fingerprint": provenance["implementation_fingerprint"],
        "rank": provenance["rank"],
        "world_size": provenance["world_size"],
        "model_version": model_version,
        "checkpoint_id": checkpoint_id,
    }


__all__ = [
    "COMPARISON_SEMANTICS_VERSION",
    "DivergenceRecord",
    "DivergenceReport",
    "SCHEMA_VERSION",
    "compute_mismatch_mask",
    "first_divergence",
    "runtime_identity",
]
