# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Offline MoE selected-logprob comparison and first-difference diagnosis.

The module consumes captured logits/logprobs. It does not run a model, dispatch
an EP collective, or certify an accelerator backend. A fixed reference is used
for strict output comparison and a separate Python/FP64 calculation is used as
an accuracy oracle.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from rl_engine.contracts.numerical import resolve_logprob_threshold

MOE_LOGPROB_DIAGNOSTICS_VERSION = "moe.logprob_diagnostics.v1"
PROBABILITY_RULE = "log_softmax_fp32_selected"


class MoELogprobError(ValueError):
    """Captured MoE logprob evidence violates the comparison contract."""


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise MoELogprobError("metadata keys must be strings")
            frozen[key] = _freeze_metadata(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise MoELogprobError("metadata must contain only finite JSON-compatible values")


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return value


def _digest(tensor: torch.Tensor) -> str:
    snapshot = tensor.detach().cpu().contiguous()
    raw = snapshot.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _snapshot(name: str, value: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise MoELogprobError(f"{name} must be a tensor")
    if value.device.type != "cpu" or value.layout != torch.strided:
        raise MoELogprobError(f"{name} must be a materialized CPU strided tensor")
    result = value.detach().clone().contiguous()
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result


def _index_vector(name: str, value: torch.Tensor, rows: int) -> torch.Tensor:
    result = _snapshot(name, value)
    if result.ndim != 1:
        raise MoELogprobError(f"{name} must be a one-dimensional vector")
    if result.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise MoELogprobError(f"{name} must use an integer dtype")
    result = result.to(dtype=torch.int64)
    if result.numel() != rows:
        raise MoELogprobError(f"{name} must contain exactly {rows} entries")
    return result


def _first_row_difference(left: torch.Tensor, right: torch.Tensor) -> int | None:
    """Return the first changed logical row for equal-shaped tensors."""

    if left.shape != right.shape or left.ndim == 0 or left.dtype != right.dtype:
        return None
    lhs = left.detach().contiguous().view(torch.uint8).reshape(left.shape[0], -1)
    rhs = right.detach().contiguous().view(torch.uint8).reshape(right.shape[0], -1)
    changed = torch.nonzero((lhs != rhs).any(dim=1), as_tuple=False)
    return int(changed[0].item()) if changed.numel() else None


@dataclass(frozen=True)
class MoELogprobCase:
    """Captured logical inputs and routing identity for one selected-logprob run."""

    logits: torch.Tensor
    target_ids: torch.Tensor
    layer_indices: torch.Tensor
    token_indices: torch.Tensor
    slot_indices: torch.Tensor
    expert_indices: torch.Tensor
    rank_indices: torch.Tensor
    checkpoint_id: str
    model_version: str
    mapping_run_id: str
    routing_weights: torch.Tensor | None = None
    probability_rule: str = PROBABILITY_RULE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        logits = _snapshot("logits", self.logits)
        if logits.ndim != 2 or not logits.shape[0] or not logits.shape[1]:
            raise MoELogprobError("logits must be a non-empty [rows, vocab] tensor")
        if logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise MoELogprobError("logits must use float16, bfloat16 or float32")
        if not bool(torch.isfinite(logits).all()):
            raise MoELogprobError("logits must be finite floating-point values")
        rows, vocab = logits.shape
        target_ids = _index_vector("target_ids", self.target_ids, rows)
        if bool((target_ids < 0).any()) or bool((target_ids >= vocab).any()):
            raise MoELogprobError("target_ids must be in the logits vocabulary range")
        for name in (
            "layer_indices",
            "token_indices",
            "slot_indices",
            "expert_indices",
            "rank_indices",
        ):
            values = _index_vector(name, getattr(self, name), rows)
            if bool((values < 0).any()):
                raise MoELogprobError(f"{name} must be non-negative")
            object.__setattr__(self, name, values)
        if self.routing_weights is not None:
            weights = _snapshot("routing_weights", self.routing_weights).reshape(-1)
            if weights.numel() != rows or not weights.is_floating_point():
                raise MoELogprobError("routing_weights must be a floating-point [rows] tensor")
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                raise MoELogprobError("routing_weights must be finite and non-negative")
            object.__setattr__(self, "routing_weights", weights)
        if not all(
            isinstance(value, str) and value.strip()
            for value in (self.checkpoint_id, self.model_version, self.mapping_run_id)
        ):
            raise MoELogprobError("checkpoint_id, model_version and mapping_run_id are required")
        if self.probability_rule != PROBABILITY_RULE:
            raise MoELogprobError(f"unsupported probability_rule {self.probability_rule!r}")
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "target_ids", target_ids)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def rows(self) -> int:
        return int(self.logits.shape[0])

    def provenance(self) -> dict[str, Any]:
        """Return bounded evidence; raw logits and weights stay in artifacts."""

        return {
            "schema_version": MOE_LOGPROB_DIAGNOSTICS_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "model_version": self.model_version,
            "mapping_run_id": self.mapping_run_id,
            "probability_rule": self.probability_rule,
            "rows": self.rows,
            "vocab_size": int(self.logits.shape[1]),
            "logits_dtype": str(self.logits.dtype),
            "logits_sha256": _digest(self.logits),
            "target_ids_sha256": _digest(self.target_ids),
            "layer_indices_sha256": _digest(self.layer_indices),
            "token_indices_sha256": _digest(self.token_indices),
            "slot_indices_sha256": _digest(self.slot_indices),
            "expert_indices_sha256": _digest(self.expert_indices),
            "rank_indices_sha256": _digest(self.rank_indices),
            "routing_weights_present": self.routing_weights is not None,
            "routing_weights_sha256": (
                _digest(self.routing_weights) if self.routing_weights is not None else None
            ),
            "metadata": _thaw_metadata(self.metadata),
        }


@dataclass(frozen=True)
class MoELogprobObservation:
    """One captured implementation output and its runtime identity."""

    values: torch.Tensor
    requested_backend: str
    actual_backend: str
    implementation_fingerprint: str
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        values = _snapshot("selected_logprobs", self.values)
        if values.ndim != 1:
            raise MoELogprobError("selected_logprobs must be a one-dimensional vector")
        if values.dtype != torch.float32:
            raise MoELogprobError("selected_logprobs must be float32")
        if not bool(torch.isfinite(values).all()):
            raise MoELogprobError("selected_logprobs must be finite")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.requested_backend,
                self.actual_backend,
                self.implementation_fingerprint,
            )
        ):
            raise MoELogprobError("backend and implementation identities are required")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise MoELogprobError("rank must be non-negative")
        if (
            isinstance(self.world_size, bool)
            or not isinstance(self.world_size, int)
            or self.world_size < 1
            or self.rank >= self.world_size
        ):
            raise MoELogprobError("world_size must be positive and contain rank")
        object.__setattr__(self, "values", values)


def independently_recompute_logprobs(case: MoELogprobCase) -> torch.Tensor:
    """Recompute selected logprobs with Python math/fsum, independently of torch log_softmax."""

    values: list[float] = []
    for row, target in zip(case.logits.double().tolist(), case.target_ids.tolist(), strict=True):
        maximum = max(row)
        log_sum_exp = math.log(math.fsum(math.exp(value - maximum) for value in row))
        values.append((row[target] - maximum) - log_sum_exp)
    return torch.tensor(values, dtype=torch.float64)


def _fixed_mismatch(expected: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    if expected.dtype != actual.dtype:
        return torch.ones(expected.shape, dtype=torch.bool)
    left = (
        expected.contiguous().view(torch.uint8).reshape(expected.numel(), expected.element_size())
    )
    right = actual.contiguous().view(torch.uint8).reshape(actual.numel(), actual.element_size())
    return (left != right).any(dim=1)


def _coordinate(case: MoELogprobCase, index: int) -> dict[str, int]:
    return {
        "layer": int(case.layer_indices[index]),
        "token": int(case.token_indices[index]),
        "slot": int(case.slot_indices[index]),
        "expert": int(case.expert_indices[index]),
        "rank": int(case.rank_indices[index]),
    }


def _first_record(
    case: MoELogprobCase,
    index: int,
    *,
    reason: str,
    expected: float | None = None,
    actual: float | None = None,
    independent: float | None = None,
) -> dict[str, Any]:
    absolute_error = None
    if expected is not None and actual is not None:
        absolute_error = abs(actual - expected)
    independent_absolute_error = None
    if independent is not None and actual is not None:
        independent_absolute_error = abs(actual - independent)
    record = {
        **_coordinate(case, index),
        "flat_index": index,
        "reason": reason,
        "expected": expected,
        "actual": actual,
        "independent": independent,
        "absolute_error": absolute_error,
        "independent_absolute_error": independent_absolute_error,
    }
    return record


def _identity_mismatch(
    reference: MoELogprobCase, candidate: MoELogprobCase
) -> tuple[str, int] | None:
    if reference.checkpoint_id != candidate.checkpoint_id:
        return "checkpoint_mismatch", 0
    if reference.model_version != candidate.model_version:
        return "model_version_mismatch", 0
    if reference.mapping_run_id != candidate.mapping_run_id:
        return "mapping_run_id_mismatch", 0
    if reference.probability_rule != candidate.probability_rule:
        return "probability_rule_mismatch", 0
    for name in (
        "layer_indices",
        "target_ids",
        "token_indices",
        "slot_indices",
        "expert_indices",
        "rank_indices",
    ):
        left = getattr(reference, name)
        right = getattr(candidate, name)
        if left.shape != right.shape or not torch.equal(left, right):
            count = min(left.numel(), right.numel())
            for index in range(count):
                if left.reshape(-1)[index] != right.reshape(-1)[index]:
                    return f"{name}_mismatch", index
            return f"{name}_shape_mismatch", count
    if reference.routing_weights is None or candidate.routing_weights is None:
        if reference.routing_weights is not None or candidate.routing_weights is not None:
            return "routing_weights_presence_mismatch", 0
    elif (
        reference.routing_weights.shape != candidate.routing_weights.shape
        or reference.routing_weights.dtype != candidate.routing_weights.dtype
        or not torch.equal(reference.routing_weights, candidate.routing_weights)
    ):
        row_index = _first_row_difference(reference.routing_weights, candidate.routing_weights)
        return "routing_weights_mismatch", 0 if row_index is None else row_index
    if (
        reference.logits.shape != candidate.logits.shape
        or reference.logits.dtype != candidate.logits.dtype
        or not torch.equal(reference.logits, candidate.logits)
    ):
        row_index = _first_row_difference(reference.logits, candidate.logits)
        return "input_logits_mismatch", 0 if row_index is None else row_index
    return None


def compare_moe_logprobs(
    reference_case: MoELogprobCase,
    reference_values: torch.Tensor,
    candidate_case: MoELogprobCase,
    candidate: MoELogprobObservation,
    *,
    expected_mapping_run_id: str,
    expected_model_version: str,
) -> dict[str, Any]:
    """Compare a candidate to a fixed capture and an independent recomputation.

    Mapping/input identity errors are reported separately from numeric errors.
    The fixed result is strict byte comparison; the independent result uses the
    shared logprob accuracy threshold. ``passed`` requires both verdicts.
    """

    if reference_case.mapping_run_id != expected_mapping_run_id:
        raise MoELogprobError("reference case has stale mapping_run_id")
    if reference_case.model_version != expected_model_version:
        raise MoELogprobError("reference case has stale model_version")
    fixed = _snapshot("reference_values", reference_values)
    if fixed.ndim != 1:
        raise MoELogprobError("reference_values must be a one-dimensional vector")
    if fixed.dtype != torch.float32:
        raise MoELogprobError("reference_values must be float32")
    if not bool(torch.isfinite(fixed).all()):
        raise MoELogprobError("reference_values must be finite")
    if fixed.numel() != reference_case.rows:
        raise MoELogprobError("reference_values must have one value per captured row")
    identity = _identity_mismatch(reference_case, candidate_case)
    if reference_case.routing_weights is None or candidate_case.routing_weights is None:
        weights_preserved = reference_case.routing_weights is candidate_case.routing_weights
    else:
        weights_preserved = (
            reference_case.routing_weights.shape == candidate_case.routing_weights.shape
            and reference_case.routing_weights.dtype == candidate_case.routing_weights.dtype
            and torch.equal(reference_case.routing_weights, candidate_case.routing_weights)
        )
    base = {
        "schema_version": MOE_LOGPROB_DIAGNOSTICS_VERSION,
        "passed": False,
        "reference_identity_verified": identity is None,
        "mapping_run_id_verified": (
            reference_case.mapping_run_id
            == candidate_case.mapping_run_id
            == expected_mapping_run_id
        ),
        "model_version_verified": (
            reference_case.model_version == candidate_case.model_version == expected_model_version
        ),
        "inputs_preserved": identity is None,
        "routing_weights_preserved": weights_preserved,
        "device_execution_verified": False,
        "producer_identity_verified": False,
        "parameter_content_verified": False,
        "candidate_backend": {
            "requested": candidate.requested_backend,
            "actual": candidate.actual_backend,
            "implementation_fingerprint": candidate.implementation_fingerprint,
            "rank": candidate.rank,
            "world_size": candidate.world_size,
        },
        "reference_provenance": reference_case.provenance(),
        "candidate_provenance": candidate_case.provenance(),
        "independent_reference": "python_math_fsum_fp64",
        "fixed_reference": "captured_reference_bytes",
        "fixed_reference_sha256": _digest(fixed),
        "candidate_values_sha256": _digest(candidate.values),
        "first_divergence": None,
    }
    if identity is not None:
        reason, index = identity
        if index >= reference_case.rows:
            index = 0
        base.update(
            status="invalid_identity",
            compared_count=0,
            fixed_mismatch_count=0,
            accuracy_mismatch_count=0,
            fixed_tree_passed=False,
            accuracy_passed=False,
            first_divergence=_first_record(reference_case, index, reason=reason),
        )
        return base
    if candidate.values.numel() != reference_case.rows:
        base.update(
            status="invalid_artifact",
            compared_count=0,
            fixed_mismatch_count=0,
            accuracy_mismatch_count=0,
            fixed_tree_passed=False,
            accuracy_passed=False,
            first_divergence=_first_record(reference_case, 0, reason="shape_mismatch"),
        )
        return base
    independent = independently_recompute_logprobs(reference_case)
    fixed_mask = _fixed_mismatch(fixed, candidate.values)
    threshold = resolve_logprob_threshold(candidate.values.dtype)
    accuracy_mask = (candidate.values.double() - independent).abs() > threshold
    first = torch.nonzero(fixed_mask | accuracy_mask, as_tuple=False)
    first_record = None
    if first.numel():
        index = int(first[0].item())
        reason = "bit_mismatch" if bool(fixed_mask[index]) else "accuracy_mismatch"
        first_record = _first_record(
            reference_case,
            index,
            reason=reason,
            expected=float(fixed[index]),
            actual=float(candidate.values[index]),
            independent=float(independent[index]),
        )
    base.update(
        status="pass" if not bool(fixed_mask.any() or accuracy_mask.any()) else "fail",
        compared_count=reference_case.rows,
        fixed_mismatch_count=int(fixed_mask.sum()),
        accuracy_mismatch_count=int(accuracy_mask.sum()),
        fixed_tree_passed=not bool(fixed_mask.any()),
        accuracy_passed=not bool(accuracy_mask.any()),
        accuracy_atol=threshold,
        first_divergence=first_record,
    )
    base["passed"] = base["fixed_tree_passed"] and base["accuracy_passed"]
    return base


__all__ = [
    "MOE_LOGPROB_DIAGNOSTICS_VERSION",
    "PROBABILITY_RULE",
    "MoELogprobCase",
    "MoELogprobError",
    "MoELogprobObservation",
    "compare_moe_logprobs",
    "independently_recompute_logprobs",
]
