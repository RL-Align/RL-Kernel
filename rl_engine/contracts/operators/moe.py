# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Contracts for the DSV4 MoE output-combine boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import torch


class MoEOutputCombineContractError(ValueError):
    """Raised when a P6 output-combine request is malformed or ambiguous."""


def _dtype_name(value: str | torch.dtype) -> str:
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    normalized = str(value).strip().lower().removeprefix("torch.")
    if not normalized:
        raise MoEOutputCombineContractError("dtype must not be empty")
    return normalized


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_value(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return value


def _freeze_value(value: Any) -> Any:
    """Deep-freeze metadata so the plan fingerprint remains stable."""
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True)
class MoEOutputCombinePlan:
    """Versioned semantic plan shared by training and rollout P6 callers.

    When ``expert_weight_applied`` is false, ``routing_weights`` are consumed
    as producer-supplied scaling factors.  P6 validates their dtype and
    finiteness, but does not impose simplex normalization; DSV4 routing can
    intentionally carry a separate scaling factor.
    """

    top_k: int
    hidden_size: int
    token_index_dtype: str | torch.dtype = "int64"
    weight_dtype: str | torch.dtype = "float32"
    accumulation_dtype: str | torch.dtype = "float32"
    output_dtype: str | torch.dtype = "bfloat16"
    expert_weight_applied: bool = False
    merge_order: tuple[int, ...] | None = None
    shared_expert_mode: str = "excluded"
    residual_mode: str = "excluded"
    mapping_schema_version: str = "p4-token-slot.v1"
    plan_version: str = "p6-output-combine.v1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0:
            raise MoEOutputCombineContractError("top_k must be a positive integer")
        if isinstance(self.hidden_size, bool) or not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise MoEOutputCombineContractError("hidden_size must be a positive integer")
        if self.merge_order is None:
            merge_order = tuple(range(self.top_k))
        else:
            if not isinstance(self.merge_order, (tuple, list)) or any(
                isinstance(item, bool) or not isinstance(item, int) for item in self.merge_order
            ):
                raise MoEOutputCombineContractError("merge_order must contain integer slot indices")
            merge_order = tuple(self.merge_order)
        if merge_order != tuple(range(self.top_k)):
            raise MoEOutputCombineContractError(
                f"merge_order must be the fixed slot order 0..{self.top_k - 1}"
            )
        object.__setattr__(self, "merge_order", merge_order)
        for name in ("token_index_dtype", "weight_dtype", "accumulation_dtype", "output_dtype"):
            object.__setattr__(self, name, _dtype_name(getattr(self, name)))
        if self.token_index_dtype not in {"int32", "int64"}:
            raise MoEOutputCombineContractError("token_index_dtype must be int32 or int64")
        if self.weight_dtype != "float32":
            raise MoEOutputCombineContractError("P6 routing weights must be float32")
        if self.accumulation_dtype != "float32":
            raise MoEOutputCombineContractError("P6 accumulation must be float32")
        if self.output_dtype not in {"float32", "bfloat16", "float16"}:
            raise MoEOutputCombineContractError(
                "output_dtype must be float32, bfloat16 or float16"
            )
        if not isinstance(self.expert_weight_applied, bool):
            raise MoEOutputCombineContractError("expert_weight_applied must be a bool")
        if self.shared_expert_mode not in {"excluded", "separate"}:
            raise MoEOutputCombineContractError(
                "shared_expert_mode must be excluded or separate"
            )
        if self.residual_mode not in {"excluded", "external"}:
            raise MoEOutputCombineContractError(
                "residual_mode must be excluded or external"
            )
        if not isinstance(self.mapping_schema_version, str) or not isinstance(self.plan_version, str):
            raise MoEOutputCombineContractError("mapping and plan versions must be strings")
        if not self.mapping_schema_version.strip() or not self.plan_version.strip():
            raise MoEOutputCombineContractError("mapping and plan versions must not be empty")
        if not isinstance(self.metadata, Mapping):
            raise MoEOutputCombineContractError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "hidden_size": self.hidden_size,
            "token_index_dtype": self.token_index_dtype,
            "weight_dtype": self.weight_dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "output_dtype": self.output_dtype,
            "expert_weight_applied": self.expert_weight_applied,
            "merge_order": list(self.merge_order or ()),
            "shared_expert_mode": self.shared_expert_mode,
            "residual_mode": self.residual_mode,
            "mapping_schema_version": self.mapping_schema_version,
            "plan_version": self.plan_version,
            "metadata": _json_value(self.metadata),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MoEOutputCombineInputs:
    """P4/P5 tensors and metadata entering the P6 boundary.

    ``(token_indices, slot_indices)`` is the canonical identity of a routed
    slot. ``expert_indices`` records the producing expert and is validated as
    metadata; it is not used as a second uniqueness key.
    """

    expert_output: torch.Tensor
    token_indices: torch.Tensor
    expert_indices: torch.Tensor
    slot_indices: torch.Tensor
    routing_weights: torch.Tensor | None = None
    shared_output: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    mapping_schema_version: str = "p4-token-slot.v1"
    mapping_run_id: str = "unknown"
    producer_backend: str = "unknown"
    producer_build_fingerprint: str = "unknown"
    num_tokens: int | None = None

    def validate(self, plan: MoEOutputCombinePlan) -> None:
        if self.expert_output.device.type == "meta":
            raise MoEOutputCombineContractError(
                "meta tensors are not supported by P6 validation"
            )
        if self.expert_output.ndim != 2 or self.expert_output.shape[1] != plan.hidden_size:
            raise MoEOutputCombineContractError(
                "expert_output must have shape [num_slots, hidden_size]"
            )
        if _dtype_name(self.expert_output.dtype) != plan.output_dtype:
            raise MoEOutputCombineContractError("expert_output dtype disagrees with the plan")
        num_slots = self.expert_output.shape[0]
        for name, tensor in (
            ("token_indices", self.token_indices),
            ("expert_indices", self.expert_indices),
            ("slot_indices", self.slot_indices),
        ):
            if tensor.ndim != 1 or tensor.shape[0] != num_slots:
                raise MoEOutputCombineContractError(
                    f"{name} must have one entry per expert output slot"
                )
            if tensor.device != self.expert_output.device:
                raise MoEOutputCombineContractError(f"{name} must share expert_output device")
        if _dtype_name(self.token_indices.dtype) != plan.token_index_dtype:
            raise MoEOutputCombineContractError("token_indices dtype disagrees with the plan")
        for name, tensor in (
            ("expert_indices", self.expert_indices),
            ("slot_indices", self.slot_indices),
        ):
            if _dtype_name(tensor.dtype) != plan.token_index_dtype:
                raise MoEOutputCombineContractError(f"{name} dtype disagrees with the plan")
        if self.mapping_schema_version != plan.mapping_schema_version:
            raise MoEOutputCombineContractError("mapping schema version disagrees with the plan")
        if not isinstance(self.mapping_run_id, str) or not self.mapping_run_id.strip():
            raise MoEOutputCombineContractError("mapping_run_id must not be empty")
        if self.num_tokens is not None and (
            isinstance(self.num_tokens, bool) or not isinstance(self.num_tokens, int) or self.num_tokens <= 0
        ):
            raise MoEOutputCombineContractError("num_tokens must be a positive integer when provided")
        if self.num_tokens is not None and bool(torch.any(self.token_indices >= self.num_tokens)):
            raise MoEOutputCombineContractError("token_indices must be within num_tokens")
        for name, tensor in (("token_indices", self.token_indices),
                             ("expert_indices", self.expert_indices),
                             ("slot_indices", self.slot_indices)):
            if bool(torch.any(tensor < 0)):
                raise MoEOutputCombineContractError(f"{name} must be non-negative")
        if bool(torch.any(self.slot_indices >= plan.top_k)):
            raise MoEOutputCombineContractError("slot_indices must be within top_k")
        pairs = torch.stack((self.token_indices, self.slot_indices), dim=1)
        if torch.unique(pairs, dim=0).shape[0] != num_slots:
            raise MoEOutputCombineContractError("duplicate token and slot pairs are not allowed")
        if self.num_tokens is not None and num_slots != self.num_tokens * plan.top_k:
            raise MoEOutputCombineContractError(
                "expert output slot count must equal num_tokens * top_k"
            )
        if not torch.isfinite(self.expert_output).all():
            raise MoEOutputCombineContractError("expert_output must contain only finite values")

        if self.routing_weights is not None:
            if plan.expert_weight_applied:
                raise MoEOutputCombineContractError(
                    "routing_weights must be omitted when expert weights are already applied"
                )
            if self.routing_weights.ndim != 1 or self.routing_weights.shape[0] != num_slots:
                raise MoEOutputCombineContractError(
                    "routing_weights must have one float32 value per expert output slot"
                )
            if self.routing_weights.dtype != torch.float32:
                raise MoEOutputCombineContractError("routing_weights must be float32")
            if self.routing_weights.device != self.expert_output.device:
                raise MoEOutputCombineContractError("routing_weights must share expert_output device")
            if not torch.isfinite(self.routing_weights).all():
                raise MoEOutputCombineContractError("routing_weights must contain only finite values")
        elif not plan.expert_weight_applied:
            raise MoEOutputCombineContractError(
                "routing_weights are required when expert weights are not applied"
            )

        for name, tensor, mode in (
            ("shared_output", self.shared_output, plan.shared_expert_mode),
            ("residual", self.residual, plan.residual_mode),
        ):
            if ((name == "shared_output" and mode == "separate") or
                    (name == "residual" and mode == "external")) and tensor is None:
                raise MoEOutputCombineContractError(f"{name} is required by the plan")
            if mode == "excluded" and tensor is not None:
                raise MoEOutputCombineContractError(
                    f"{name} must be omitted when {name} mode is excluded"
                )
            if tensor is not None:
                if mode in {"separate", "external"} and self.num_tokens is None:
                    raise MoEOutputCombineContractError(
                        "num_tokens is required when shared_expert_mode or residual_mode is not excluded"
                    )
                num_tokens = self.num_tokens
                if tensor.shape != (num_tokens, plan.hidden_size):
                    raise MoEOutputCombineContractError(
                        f"{name} must have shape [num_tokens, hidden_size]"
                    )
                if tensor.device != self.expert_output.device:
                    raise MoEOutputCombineContractError(f"{name} must share expert_output device")
                if tensor.dtype != self.expert_output.dtype:
                    raise MoEOutputCombineContractError(
                        f"{name} dtype must match expert_output dtype"
                    )

    def provenance(self, plan: MoEOutputCombinePlan) -> dict[str, Any]:
        self.validate(plan)
        return {
            "semantic_op": "moe_output_combine",
            "plan_version": plan.plan_version,
            "plan_fingerprint": plan.fingerprint,
            "mapping_schema_version": plan.mapping_schema_version,
            "mapping_run_id": self.mapping_run_id,
            "mapping_run_id_verified": False,
            "top_k": plan.top_k,
            "hidden_size": plan.hidden_size,
            "num_slots": int(self.expert_output.shape[0]),
            "num_tokens": self.num_tokens,
            "producer_backend": self.producer_backend,
            "producer_build_fingerprint": self.producer_build_fingerprint,
            "device": str(self.expert_output.device),
            "expert_output_shape": list(self.expert_output.shape),
            "expert_output_dtype": _dtype_name(self.expert_output.dtype),
            "expert_output_stride": list(self.expert_output.stride()),
        }

