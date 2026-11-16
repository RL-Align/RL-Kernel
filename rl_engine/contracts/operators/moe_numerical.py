# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Device-neutral MoE numerical rules for issue #563, task 31.

This is an explicit softmax/top-k profile, not a claim about every Qwen variant.
Native routing/dispatch/combine implementations belong to tasks 13--16. A matching
fingerprint identifies rules; it does not qualify an implementation or a device.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance

MOE_CONTRACT_VERSION = "moe.softmax_fixed_tree.v1"
MOE_PHASES = ("train", "prefill", "decode", "verify")
_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class MoEContractError(ValueError):
    """A numerical profile or a logical MoE input is invalid."""


def _count(name: str, value: int, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise MoEContractError(
            f"{name} must be a {'non-negative' if allow_zero else 'positive'} integer"
        )


def _tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...] | None = None,
    device: torch.device | None = None,
) -> None:
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise MoEContractError(f"{name} must be a strided tensor")
    if value.device.type == "meta" or (device is not None and value.device != device):
        expected = f" on {device}" if device is not None else ""
        raise MoEContractError(f"{name} must be materialized{expected}")
    if shape is not None and tuple(value.shape) != shape:
        raise MoEContractError(f"{name} must have shape {shape}")


def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise MoEContractError(f"{name} must contain only finite values")


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MoENumericalContract:
    """Versioned rules independent of batch size, EP ownership and phase.

    FP32 softmax is reduced by adjacent pairs in ascending expert-ID order.
    Slots use descending scores, ties by ascending expert ID. Odd tree nodes
    carry unchanged. Optional selected-weight normalization precedes scaling.
    Combine multiplies in FP32 (unless already weighted), then reduces by the
    same adjacent-pair tree in slot order. Shared then residual are added in
    FP32, followed by one output cast. See docs/contracts/moe.md for boundaries.
    """

    num_experts: int
    top_k: int
    hidden_size: int
    input_dtype: str = "bfloat16"
    output_dtype: str = "bfloat16"
    normalize_selected_weights: bool = True
    routing_scale: float = 1.0
    expert_weight_applied: bool = False
    shared_expert_mode: str = "excluded"
    residual_mode: str = "excluded"
    version: str = MOE_CONTRACT_VERSION
    score_transform: str = "softmax_fp32"
    tie_break: str = "expert_id_ascending"
    accumulation_dtype: str = "float32"
    reduction_tree: str = "adjacent_pairs_odd_carry"
    rounding: str = "nearest_even_final_cast"
    quantization: str = "none"

    def __post_init__(self) -> None:
        for name in ("num_experts", "top_k", "hidden_size"):
            _count(name, getattr(self, name))
        if self.top_k > self.num_experts:
            raise MoEContractError("top_k must not exceed num_experts")
        for name in ("input_dtype", "output_dtype"):
            if getattr(self, name) not in _DTYPES:
                raise MoEContractError(f"{name} must be float16, bfloat16 or float32")
        for name in ("normalize_selected_weights", "expert_weight_applied"):
            if not isinstance(getattr(self, name), bool):
                raise MoEContractError(f"{name} must be bool")
        if (
            isinstance(self.routing_scale, bool)
            or not isinstance(self.routing_scale, (int, float))
            or not math.isfinite(self.routing_scale)
            or not 0 < self.routing_scale <= torch.finfo(torch.float32).max
        ):
            raise MoEContractError("routing_scale must be positive and representable in FP32")
        # Store the effective FP32 constant, so equal fingerprints mean equal arithmetic.
        scale = torch.tensor(self.routing_scale, dtype=torch.float32).item()
        if scale == 0:
            raise MoEContractError("routing_scale underflows FP32")
        object.__setattr__(self, "routing_scale", scale)
        for name, allowed in (
            ("shared_expert_mode", ("excluded", "separate")),
            ("residual_mode", ("excluded", "external")),
        ):
            if getattr(self, name) not in allowed:
                raise MoEContractError(f"{name} must be one of {allowed}")
        for name, expected in (
            ("version", MOE_CONTRACT_VERSION),
            ("score_transform", "softmax_fp32"),
            ("tie_break", "expert_id_ascending"),
            ("accumulation_dtype", "float32"),
            ("reduction_tree", "adjacent_pairs_odd_carry"),
            ("rounding", "nearest_even_final_cast"),
            ("quantization", "none"),
        ):
            if getattr(self, name) != expected:
                raise MoEContractError(f"unsupported {name}: expected {expected!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def validate_logits(self, logits: torch.Tensor) -> None:
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise MoEContractError("logits must be [num_tokens, num_experts]")
        _tensor("logits", logits)
        if logits.shape[1] != self.num_experts:
            raise MoEContractError(
                f"logits must have shape [num_tokens, {self.num_experts}]"
            )
        if logits.dtype != _DTYPES[self.input_dtype]:
            raise MoEContractError("logits dtype must match input_dtype")
        _finite("logits", logits)


@dataclass(frozen=True)
class MoECombineInputs:
    """Physical rows plus an independently captured canonical routing result.

    Canonical routing tensors are [token, slot]; weights are always FP32 and
    retained for auditing even when the producer already weighted the rows.
    Tensors remain caller-owned; consumers validate them on every invocation.
    Shared output is already shared-gate weighted. Residual is the model's
    explicitly external forward branch; this does not implement RMS backward.
    """

    num_tokens: int
    expert_outputs: torch.Tensor
    token_indices: torch.Tensor
    slot_indices: torch.Tensor
    expert_indices: torch.Tensor
    routing_expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    mapping_run_id: str
    shared_output: torch.Tensor | None = None
    residual: torch.Tensor | None = None

    def validate(self, contract: MoENumericalContract, *, expected_mapping_run_id: str) -> None:
        _count("num_tokens", self.num_tokens, allow_zero=True)
        if not isinstance(expected_mapping_run_id, str) or not expected_mapping_run_id.strip():
            raise MoEContractError("expected_mapping_run_id must be explicit and non-empty")
        if self.mapping_run_id != expected_mapping_run_id:
            raise MoEContractError("mapping_run_id does not match expected_mapping_run_id")
        if not isinstance(self.expert_outputs, torch.Tensor):
            raise MoEContractError("expert_outputs must be a tensor")
        device = self.expert_outputs.device
        rows = self.num_tokens * contract.top_k
        _tensor("expert_outputs", self.expert_outputs, (rows, contract.hidden_size), device)
        if self.expert_outputs.dtype != _DTYPES[contract.input_dtype]:
            raise MoEContractError("expert_outputs dtype must match input_dtype")
        _finite("expert_outputs", self.expert_outputs)
        for name, shape in (
            ("token_indices", (rows,)),
            ("slot_indices", (rows,)),
            ("expert_indices", (rows,)),
            ("routing_expert_ids", (self.num_tokens, contract.top_k)),
        ):
            tensor = getattr(self, name)
            _tensor(name, tensor, shape, device)
            if tensor.dtype != torch.int64:
                raise MoEContractError(f"{name} must be int64")
        _tensor("routing_weights", self.routing_weights, (self.num_tokens, contract.top_k), device)
        if self.routing_weights.dtype != torch.float32:
            raise MoEContractError("routing_weights must be float32")
        _finite("routing_weights", self.routing_weights)
        if bool((self.routing_weights < 0).any()):
            raise MoEContractError("routing_weights must be non-negative")
        for name, upper in (
            ("token_indices", self.num_tokens),
            ("slot_indices", contract.top_k),
            ("expert_indices", contract.num_experts),
            ("routing_expert_ids", contract.num_experts),
        ):
            tensor = getattr(self, name)
            if bool(((tensor < 0) | (tensor >= upper)).any()):
                raise MoEContractError(f"{name} is out of range")
        keys = self.token_indices * contract.top_k + self.slot_indices
        if keys.unique().numel() != rows:
            raise MoEContractError("token/slot mapping must cover every pair exactly once")
        sorted_ids = self.routing_expert_ids.sort(dim=1).values
        if bool((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()):
            raise MoEContractError("routing_expert_ids must select distinct experts per token")
        if not torch.equal(
            self.expert_indices, self.routing_expert_ids[self.token_indices, self.slot_indices]
        ):
            raise MoEContractError("expert_indices disagree with canonical routing")
        for name, included in (
            ("shared_output", contract.shared_expert_mode == "separate"),
            ("residual", contract.residual_mode == "external"),
        ):
            tensor = getattr(self, name)
            if (tensor is not None) != included:
                raise MoEContractError(f"{name} {'is required' if included else 'must be omitted'}")
            if tensor is not None:
                _tensor(name, tensor, (self.num_tokens, contract.hidden_size), device)
                if tensor.dtype != _DTYPES[contract.input_dtype]:
                    raise MoEContractError(f"{name} dtype must match input_dtype")
                _finite(name, tensor)


def moe_rule_manifest(contract: MoENumericalContract) -> dict:
    """Resolve shared judgment rows; this is a rule binding, not device evidence.

    GEMM/Norm/combine are reduction-class outputs; logprob remains FP32. These
    threshold links do not replace each operator's arithmetic/execution plan.
    Quantized profiles are rejected until scale rules and references exist.
    """
    shared = load_contract()
    rules = {}
    for op, op_class in (
        ("attention", "attention"),
        ("gemm", "reduction"),
        ("norm", "reduction"),
        ("logprob", "logprob"),
        ("moe_combine", "reduction"),
    ):
        dtype = "float32" if op == "logprob" else contract.output_dtype
        rules[op] = {
            judgment: resolve_tolerance(
                shared, judgment=judgment, op_class=op_class, dtype=dtype
            ).to_dict()
            for judgment in (
                "forward_accuracy",
                "forward_invariance",
                "gradient_accuracy",
                "gradient_invariance",
            )
        }
    return {
        "moe_contract": contract.to_dict(),
        "moe_fingerprint": contract.fingerprint,
        "shared_contract_version": shared["version"],
        "shared_contract_fingerprint": _fingerprint(shared),
        "rules": rules,
    }
