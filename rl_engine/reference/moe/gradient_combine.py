# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reference fixed-order local input-gradient combination for P6 task 56."""

from __future__ import annotations

import torch

from rl_engine.contracts.operators.moe import (
    MoEOutputCombineContractError,
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)
from rl_engine.reference.moe.inverse_mapping import inverse_map_expert_output


def _validate_token_gradient(
    name: str,
    value: torch.Tensor | None,
    *,
    num_tokens: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if value is None:
        return
    if value.device.type != "cpu":
        raise MoEOutputCombineContractError(f"{name} must be on CPU for the reference path")
    if value.device != device:
        raise MoEOutputCombineContractError(f"{name} must share the expert gradient device")
    if value.shape != (num_tokens, hidden_size):
        raise MoEOutputCombineContractError(f"{name} must have shape [num_tokens, hidden_size]")
    if value.dtype != dtype:
        raise MoEOutputCombineContractError(f"{name} dtype must match expert_input_gradient")
    if not torch.isfinite(value).all():
        raise MoEOutputCombineContractError(f"{name} must contain only finite values")


def combine_local_input_gradients(
    plan: MoEOutputCombinePlan,
    inputs: MoEOutputCombineInputs,
    expert_input_gradient: torch.Tensor,
    *,
    num_tokens: int,
    shared_expert_gradient: torch.Tensor | None = None,
    gate_gradient: torch.Tensor | None = None,
    residual_gradient: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine local MoE input gradients in the declared deterministic order.

    ``expert_input_gradient`` uses the same physical slot order and mapping
    metadata as ``inputs.expert_output``.  The reference path restores it to
    token-major/slot-minor order, accumulates slots in ``plan.merge_order``,
    then adds the shared-expert, gate, and residual branches in that order.
    Every addition is performed in ``plan.accumulation_dtype`` (FP32 for the
    current P6 plan); the returned tensor is converted to ``plan.output_dtype``.
    The function is CPU-only until a ROCm implementation has independent
    device evidence.
    """

    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens <= 0:
        raise MoEOutputCombineContractError("num_tokens must be a positive integer")
    if plan.accumulation_dtype != "float32":
        raise MoEOutputCombineContractError("P6 gradient accumulation must be float32")
    if expert_input_gradient.device.type != "cpu":
        raise MoEOutputCombineContractError("P6 reference input gradients must be on CPU")
    if expert_input_gradient.ndim != 2:
        raise MoEOutputCombineContractError(
            "expert_input_gradient must have shape [num_slots, hidden_size]"
        )
    if expert_input_gradient.shape[1] != plan.hidden_size:
        raise MoEOutputCombineContractError(
            "expert_input_gradient must have shape [num_slots, hidden_size]"
        )
    if expert_input_gradient.dtype != getattr(torch, plan.output_dtype):
        raise MoEOutputCombineContractError("expert_input_gradient dtype disagrees with the plan")
    if not torch.isfinite(expert_input_gradient).all():
        raise MoEOutputCombineContractError("expert_input_gradient must contain only finite values")

    _validate_token_gradient(
        "shared_expert_gradient",
        shared_expert_gradient,
        num_tokens=num_tokens,
        hidden_size=plan.hidden_size,
        device=expert_input_gradient.device,
        dtype=expert_input_gradient.dtype,
    )
    _validate_token_gradient(
        "gate_gradient",
        gate_gradient,
        num_tokens=num_tokens,
        hidden_size=plan.hidden_size,
        device=expert_input_gradient.device,
        dtype=expert_input_gradient.dtype,
    )
    _validate_token_gradient(
        "residual_gradient",
        residual_gradient,
        num_tokens=num_tokens,
        hidden_size=plan.hidden_size,
        device=expert_input_gradient.device,
        dtype=expert_input_gradient.dtype,
    )

    mapped_inputs = inputs.__class__(
        expert_output=expert_input_gradient,
        token_indices=inputs.token_indices,
        expert_indices=inputs.expert_indices,
        slot_indices=inputs.slot_indices,
        routing_weights=inputs.routing_weights,
        shared_output=inputs.shared_output,
        residual=inputs.residual,
        mapping_schema_version=inputs.mapping_schema_version,
        mapping_run_id=inputs.mapping_run_id,
        producer_backend=inputs.producer_backend,
        producer_build_fingerprint=inputs.producer_build_fingerprint,
        num_tokens=inputs.num_tokens,
    )
    expert_major = inverse_map_expert_output(
        plan,
        mapped_inputs,
        num_tokens=num_tokens,
    )
    combined = torch.zeros(
        (num_tokens, plan.hidden_size),
        dtype=torch.float32,
        device=expert_input_gradient.device,
    )
    for slot in plan.merge_order:
        combined.add_(expert_major[:, slot, :].to(torch.float32))
    for branch in (shared_expert_gradient, gate_gradient, residual_gradient):
        if branch is not None:
            combined.add_(branch.to(torch.float32))
    return combined.to(dtype=expert_input_gradient.dtype)


__all__ = ["combine_local_input_gradients"]
