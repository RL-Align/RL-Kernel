# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reference fixed-order local input-gradient combination for P6 task 56."""

from __future__ import annotations

from dataclasses import replace

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
) -> None:
    if value is None:
        return
    if value.device.type != "cpu":
        raise MoEOutputCombineContractError(f"{name} must be on CPU for the reference path")
    if value.device != device:
        raise MoEOutputCombineContractError(f"{name} must share the expert gradient device")
    if value.shape != (num_tokens, hidden_size):
        raise MoEOutputCombineContractError(f"{name} must have shape [num_tokens, hidden_size]")
    if not value.is_floating_point():
        raise MoEOutputCombineContractError(f"{name} must use a floating-point dtype")
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
    expected_mapping_run_id: str | None = None,
) -> torch.Tensor:
    """Combine local MoE input gradients in the declared deterministic order.

    ``expert_input_gradient`` uses the same physical slot order and mapping
    metadata as ``inputs.expert_output``.  The reference path restores it to
    token-major/slot-minor order, accumulates slots in ``plan.merge_order``,
    then adds the shared-expert and gate branches in that order.  The returned
    value is ``dX_norm`` and must be passed through the P1 RMSNorm backward
    before an original residual gradient is added with
    :func:`merge_residual_gradient_after_rms`.
    Every addition is performed in ``plan.accumulation_dtype`` (FP32 for the
    current P6 plan); the returned tensor uses the expert gradient dtype.
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

    if plan.shared_expert_mode == "separate" and shared_expert_gradient is None:
        raise MoEOutputCombineContractError("shared_expert_gradient is required by the plan")
    if plan.shared_expert_mode == "excluded" and shared_expert_gradient is not None:
        raise MoEOutputCombineContractError(
            "shared_expert_gradient must be omitted when shared expert mode is excluded"
        )
    if plan.residual_mode == "external" and inputs.residual is None:
        raise MoEOutputCombineContractError("residual metadata is required by the plan")
    if plan.residual_mode == "excluded" and inputs.residual is not None:
        raise MoEOutputCombineContractError(
            "residual metadata must be omitted when residual mode is excluded"
        )
    _validate_token_gradient(
        "shared_expert_gradient",
        shared_expert_gradient,
        num_tokens=num_tokens,
        hidden_size=plan.hidden_size,
        device=expert_input_gradient.device,
    )
    _validate_token_gradient(
        "gate_gradient",
        gate_gradient,
        num_tokens=num_tokens,
        hidden_size=plan.hidden_size,
        device=expert_input_gradient.device,
    )

    mapped_inputs = replace(inputs, expert_output=expert_input_gradient)
    expert_major = inverse_map_expert_output(
        plan,
        mapped_inputs,
        num_tokens=num_tokens,
        expected_mapping_run_id=expected_mapping_run_id,
    )
    combined = torch.zeros(
        (num_tokens, plan.hidden_size),
        dtype=torch.float32,
        device=expert_input_gradient.device,
    )
    for slot in plan.merge_order:
        combined.add_(expert_major[:, slot, :].to(torch.float32))
    for branch in (shared_expert_gradient, gate_gradient):
        if branch is not None:
            combined.add_(branch.to(torch.float32))
    return combined.to(dtype=expert_input_gradient.dtype)


def merge_residual_gradient_after_rms(
    rms_input_gradient: torch.Tensor,
    residual_gradient: torch.Tensor,
) -> torch.Tensor:
    """Add the original residual gradient after P1 RMSNorm backward."""

    if rms_input_gradient.device.type != "cpu" or residual_gradient.device.type != "cpu":
        raise MoEOutputCombineContractError("P6 reference gradients must be on CPU")
    if rms_input_gradient.shape != residual_gradient.shape:
        raise MoEOutputCombineContractError("residual_gradient must match the RMS gradient shape")
    for name, value in (
        ("rms_input_gradient", rms_input_gradient),
        ("residual_gradient", residual_gradient),
    ):
        if not value.is_floating_point():
            raise MoEOutputCombineContractError(f"{name} must use a floating-point dtype")
        if not torch.isfinite(value).all():
            raise MoEOutputCombineContractError(f"{name} must contain only finite values")
    return (rms_input_gradient.to(torch.float32) + residual_gradient.to(torch.float32)).to(
        rms_input_gradient.dtype
    )


def local_input_gradient_provenance(
    plan: MoEOutputCombinePlan,
    inputs: MoEOutputCombineInputs,
    expert_input_gradient: torch.Tensor,
    *,
    num_tokens: int,
    shared_expert_gradient: torch.Tensor | None = None,
    gate_gradient: torch.Tensor | None = None,
    expected_mapping_run_id: str | None = None,
) -> dict[str, object]:
    """Record checks and deterministic choices for a successful P6 combine."""

    output = combine_local_input_gradients(
        plan,
        inputs,
        expert_input_gradient,
        num_tokens=num_tokens,
        shared_expert_gradient=shared_expert_gradient,
        gate_gradient=gate_gradient,
        expected_mapping_run_id=expected_mapping_run_id,
    )
    return {
        "semantic_op": "moe_input_gradient_combine",
        "plan_version": plan.plan_version,
        "plan_fingerprint": plan.fingerprint,
        "mapping_schema_version": plan.mapping_schema_version,
        "mapping_run_id": inputs.mapping_run_id,
        "mapping_run_id_verified": expected_mapping_run_id is not None,
        "expected_mapping_run_id": expected_mapping_run_id,
        "merge_order": list(plan.merge_order),
        "branches": {
            "expert": True,
            "shared_expert": shared_expert_gradient is not None,
            "gate": gate_gradient is not None,
        },
        "accumulation_dtype": plan.accumulation_dtype,
        "output_dtype": str(output.dtype).removeprefix("torch."),
        "output_shape": list(output.shape),
        "output_stride": list(output.stride()),
    }


__all__ = [
    "combine_local_input_gradients",
    "local_input_gradient_provenance",
    "merge_residual_gradient_after_rms",
]
