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

_SUPPORTED_GRADIENT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _dtype_name(value: torch.dtype) -> str:
    return str(value).removeprefix("torch.")


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
    if value.dtype not in _SUPPORTED_GRADIENT_DTYPES:
        raise MoEOutputCombineContractError(f"{name} uses an unsupported gradient dtype")
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
    if expert_input_gradient.dtype not in _SUPPORTED_GRADIENT_DTYPES:
        raise MoEOutputCombineContractError("expert_input_gradient uses an unsupported dtype")
    if not torch.isfinite(expert_input_gradient).all():
        raise MoEOutputCombineContractError("expert_input_gradient must contain only finite values")

    inputs.validate(plan)
    if shared_expert_gradient is gate_gradient and shared_expert_gradient is not None:
        raise MoEOutputCombineContractError("shared and gate gradients must be separate tensors")
    if plan.shared_expert_mode == "separate" and shared_expert_gradient is None:
        raise MoEOutputCombineContractError("shared_expert_gradient is required by the plan")
    if plan.shared_expert_mode == "excluded" and shared_expert_gradient is not None:
        raise MoEOutputCombineContractError(
            "shared_expert_gradient must be omitted when shared expert mode is excluded"
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

    mapping_plan = replace(
        plan,
        output_dtype=_dtype_name(expert_input_gradient.dtype),
        shared_expert_mode="excluded",
        residual_mode="excluded",
    )
    mapped_inputs = replace(
        inputs,
        expert_output=expert_input_gradient,
        shared_output=None,
        residual=None,
    )
    expert_major = inverse_map_expert_output(
        mapping_plan,
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
    plan: MoEOutputCombinePlan,
    rms_input_gradient: torch.Tensor,
    residual_gradient: torch.Tensor,
) -> torch.Tensor:
    """Add external residual dH after P1 RMSNorm backward.

    The RMS gradient dtype determines the returned dtype.  Both inputs are
    accumulated in FP32 before the final conversion at this boundary.
    """

    if plan.residual_mode != "external":
        raise MoEOutputCombineContractError("residual gradient requires residual_mode='external'")
    if plan.accumulation_dtype != "float32":
        raise MoEOutputCombineContractError("P6 gradient accumulation must be float32")
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
        if value.dtype not in _SUPPORTED_GRADIENT_DTYPES:
            raise MoEOutputCombineContractError(f"{name} uses an unsupported gradient dtype")
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
    provenance = inputs.provenance(plan)
    provenance.update(
        {
            "semantic_op": "moe_input_gradient_combine",
            "mapping_run_id_verified": expected_mapping_run_id is not None,
            "expected_mapping_run_id": expected_mapping_run_id,
            "merge_order": list(plan.merge_order),
            "branches": {
                "expert": {"present": True, "dtype": _dtype_name(expert_input_gradient.dtype)},
                "shared_expert": {
                    "present": shared_expert_gradient is not None,
                    "dtype": (
                        _dtype_name(shared_expert_gradient.dtype)
                        if shared_expert_gradient is not None
                        else None
                    ),
                },
                "gate": {
                    "present": gate_gradient is not None,
                    "dtype": (
                        _dtype_name(gate_gradient.dtype) if gate_gradient is not None else None
                    ),
                },
            },
            "accumulation_dtype": plan.accumulation_dtype,
            "output_dtype": str(output.dtype).removeprefix("torch."),
            "output_shape": list(output.shape),
            "output_stride": list(output.stride()),
        }
    )
    return provenance


__all__ = [
    "combine_local_input_gradients",
    "local_input_gradient_provenance",
    "merge_residual_gradient_after_rms",
]
