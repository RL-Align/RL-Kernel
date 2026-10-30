# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Reference inverse mapping for the P6 expert-output boundary."""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.contracts.operators.moe import (
    MoEOutputCombineContractError,
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)


def inverse_map_expert_output(
    plan: MoEOutputCombinePlan,
    inputs: MoEOutputCombineInputs,
    *,
    num_tokens: int,
    num_experts: int | None = None,
    expected_mapping_run_id: str | None = None,
) -> torch.Tensor:
    """Restore shuffled expert slots to ``[token, slot, hidden]`` order.

    This is a correctness-first reference implementation.  It validates the
    complete ``(token, slot)`` domain on the host, then gathers rows in a
    canonical token-major order.  A future HIP/Triton implementation must
    preserve this exact mapping and failure contract.
    ``expected_mapping_run_id`` is an optional cross-request guard. When it is
    omitted, the input run ID is carried through without claiming an external
    identity comparison.
    """

    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens <= 0:
        raise MoEOutputCombineContractError("num_tokens must be a positive integer")
    if num_experts is not None and (
        isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts <= 0
    ):
        raise MoEOutputCombineContractError("num_experts must be a positive integer")
    if inputs.expert_output.device.type != "cpu":
        raise MoEOutputCombineContractError("P6 reference inverse mapping inputs must be on CPU")
    inputs.validate(plan)
    if inputs.num_tokens is not None and inputs.num_tokens != num_tokens:
        raise MoEOutputCombineContractError("num_tokens disagrees with the input mapping contract")
    if expected_mapping_run_id is not None and inputs.mapping_run_id != expected_mapping_run_id:
        raise MoEOutputCombineContractError("mapping run id disagrees with the expected request")

    expected_slots = num_tokens * plan.top_k
    if inputs.expert_output.shape[0] != expected_slots:
        raise MoEOutputCombineContractError(
            "expert output slot count must equal num_tokens * top_k"
        )

    token_indices = inputs.token_indices.detach().cpu().tolist()
    slot_indices = inputs.slot_indices.detach().cpu().tolist()
    expert_indices = inputs.expert_indices.detach().cpu().tolist()
    order: dict[tuple[int, int], int] = {}
    for row, (token, slot, expert) in enumerate(
        zip(token_indices, slot_indices, expert_indices, strict=True)
    ):
        token = int(token)
        slot = int(slot)
        expert = int(expert)
        if token < 0 or token >= num_tokens:
            raise MoEOutputCombineContractError(f"token index {token} is outside [0, {num_tokens})")
        if slot < 0 or slot >= plan.top_k:
            raise MoEOutputCombineContractError(f"slot index {slot} is outside [0, {plan.top_k})")
        if num_experts is not None and (expert < 0 or expert >= num_experts):
            raise MoEOutputCombineContractError(
                f"expert index {expert} is outside [0, {num_experts})"
            )
        key = (token, slot)
        if key in order:
            raise MoEOutputCombineContractError(
                f"duplicate expert result for token={token}, slot={slot}"
            )
        order[key] = row

    expected_keys = {(token, slot) for token in range(num_tokens) for slot in range(plan.top_k)}
    missing = sorted(expected_keys.difference(order))
    if missing:
        preview = ", ".join(f"({token},{slot})" for token, slot in missing[:4])
        suffix = "..." if len(missing) > 4 else ""
        raise MoEOutputCombineContractError(f"missing expert results: {preview}{suffix}")

    canonical_rows = [
        order[(token, slot)] for token in range(num_tokens) for slot in range(plan.top_k)
    ]
    row_index = torch.tensor(canonical_rows, dtype=torch.int64, device=inputs.expert_output.device)
    restored = inputs.expert_output.index_select(0, row_index)
    return restored.reshape(num_tokens, plan.top_k, plan.hidden_size)


def inverse_mapping_provenance(
    plan: MoEOutputCombinePlan,
    inputs: MoEOutputCombineInputs,
    *,
    num_tokens: int,
    num_experts: int | None = None,
    expected_mapping_run_id: str | None = None,
) -> dict[str, Any]:
    """Return provenance for a successful inverse-map reference execution."""

    output = inverse_map_expert_output(
        plan,
        inputs,
        num_tokens=num_tokens,
        num_experts=num_experts,
        expected_mapping_run_id=expected_mapping_run_id,
    )
    provenance = inputs.provenance(plan)
    provenance.update(
        {
            "inverse_mapping": "token_major_slot_minor",
            "num_tokens": num_tokens,
            "restored_shape": list(output.shape),
            "restored_stride": list(output.stride()),
            "mapping_run_id_verified": expected_mapping_run_id is not None,
            "expected_mapping_run_id": expected_mapping_run_id,
        }
    )
    return provenance


__all__ = ["inverse_map_expert_output", "inverse_mapping_provenance"]
