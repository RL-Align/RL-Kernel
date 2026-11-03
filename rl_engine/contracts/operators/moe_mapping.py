# SPDX-License-Identifier: Apache-2.0
"""Shared mapping validation for the task 52 reference and task 55 backend."""

from __future__ import annotations

import torch

from rl_engine.contracts.operators.moe import (
    MoEOutputCombineContractError,
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)


def canonical_expert_rows(
    plan: MoEOutputCombinePlan,
    inputs: MoEOutputCombineInputs,
    *,
    num_tokens: int,
    num_experts: int | None = None,
    expected_mapping_run_id: str | None = None,
) -> torch.Tensor:
    """Validate task 51/52 mapping and return canonical-to-source row indices.

    Preparation may synchronize metadata/validation to the host. Only the small
    row map is allocated; expert values are not expanded to [T,K,H]. Call before
    Graph capture, then keep the routing metadata fixed for that capture.
    """

    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens <= 0:
        raise MoEOutputCombineContractError("num_tokens must be a positive integer")
    if num_experts is not None and (
        isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts <= 0
    ):
        raise MoEOutputCombineContractError("num_experts must be a positive integer")
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
    return row_index.reshape(num_tokens, plan.top_k)
