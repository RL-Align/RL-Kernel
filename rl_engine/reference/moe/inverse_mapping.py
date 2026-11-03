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
from rl_engine.contracts.operators.moe_mapping import canonical_expert_rows


def inverse_map_expert_output(
    plan: MoEOutputCombinePlan,
    inputs: MoEOutputCombineInputs,
    *,
    num_tokens: int,
    num_experts: int | None = None,
    expected_mapping_run_id: str | None = None,
) -> torch.Tensor:
    """Restore complete expert rows in token-major/slot-minor order on CPU."""
    if inputs.expert_output.device.type != "cpu":
        raise MoEOutputCombineContractError("P6 reference inverse mapping inputs must be on CPU")
    rows = canonical_expert_rows(
        plan,
        inputs,
        num_tokens=num_tokens,
        num_experts=num_experts,
        expected_mapping_run_id=expected_mapping_run_id,
    )
    return inputs.expert_output.index_select(0, rows.flatten()).reshape(
        num_tokens, plan.top_k, plan.hidden_size
    )


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
