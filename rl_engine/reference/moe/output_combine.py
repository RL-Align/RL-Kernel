# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""P6 output-combine request preparation.

The first P6 implementation intentionally stops at a validated request. It
does not perform inverse permutation or reduction; later P6 tasks consume the
same plan for those operations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
from types import MappingProxyType

from rl_engine.contracts.operators.moe import (
    MoEOutputCombineContractError,
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)


@dataclass(frozen=True)
class PreparedMoEOutputCombine:
    """Validated P6 snapshot shared by reference and future ROCm backends."""

    plan: MoEOutputCombinePlan
    inputs: MoEOutputCombineInputs
    provenance: MappingProxyType


class MoEOutputCombineReferenceOp:
    """Semantic-registry adapter for the validated P6 boundary."""

    backend_id = "rlkernel.reference.moe_output_combine.v1"

    def prepare(
        self,
        plan: MoEOutputCombinePlan,
        inputs: MoEOutputCombineInputs,
    ) -> PreparedMoEOutputCombine:
        return prepare_output_combine(plan, inputs)


def prepare_output_combine(
    plan: MoEOutputCombinePlan,
    inputs: MoEOutputCombineInputs,
) -> PreparedMoEOutputCombine:
    """Validate and freeze the semantic boundary before backend execution."""

    if not isinstance(plan, MoEOutputCombinePlan):
        raise MoEOutputCombineContractError("plan must be a MoEOutputCombinePlan")
    if not isinstance(inputs, MoEOutputCombineInputs):
        raise MoEOutputCombineContractError("inputs must be a MoEOutputCombineInputs")
    if inputs.expert_output.device.type != "cpu":
        raise MoEOutputCombineContractError(
            "P6 reference output-combine inputs must be on CPU"
        )
    if inputs.num_tokens is None:
        raise MoEOutputCombineContractError(
            "P6 reference preparation requires explicit num_tokens"
        )
    if plan.plan_version != "p6-output-combine.v1":
        raise MoEOutputCombineContractError(
            f"unsupported P6 plan version: {plan.plan_version}"
        )

    snapshot = replace(
        inputs,
        expert_output=inputs.expert_output.clone(),
        token_indices=inputs.token_indices.clone(),
        expert_indices=inputs.expert_indices.clone(),
        slot_indices=inputs.slot_indices.clone(),
        routing_weights=(inputs.routing_weights.clone() if inputs.routing_weights is not None else None),
        shared_output=(inputs.shared_output.clone() if inputs.shared_output is not None else None),
        residual=(inputs.residual.clone() if inputs.residual is not None else None),
    )
    provenance = snapshot.provenance(plan)
    return PreparedMoEOutputCombine(
        plan=plan, inputs=snapshot, provenance=MappingProxyType(provenance)
    )


__all__ = [
    "MoEOutputCombineReferenceOp",
    "PreparedMoEOutputCombine",
    "prepare_output_combine",
]

