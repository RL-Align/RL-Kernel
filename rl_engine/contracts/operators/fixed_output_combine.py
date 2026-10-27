# SPDX-License-Identifier: Apache-2.0
"""Strict DSV4 subset of the task 51 plan, prepared outside Graph capture."""

from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from rl_engine.contracts.operators.moe import (
    MoEOutputCombineContractError,
    MoEOutputCombineInputs,
    MoEOutputCombinePlan,
)
from rl_engine.contracts.operators.moe_mapping import canonical_expert_rows

PROFILE = "dsv4-output-combine-fp32-slot012345-shared-last-v1"


@dataclass(frozen=True)
class OutputCombineTrace:
    output: Tensor
    slot_steps: Tensor
    before_cast: Tensor


@dataclass(frozen=True)
class PreparedFixedOutputCombine:
    plan: MoEOutputCombinePlan
    num_tokens: int
    mapping_run_id: str
    provenance: MappingProxyType
    _rows: Tensor
    _tokens: Tensor
    _versions: tuple[int, int]

    def canonical_rows(self) -> Tensor:
        """Diagnostic copy; callers cannot mutate the execution map through it."""
        return self._rows.clone()

    def validate_execution(self, expert_output, shared_output, mapping_run_id):
        if mapping_run_id != self.mapping_run_id:
            raise MoEOutputCombineContractError("mapping run id disagrees with prepared execution")
        if (self._rows._version, self._tokens._version) != self._versions:
            raise MoEOutputCombineContractError("prepared mapping was modified; prepare again")
        dtype = getattr(torch, str(self.plan.output_dtype))
        for name, value, shape in (
            ("expert_output", expert_output, (self.num_tokens * 6, self.plan.hidden_size)),
            ("shared_output", shared_output, (self.num_tokens, self.plan.hidden_size)),
        ):
            if name == "shared_output":
                if self.plan.shared_expert_mode == "excluded":
                    if value is not None:
                        raise MoEOutputCombineContractError("shared output is excluded by the plan")
                    continue
                if value is None:
                    raise MoEOutputCombineContractError("shared output is required by the plan")
            if (
                value.layout != torch.strided
                or tuple(value.shape) != shape
                or value.dtype != dtype
                or value.device != self._rows.device
            ):
                raise MoEOutputCombineContractError(f"{name} shape/dtype/device/layout mismatch")


def prepare_fixed_output_combine(
    plan: MoEOutputCombinePlan,
    inputs: MoEOutputCombineInputs,
    *,
    expected_mapping_run_id: str,
    num_experts: int | None = None,
) -> PreparedFixedOutputCombine:
    """Validate a complete Top-6 handoff and retain only O(T*K) mapping storage.

    Initial value validation may synchronize. Execution permits value updates
    but keeps this mapping snapshot and run identity fixed; prepare a new object
    and recapture for a new routing plan. No expert-value snapshot is retained.
    """
    if not isinstance(plan, MoEOutputCombinePlan) or not isinstance(inputs, MoEOutputCombineInputs):
        raise TypeError("use the task 51 MoEOutputCombinePlan and MoEOutputCombineInputs")
    if plan.top_k != 6 or plan.merge_order != tuple(range(6)):
        raise MoEOutputCombineContractError("DSV4 combine requires fixed slots 0..5")
    if not plan.expert_weight_applied or inputs.routing_weights is not None:
        raise MoEOutputCombineContractError("P5 must apply route weights; P6 never reapplies them")
    if plan.residual_mode != "excluded" or inputs.residual is not None:
        raise MoEOutputCombineContractError("P6 does not add residuals; P1 mhc_post owns them")
    if plan.output_dtype not in ("bfloat16", "float32"):
        raise MoEOutputCombineContractError("combine accepts BF16 or FP32 expert outputs")
    if (
        plan.plan_version != "p6-output-combine.v1"
        or plan.mapping_schema_version != "p4-token-slot.v1"
    ):
        raise MoEOutputCombineContractError("unsupported combine plan or mapping schema version")
    if not isinstance(expected_mapping_run_id, str) or expected_mapping_run_id.strip() in (
        "",
        "unknown",
    ):
        raise MoEOutputCombineContractError("an explicit expected mapping run id is required")
    if inputs.num_tokens is None:
        raise MoEOutputCombineContractError("num_tokens is required")
    for value in (
        inputs.expert_output,
        inputs.token_indices,
        inputs.expert_indices,
        inputs.slot_indices,
        inputs.shared_output,
    ):
        if value is not None and value.layout != torch.strided:
            raise MoEOutputCombineContractError("combine inputs must be strided tensors")
    rows = canonical_expert_rows(
        plan,
        inputs,
        num_tokens=inputs.num_tokens,
        num_experts=num_experts,
        expected_mapping_run_id=expected_mapping_run_id,
    )
    if inputs.shared_output is not None and not bool(torch.isfinite(inputs.shared_output).all()):
        raise MoEOutputCombineContractError("shared_output must contain only finite values")
    # Standard tensors retain version counters even if preparation is called
    # from an inference-mode rollout. Original metadata is never consulted again.
    with torch.inference_mode(False):
        rows = rows.detach().clone()
        tokens = inputs.token_indices.detach().to(torch.int64).clone()
    provenance = inputs.provenance(plan)
    provenance.update(profile=PROFILE, mapping_run_id_verified=True, residual_owner="P1.mhc_post")
    return PreparedFixedOutputCombine(
        plan,
        inputs.num_tokens,
        inputs.mapping_run_id,
        MappingProxyType(provenance),
        rows,
        tokens,
        (rows._version, tokens._version),
    )
