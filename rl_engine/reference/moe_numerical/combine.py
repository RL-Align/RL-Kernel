# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Forward combine references with distinct accuracy and fixed-order arithmetic."""

import math

import torch

from rl_engine.contracts.operators.moe_numerical import (
    MoECombineInputs,
    MoEContractError,
    MoENumericalContract,
)

from ._tree import adjacent_tree, require_cpu


def combine_fixed_tree(
    contract: MoENumericalContract, inputs: MoECombineInputs, *, expected_mapping_run_id: str
) -> torch.Tensor:
    """Canonicalize physical rows, multiply in FP32, tree-reduce slots, cast once.

    The producer declares whether expert rows are already weighted; values
    cannot prove that claim. Shared output is already shared-gate weighted.
    This is forward arithmetic only; no RMSNorm backward or EP collective runs.
    """
    require_cpu(inputs.expert_outputs)
    inputs.validate(contract, expected_mapping_run_id=expected_mapping_run_id)
    keys = inputs.token_indices * contract.top_k + inputs.slot_indices
    slots = (
        inputs.expert_outputs[torch.argsort(keys)]
        .float()
        .reshape(inputs.num_tokens, contract.top_k, contract.hidden_size)
    )
    if not contract.expert_weight_applied:
        slots = slots * inputs.routing_weights.unsqueeze(-1)
    output = adjacent_tree(slots)
    for branch in (inputs.shared_output, inputs.residual):
        if branch is not None:
            output = output + branch.float()
    output = output.to(getattr(torch, contract.output_dtype))
    if not bool(torch.isfinite(output).all()):
        raise MoEContractError("combine arithmetic or final cast produced non-finite output")
    return output


def combine_accuracy_reference(
    contract: MoENumericalContract, inputs: MoECombineInputs, *, expected_mapping_run_id: str
) -> torch.Tensor:
    """FP64 products and math.fsum; independent of tree, sorting and output cast.

    Shares input validation, but neither the fixed-tree implementation nor its
    arithmetic helpers. Evaluates the supplied contributions, not the expert
    MLP or projection which produced them. No backward accuracy is claimed.
    """
    require_cpu(inputs.expert_outputs)
    inputs.validate(contract, expected_mapping_run_id=expected_mapping_run_id)
    rows = {
        (token, slot): values
        for token, slot, values in zip(
            inputs.token_indices.tolist(),
            inputs.slot_indices.tolist(),
            inputs.expert_outputs.detach().double().tolist(),
            strict=True,
        )
    }
    weights = inputs.routing_weights.detach().double().tolist()
    branches = [
        branch.detach().double().tolist()
        for branch in (inputs.shared_output, inputs.residual)
        if branch is not None
    ]
    output = []
    for token in range(inputs.num_tokens):
        result = []
        for column in range(contract.hidden_size):
            terms = [
                rows[token, slot][column]
                * (1.0 if contract.expert_weight_applied else weights[token][slot])
                for slot in range(contract.top_k)
            ]
            terms.extend(branch[token][column] for branch in branches)
            result.append(math.fsum(terms))
        output.append(result)
    return torch.tensor(output, dtype=torch.float64).reshape(
        inputs.num_tokens, contract.hidden_size
    )
