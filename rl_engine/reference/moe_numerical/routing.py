# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Stable top-k profiles from logits; no projection, expert kernel or dispatch."""

import math
from dataclasses import dataclass

import torch

from rl_engine.contracts.operators.moe_numerical import MoEContractError, MoENumericalContract

from ._tree import adjacent_tree, require_cpu


@dataclass(frozen=True)
class RoutingResult:
    expert_ids: torch.Tensor
    weights: torch.Tensor


def route_fixed_tree(contract: MoENumericalContract, logits: torch.Tensor) -> RoutingResult:
    """CPU FP32 softmax, stable top-k, optional renormalization, then scaling.

    Selection is by input logits (before softmax rounding), ties by expert ID.
    Projection and auxiliary router losses are owned by task 13. Autograd can
    differentiate weights with choices held fixed; no native backward is claimed.
    """
    require_cpu(logits)
    contract.validate_logits(logits)
    scores = logits.float()
    ids = torch.argsort(scores, dim=1, descending=True, stable=True)[:, : contract.top_k]
    # Token-local exp avoids batch-shape dependent vectorization in the oracle.
    weights = []
    for token in range(scores.shape[0]):
        row = scores[token : token + 1]
        exponentials = torch.exp(row - row.max(dim=1, keepdim=True).values)
        probabilities = exponentials / adjacent_tree(exponentials).unsqueeze(1)
        selected = probabilities.gather(1, ids[token : token + 1])
        if contract.normalize_selected_weights:
            selected = selected / adjacent_tree(selected).unsqueeze(1)
        weights.append(selected * contract.routing_scale)
    result = torch.cat(weights) if weights else scores.new_empty((0, contract.top_k))
    if not bool(torch.isfinite(result).all()):
        raise MoEContractError("routing arithmetic produced non-finite weights")
    return RoutingResult(ids, result)


def route_accuracy_reference(contract: MoENumericalContract, logits: torch.Tensor) -> RoutingResult:
    """Independent FP64/math.fsum oracle; not a bitwise target or backward path."""
    require_cpu(logits)
    contract.validate_logits(logits)
    all_ids, all_weights = [], []
    for row in logits.detach().double().tolist():
        ids = sorted(range(contract.num_experts), key=lambda expert: (-row[expert], expert))[
            : contract.top_k
        ]
        maximum = max(row)
        exponentials = [math.exp(value - maximum) for value in row]
        total = math.fsum(exponentials)
        selected = [exponentials[expert] / total for expert in ids]
        if contract.normalize_selected_weights:
            denominator = math.fsum(selected)
            selected = [value / denominator for value in selected]
        all_ids.append(ids)
        all_weights.append([value * contract.routing_scale for value in selected])
    return RoutingResult(
        torch.tensor(all_ids, dtype=torch.int64).reshape(-1, contract.top_k),
        torch.tensor(all_weights, dtype=torch.float64).reshape(-1, contract.top_k),
    )
