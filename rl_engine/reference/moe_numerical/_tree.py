# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Arithmetic helper shared only by the fixed-order CPU references."""

import torch

from rl_engine.contracts.operators.moe_numerical import MoEContractError


def require_cpu(value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise MoEContractError("MoE numerical references require CPU tensors")


def adjacent_tree(values: torch.Tensor) -> torch.Tensor:
    """Reduce dimension 1 using explicit FP32 pair adds, carrying odd nodes.

    No zero padding: adding +0 can change the sign of a -0 leaf. No in-place
    accumulation or implicit reduction primitive may change this grouping.
    """
    nodes = list(values.unbind(dim=1))
    while len(nodes) > 1:
        merged = [nodes[i] + nodes[i + 1] for i in range(0, len(nodes) - 1, 2)]
        if len(nodes) % 2:
            merged.append(nodes[-1])
        nodes = merged
    return nodes[0]
