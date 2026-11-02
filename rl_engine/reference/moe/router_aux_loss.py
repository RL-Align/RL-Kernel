# SPDX-License-Identifier: Apache-2.0
"""Independent FP64 normal selection/admission and mathematical aux-loss graph."""

import torch

from rl_engine.reference.moe.learned_route import learned_route_reference, weights_fp64


def router_aux_loss_fp64(logits, logical_ids, bias, capacities, alpha, global_ids=None):
    _, experts, columns, keep, positions = learned_route_reference(
        logits, logical_ids, bias, capacities, global_ids=global_ids, scale=1.0
    )
    probabilities = weights_fp64(logits, columns.to(logits.device), keep.to(logits.device), 1.0)
    counts = torch.tensor([(experts == e).sum().item() for e in range(256)], dtype=torch.int64)
    masses = torch.stack(
        [
            torch.where((experts == e).to(logits.device), probabilities, 0.0).sum()
            for e in range(256)
        ]
    )
    n = logits.shape[0]
    if n:
        loss = alpha * 256 * ((counts.to(logits.device).double() / (6 * n)) * (masses / n)).sum()
    else:
        loss = probabilities.sum() * 0
    return loss, experts, columns, keep, positions, counts, masses
