# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Synthetic Torch forward producers for T06 handoff tests.

These are test fixtures, not the T03/T04 providers or a P3 ABI implementation.
The weights retain a Torch graph for semantic checks; the saved tensors are
detached snapshots passed directly to route_backward_core(dweights, *saved).
SyntheticSavedRoute carries no identity, checksum, or sealed-state guarantee.
"""

from typing import NamedTuple

import torch

from examples.p3_router_backward.prototype import fixed_tree6


class SyntheticSavedRoute(NamedTuple):
    ids: torch.Tensor
    p: torch.Tensor
    z: torch.Tensor
    row_active: torch.Tensor


def _normalize(scores, active_ids, row_active):
    rows = row_active.nonzero(as_tuple=True)[0]
    selected = scores[rows].gather(1, active_ids.long())
    z = fixed_tree6(selected) + 1e-20
    p = selected / z[:, None]
    weights = scores.new_zeros((scores.shape[0], 6)).index_copy(0, rows, p * 1.5)
    ids = torch.full(weights.shape, -1, dtype=torch.int32, device=scores.device)
    ids[rows] = active_ids.to(torch.int32)
    saved_p = torch.zeros_like(weights).index_copy(0, rows, p)
    saved_z = scores.new_zeros(scores.shape[0]).index_copy(0, rows, z)
    saved = SyntheticSavedRoute(
        *(value.detach().clone().contiguous() for value in (ids, saved_p, saved_z, row_active))
    )
    return weights, saved


def hash_forward_reference(scores, token_ids, table, row_active):
    """Look up active token IDs, preserving all six table slots and duplicates."""
    active_ids = table[token_ids[row_active].long()]
    return _normalize(scores, active_ids, row_active)


def learned_forward_reference(scores, bias, row_active):
    """Select by score+bias; normalize pre-bias scores with no selection gradient.

    Columns already follow ascending expert ID, so stable descending sort breaks
    ties by ascending ID. Only indices leave this block, never post-bias scores.
    """
    with torch.no_grad():
        q = scores[row_active] + bias
        active_ids = torch.argsort(q, dim=1, descending=True, stable=True)[:, :6]
    return _normalize(scores, active_ids, row_active)
