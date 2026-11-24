# SPDX-License-Identifier: Apache-2.0
"""CUDA fault injectors for the INK-24 top-k router contract."""

from __future__ import annotations

import torch

from .router import RoutingResult


def route_with_bias_in_weights(
    logits: torch.Tensor, selection_bias: torch.Tensor, *, top_k: int = 8
) -> RoutingResult:
    """Fault A51.1 applies selection bias to route weights after ownership."""
    probabilities = torch.sigmoid(logits.float())
    scores = probabilities + selection_bias.float().view(1, -1)
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :top_k]
    selected = probabilities.gather(1, indices) + selection_bias.float().gather(
        0, indices.reshape(-1)
    ).reshape_as(indices)
    weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return RoutingResult(indices, weights * 8.0)


def route_with_reverse_tie_break(
    logits: torch.Tensor, selection_bias: torch.Tensor, *, top_k: int = 8
) -> RoutingResult:
    """Fault A51.2 reverses the expert-ID tie-break order."""
    probabilities = torch.sigmoid(logits.float())
    scores = probabilities + selection_bias.float().view(1, -1)
    reverse_ids = torch.arange(logits.shape[1] - 1, -1, -1, device=logits.device)
    reversed_scores = scores.index_select(1, reverse_ids)
    positions = torch.argsort(reversed_scores, dim=-1, descending=True, stable=True)[:, :top_k]
    indices = reverse_ids.index_select(0, positions.reshape(-1)).reshape_as(positions)
    selected = probabilities.gather(1, indices)
    weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return RoutingResult(indices, weights * 8.0)


def route_with_wrong_scale(
    logits: torch.Tensor, selection_bias: torch.Tensor, *, top_k: int = 8
) -> RoutingResult:
    """Fault A51.3 normalizes routes before applying an incorrect scale."""
    probabilities = torch.sigmoid(logits.float())
    scores = probabilities + selection_bias.float().view(1, -1)
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :top_k]
    selected = probabilities.gather(1, indices)
    weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return RoutingResult(indices, weights * 1.0)
