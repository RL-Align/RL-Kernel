# SPDX-License-Identifier: Apache-2.0
"""Deterministic Inkling 2 Mini top-8 sigmoid router contract (INK-24)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RoutingResult:
    """Stable route ownership and normalized route weights."""

    expert_indices: torch.Tensor
    weights: torch.Tensor


def route_topk_sigmoid(
    logits: torch.Tensor,
    selection_bias: torch.Tensor | None = None,
    *,
    top_k: int = 8,
    route_scale: float = 8.0,
) -> RoutingResult:
    """Select experts with sigmoid scores and a selection-only bias.

    Router probabilities are computed in FP32. ``selection_bias`` affects the
    ordering only. Selected route weights use the unbiased sigmoid scores,
    normalize across the selected experts, and then apply ``route_scale``.
    Stable descending sort preserves ascending expert IDs for ties.
    """
    if logits.ndim != 2:
        raise ValueError("router logits must have shape [tokens, experts]")
    if not logits.is_floating_point():
        raise TypeError("router logits must be floating point")
    tokens, experts = logits.shape
    if top_k < 1 or top_k > experts:
        raise ValueError(f"top_k must be in [1, {experts}]")
    if selection_bias is None:
        bias = torch.zeros(experts, device=logits.device, dtype=torch.float32)
    else:
        if selection_bias.shape != (experts,):
            raise ValueError("selection bias must have shape [experts]")
        if selection_bias.device != logits.device:
            raise ValueError("selection bias must share the logits device")
        if not selection_bias.is_floating_point():
            raise TypeError("selection bias must be floating point")
        bias = selection_bias.float()

    probabilities = torch.sigmoid(logits.float())
    selection_scores = probabilities + bias.view(1, experts)
    order = torch.argsort(selection_scores, dim=-1, descending=True, stable=True)
    expert_indices = order[:, :top_k]
    selected = probabilities.gather(1, expert_indices)
    normalized = selected / selected.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    weights = normalized * float(route_scale)
    return RoutingResult(expert_indices=expert_indices, weights=weights)


def route_topk_sigmoid_cuda(
    logits: torch.Tensor,
    selection_bias: torch.Tensor | None = None,
    *,
    top_k: int = 8,
    route_scale: float = 8.0,
) -> RoutingResult:
    """Run the native CUDA deterministic top-k router forward kernel."""
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_router_forward"):
        raise RuntimeError("Inkling 2 CUDA router extension is unavailable")
    if not logits.is_cuda or logits.ndim != 2:
        raise ValueError("native router logits must be a 2-D CUDA tensor")
    experts = logits.shape[1]
    if selection_bias is None:
        selection_bias = torch.zeros(experts, device=logits.device, dtype=torch.float32)
    else:
        selection_bias = selection_bias.to(device=logits.device, dtype=torch.float32).contiguous()
    expert_indices, weights = _C.inkling2_router_forward(
        logits.contiguous(), selection_bias, int(top_k), float(route_scale)
    )
    return RoutingResult(expert_indices=expert_indices, weights=weights)


def route_topk_sigmoid_rocm(
    logits: torch.Tensor,
    selection_bias: torch.Tensor | None = None,
    *,
    top_k: int = 8,
    route_scale: float = 8.0,
) -> RoutingResult:
    """Run the native ROCm wave64 deterministic top-k router forward kernel."""
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if torch.version.hip is None:
        raise RuntimeError("route_topk_sigmoid_rocm requires a ROCm (HIP) build of PyTorch")
    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_router_forward"):
        raise RuntimeError("Inkling 2 ROCm router extension is unavailable")
    if not logits.is_cuda or logits.ndim != 2:
        raise ValueError("native router logits must be a 2-D ROCm tensor")
    experts = logits.shape[1]
    if selection_bias is None:
        selection_bias = torch.zeros(experts, device=logits.device, dtype=torch.float32)
    else:
        selection_bias = selection_bias.to(device=logits.device, dtype=torch.float32).contiguous()
    expert_indices, weights = _C.inkling2_router_forward(
        logits.contiguous(), selection_bias, int(top_k), float(route_scale)
    )
    return RoutingResult(expert_indices=expert_indices, weights=weights)
