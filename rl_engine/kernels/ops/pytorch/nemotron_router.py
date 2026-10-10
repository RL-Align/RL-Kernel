# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Nemotron Nano routing specification/reference, not a strict GPU provider.

The ordering/dispatch ABI is a proposal for RFC #434. GEMM and autograd here
are numerical baselines only: CUDA matmul and gather backward are not claimed
batch invariant. The independent scalar oracle belongs in tests.
"""

from dataclasses import dataclass

import torch


@dataclass
class RouterResult:
    expert_ids: torch.Tensor  # [T, K], ascending expert id within each row
    weights: torch.Tensor  # [T, K], aligned with expert_ids
    permutation: torch.Tensor  # [T*K], packed position -> original flat route slot
    expert_offsets: torch.Tensor  # [E+1], exclusive prefix counts
    packed_tokens: torch.Tensor  # [T*K, H], unweighted


def route_scores(scores, correction_bias, top_k=6, scaling=2.5):
    """Select by corrected scores; weight using original scores.

    Proposed tie policy: lower expert id wins. Selected experts are then
    sorted by id; normalization sums in that order, then adds 1e-20.
    Top-k selection is discrete; bias has no differentiable output path.
    """
    if scores.ndim != 2 or not scores.is_floating_point():
        raise ValueError("scores must be a floating [tokens, experts] tensor")
    if correction_bias.shape != (scores.shape[1],):
        raise ValueError("correction_bias must have shape [experts]")
    if correction_bias.device != scores.device or correction_bias.dtype != scores.dtype:
        raise ValueError("scores and correction_bias must share dtype and device")
    if not 1 <= top_k <= scores.shape[1]:
        raise ValueError("top_k must be between 1 and the number of experts")
    if not torch.isfinite(scores).all() or not torch.isfinite(correction_bias).all():
        raise ValueError("non-finite routing inputs are unsupported")
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError("scores must lie in [0, 1]")
    with torch.no_grad():
        choice = scores + correction_bias
        if not torch.isfinite(choice).all():
            raise ValueError("corrected score overflow")
        ids = torch.argsort(choice, dim=-1, descending=True, stable=True)[:, :top_k]
        ids = ids.sort(dim=-1).values
    selected = scores.gather(1, ids)
    denominator = torch.zeros_like(selected[:, :1])
    for slot in range(top_k):
        denominator = denominator + selected[:, slot : slot + 1]
    weights = (selected / (denominator + 1e-20)) * scaling
    return ids, weights


def dispatch_tokens(hidden_states, expert_ids, num_experts):
    """Proposed ABI: expert-major, stable token-major unweighted dispatch.

    Permutation stores flattened (token, route-slot) indices, retaining enough
    information to invert dispatch. No padding, capacity truncation or dropping.
    """
    if hidden_states.ndim != 2 or expert_ids.ndim != 2:
        raise ValueError("hidden_states and expert_ids must be rank two")
    if expert_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("token counts must agree")
    if expert_ids.dtype != torch.int64 or expert_ids.device != hidden_states.device:
        raise ValueError("expert_ids must be int64 on the input device")
    if num_experts < 1 or expert_ids.shape[1] < 1:
        raise ValueError("positive expert and route counts are required")
    if ((expert_ids < 0) | (expert_ids >= num_experts)).any():
        raise ValueError("expert id out of range")
    if expert_ids.shape[1] > 1 and (expert_ids[:, 1:] <= expert_ids[:, :-1]).any():
        raise ValueError("expert ids must be unique and ascending per token")
    flat = expert_ids.reshape(-1)
    permutation = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    packed = hidden_states.index_select(0, permutation // expert_ids.shape[1])
    return permutation, offsets, packed


def router_reference(hidden_states, router_weight, correction_bias, top_k=6, scaling=2.5):
    """Differentiable baseline. FP64 inputs retain FP64 for gradient checks.

    Otherwise projection, sigmoid and routing weights use FP32 as in the
    pinned checkpoint. Packed payload retains the input dtype.
    """
    if hidden_states.ndim != 2 or router_weight.ndim != 2:
        raise ValueError("expected [T,H] input and [E,H] router weights")
    if hidden_states.shape[1] != router_weight.shape[1]:
        raise ValueError("hidden widths must agree")
    if not hidden_states.is_floating_point() or not router_weight.is_floating_point():
        raise ValueError("input and weight must be floating tensors")
    dtype = (
        torch.float64
        if hidden_states.dtype == router_weight.dtype == torch.float64
        else torch.float32
    )
    logits = hidden_states.to(dtype) @ router_weight.to(dtype).T
    ids, weights = route_scores(logits.sigmoid(), correction_bias.to(dtype), top_k, scaling)
    permutation, offsets, packed = dispatch_tokens(hidden_states, ids, router_weight.shape[0])
    return RouterResult(ids, weights, permutation, offsets, packed)
