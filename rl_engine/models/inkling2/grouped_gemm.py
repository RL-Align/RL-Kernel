# SPDX-License-Identifier: Apache-2.0
"""Stable grouped expert GEMM contract for Inkling 2 Mini (INK-27)."""

from __future__ import annotations

import torch


def grouped_expert_gemm(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one deterministic linear projection per token's selected expert.

    Tokens are stably sorted by expert ID before grouped projections and copied
    back to their original positions. The result is therefore invariant to a
    permutation of the input token batch after the caller restores that batch's
    token order.

    ``tokens`` is ``[tokens, hidden]``, ``expert_indices`` is ``[tokens]`` and
    ``weights`` is ``[experts, output, hidden]``.
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [tokens, hidden]")
    if expert_indices.ndim != 1 or expert_indices.numel() != tokens.shape[0]:
        raise ValueError("expert_indices must have one entry per token")
    if weights.ndim != 3 or weights.shape[2] != tokens.shape[1]:
        raise ValueError("weights must have shape [experts, output, hidden]")
    if not tokens.is_floating_point() or not weights.is_floating_point():
        raise TypeError("tokens and weights must be floating point tensors")
    if tokens.device != expert_indices.device or tokens.device != weights.device:
        raise ValueError("tokens, expert_indices and weights must share a device")
    if bias is not None:
        if bias.shape != (weights.shape[0], weights.shape[1]):
            raise ValueError("bias must have shape [experts, output]")
        if bias.device != tokens.device:
            raise ValueError("bias must share the token device")

    token_count, _ = tokens.shape
    output_size = weights.shape[1]
    output_dtype = torch.promote_types(tokens.dtype, weights.dtype)
    if token_count == 0:
        return torch.empty((0, output_size), device=tokens.device, dtype=output_dtype)

    if expert_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_indices must be int32 or int64")
    if int(expert_indices.min().item()) < 0 or int(expert_indices.max().item()) >= weights.shape[0]:
        raise ValueError("expert index is outside the available expert range")

    order = torch.argsort(expert_indices, stable=True)
    sorted_tokens = tokens.index_select(0, order).float()
    sorted_experts = expert_indices.index_select(0, order)
    unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)

    grouped_outputs = []
    offset = 0
    for expert, count in zip(unique_experts.tolist(), counts.tolist(), strict=False):
        group = sorted_tokens[offset : offset + count]
        expert_weight = weights[expert].float()
        result = torch.zeros((count, output_size), device=tokens.device, dtype=torch.float32)
        for hidden_index in range(tokens.shape[1]):
            result = result + group[:, hidden_index : hidden_index + 1] * expert_weight[
                :, hidden_index
            ].view(1, output_size)
        if bias is not None:
            result = result + bias[expert].float()
        grouped_outputs.append(result)
        offset += count

    sorted_output = torch.cat(grouped_outputs, dim=0).to(output_dtype)
    return torch.empty_like(sorted_output).index_copy(0, order, sorted_output)
