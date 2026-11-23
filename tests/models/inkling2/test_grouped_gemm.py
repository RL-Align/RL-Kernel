# SPDX-License-Identifier: Apache-2.0
"""INK-27 grouped expert GEMM contract tests."""

import pytest
import torch

from rl_engine.models.inkling2.grouped_gemm import grouped_expert_gemm


def test_grouped_gemm_matches_per_token_reference():
    tokens = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10
    expert_indices = torch.tensor([1, 0, 1, 2, 0])
    weights = torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3) / 20
    bias = torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2) / 10
    output = grouped_expert_gemm(tokens, expert_indices, weights, bias)

    expected = torch.stack(
        [
            tokens[row] @ weights[expert].transpose(0, 1) + bias[expert]
            for row, expert in enumerate(expert_indices)
        ]
    )
    assert torch.equal(output, expected)


def test_grouped_gemm_is_invariant_to_token_permutation():
    torch.manual_seed(7)
    tokens = torch.randn(12, 6)
    experts = torch.randint(0, 4, (12,))
    weights = torch.randn(4, 5, 6)
    bias = torch.randn(4, 5)
    baseline = grouped_expert_gemm(tokens, experts, weights, bias)

    permutation = torch.randperm(tokens.shape[0])
    shuffled = grouped_expert_gemm(tokens[permutation], experts[permutation], weights, bias)
    restored = torch.empty_like(shuffled)
    restored[permutation] = shuffled
    assert torch.equal(restored, baseline)


def test_grouped_gemm_backward_flows_through_tokens_and_weights():
    tokens = torch.randn(8, 4, requires_grad=True)
    weights = torch.randn(3, 2, 4, requires_grad=True)
    experts = torch.tensor([0, 1, 2, 0, 2, 1, 1, 0])
    output = grouped_expert_gemm(tokens, experts, weights)
    output.square().mean().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


def test_grouped_gemm_rejects_invalid_expert_ids():
    with pytest.raises(ValueError, match="outside"):
        grouped_expert_gemm(torch.randn(2, 3), torch.tensor([0, 4]), torch.randn(4, 2, 3))
