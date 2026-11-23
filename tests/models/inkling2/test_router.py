# SPDX-License-Identifier: Apache-2.0
"""INK-24 router contract tests."""

import torch

from rl_engine.models.inkling2.router import route_topk_sigmoid


def test_router_uses_stable_expert_id_tie_break():
    result = route_topk_sigmoid(torch.zeros(2, 10), top_k=4)
    expected_indices = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    assert torch.equal(result.expert_indices, expected_indices)
    assert torch.equal(result.weights, torch.full((2, 4), 2.0))


def test_selection_bias_changes_ownership_but_not_route_weight_source():
    logits = torch.tensor([[4.0, 3.0, -2.0, -3.0]])
    bias = torch.tensor([-10.0, -10.0, 10.0, -10.0])
    result = route_topk_sigmoid(logits, bias, top_k=2)
    assert torch.equal(result.expert_indices, torch.tensor([[2, 0]]))

    probabilities = torch.sigmoid(logits)
    expected = probabilities[:, [2, 0]]
    expected = expected / expected.sum(dim=-1, keepdim=True) * 8.0
    assert torch.equal(result.weights, expected)


def test_router_outputs_fp32_and_normalizes_to_route_scale():
    logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.bfloat16)
    result = route_topk_sigmoid(logits, top_k=3, route_scale=8.0)
    assert result.expert_indices.dtype == torch.int64
    assert result.weights.dtype == torch.float32
    torch.testing.assert_close(result.weights.sum(dim=-1), torch.tensor([8.0]))


def test_router_backward_flows_through_unbiased_scores():
    logits = torch.randn(3, 12, requires_grad=True)
    result = route_topk_sigmoid(logits, top_k=4)
    result.weights.square().sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
