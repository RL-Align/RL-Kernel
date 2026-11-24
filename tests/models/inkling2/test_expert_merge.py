# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from rl_engine.models.inkling2.expert_merge import (
    merge_expert_rows,
    update_block_residual,
    weight_before_down,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_slot_order_and_gradients(device):
    torch.manual_seed(28)
    rows = torch.randn(16, 9, device=device, requires_grad=True)
    weights = torch.randn(2, 8, device=device, requires_grad=True)
    mapping = torch.randperm(16, device=device).reshape(2, 8)
    out = merge_expert_rows(rows, mapping, route_weights=weights, out_dtype=torch.float32)
    expected = (rows[mapping] * weights.unsqueeze(-1)).sum(1)
    torch.testing.assert_close(out, expected)
    g = torch.randn_like(out)
    for a, b in zip(torch.autograd.grad(out, (rows, weights), g),
                    torch.autograd.grad(expected, (rows, weights), g), strict=True):
        torch.testing.assert_close(a, b)
    order = torch.randperm(16, device=device)
    inverse = order.argsort()
    reordered = merge_expert_rows(rows[order], inverse[mapping], route_weights=weights,
                                  out_dtype=torch.float32)
    assert torch.equal(out, reordered)


@pytest.mark.parametrize("device", DEVICES)
def test_preweighted_rows_and_residual_order(device):
    x = torch.ones(8, 4, device=device, requires_grad=True)
    w = torch.arange(1, 9, device=device).float().requires_grad_()
    weighted = weight_before_down(x, w)
    out = merge_expert_rows(weighted, torch.arange(8, device=device)[None], route_weights=None)
    assert torch.equal(out, torch.full_like(out, 36))
    out.float().sum().backward()
    assert torch.equal(w.grad, torch.full_like(w, 4))
    p = torch.full((1, 4), 1e8, device=device).bfloat16()
    a = -p
    m = torch.ones_like(p)
    assert torch.equal(update_block_residual(p, a, m, block_boundary=False), m)
    assert torch.equal(update_block_residual(p, a, m, block_boundary=True), a)


def test_duplicate_slot_rejected():
    with pytest.raises(ValueError, match="permutation"):
        merge_expert_rows(torch.ones(8, 4), torch.zeros(1, 8, dtype=torch.long),
                          route_weights=None)
