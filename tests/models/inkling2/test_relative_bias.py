# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from rl_engine.models.inkling2.relative_bias import lookup_relative_bias, project_relative_logits

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_projection_and_gradients(device):
    torch.manual_seed(11)
    r = torch.randint(-3, 4, (3, 24, 16), device=device).bfloat16().requires_grad_()
    w = torch.randint(-3, 4, (16, 1024), device=device).float().requires_grad_()
    actual = project_relative_logits(r, w)
    expected = (r.float() @ w.bfloat16().float()).bfloat16()
    assert actual.shape == (3, 24, 1024)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[1:2], project_relative_logits(r[1:2], w))
    actual.float().sum().backward()
    assert torch.isfinite(r.grad).all() and torch.isfinite(w.grad).all()
    assert torch.equal(r.grad, w.detach().bfloat16().float().sum(-1).bfloat16().expand_as(r))
    expected_dw = r.detach().float().sum((0, 1))[:, None].expand_as(w)
    assert torch.equal(w.grad, expected_dw.bfloat16().float())


@pytest.mark.parametrize("device", DEVICES)
def test_distance_boundary_and_future_keys(device):
    logits = (torch.arange(1024, device=device).float() + 1).expand(2, 3, -1)
    q = torch.tensor([1023, 1024], device=device)
    k = torch.tensor([0, 1, 1023, 1024, 1025], device=device)
    actual = lookup_relative_bias(logits, q, k)
    expected = torch.tensor([[1024, 1023, 1, 0, 0], [0, 1024, 2, 1, 0]], device=device)
    assert torch.equal(actual, expected[:, None, :].expand_as(actual))


def test_wrong_geometry_rejected():
    with pytest.raises(ValueError, match="16"):
        project_relative_logits(torch.ones(2, 3, 8), torch.ones(16, 1024))
    with pytest.raises(ValueError, match="integer"):
        lookup_relative_bias(torch.ones(2, 3, 1024), torch.ones(2), torch.zeros(4))
