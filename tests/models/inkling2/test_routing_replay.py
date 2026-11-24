# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from rl_engine.models.inkling2.routing_replay import VisitRouteReplay, replay_route_weights

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    ),
]


def _trace(device):
    keys = torch.tensor([20, 5, 99], device=device)
    ids = torch.arange(8, device=device).view(1, 1, 8).expand(3, 48, 8).clone()
    ids = (ids + torch.arange(48, device=device)[None, :, None] * 2) % 128
    ids[1] = (ids[1] + 9) % 128
    return keys, ids


@pytest.mark.parametrize("device", DEVICES)
def test_replay_preserves_visit_and_token_identity(device):
    keys, routes = _trace(device)
    replay = VisitRouteReplay(keys, routes, weight_version="v1")
    wanted = keys[[2, 0, 1]]
    for visit in (6, 18, 30, 47):
        assert torch.equal(replay.for_visit(visit, wanted, weight_version="v1"),
                           routes[[2, 0, 1], visit])
    snapshot = routes[0:1, 18].clone()
    routes.zero_()
    assert torch.equal(replay.for_visit(18, keys[:1], weight_version="v1"), snapshot)


@pytest.mark.parametrize("device", DEVICES)
def test_recomputed_weights_and_gradients(device):
    torch.manual_seed(25)
    x = torch.randn(3, 128, device=device, requires_grad=True)
    ids = torch.tensor([[7, 2, 3, 9, 15, 20, 1, 0]], device=device).expand(3, -1)
    y = replay_route_weights(x, ids)
    scores = x.sigmoid().gather(1, ids)
    expected = scores / (scores.sum(-1, keepdim=True) + 1e-20) * 8
    torch.testing.assert_close(y, expected)
    assert torch.equal(y[1:2], replay_route_weights(x[1:2], ids[1:2]))
    g = torch.randn_like(y)
    actual = torch.autograd.grad(y, x, g)[0]
    ref = torch.autograd.grad(expected, x, g)[0]
    torch.testing.assert_close(actual, ref, atol=2e-7, rtol=2e-5)
    mask = torch.ones_like(x, dtype=torch.bool).scatter_(1, ids, False)
    assert torch.count_nonzero(actual[mask]) == 0


def test_invalid_trace_and_stale_version_rejected():
    keys, routes = _trace("cpu")
    with pytest.raises(ValueError, match="48"):
        VisitRouteReplay(keys, routes[:, :24], weight_version="v1")
    replay = VisitRouteReplay(keys, routes, weight_version="v1")
    with pytest.raises(ValueError, match="weight version"):
        replay.for_visit(1, keys, weight_version="v2")
    with pytest.raises(ValueError, match="missing"):
        replay.for_visit(1, torch.tensor([6]), weight_version="v1")
    with pytest.raises(ValueError, match="duplicate expert"):
        replay_route_weights(torch.ones(1, 128), torch.zeros(1, 8, dtype=torch.long))
