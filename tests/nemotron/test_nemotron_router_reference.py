# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Independent scalar checks and differentiability of the draft reference."""

import math

import pytest
import torch

from rl_engine.kernels.ops.pytorch.nemotron_router import (
    dispatch_tokens,
    route_scores,
    router_reference,
)


def scalar_oracle(x, w, bias, k):
    ids, weights = [], []
    for row in x.tolist():
        logits = [math.fsum(a * b for a, b in zip(row, wr)) for wr in w.tolist()]
        scores = [1 / (1 + math.exp(-v)) for v in logits]
        chosen = sorted(sorted(range(len(scores)), key=lambda e: (-(scores[e] + bias[e]), e))[:k])
        denom = math.fsum(scores[e] for e in chosen) + 1e-20
        ids.append(chosen)
        weights.append([2.5 * scores[e] / denom for e in chosen])
    return torch.tensor(ids), torch.tensor(weights, dtype=torch.float64)


def test_independent_scalar_and_dispatch():
    g = torch.Generator().manual_seed(434)
    x = torch.randn(7, 9, generator=g, dtype=torch.float64)
    w = torch.randn(8, 9, generator=g, dtype=torch.float64)
    b = torch.linspace(-0.2, 0.2, 8, dtype=torch.float64)
    result = router_reference(x, w, b, top_k=3)
    ids, weights = scalar_oracle(x, w, b.tolist(), 3)
    assert torch.equal(result.expert_ids, ids)
    torch.testing.assert_close(result.weights, weights, atol=1e-14, rtol=1e-14)
    slots = [(e, t, s) for t, row in enumerate(ids.tolist()) for s, e in enumerate(row)]
    slots.sort()
    assert result.permutation.tolist() == [t * 3 + s for e, t, s in slots]
    assert torch.equal(result.packed_tokens, x[[t for e, t, s in slots]])
    for e in range(8):
        lo, hi = result.expert_offsets[e : e + 2].tolist()
        assert hi - lo == sum(v[0] == e for v in slots)


def test_ties_choose_lowest_ids():
    ids, weights = route_scores(torch.full((3, 128), 0.5), torch.zeros(128))
    assert torch.equal(ids, torch.arange(6).expand(3, -1))
    torch.testing.assert_close(weights.sum(1), torch.full((3,), 2.5))


def test_bias_is_selection_only():
    ids, weights = route_scores(torch.tensor([[0.1, 0.2, 0.7]]), torch.tensor([1.0, 1.0, 0.0]), 2)
    assert ids.tolist() == [[0, 1]]
    torch.testing.assert_close(weights, torch.tensor([[2.5 / 3, 5.0 / 3]]))


def test_zero_scores_and_empty_tokens():
    ids, weights = route_scores(torch.zeros(1, 128), torch.zeros(128))
    assert torch.equal(weights, torch.zeros_like(weights))
    result = router_reference(torch.empty(0, 4), torch.zeros(128, 4), torch.zeros(128))
    assert result.packed_tokens.shape == (0, 4)
    assert result.expert_offsets.tolist() == [0] * 129


def test_score_batch_position_invariance():
    scores = torch.rand(17, 128, generator=torch.Generator().manual_seed(1))
    bias = torch.zeros(128)
    ids, weights = route_scores(scores, bias)
    for t in [0, 7, 16]:
        one_ids, one_weights = route_scores(scores[t : t + 1], bias)
        assert torch.equal(one_ids[0], ids[t])
        assert torch.equal(one_weights[0], weights[t])


def test_gradcheck_weights_and_payload():
    g = torch.Generator().manual_seed(4)
    x = (torch.randn(3, 4, generator=g, dtype=torch.float64) * 0.1).requires_grad_()
    w = (torch.randn(8, 4, generator=g, dtype=torch.float64) * 0.1).requires_grad_()
    # Well-separated selection keeps finite differences away from top-k discontinuities.
    bias = torch.arange(8, dtype=torch.float64) * 0.2

    def fn(a, b):
        r = router_reference(a, b, bias, top_k=3)
        return r.weights, r.packed_tokens

    assert torch.autograd.gradcheck(fn, (x, w))


def test_payload_gradient_counts_all_routes():
    x = torch.randn(2, 3, requires_grad=True)
    _, _, packed = dispatch_tokens(x, torch.tensor([[0, 2], [1, 2]]), 4)
    packed.sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_reject_invalid_scores(bad):
    with pytest.raises(ValueError):
        route_scores(torch.tensor([[bad, 0.5]]), torch.zeros(2), 1)


def test_reject_duplicate_experts():
    with pytest.raises(ValueError):
        dispatch_tokens(torch.zeros(1, 3), torch.tensor([[1, 1]]), 2)


def test_registry_fails_closed_and_selects_only_router(monkeypatch):
    from rl_engine.kernels.registry import KernelRegistry, OpBackend

    registry = KernelRegistry()
    for device in ("cpu", "mps"):
        with pytest.raises(RuntimeError, match="SM90"):
            registry.get_op("nemotron_router_dispatch", device=device)
    sentinel = object()
    calls = []

    def load(backend):
        calls.append(backend)
        return sentinel

    monkeypatch.setattr(registry, "_get_or_create_backend", load)
    monkeypatch.setattr(registry, "_platform_for_device", lambda device: "cuda")
    assert registry.get_op("nemotron_router_dispatch", "cuda") is sentinel
    assert calls == [OpBackend.TRITON_NEMOTRON_ROUTER]
    monkeypatch.setattr(registry, "_get_or_create_backend", lambda backend: None)
    with pytest.raises(RuntimeError, match="No functional backend"):
        registry.get_op("nemotron_router_dispatch", "cuda")
