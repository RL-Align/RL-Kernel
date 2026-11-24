# SPDX-License-Identifier: Apache-2.0
"""INK-51 ROCm top-8 routing fault-injection tests against the wave64 router."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm GPU is unavailable",
)

from rl_engine.backends.extension import _C, _EXT_AVAILABLE  # noqa: E402
from rl_engine.models.inkling2.router import (  # noqa: E402
    RoutingResult,
    route_topk_sigmoid,
    route_topk_sigmoid_rocm,
)
from rl_engine.models.inkling2.router_ablation import (  # noqa: E402
    route_with_bias_in_weights,
    route_with_reverse_tie_break,
    route_with_wrong_scale,
)

if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_router_forward"):
    pytest.skip("requires the native ROCm router extension", allow_module_level=True)


def _fixture():
    logits = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, -2.0, -3.0, -4.0, -5.0], [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    bias = torch.tensor([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], device="cuda")
    return logits, bias


def _production(num_tokens=64, experts=128, seed=51):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(num_tokens, experts, device="cuda", generator=gen).bfloat16()
    bias = torch.randn(experts, device="cuda", generator=gen) * 0.1
    return logits, bias


def _detect(baseline: RoutingResult, candidate: RoutingResult):
    """Return the first drifting (token, rank) of a candidate, or None when bitwise equal."""
    bad = (baseline.expert_indices != candidate.expert_indices) | (
        baseline.weights != candidate.weights
    )
    if not bad.any():
        return None
    token = int(bad.any(dim=1).nonzero()[0])
    rank = int(bad[token].nonzero()[0])
    return token, rank


def test_clean_control_matches_reference_bitwise():
    logits, bias = _production()
    native = route_topk_sigmoid_rocm(logits, bias)
    again = route_topk_sigmoid_rocm(logits, bias)
    assert _detect(native, again) is None
    reference = route_topk_sigmoid(logits, bias)
    assert torch.equal(native.expert_indices, reference.expert_indices)
    torch.testing.assert_close(native.weights, reference.weights, atol=1e-6, rtol=1e-6)


def test_bias_as_weight_fault_detected():
    logits, bias = _fixture()
    baseline = route_topk_sigmoid_rocm(logits, bias, top_k=4)
    faulty = route_with_bias_in_weights(logits, bias, top_k=4)
    assert torch.equal(faulty.expert_indices, baseline.expert_indices)
    assert not torch.equal(faulty.weights, baseline.weights)
    assert _detect(baseline, faulty) is not None
    assert torch.allclose(baseline.weights.sum(-1), torch.full((2,), 8.0, device="cuda"))


def test_expert_id_tie_break_fault_detected():
    logits, bias = _fixture()
    zero = torch.zeros_like(bias)
    baseline = route_topk_sigmoid_rocm(logits, zero, top_k=4)
    faulty = route_with_reverse_tie_break(logits, zero, top_k=4)
    assert baseline.expert_indices[0].tolist() == [0, 1, 2, 3]
    assert faulty.expert_indices[0].tolist() == [3, 2, 1, 0]
    assert _detect(baseline, faulty) == (0, 0)


def test_tie_break_fault_detected_across_wave_lanes():
    # Ties between experts owned by different lanes (and the same lane, 64 apart).
    logits = torch.full((1, 192), -9.0, device="cuda")
    logits[0, [5, 69, 133, 70]] = 1.0
    zero = torch.zeros(192, device="cuda")
    baseline = route_topk_sigmoid_rocm(logits, zero, top_k=4)
    faulty = route_with_reverse_tie_break(logits, zero, top_k=4)
    assert baseline.expert_indices[0].tolist() == [5, 69, 70, 133]
    assert faulty.expert_indices[0].tolist() == [133, 70, 69, 5]
    assert _detect(baseline, faulty) == (0, 0)


def test_route_scale_fault_detected():
    logits, bias = _fixture()
    baseline = route_topk_sigmoid_rocm(logits, bias, top_k=4, route_scale=8.0)
    faulty = route_with_wrong_scale(logits, bias, top_k=4)
    assert torch.equal(faulty.expert_indices, baseline.expert_indices)
    assert _detect(baseline, faulty) == (0, 0)
    assert torch.allclose(faulty.weights.sum(-1), torch.ones(2, device="cuda"))


def test_rank_order_fault_detected():
    # Fault: selected experts emitted in ascending expert-id order instead of score order.
    logits, bias = _production()
    baseline = route_topk_sigmoid_rocm(logits, bias)
    order = baseline.expert_indices.argsort(dim=-1)
    faulty = RoutingResult(
        baseline.expert_indices.gather(1, order), baseline.weights.gather(1, order)
    )
    location = _detect(baseline, faulty)
    assert location is not None
    # Same experts, same total weight: only the rank order drifted.
    assert torch.equal(
        faulty.expert_indices.sort(dim=-1).values, baseline.expert_indices.sort(dim=-1).values
    )
    assert location[1] < 8


def test_replay_fault_detected():
    # Fault: the replayed route table is stale (rolled by one token / earlier step logits).
    logits, bias = _production()
    baseline = route_topk_sigmoid_rocm(logits, bias)
    stale = route_topk_sigmoid_rocm(logits.roll(1, dims=0), bias)
    location = _detect(baseline, stale)
    assert location is not None
    assert location[0] == 0
    # A skewed (non-uniform) bias replay also drifts.
    skewed = route_topk_sigmoid_rocm(logits, bias.flip(0) * 4.0)
    assert _detect(baseline, skewed) is not None
    # A faithful replay is bitwise identical, independent of batch composition.
    replay = route_topk_sigmoid_rocm(logits[10:20], bias)
    assert torch.equal(replay.expert_indices, baseline.expert_indices[10:20])
    assert torch.equal(replay.weights, baseline.weights[10:20])
