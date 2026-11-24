# SPDX-License-Identifier: Apache-2.0
"""INK-51 CUDA top-8 routing fault-injection tests."""

import pytest
import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.models.inkling2.router import route_topk_sigmoid_cuda
from rl_engine.models.inkling2.router_ablation import (
    route_with_bias_in_weights,
    route_with_reverse_tie_break,
    route_with_wrong_scale,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not _EXT_AVAILABLE
    or not hasattr(_C, "inkling2_router_forward"),
    reason="requires the native INK-24 router extension",
)


def _fixture():
    logits = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, -2.0, -3.0, -4.0, -5.0], [4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    bias = torch.tensor([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0], device="cuda")
    return logits, bias


def test_selection_only_bias_fault_changes_weight_source():
    logits, bias = _fixture()
    baseline = route_topk_sigmoid_cuda(logits, bias, top_k=4)
    faulty = route_with_bias_in_weights(logits, bias, top_k=4)
    assert torch.equal(faulty.expert_indices, baseline.expert_indices)
    assert not torch.equal(faulty.weights, baseline.weights)
    assert torch.allclose(baseline.weights.sum(-1), torch.full((2,), 8.0, device="cuda"))


def test_expert_id_tie_break_fault_changes_equal_scores():
    logits, bias = _fixture()
    baseline = route_topk_sigmoid_cuda(logits, torch.zeros_like(bias), top_k=4)
    faulty = route_with_reverse_tie_break(logits, torch.zeros_like(bias), top_k=4)
    assert torch.equal(baseline.expert_indices[0], torch.tensor([0, 1, 2, 3], device="cuda"))
    assert torch.equal(faulty.expert_indices[0], torch.tensor([3, 2, 1, 0], device="cuda"))


def test_route_scale_fault_is_detected_after_normalization():
    logits, bias = _fixture()
    baseline = route_topk_sigmoid_cuda(logits, bias, top_k=4, route_scale=8.0)
    faulty = route_with_wrong_scale(logits, bias, top_k=4)
    assert torch.equal(faulty.expert_indices, baseline.expert_indices)
    assert not torch.equal(faulty.weights, baseline.weights)
    assert torch.allclose(faulty.weights.sum(-1), torch.ones(2, device="cuda"))
