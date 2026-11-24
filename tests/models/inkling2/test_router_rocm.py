# SPDX-License-Identifier: Apache-2.0
"""INK-24 ROCm wave64 router tests."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm GPU is unavailable",
)

from rl_engine.models.inkling2.router import (  # noqa: E402
    route_topk_sigmoid,
    route_topk_sigmoid_rocm,
)


def _binding_or_skip():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_router_forward"):
        pytest.skip("Inkling 2 ROCm router extension is unavailable")


@pytest.fixture(autouse=True)
def _require_binding():
    _binding_or_skip()


def _sequential_reference(logits, bias, top_k, route_scale):
    """Reference with the weight sum taken in selected rank order 0..k-1."""
    ref = route_topk_sigmoid(logits, bias, top_k=top_k, route_scale=route_scale)
    probs = torch.sigmoid(logits.float()).gather(1, ref.expert_indices)
    total = torch.zeros_like(probs[:, 0])
    for i in range(top_k):
        total = total + probs[:, i]
    total = total.clamp_min(torch.finfo(torch.float32).tiny)
    return ref.expert_indices, (probs / total.unsqueeze(1)) * float(route_scale)


def _make(tokens, experts, dtype, seed=24):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(tokens, experts, device="cuda", generator=gen).mul_(3).to(dtype)
    bias = torch.randn(experts, device="cuda", generator=gen) * 0.1
    return logits, bias


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tokens,experts,top_k",
    [
        (64, 128, 8),
        (257, 128, 8),
        (1, 128, 8),
        (5, 100, 8),
        (7, 63, 3),
        (9, 65, 8),
        (3, 8, 8),
        (4, 200, 1),
        (6, 129, 5),
    ],
)
def test_matches_reference_bitwise(dtype, tokens, experts, top_k):
    logits, bias = _make(tokens, experts, dtype)
    out = route_topk_sigmoid_rocm(logits, bias, top_k=top_k, route_scale=2.5)
    idx, weights = _sequential_reference(logits, bias, top_k, 2.5)
    assert out.expert_indices.dtype == torch.int64
    assert out.weights.dtype == torch.float32
    assert torch.equal(out.expert_indices, idx)
    assert torch.equal(out.weights, weights)
    ref = route_topk_sigmoid(logits, bias, top_k=top_k, route_scale=2.5)
    torch.testing.assert_close(out.weights, ref.weights, atol=1e-6, rtol=1e-6)


def test_no_bias_and_ties():
    out = route_topk_sigmoid_rocm(torch.zeros(3, 200, device="cuda"), top_k=8)
    expected = torch.arange(8, device="cuda").repeat(3, 1)
    assert torch.equal(out.expert_indices, expected)
    assert torch.equal(out.weights, torch.full((3, 8), 1.0, device="cuda"))
    # Ties spread across lanes: equal scores in experts 70, 5, 130, 64.
    logits = torch.full((1, 192), -9.0, device="cuda")
    logits[0, [70, 5, 130, 64]] = 1.0
    out = route_topk_sigmoid_rocm(logits, top_k=6)
    assert out.expert_indices[0, :4].tolist() == [5, 64, 70, 130]


def test_selection_bias_only_affects_ownership():
    logits = torch.tensor([[4.0, 3.0, -2.0, -3.0]], device="cuda")
    bias = torch.tensor([-10.0, -10.0, 10.0, -10.0], device="cuda")
    out = route_topk_sigmoid_rocm(logits, bias, top_k=2)
    assert out.expert_indices.tolist() == [[2, 0]]
    idx, weights = _sequential_reference(logits, bias, 2, 8.0)
    assert torch.equal(out.weights, weights)


def test_nan_and_inf_match_reference_semantics():
    logits, bias = _make(8, 128, torch.float32, seed=3)
    logits[0, 100] = float("nan")
    logits[1, [3, 70]] = float("nan")
    logits[2, 10] = float("inf")
    logits[3, :] = float("-inf")
    out = route_topk_sigmoid_rocm(logits, bias, top_k=8)
    ref = route_topk_sigmoid(logits, bias, top_k=8)
    assert torch.equal(out.expert_indices, ref.expert_indices)
    torch.testing.assert_close(out.weights, ref.weights, atol=1e-6, rtol=1e-6, equal_nan=True)
    assert out.expert_indices[0, 0].item() == 100
    assert out.expert_indices[1, :2].tolist() == [3, 70]
    # All-underflow row hits the FLT_MIN floor: finite zero weights, no NaN.
    assert torch.isfinite(out.weights[3]).all()


def test_empty_batch():
    out = route_topk_sigmoid_rocm(torch.empty(0, 128, device="cuda"), top_k=8)
    assert out.expert_indices.shape == (0, 8)
    assert out.weights.shape == (0, 8)


def test_rejects_invalid_arguments():
    logits = torch.randn(2, 16, device="cuda")
    with pytest.raises(RuntimeError):
        route_topk_sigmoid_rocm(logits, top_k=9)
    with pytest.raises(RuntimeError):
        route_topk_sigmoid_rocm(logits, top_k=0)
    with pytest.raises(ValueError):
        route_topk_sigmoid_rocm(logits.cpu())


def test_run_to_run_deterministic_and_batch_invariant():
    logits, bias = _make(300, 128, torch.bfloat16, seed=9)
    first = route_topk_sigmoid_rocm(logits, bias)
    for _ in range(5):
        again = route_topk_sigmoid_rocm(logits, bias)
        assert torch.equal(first.expert_indices, again.expert_indices)
        assert torch.equal(first.weights, again.weights)
    # Row output independent of batch size and of other rows' contents.
    single = route_topk_sigmoid_rocm(logits[137:138], bias)
    assert torch.equal(single.expert_indices, first.expert_indices[137:138])
    assert torch.equal(single.weights, first.weights[137:138])
    other = logits.clone()
    other[:137] = torch.randn_like(other[:137])
    mixed = route_topk_sigmoid_rocm(other, bias)
    assert torch.equal(mixed.expert_indices[137:], first.expert_indices[137:])
    assert torch.equal(mixed.weights[137:], first.weights[137:])
    sliced = route_topk_sigmoid_rocm(logits[:50], bias)
    assert torch.equal(sliced.weights, first.weights[:50])


def test_requires_rocm_device_tensors():
    with pytest.raises(ValueError):
        route_topk_sigmoid_rocm(torch.zeros(1, 8), top_k=2)
