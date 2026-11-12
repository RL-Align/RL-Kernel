# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU regressions for the provider-specific references extracted from PR #469."""

# Adapted from PR #469, 2b64780b2e1be5d97e936405bb1e899fea14215c.

import pytest
import torch

from rl_engine.reference.linear_attn import CausalConv1dUpdateOp, GatedDeltaRuleRecurrentStepOp


def _call(kind, indices, heads=1):
    batch = indices.numel()
    if kind == "conv":
        state = torch.randn(4, 2, 3)
        before = state.clone()
        result = CausalConv1dUpdateOp()(torch.randn(batch, 2), state, torch.randn(2, 4), indices)
    else:
        state = torch.randn(4, 2, 2, 32)
        before = state.clone()
        result = GatedDeltaRuleRecurrentStepOp()(
            torch.randn(batch, 68),
            torch.randn(batch, 2),
            torch.randn(batch, 2),
            torch.zeros(2),
            torch.zeros(2),
            state,
            indices,
            scale=32**-0.5,
            num_k_heads=heads,
        )
    assert torch.equal(state, before)
    return result, before


@pytest.mark.parametrize("kind", ["conv", "gdn"])
@pytest.mark.parametrize(
    "indices,message",
    [
        (torch.tensor([1.5, 2.0]), "int32 or int64"),
        (torch.tensor([True, False]), "int32 or int64"),
        (torch.tensor([1, 1]), "unique"),
        (torch.tensor([1, 4]), "out of range"),
    ],
)
def test_invalid_cache_indices_are_rejected(kind, indices, message):
    with pytest.raises(ValueError, match=message):
        _call(kind, indices)


@pytest.mark.parametrize(
    "kind, indices",
    [
        ("conv", torch.tensor([0, 0, 0])),
        # The two ops differ on purpose: the GDN provider skips every index <= 0, while
        # vLLM's conv provider skips only 0 and a negative conv index is rejected below
        # (see NULL_BLOCK_ID in gated_delta_rule.py).
        ("gdn", torch.tensor([0, -1, 0])),
        ("conv", torch.empty(0, dtype=torch.int64)),
        ("gdn", torch.empty(0, dtype=torch.int64)),
    ],
)
def test_inactive_and_empty_batches_preserve_cache(kind, indices):
    (out, state), before = _call(kind, indices)
    assert torch.equal(state, before)
    assert torch.count_nonzero(out) == 0


def test_conv_rejects_negative_state_indices():
    """vLLM skips only the null block id, so a negative entry must not pass as inactive."""
    with pytest.raises(ValueError, match="must be >= 0"):
        _call("conv", torch.tensor([1, -1, 2]))


@pytest.mark.parametrize("heads", [0, -1, 1.5, True])
def test_gdn_requires_positive_integer_heads(heads):
    with pytest.raises(ValueError, match="positive integer"):
        _call("gdn", torch.tensor([1, 2]), heads=heads)


def test_conv_bias_is_accumulated_before_taps():
    # Two products of 3.0 and a bias of 1e8 (fp32 spacing 8 there). Bias first: 1e8 + 3
    # and 1e8 + 3 both round back to 1e8. Bias last: 3 + 3 = 6 survives, and 6 + 1e8
    # rounds up to 100000008.
    state = torch.zeros(2, 1, 1)
    state[1] = 3.0
    out, _ = CausalConv1dUpdateOp()(
        torch.tensor([[3.0]]),
        state,
        torch.ones(1, 2),
        torch.tensor([1]),
        bias=torch.tensor([1e8]),
        activation=None,
    )
    assert out.item() == 1e8
    assert out.item() != (3.0 + 3.0) + 1e8  # what adding the bias last would give


def test_conv_bf16_products_round_before_fp32_accumulation():
    state = torch.zeros(2, 1, 1, dtype=torch.bfloat16)
    state[1, 0, 0] = 1.0078125
    out, _ = CausalConv1dUpdateOp().forward_fp32(
        torch.tensor([[-1.015625]], dtype=torch.bfloat16),
        state,
        torch.tensor([[1.0078125, 1.0]], dtype=torch.bfloat16),
        torch.tensor([1]),
        activation=None,
    )
    # The first exact product is 1.015686..., rounded to 1.015625 in BF16.
    assert out.item() == 0.0


@pytest.mark.parametrize("cache_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_conv_mixed_dtype_preserves_provider_product_promotion(cache_dtype, weight_dtype):
    # vLLM 0.30 casts the token, not the weights. An FP32 weight must not be
    # narrowed even with a BF16 cache (PR #469, issuecomment-6429237865).
    state = torch.zeros(2, 1, 1, dtype=cache_dtype)
    state[1, 0, 0] = 1.0078125
    before = state.clone()
    x = torch.tensor([[-1.0157]])
    weight = torch.tensor([[1.0079, 1.0]], dtype=weight_dtype)
    bias = torch.tensor([0.125])
    out, updated = CausalConv1dUpdateOp().forward_fp32(
        x, state, weight, torch.tensor([1]), bias=bias, activation=None
    )
    product_dtype = torch.promote_types(cache_dtype, weight_dtype)
    window = [state[1, 0, 0], x[0, 0].to(cache_dtype)]
    expected = bias[0].clone()
    for tap in range(2):
        product = (window[tap].double() * weight[0, tap].double()).to(product_dtype)
        expected = expected + product.float()
    assert torch.equal(out[0, 0], expected)
    assert torch.equal(state, before)
    assert torch.equal(updated[0], before[0])
    assert torch.equal(updated[1, 0, 0], x[0, 0].to(cache_dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("widen", [False, True])
def test_gdn_returned_state_is_independent_and_differentiable(dtype, widen):
    # Noncontiguous cache: both same-dtype copying and widening must preserve
    # values, isolate writes, and carry gradients through untouched blocks.
    state = torch.arange(24, dtype=torch.float32).reshape(3, 2, 2, 2)
    state = state.transpose(-1, -2).to(dtype).detach().requires_grad_()
    before = state.detach().clone()
    op = GatedDeltaRuleRecurrentStepOp()
    forward = op.forward_fp32 if widen else op.forward
    out, updated = forward(
        torch.ones(2, 12),
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.zeros(2),
        torch.zeros(2),
        state,
        torch.tensor([0, -1]),
        scale=1.0,
        num_k_heads=2,
    )
    target_dtype = torch.float32 if widen else dtype
    assert updated.dtype == target_dtype
    assert torch.equal(updated, before.to(target_dtype))
    assert updated.data_ptr() != state.data_ptr()
    assert torch.count_nonzero(out) == 0
    updated.sum().backward()
    assert torch.equal(state.grad, torch.ones_like(state))
    with torch.no_grad():
        updated[0] = -1
    assert torch.equal(state, before)


def test_softplus_large_branch_has_finite_gradient():
    from rl_engine.reference.linear_attn.gated_delta_rule import _softplus

    x = torch.tensor([100.0, 21.0, 0.0], requires_grad=True)
    _softplus(x).sum().backward()
    assert torch.equal(x.grad, torch.tensor([1.0, 1.0, 0.5]))


def test_chunked_sum_keeps_fixed_order_for_non_multiple_of_32_width():
    from rl_engine.reference.linear_attn.gated_delta_rule import _REDUCTION_CHUNK, _chunked_sum

    torch.manual_seed(0)
    x = torch.randn(3, 5, 2 * _REDUCTION_CHUNK + 7)
    padded = torch.nn.functional.pad(x, (0, _REDUCTION_CHUNK - 7))
    chunked = padded.reshape(3, 5, 3, _REDUCTION_CHUNK).sum(dim=-1).sum(dim=-1)
    assert torch.equal(_chunked_sum(x), chunked)
    # The reduction must not depend on the leading shape either.
    assert torch.equal(_chunked_sum(x[:1]), _chunked_sum(x)[:1])


@pytest.mark.parametrize("tap_weight", [-100.0, -30.0, 0.5, 30.0, 100.0])
def test_conv_silu_backward_is_finite_and_matches_torch_silu(tap_weight):
    # exp(-acc) overflows FP32 for acc < -88.7; the forward value must stay the
    # provider-matching one while the gradient stays finite and equal to F.silu's.
    dim, width = 4, 4
    state = torch.zeros(2, dim, width - 1)
    weight = torch.zeros(dim, width)
    weight[:, -1] = tap_weight
    indices = torch.tensor([1])
    x = torch.ones(1, dim, requires_grad=True)
    out, _ = CausalConv1dUpdateOp().forward_fp32(x, state, weight, indices)
    out.sum().backward()
    acc = torch.full((1, dim), tap_weight, requires_grad=True)
    torch.nn.functional.silu(acc).sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, acc.grad * tap_weight, rtol=1e-5, atol=1e-30)
    plain = acc.detach() / (1.0 + torch.exp(-acc.detach()))
    assert torch.equal(out.detach(), plain)


def test_state_index_helper_skip_rule_is_explicit():
    # The recurrent rule skips every index <= 0; the conv rule skips only 0, so a negative
    # index is an active row and fails the range check instead of being dropped.
    from rl_engine.reference.linear_attn.gated_delta_rule import _validate_state_indices

    indices = torch.tensor([0, -1, 2])
    _validate_state_indices(indices, 3, 4, indices.device)
    with pytest.raises(ValueError, match="out of range"):
        _validate_state_indices(indices, 3, 4, indices.device, skip_negative=False)
