"""Provider-independent cache contracts, collected in the ordinary CPU suite."""

import pytest
import torch

from rl_engine.kernels.ops.pytorch.linear_attn import (
    CausalConv1dUpdateOp,
    GatedDeltaRuleRecurrentStepOp,
)


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


@pytest.mark.parametrize("kind", ["conv", "gdn"])
@pytest.mark.parametrize("indices", [torch.tensor([0, -1, 0]), torch.empty(0, dtype=torch.int64)])
def test_inactive_and_empty_batches_preserve_cache(kind, indices):
    (out, state), before = _call(kind, indices)
    assert torch.equal(state, before)
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("heads", [0, -1, 1.5, True])
def test_gdn_requires_positive_integer_heads(heads):
    with pytest.raises(ValueError, match="positive integer"):
        _call("gdn", torch.tensor([1, 2]), heads=heads)


def test_conv_bias_is_accumulated_before_taps():
    # Adding bias last would yield 1.0 instead of 0.0.
    out, _ = CausalConv1dUpdateOp()(
        torch.tensor([[-1e8]]),
        torch.zeros(2, 1, 1),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([1]),
        bias=torch.tensor([1e8]),
        activation=None,
    )
    assert out.item() == 0.0
    out, _ = CausalConv1dUpdateOp()(
        torch.tensor([[-1e8]]),
        torch.ones(2, 1, 1),
        torch.ones(1, 2),
        torch.tensor([1]),
        bias=torch.tensor([1e8]),
        activation=None,
    )
    assert out.item() == 0.0


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
