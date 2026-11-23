# SPDX-License-Identifier: Apache-2.0
"""INK-10 short-convolution contract tests."""

import pytest
import torch

from rl_engine.models.inkling2.short_conv import (
    STREAMS,
    causal_short_conv,
    causal_short_conv_streams,
)


def test_causal_short_conv_matches_explicit_reference():
    x = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3) / 10
    weight = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4) / 20
    out, state = causal_short_conv(x, weight)

    history = torch.cat((torch.zeros(2, 3, 3), x.transpose(1, 2)), dim=-1)
    expected = torch.zeros_like(out)
    for tap in range(4):
        expected = expected + (
            history[..., tap : tap + x.shape[1]].transpose(1, 2) * weight[:, tap]
        )
    assert torch.equal(out, expected)
    assert torch.equal(state, history[..., -3:])


def test_chunked_execution_matches_full_sequence():
    torch.manual_seed(4)
    x = torch.randn(2, 9, 5)
    weight = torch.randn(5, 4)
    full, full_state = causal_short_conv(x, weight)

    first, state = causal_short_conv(x[:, :4], weight)
    second, state = causal_short_conv(x[:, 4:], weight, state)
    assert torch.equal(torch.cat((first, second), dim=1), full)
    assert torch.equal(state, full_state)


def test_four_streams_keep_independent_state_and_weights():
    inputs = {name: torch.randn(1, 6, 4) for name in STREAMS}
    weights = {name: torch.randn(4, 4) for name in STREAMS}
    outputs, states = causal_short_conv_streams(inputs, weights)
    assert tuple(outputs) == STREAMS
    assert tuple(states) == STREAMS
    for name in STREAMS:
        expected, expected_state = causal_short_conv(inputs[name], weights[name])
        assert torch.equal(outputs[name], expected)
        assert torch.equal(states[name], expected_state)


def test_short_conv_backward_is_defined():
    x = torch.randn(2, 5, 3, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)
    output, _ = causal_short_conv(x, weight)
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_cuda_short_conv_matches_reference_and_backward():
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE
    from rl_engine.models.inkling2.short_conv import causal_short_conv_cuda

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_short_conv_forward_with_state"):
        pytest.skip("Inkling 2 CUDA extension is unavailable")

    x = torch.randn(2, 7, 5, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(5, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference, reference_state = causal_short_conv(x, weight)
    output, state = causal_short_conv_cuda(x, weight)
    torch.testing.assert_close(output, reference, atol=0, rtol=0)
    torch.testing.assert_close(state, reference_state, atol=0, rtol=0)
    output.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()
    assert weight.grad is not None and torch.isfinite(weight.grad.float()).all()
