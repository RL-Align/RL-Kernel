# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Required Qwen3-Next CUDA attention gates; unavailable CUDA is a failure."""

import pytest
import torch

from rl_engine.backends.cuda.attention.deterministic_attn import DeterministicAttentionOp
from rl_engine.backends.extension import _C
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal as exact


def inputs(tokens, *, batch=1, query_tokens=None, device="cuda", grad=False):
    assert torch.cuda.is_available(), "Mandatory attention CUDA gate cannot skip"
    gen = torch.Generator(device=device).manual_seed(920)
    query_tokens = tokens if query_tokens is None else query_tokens
    return tuple(
        torch.randn(batch, heads, length, 256, device=device, dtype=torch.bfloat16, generator=gen)
        .mul_(0.25)
        .requires_grad_(grad)
        for heads, length in ((4, query_tokens), (1, tokens), (1, tokens))
    )


@pytest.mark.parametrize("tokens", [8, 64, 256, 1024])
@torch.no_grad()
def test_d256_full_chunk_decode_and_batch_are_raw_bit_equal(tokens):
    q, k, v = inputs(tokens, batch=4)
    op = DeterministicAttentionOp()
    whole, lse = op.forward_with_lse(q, k, v)
    order = torch.tensor([3, 1, 0, 2], device=q.device)
    exact(op(q[order], k[order], v[order]), whole[order])
    for row in range(4):
        exact(op(q[row : row + 1], k[row : row + 1], v[row : row + 1]), whole[row : row + 1])
    starts = [0, 1, 7, tokens - 1, tokens]
    starts = sorted(set(starts))
    for start, end in zip(starts[:-1], starts[1:]):
        out, part_lse = op.forward_with_lse(q[:, :, start:end], k[:, :, :end], v[:, :, :end])
        exact(out, whole[:, :, start:end])
        exact(part_lse, lse[:, :, start:end])


def test_d256_backward_with_cached_prefix_matches_float64_formula():
    q, k, v = inputs(19, query_tokens=3, grad=True)
    op = DeterministicAttentionOp()
    out = op(q, k, v)
    grad = torch.linspace(-1, 1, out.numel(), device=q.device).reshape_as(out).to(out.dtype)
    out.backward(grad)
    qr, kr, vr = (x.detach().double().requires_grad_() for x in (q, k, v))
    scores = qr @ kr.repeat_interleave(4, dim=1).transpose(-1, -2) / 16
    causal = (
        torch.arange(19, device=q.device)[None, :] <= torch.arange(16, 19, device=q.device)[:, None]
    )
    expected = scores.masked_fill(~causal, -torch.inf).softmax(-1) @ vr.repeat_interleave(4, dim=1)
    expected.backward(grad.double())
    # Formula accuracy is separate from the raw-bit provider comparisons above.
    for actual, reference in zip((q, k, v), (qr, kr, vr)):
        assert torch.isfinite(actual.grad).all() and actual.grad.abs().max() > 0
        torch.testing.assert_close(
            actual.grad.float(), reference.grad.float(), atol=0.002, rtol=0.016
        )
    assert k.grad[:, :, :16].abs().max() > 0
    assert v.grad[:, :, :16].abs().max() > 0


def test_native_backward_rejects_malformed_probability_and_gradient_buffers():
    q, k, v = inputs(8)
    out, _, probabilities = _C.deterministic_attention_forward(q, k, v, True, 1 / 16, None)
    with pytest.raises(RuntimeError, match="P must be FP32"):
        _C.deterministic_attention_backward(
            out, q, k, v, probabilities.to(q.dtype), True, 1 / 16, None
        )
    with pytest.raises(RuntimeError, match="grad_output must have q shape"):
        _C.deterministic_attention_backward(
            out[:, :, :1], q, k, v, probabilities, True, 1 / 16, None
        )
    with pytest.raises(RuntimeError, match="P must have shape"):
        _C.deterministic_attention_backward(
            out, q, k, v, probabilities[:, :, :1], True, 1 / 16, None
        )


def test_d256_input_device_guard_and_nondefault_stream():
    assert torch.cuda.device_count() >= 2, "Mandatory cross-device gate requires allocated GPUs"
    previous = torch.cuda.current_device()
    try:
        q, k, v = inputs(8, device="cuda:1", grad=True)
        op = DeterministicAttentionOp()
        torch.cuda.set_device(1)
        expected = op(q, k, v)
        expected.sum().backward()
        gradients = [x.grad.clone() for x in (q, k, v)]
        for x in (q, k, v):
            x.grad = None
        torch.cuda.synchronize(1)
        stream = torch.cuda.Stream(device=1)
        with torch.cuda.stream(stream):
            torch.cuda.set_device(0)
            actual = op(q, k, v)
            actual.sum().backward()
            assert torch.cuda.current_device() == 0
        stream.synchronize()
        exact(actual, expected)
        for x, expected_grad in zip((q, k, v), gradients):
            exact(x.grad, expected_grad)
        with pytest.raises(ValueError, match="same device"):
            op(q, k.to("cuda:0"), v)
    finally:
        torch.cuda.set_device(previous)
