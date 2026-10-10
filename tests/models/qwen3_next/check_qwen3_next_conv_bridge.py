# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The convolution provider must agree across batch, chunk and cache layouts."""

import pytest
import torch

from rl_engine.integrations.engines.train.vllm.qwen3_next_conv import (
    _provider,
    causal_conv_sequence,
)
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal as exact


def inputs(lengths):
    assert torch.cuda.is_available(), "Required provider tests cannot skip CUDA"
    gen = torch.Generator(device="cuda").manual_seed(17)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=gen)

    return (
        rand(sum(lengths), 2048),
        rand(len(lengths) + 1, 2048, 3),
        rand(2048, 4),
        torch.arange(1, len(lengths) + 1, device="cuda", dtype=torch.int32),
        torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()]),
    )


@torch.no_grad()
@pytest.mark.parametrize("lengths", [(1,), (64, 8, 0, 16), (1024, 256, 64, 8)])
def test_varlen_matches_repeated_decode_and_preserves_inputs(lengths):
    x, state, weight, indices, cu = inputs(lengths)
    saved = [tensor.clone() for tensor in (x, state, weight)]
    output, final_state = causal_conv_sequence(x, state, weight, indices, cu)
    reference_state = state.clone()
    reference_rows = []
    for seq in range(len(lengths)):
        for token in range(cu[seq], cu[seq + 1]):
            value = x[token : token + 1].clone()
            reference_rows.append(
                _provider()(
                    value,
                    reference_state,
                    weight,
                    activation="silu",
                    conv_state_indices=indices[seq : seq + 1],
                )
            )
    exact(output, torch.cat(reference_rows))
    exact(final_state, reference_state)
    for original, before in zip((x, state, weight), saved):
        exact(original, before)


@torch.no_grad()
def test_chunked_and_reordered_sequences_match():
    x, state, weight, indices, cu = inputs((1024, 64))
    whole, whole_state = causal_conv_sequence(x, state, weight, indices, cu)
    # Continue both sequences after 32 tokens, in the opposite batch order.
    prompt = torch.cat((x[:32], x[1024:1056]))
    first, next_state = causal_conv_sequence(
        prompt, state, weight, indices, torch.tensor([0, 32, 64])
    )
    response = torch.cat((x[1056:], x[32:1024]))
    second, next_state = causal_conv_sequence(
        response, next_state, weight, indices.flip(0), torch.tensor([0, 32, 1024])
    )
    restored = torch.cat((first[:32], second[32:], first[32:], second[:32]))
    exact(restored, whole)
    exact(next_state, whole_state)


def test_response_gradient_reaches_prompt_and_convolution_weight():
    x, state, weight, indices, _ = inputs((8,))
    x.requires_grad_()
    state.requires_grad_()
    weight.requires_grad_()
    _, prompt_state = causal_conv_sequence(x[:4], state, weight, indices, torch.tensor([0, 4]))
    out, final_state = causal_conv_sequence(
        x[4:], prompt_state, weight, indices, torch.tensor([0, 4])
    )
    (out.float().square().sum() + final_state.float().square().sum() * 0.01).backward()
    for tensor in (x, weight):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().max() > 0
    assert x.grad[:4].abs().max() > 0


def test_empty_sequence_preserves_cache_and_gradients():
    x, state, weight, indices, cu = inputs((0,))
    state.requires_grad_()
    out, next_state = causal_conv_sequence(x, state, weight, indices, cu)
    assert out.shape == x.shape
    exact(next_state, state)
    next_state.float().sum().backward()
    exact(state.grad, torch.ones_like(state))
