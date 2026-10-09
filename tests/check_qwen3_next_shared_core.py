# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Combined provider contract; does not instantiate a complete Qwen model."""

from dataclasses import replace

import pytest
import torch

from rl_engine.integrations.qwen3_next_provider import GDNProviderConfig, GDNState, shared_gdn_core
from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal as exact


def inputs(tokens, grad=False):
    assert torch.cuda.is_available(), "Required CUDA provider suite cannot skip"
    gen = torch.Generator(device="cuda").manual_seed(73)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=gen).requires_grad_(grad)

    args = [
        rand(tokens, 2048),
        rand(tokens, 8),
        rand(tokens, 8),
        rand(tokens, 8, 128),
        rand(8, dtype=torch.float32),
        rand(8, dtype=torch.float32),
        rand(2048, 4),
        rand(128),
        GDNState(rand(2, 2048, 3), rand(2, 8, 128, 128, dtype=torch.float32)),
    ]
    return args, torch.tensor([1], device="cuda", dtype=torch.int32)


@pytest.mark.parametrize("tokens", [8, 1024])
@torch.no_grad()
def test_shared_core_chunk_and_decode_state_match(tokens):
    args, indices = inputs(tokens)
    config = GDNProviderConfig()
    output, state = shared_gdn_core(
        *args, indices, torch.tensor([0, tokens]), config=config, num_k_heads=4
    )
    pieces, next_state = [], args[-1]
    for start, end in ((0, 3), (3, tokens - 1), (tokens - 1, tokens)):
        chunk = [*(value[start:end] for value in args[:4]), *args[4:8], next_state]
        part, next_state = shared_gdn_core(
            *chunk, indices, torch.tensor([0, end - start]), config=config, num_k_heads=4
        )
        pieces.append(part)
    exact(torch.cat(pieces), output)
    exact(next_state.convolution, state.convolution)
    exact(next_state.recurrent, state.recurrent)


def test_independent_training_forward_matches_inference_and_retains_gradients():
    args, indices = inputs(8, grad=True)
    config = GDNProviderConfig()
    with torch.no_grad():
        reference, _ = shared_gdn_core(
            *args, indices, torch.tensor([0, 8]), config=config, num_k_heads=4
        )
    prompt = [*(value[:4] for value in args[:4]), *args[4:]]
    _, prompt_state = shared_gdn_core(
        *prompt, indices, torch.tensor([0, 4]), config=config, num_k_heads=4
    )
    prompt_state.recurrent.retain_grad()
    response = [*(value[4:] for value in args[:4]), *args[4:8], prompt_state]
    actual, _ = shared_gdn_core(
        *response, indices, torch.tensor([0, 4]), config=config, num_k_heads=4
    )
    exact(actual, reference[4:])
    actual.float().square().sum().backward()
    for value in args[:8]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().max() > 0
    assert args[0].grad[:4].abs().max() > 0
    assert prompt_state.recurrent.grad.abs().max() > 0


def test_unimplemented_provider_profile_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        replace(GDNProviderConfig(), recurrence="unverified-prefill").validate()
