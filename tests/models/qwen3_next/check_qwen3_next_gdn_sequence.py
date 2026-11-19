# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Packed decode recurrence boundary checks; no complete-model L2 claim."""

import pytest
import torch

from rl_engine.integrations.engines.train.vllm.qwen3_next_gdn import packed_recurrent_sequence
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal as exact


def inputs(lengths, seed=7):
    assert torch.cuda.is_available(), "CUDA is required; this acceptance suite cannot skip"
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    tokens = sum(lengths)
    # Actual TP4 shard: 16 / 4 key heads, 32 / 4 value heads, K=V=128.
    tensors = [
        rand(tokens, 2048),
        rand(tokens, 8),
        rand(tokens, 8),
        rand(8, dtype=torch.float32),
        rand(8, dtype=torch.float32),
        rand(len(lengths) + 1, 8, 128, 128, dtype=torch.float32) * 0.01,
    ]
    indices = torch.arange(1, len(lengths) + 1, device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    return tensors, indices, cu


def run(tensors, indices, cu):
    return packed_recurrent_sequence(*tensors, indices, cu, num_k_heads=4)


@torch.no_grad()
@pytest.mark.parametrize("lengths", [(8,), (64, 8, 0, 16), (1024, 256, 64, 8)])
def test_packed_matches_independent_sequences_and_reordering(lengths):
    tensors, indices, cu = inputs(lengths)
    original_state = tensors[-1].clone()
    output, state = run(tensors, indices, cu)
    for seq, length in enumerate(lengths):
        start, end = cu[seq : seq + 2].tolist()
        one = [*(value[start:end] for value in tensors[:3]), *tensors[3:]]
        expected, expected_state = run(one, indices[seq : seq + 1], torch.tensor([0, length]))
        exact(output[start:end], expected)
        exact(state[seq + 1], expected_state[seq + 1])
    permutation = list(reversed(range(len(lengths))))
    rows = torch.cat([torch.arange(cu[seq], cu[seq + 1], device="cuda") for seq in permutation])
    changed = [*(value.index_select(0, rows) for value in tensors[:3]), *tensors[3:]]
    reordered_cu = torch.tensor([0, *torch.tensor([lengths[i] for i in permutation]).cumsum(0)])
    reordered, reordered_state = run(changed, indices[permutation], reordered_cu)
    exact(reordered, output.index_select(0, rows))
    exact(reordered_state, state)
    exact(tensors[-1], original_state)


@torch.no_grad()
def test_1024_token_chunk_continuation_is_bitwise_exact():
    tensors, indices, cu = inputs((1024,))
    whole, whole_state = run(tensors, indices, cu)
    state, pieces = tensors[-1], []
    cuts = (0, 1, 8, 64, 256, 513, 1024)
    for start, end in zip(cuts, cuts[1:]):
        chunk = [*(value[start:end] for value in tensors[:3]), *tensors[3:5], state]
        output, state = run(chunk, indices, torch.tensor([0, end - start]))
        pieces.append(output)
    exact(torch.cat(pieces), whole)
    exact(state, whole_state)


def test_chunk_boundary_retains_prompt_and_initial_state_gradients():
    tensors, indices, cu = inputs((8,))
    tensors = [tensor.requires_grad_() for tensor in tensors]
    prompt = [*(value[:4] for value in tensors[:3]), *tensors[3:]]
    _, prompt_state = run(prompt, indices, torch.tensor([0, 4]))
    prompt_state.retain_grad()
    response = [*(value[4:] for value in tensors[:3]), *tensors[3:5], prompt_state]
    out, state = run(response, indices, torch.tensor([0, 4]))
    (out.float().square().sum() + state.square().sum() * 0.01).backward()
    assert prompt_state.grad is not None and prompt_state.grad.abs().max() > 0
    for value in tensors:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().max() > 0
    assert tensors[0].grad[:4].abs().max() > 0


@pytest.mark.parametrize("bad", [[1, 8], [0, 9], [0, 6, 5, 8]])
def test_bad_packing_is_rejected(bad):
    tensors, indices, _ = inputs((8,))
    with pytest.raises(ValueError, match="cu_seqlens"):
        run(tensors, indices, torch.tensor(bad))


def test_reserved_and_duplicate_slots_are_rejected():
    tensors, indices, cu = inputs((4, 4))
    for bad in ([0, 1], [1, 1]):
        with pytest.raises(ValueError):
            run(tensors, torch.tensor(bad, device="cuda", dtype=torch.int32), cu)
