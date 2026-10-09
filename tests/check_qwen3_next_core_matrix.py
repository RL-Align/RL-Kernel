# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Mandatory TP4-shape GDN state tests; no full-model claim or optional skips."""

import pytest
import torch

from rl_engine.integrations.qwen3_next_provider import GDNProviderConfig, GDNState, shared_gdn_core
from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal as exact


def _inputs(lengths):
    assert torch.cuda.is_available(), "Required shared-provider CUDA tests cannot skip"
    generator = torch.Generator(device="cuda").manual_seed(20261002)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator) * 0.1

    tokens = sum(lengths)
    values = [rand(tokens, 2048), rand(tokens, 8), rand(tokens, 8), rand(tokens, 8, 128)]
    weights = [rand(8, dtype=torch.float32), rand(8, dtype=torch.float32), rand(2048, 4), rand(128)]
    state = GDNState(
        rand(len(lengths) + 2, 2048, 3),
        rand(len(lengths) + 2, 8, 128, 128, dtype=torch.float32),
    )
    indices = torch.arange(1, len(lengths) + 1, device="cuda", dtype=torch.int32)
    return values, weights, state, indices


def _run(values, weights, state, indices, lengths):
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64)
    return shared_gdn_core(
        *values, *weights, state, indices, boundaries, config=GDNProviderConfig(), num_k_heads=4
    )


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("length", [8, 64, 256, 1024])
@torch.no_grad()
def test_full_chunked_reordered_shared_core(batch, length):
    lengths = [length] * batch
    values, weights, initial, indices = _inputs(lengths)
    whole, final = _run(values, weights, initial, indices, lengths)
    for cut in (1, 63, 64, 65, 100, 528, 576):
        cut = min(cut, length)
        state = initial
        pieces = [[] for _ in lengths]
        for start, end in ((0, cut), (cut, length)):
            # Reorder the response-bearing step, preserving sequence/cache identity.
            order = list(range(batch)) if start == 0 else list(reversed(range(batch)))
            rows = [row for seq in order for row in range(seq * length + start, seq * length + end)]
            row_ids = torch.tensor(rows, device="cuda", dtype=torch.int64)
            chunk = [value.index_select(0, row_ids) for value in values]
            part, state = _run(chunk, weights, state, indices[order], [end - start] * batch)
            for position, seq in enumerate(order):
                pieces[seq].append(part[position * (end - start) : (position + 1) * (end - start)])
        exact(torch.cat([torch.cat(seq) for seq in pieces]), whole, name=f"output cut={cut}")
        exact(state.convolution, final.convolution, name=f"conv state cut={cut}")
        exact(state.recurrent, final.recurrent, name=f"recurrent state cut={cut}")
    exact(final.convolution[0], initial.convolution[0], name="reserved conv slot")
    exact(final.recurrent[0], initial.recurrent[0], name="reserved recurrent slot")


@torch.no_grad()
def test_variable_lengths_empty_and_mixed_step_match_independent_sequences():
    lengths = [65, 1, 0, 17]
    values, weights, initial, indices = _inputs(lengths)
    mixed, mixed_state = _run(values, weights, initial, indices, lengths)
    start = 0
    for seq, length in enumerate(lengths):
        end = start + length
        output, state = _run(
            [value[start:end] for value in values],
            weights,
            initial,
            indices[seq : seq + 1],
            [length],
        )
        exact(output, mixed[start:end], name=f"sequence {seq}")
        exact(state.convolution[seq + 1], mixed_state.convolution[seq + 1], name="conv state")
        exact(state.recurrent[seq + 1], mixed_state.recurrent[seq + 1], name="recurrent state")
        start = end
    exact(mixed_state.recurrent[3], initial.recurrent[3], name="empty sequence state")


@pytest.mark.parametrize("batch", [1, 4])
@torch.no_grad()
def test_shared_core_one_hundred_repetitions(batch):
    lengths = [8] * batch
    values, weights, initial, indices = _inputs(lengths)
    expected, expected_state = _run(values, weights, initial, indices, lengths)
    for repeat in range(100):
        output, state = _run(values, weights, initial, indices, lengths)
        exact(output, expected, name=f"output repeat={repeat}")
        exact(state.convolution, expected_state.convolution, name="conv state")
        exact(state.recurrent, expected_state.recurrent, name="recurrent state")
