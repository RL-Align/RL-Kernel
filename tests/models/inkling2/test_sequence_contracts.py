# SPDX-License-Identifier: Apache-2.0
"""INK-23, INK-34 and INK-71 sequence contract tests."""

import pytest
import torch

from rl_engine.models.inkling2.sequence_contracts import (
    ChunkRange,
    LongContextPlan,
    SequenceContractError,
    build_recompute_schedule,
    causal_position_mask,
    chunked_prefill_ranges,
    decode_position_ids,
    pack_rows,
    recompute_forward,
    unpack_rows,
    validate_unrelated_row_invariance,
)


def test_chunked_prefill_and_cached_decode_positions():
    assert chunked_prefill_ranges(10, 4) == (ChunkRange(0, 4), ChunkRange(4, 8), ChunkRange(8, 10))
    cache_lengths = torch.tensor([0, 7, 12])
    assert torch.equal(decode_position_ids(cache_lengths), cache_lengths[:, None])
    assert torch.equal(
        decode_position_ids(cache_lengths, step=2),
        torch.tensor([[0, 1], [7, 8], [12, 13]]),
    )


def test_packing_round_trip_and_unrelated_row_invariance():
    rows = (torch.arange(6).reshape(3, 2), torch.arange(8, 12).reshape(2, 2))
    packed = pack_rows(rows)
    assert torch.equal(packed.offsets, torch.tensor([0, 3, 5]))
    assert torch.equal(packed.row_ids, torch.tensor([0, 0, 0, 1, 1]))
    assert torch.equal(packed.position_ids, torch.tensor([0, 1, 2, 0, 1]))
    unpacked = unpack_rows(packed)
    for actual, expected in zip(unpacked, rows, strict=True):
        assert torch.equal(actual, expected)
    reference = torch.stack((torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])))
    changed = reference.clone()
    changed[1] += 5
    validate_unrelated_row_invariance(reference, changed, row=1)


def test_causal_and_local_position_masks():
    query = torch.tensor([[0, 1, 4]])
    key = torch.tensor([[0, 1, 2, 4]])
    causal = causal_position_mask(query, key)
    local = causal_position_mask(query, key, window=2)
    assert torch.equal(
        causal,
        torch.tensor(
            [[[True, False, False, False], [True, True, False, False], [True, True, True, True]]]
        ),
    )
    assert torch.equal(
        local,
        torch.tensor(
            [[[True, False, False, False], [True, True, False, False], [False, False, False, True]]]
        ),
    )


def test_recompute_preserves_forward_and_backward():
    x = torch.randn(5, 7, requires_grad=True)
    weight = torch.randn(7, 3, requires_grad=True)
    output = recompute_forward(lambda value, matrix: torch.tanh(value @ matrix), x, weight)
    reference = torch.tanh(x @ weight)
    gradient = torch.randn_like(output)
    actual = torch.autograd.grad(output, (x, weight), gradient)
    expected = torch.autograd.grad(reference, (x, weight), gradient)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)


def test_three_pass_schedule_has_stable_logical_visit_ids():
    schedule = build_recompute_schedule(2, repeated_passes=3)
    assert [
        (entry.physical_layer, entry.pass_index, entry.logical_visit) for entry in schedule
    ] == [
        (0, 0, 0),
        (1, 0, 1),
        (0, 1, 2),
        (1, 1, 3),
        (0, 2, 4),
        (1, 2, 5),
    ]


def test_long_context_plan_and_relative_clipping():
    plan = LongContextPlan()
    plan.validate(4096)
    plan.validate(1048576)
    query = torch.tensor([[0, 2000]])
    key = torch.tensor([[0, 3000]])
    assert torch.equal(
        plan.clipped_relative_positions(query, key), torch.tensor([[[0, -1024], [1024, -1000]]])
    )
    with pytest.raises(SequenceContractError, match="outside"):
        plan.validate(8192)


def test_invalid_sequence_metadata_fails_closed():
    with pytest.raises(SequenceContractError, match="positive"):
        chunked_prefill_ranges(0, 4)
    with pytest.raises(SequenceContractError, match="nonempty"):
        decode_position_ids(torch.empty(0, dtype=torch.int64))
