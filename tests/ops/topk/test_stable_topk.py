# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Semantics of the stable_topk_per_row golden (vLLM ds4_bi selector order)."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.topk import ordered_bits, stable_topk_per_row


def _rows(*pairs):
    starts = torch.tensor([p[0] for p in pairs], dtype=torch.int32)
    ends = torch.tensor([p[1] for p in pairs], dtype=torch.int32)
    return starts, ends


def test_score_then_larger_local_index_order():
    # vLLM tests/kernels/test_ds4_bi_topk_extension.py::..._score_and_tie_order
    logits = torch.zeros(1, 16)
    logits[0, 5], logits[0, 2], logits[0, 7], logits[0, 1] = 3, 4, 4, 2
    out = stable_topk_per_row(logits, *_rows((0, 16)), 4)
    assert out.tolist() == [[7, 2, 5, 1]]
    assert out.dtype == torch.int32


def test_equal_scores_take_the_largest_request_local_indices():
    # vLLM ..._uses_standalone_topk_with_request_offsets: ids are relative to the row start.
    logits = torch.ones(3, 1024)
    starts, ends = _rows((0, 512), (17, 529), (65, 577))
    out = stable_topk_per_row(logits, starts, ends, 64)
    expected = (511 - torch.arange(64, dtype=torch.int32)).expand(3, 64)
    assert torch.equal(out, expected)


def test_short_and_empty_rows_pad_with_minus_one():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0], [5.0, 5.0, 5.0, 5.0], [9.0, 9.0, 9.0, 9.0]])
    out = stable_topk_per_row(logits, *_rows((0, 3), (2, 2), (3, 1)), 5)
    assert out.tolist() == [[1, 2, 0, -1, -1], [-1] * 5, [-1] * 5]


def test_total_order_over_special_values():
    nan_pos = torch.tensor([0x7FC00000], dtype=torch.int32).view(torch.float32).item()
    nan_neg = torch.tensor([0xFFC00000 - (1 << 32)], dtype=torch.int32).view(torch.float32).item()
    values = [-float("inf"), -1.0, -0.0, 0.0, 1e-45, 1.0, float("inf"), nan_pos, nan_neg]
    logits = torch.tensor([values])
    out = stable_topk_per_row(logits, *_rows((0, len(values))), len(values))
    # +NaN > +inf > 1 > denormal > +0 > -0 > -1 > -inf > -NaN
    assert out.tolist() == [[7, 6, 5, 4, 3, 2, 1, 0, 8]]


def test_ordered_bits_is_monotone_on_finite_values():
    x = torch.randn(4096) * torch.logspace(-40, 30, 4096)
    order_by_value = torch.sort(x, stable=True).indices
    keys = ordered_bits(x)
    assert torch.all(keys[order_by_value][1:] >= keys[order_by_value][:-1])


def test_selection_ignores_neighbours_and_packing():
    gen = torch.Generator().manual_seed(0)
    logits = torch.randint(-4, 4, (6, 300), generator=gen).float()
    starts, ends = _rows((0, 300), (10, 200), (50, 51), (0, 0), (299, 300), (7, 287))
    full = stable_topk_per_row(logits, starts, ends, 33)
    for r in range(6):
        s, e = int(starts[r]), int(ends[r])
        alone = stable_topk_per_row(logits[r : r + 1, s:e].clone(), *_rows((0, max(e - s, 0))), 33)
        assert torch.equal(alone[0], full[r]), r


def test_matches_a_brute_force_sort():
    gen = torch.Generator().manual_seed(1)
    logits = torch.randint(-3, 3, (4, 50), generator=gen).float()
    logits[1, ::3] = -0.0
    starts, ends = _rows((0, 50), (5, 45), (0, 7), (20, 50))
    out = stable_topk_per_row(logits, starts, ends, 16)
    for r in range(4):
        s, e = int(starts[r]), int(ends[r])
        keys = ordered_bits(logits[r, s:e]).tolist()
        ranked = sorted(range(e - s), key=lambda i: (keys[i], i), reverse=True)[:16]
        assert out[r].tolist() == ranked + [-1] * (16 - len(ranked)), r


def test_strided_logits_are_read_by_value():
    gen = torch.Generator().manual_seed(2)
    base = torch.randn(64, 8, generator=gen)
    starts, ends = _rows(*[(0, 64)] * 8)
    assert torch.equal(
        stable_topk_per_row(base.t(), starts, ends, 10),
        stable_topk_per_row(base.t().contiguous(), starts, ends, 10),
    )


def test_input_validation():
    logits = torch.zeros(2, 8)
    starts, ends = _rows((0, 8), (0, 8))
    with pytest.raises(TypeError):
        stable_topk_per_row(logits.double(), starts, ends, 4)
    with pytest.raises(TypeError):
        stable_topk_per_row(logits, starts.long(), ends, 4)
    with pytest.raises(ValueError):
        stable_topk_per_row(logits, starts, ends, 0)
    with pytest.raises(ValueError):
        stable_topk_per_row(logits, starts, ends, 2049)
    with pytest.raises(ValueError):
        stable_topk_per_row(logits, *_rows((0, 9), (0, 8)), 4)
    with pytest.raises(ValueError):
        stable_topk_per_row(logits, *_rows((-1, 8), (0, 8)), 4)
