# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The shared batch-invariance checks, on CPU, with negative controls.

A row-wise op passes every check. An op whose row depends on the batch size,
on the row's position or on the input strides, or whose result changes between
runs, must be reported by the check that targets that failure.
"""

import torch

from rl_engine.validation.common.batch_invariance import (
    RowwiseCase,
    bits_equal,
    differing_after_chunk_reorder_pad,
    differing_after_restride,
    differing_between,
    differing_between_runs,
    differing_in_batch,
)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _make_rows(m, seed):
    return [torch.randn(m, 8, generator=_gen(seed))]


def _rowwise(rows):
    x = rows[0]
    return [x * 2.0, (x * x).sum(-1)]


def _depends_on_batch_size(rows):
    x = rows[0]
    return [x * 2.0, (x * x).sum(-1) + x.shape[0]]


def _depends_on_position(rows):
    x = rows[0]
    return [x * 2.0 + torch.arange(x.shape[0])[:, None]]


_NAMES = ("scaled", "sum_of_squares")


def test_bits_equal_distinguishes_signed_zero_and_dtype():
    assert bits_equal(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]))
    assert not bits_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not bits_equal(torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64))
    assert not bits_equal(torch.tensor([1.0]), torch.tensor([1.0, 1.0]))


def test_rowwise_op_passes_every_check():
    case = RowwiseCase("rowwise", _NAMES, _rowwise, _make_rows)
    for m in (1, 2, 17, 64):
        assert differing_in_batch(case, m) == []
    assert differing_after_chunk_reorder_pad(case, 30) == []
    assert differing_after_restride(case, 30) == []
    assert differing_between_runs(case, 30) == []
    assert differing_between(case, case, 30) == []


def test_op_depending_on_batch_size_is_reported_in_output_order():
    case = RowwiseCase("batch_size", _NAMES, _depends_on_batch_size, _make_rows)
    assert differing_in_batch(case, 17) == ["sum_of_squares"]
    assert differing_after_chunk_reorder_pad(case, 30) == [
        "chunked sum_of_squares",
        "padded sum_of_squares",
    ]


def test_op_depending_on_row_position_is_reported():
    case = RowwiseCase("position", ("y",), _depends_on_position, _make_rows)
    assert differing_in_batch(case, 17) == ["y"]
    assert differing_after_chunk_reorder_pad(case, 30) == ["chunked y", "reordered y"]


def test_op_depending_on_strides_is_reported():
    def run(rows):
        x = rows[0]
        return [x * 2.0, x + x.stride(0)]

    case = RowwiseCase("strides", ("scaled", "shifted"), run, _make_rows)
    assert differing_after_restride(case, 30) == ["restrided shifted"]


def test_restride_handles_1d_inputs():
    def make_rows(m, seed):
        return [torch.arange(m, dtype=torch.int64), torch.randn(m, 3, generator=_gen(seed))]

    case = RowwiseCase("ids", ("y",), lambda rows: [rows[1] * rows[0][:, None]], make_rows)
    assert differing_after_restride(case, 9) == []


def test_run_to_run_and_paired_differences_are_reported():
    calls = []

    def drifting(rows):
        calls.append(None)
        return [rows[0] + len(calls)]

    case = RowwiseCase("drifting", ("y",), drifting, _make_rows)
    assert differing_between_runs(case, 4) == ["y"]

    shifted = RowwiseCase("shifted", ("y",), lambda rows: [rows[0] + 1.0], _make_rows)
    exact = RowwiseCase("exact", ("y", "extra"), lambda rows: [rows[0], rows[0]], _make_rows)
    assert differing_between(exact, shifted, 4) == ["y"]
