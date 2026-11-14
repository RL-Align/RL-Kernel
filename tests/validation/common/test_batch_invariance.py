# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The shared batch-invariance checks, on CPU, with negative controls.

A row-wise op passes every check. An op whose row depends on the batch size,
on the row's position, on the other rows' values or on the input strides, or
whose result changes between runs, must be reported by the check that targets
that failure.
"""

import pytest
import torch

from rl_engine.validation.common.batch_invariance import (
    DISTINCT_ROWS,
    RowwiseCase,
    bits_equal,
    differing_after_chunk_reorder_pad,
    differing_after_restride,
    differing_between,
    differing_between_runs,
    differing_in_batch,
    differing_with_special_values,
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


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16], ids=str
)
def test_bits_equal_compares_signed_zero_and_nan_payload_by_bits(dtype):
    zero = torch.zeros(3, dtype=dtype)
    assert bits_equal(zero, zero.clone())
    assert not bits_equal(zero, -zero)
    nan = torch.full((3,), float("nan"), dtype=dtype)
    assert bits_equal(nan, nan.clone())  # equal bits, although NaN != NaN
    bits = {torch.float64: torch.int64, torch.float32: torch.int32}.get(dtype, torch.int16)
    other_payload = nan.view(bits).clone()
    other_payload[1] += 1  # another quiet NaN
    assert torch.isnan(other_payload.view(dtype)).all()
    assert not bits_equal(nan, other_payload.view(dtype))


@pytest.mark.parametrize(
    "dtype",
    [torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64],
    ids=str,
)
def test_bits_equal_supports_integer_and_bool_outputs(dtype):
    a = torch.tensor([0, 1, 1, 0]).to(dtype)
    assert bits_equal(a, a.clone())
    assert not bits_equal(a, torch.tensor([0, 1, 0, 0]).to(dtype))


def test_bits_equal_rejects_a_dtype_without_bit_view():
    a = torch.zeros(2, dtype=torch.complex64)
    with pytest.raises(TypeError, match="no bit comparison"):
        bits_equal(a, a)


def test_rowwise_op_passes_every_check():
    case = RowwiseCase("rowwise", _NAMES, _rowwise, _make_rows)
    for m in (1, 2, 17, 64):
        assert differing_in_batch(case, m) == []
    assert differing_after_chunk_reorder_pad(case, 30) == []
    assert differing_after_restride(case, 30) == []
    assert differing_between_runs(case, 30) == []
    assert differing_between(case, case, 30) == []
    for m in (4, 5, 17):
        assert differing_with_special_values(case, m) == []


def test_op_depending_on_batch_size_is_reported_in_output_order():
    case = RowwiseCase("batch_size", _NAMES, _depends_on_batch_size, _make_rows)
    assert differing_in_batch(case, 17) == ["sum_of_squares"]
    assert differing_after_chunk_reorder_pad(case, 30) == [
        "chunked sum_of_squares",
        "zero-padded sum_of_squares",
        "random-padded sum_of_squares",
    ]


def test_op_depending_on_row_position_is_reported():
    case = RowwiseCase("position", ("y",), _depends_on_position, _make_rows)
    assert differing_in_batch(case, 17) == ["y"]
    assert differing_after_chunk_reorder_pad(case, 30) == ["chunked y", "reordered y"]


# Every row of the batch is compared, not only the first and the last: a defect at
# one middle row, or only past a tile edge, is found.
@pytest.mark.parametrize("bad_row", [lambda m: m // 2, lambda m: 16, lambda m: m - 1])
@pytest.mark.parametrize("m", [17, 64, 1000])
def test_a_single_wrong_row_anywhere_in_the_batch_is_reported(bad_row, m):
    def run(rows):
        x = rows[0]
        y = x * 2.0
        if x.shape[0] == m:
            y[bad_row(m)] += 1.0
        return [y]

    assert differing_in_batch(RowwiseCase("one_row", ("y",), run, _make_rows), m) == ["y"]


def test_reading_a_neighbouring_row_is_reported():
    def run(rows):
        x = rows[0]
        return [x + 1e-3 * torch.roll(x, 1, dims=0)]

    case = RowwiseCase("neighbour", ("y",), run, _make_rows)
    for m in (2, 17, 1000):
        assert differing_in_batch(case, m) == ["y"]


def test_in_batch_check_costs_distinct_rows_plus_one_runs():
    sizes = []

    def run(rows):
        sizes.append(rows[0].shape[0])
        return [rows[0] * 2.0]

    case = RowwiseCase("count", ("y",), run, _make_rows)
    assert differing_in_batch(case, 16384) == []
    assert sizes == [1] * DISTINCT_ROWS + [16384]
    sizes.clear()
    assert differing_in_batch(case, 3) == []
    assert sizes == [1, 1, 1, 3]


# Zero padding rows cannot find an op that depends on the padding rows' values.
def test_op_depending_on_other_rows_values_is_reported_by_random_padding():
    def run(rows):
        x = rows[0]
        return [x + (x != 0).all(dim=1).sum()]  # adds the count of rows without a zero

    case = RowwiseCase("nonzero_rows", ("y",), run, _make_rows)
    assert differing_after_chunk_reorder_pad(case, 30) == ["chunked y", "random-padded y"]


def test_special_values_leaking_into_other_rows_are_reported():
    def run(rows):
        x = rows[0]
        return [torch.nan_to_num(x) * 2.0 + torch.isnan(x).any()]  # +1 if any NaN in the batch

    case = RowwiseCase("nan_leak", ("y",), run, _make_rows)
    assert differing_with_special_values(case, 17) == [
        "other rows y",
        "+Inf row y",
        "-Inf row y",
        "-0.0 row y",
    ]


def test_signed_zero_lost_in_the_batch_only_is_reported():
    def run(rows):
        x = rows[0]
        return [x + 0.0 if x.shape[0] > 1 else x]  # -0.0 + 0.0 is +0.0

    case = RowwiseCase("signed_zero", ("y",), run, _make_rows)
    assert differing_with_special_values(case, 17) == ["-0.0 row y"]


def test_special_values_go_into_the_chosen_inputs():
    def make_rows(m, seed):
        return [torch.randn(m, 3, generator=_gen(seed + i)) for i in range(2)]

    seen = []

    def run(rows):
        seen.append([torch.isnan(t).any().item() for t in rows])
        return [rows[0] + rows[1]]

    case = RowwiseCase("two_inputs", ("y",), run, make_rows)
    assert differing_with_special_values(case, 8, inputs=(1,)) == []
    assert seen[1] == [False, True]  # the batch with the special rows

    ids = RowwiseCase("ids", ("y",), run, lambda m, seed: [torch.arange(m), torch.ones(m)])
    with pytest.raises(TypeError, match="not floating point"):
        differing_with_special_values(ids, 8)
    with pytest.raises(ValueError, match="at least 4"):
        differing_with_special_values(case, 3)


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
