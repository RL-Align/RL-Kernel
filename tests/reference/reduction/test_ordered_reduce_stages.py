# SPDX-License-Identifier: Apache-2.0
"""Eager stage interfaces preserve the declared arithmetic and independent expectations."""

import pytest
import torch

from rl_engine.reference.reduction import product_tile, sum_tile
from rl_engine.reference.reduction.ordered_reduce import (
    ordered_reduce_merge,
    ordered_reduce_partial,
    ordered_reduce_serial,
    reduce_add,
    reduce_multiply,
)
from tests.helpers.tensor_checks import assert_bitwise


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("operation", ("sum", "product"))
def test_partial_serial_and_merge_preserve_seed_padding_and_rounding(dtype, operation):
    """Frozen partials expose reassociation and accidentally seeded/padded tiles."""
    if operation == "sum":
        large = 2 / torch.finfo(dtype).eps
        data = [large, 1, -large, 1, 2, 3, 4, 5, 6]
        combine, identity, initial = reduce_add, 0, 7
        expected_partials = [1, 14, 6]
        expected_total = 28
    else:
        data = [2, 0.5, 4, 0.25, 2, 2, 0.5, 0.5, 2]
        combine, identity, initial = reduce_multiply, 1, 3
        expected_partials = [1, 1, 2]
        expected_total = 6
    # Preserve noncontiguous reads and leading batch dimensions in every stage.
    storage = torch.zeros((2, 3, 18), dtype=torch.float64)
    storage[..., ::2] = torch.tensor(data, dtype=torch.float64)
    values = storage[..., ::2]
    before = storage.clone()
    config = dict(
        block_size=4,
        max_group_size=8,
        section_count=4,
        combine_fn=combine,
        identity=identity,
        compute_dtype=dtype,
    )
    partials = torch.stack(
        [ordered_reduce_partial(values, tile, **config) for tile in range(3)], dim=-1
    )
    assert_bitwise(
        partials,
        torch.tensor(expected_partials, dtype=dtype).expand(2, 3, 3),
        "unseeded tiles, including the padded final tile",
    )
    initial = torch.tensor([initial, initial, initial], dtype=dtype)
    initial_before = initial.clone()
    expected = torch.full((2, 3), expected_total, dtype=dtype)
    assert_bitwise(ordered_reduce_serial(values, initial, **config), expected)
    for unroll in (1, 2, 3, 4, 7):
        assert_bitwise(
            ordered_reduce_merge(
                partials,
                initial,
                combine_fn=combine,
                merge_unroll=unroll,
                compute_dtype=dtype,
            ),
            expected,
        )
    assert_bitwise(storage, before)
    assert_bitwise(initial, initial_before)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.int64))
@pytest.mark.parametrize("unroll", (1, 2, 3, 4, 7))
@pytest.mark.parametrize("n_tiles", (0, 1, 3, 4, 9))
def test_partial_sum_uses_left_fold_for_every_unroll_and_tail(dtype, unroll, n_tiles):
    """Known results distinguish a partial fold from applying a new tile tree."""
    large = 2**54 if dtype != torch.float32 else 2**25
    source = torch.tensor([large, 1, -large, 1, 2, -2, 4, -4, 5], dtype=dtype)
    source = source[:n_tiles]
    frozen = {0: 0, 1: large, 3: 0, 4: 1, 9: 6}
    if dtype == torch.int64:
        frozen.update({3: 1, 4: 2, 9: 7})
    before = source.clone()
    actual = ordered_reduce_merge(
        source, initial=0, combine_fn=reduce_add, compute_dtype=dtype, merge_unroll=unroll
    )
    assert_bitwise(actual, torch.tensor(frozen[n_tiles], dtype=dtype))
    assert_bitwise(source, before)


@pytest.mark.parametrize("helper", (sum_tile, product_tile))
def test_tile_helpers_cast_before_arithmetic_and_do_not_insert_a_seed(helper):
    """A one-element tile retains negative zero and low FP64 bits."""
    values = torch.tensor([[-0.0], [1 + 2**-35]], dtype=torch.float64)
    assert_bitwise(helper(values), values[:, 0])
    assert_bitwise(helper(values, compute_dtype=None), values[:, 0])
    assert_bitwise(helper(values, compute_dtype=torch.float64), values[:, 0])
    assert_bitwise(helper(values, compute_dtype=torch.float32), values[:, 0].float())


@pytest.mark.parametrize("helper", (sum_tile, product_tile))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_tile_helpers_apply_requested_group_and_section_tree(helper, dtype):
    if helper is sum_tile:
        large = 2 / torch.finfo(dtype).eps
        values = torch.tensor([large, 1, -large, 1], dtype=dtype)
        # One sequential group returns 1; four one-element groups return 2.
        expected = (1, 2)
    else:
        exponent = {torch.float16: 10, torch.bfloat16: 80, torch.float32: 80, torch.float64: 600}[
            dtype
        ]
        values = torch.tensor([2.0**exponent] * 2 + [2.0**-exponent] * 2, dtype=dtype)
        # Sequential products overflow; half-pair products cancel the exponents.
        expected = (float("inf"), 1)
    for group_size, result in zip((4, 1), expected, strict=True):
        for dtype_kwargs in ({}, {"compute_dtype": None}, {"compute_dtype": dtype}):
            assert_bitwise(
                helper(values, max_group_size=group_size, section_count=4, **dtype_kwargs),
                torch.tensor(result, dtype=dtype),
            )


@pytest.mark.parametrize("helper", (sum_tile, product_tile))
def test_tile_helpers_infer_integer_precision_without_float_conversion(helper):
    values = torch.tensor([2**53 + 1, 1, 2, 4], dtype=torch.int64)
    expected = 2**53 + 8 if helper is sum_tile else (2**53 + 1) * 8
    for dtype_kwargs in ({}, {"compute_dtype": None}, {"compute_dtype": torch.int64}):
        assert_bitwise(helper(values, **dtype_kwargs), torch.tensor(expected, dtype=torch.int64))


def test_empty_stage_inputs_preserve_typed_seed_bits():
    """No row/tile combines occur on empty inputs; scalar seeds avoid FP32 conversion."""
    empty = torch.empty((2, 0), dtype=torch.float64)
    for seed in (-0.0, 1 + 2**-35):
        expected = torch.full((2,), seed, dtype=torch.float64)
        serial = ordered_reduce_serial(
            empty,
            seed,
            block_size=16,
            max_group_size=8,
            section_count=4,
            combine_fn=reduce_add,
            identity=0,
            compute_dtype=torch.float64,
        )
        merged = ordered_reduce_merge(
            empty, seed, combine_fn=reduce_add, merge_unroll=3, compute_dtype=torch.float64
        )
        assert_bitwise(serial, expected)
        assert_bitwise(merged, expected)


def test_partial_merge_rejects_storage_casts_and_invalid_unroll():
    with pytest.raises(ValueError, match="partial storage must match"):
        ordered_reduce_merge(
            torch.ones(3, dtype=torch.float64),
            initial=0,
            combine_fn=reduce_add,
            compute_dtype=torch.float32,
            merge_unroll=1,
        )
    for unroll in (0, -1):
        with pytest.raises(ValueError, match="merge unroll must be a positive integer"):
            ordered_reduce_merge(
                torch.empty(0),
                initial=0,
                combine_fn=reduce_add,
                compute_dtype=torch.float32,
                merge_unroll=unroll,
            )
