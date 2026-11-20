# SPDX-License-Identifier: Apache-2.0
"""Fixed-order reduction conformance, independent of model/operator tests."""

import pytest
import torch

from rl_engine.reference.reduction.ordered_reduce import (
    ordered_reduce_merge,
    ordered_reduce_partial,
    ordered_reduce_serial,
    ordered_tile_reduce,
    reduce_add,
    reduce_multiply,
)
from tests.helpers.tensor_checks import assert_bitwise

_TILE_WIDTHS = tuple(1 << exponent for exponent in range(14))
_TREE_CONFIG = {"max_group_size": 8}
"""Eight-value sequential groups, then one tree over all group totals."""


@pytest.mark.parametrize("max_group_size", (0, -1, 3))
def test_rejects_invalid_group_size(max_group_size):
    with pytest.raises(ValueError, match="maximum group size"):
        ordered_tile_reduce(torch.ones(16), max_group_size=max_group_size, combine_fn=torch.add)


@pytest.mark.parametrize("max_group_size", (1, 8))
@pytest.mark.parametrize(
    "dtype,precision_bits",
    ((torch.float16, 11), (torch.bfloat16, 8), (torch.float32, 24), (torch.float64, 53)),
)
def test_group_totals_follow_one_halving_tree(max_group_size, dtype, precision_bits):
    """Freeze pairs independently of the reference implementation.

    Eight group totals first pair 0+4, 1+5, 2+6, 3+7. The first three rows
    cancel large values before adding the ones; the last row rounds both
    ones away in that first step. Adjacent-first and left-fold trees differ.
    """
    large = 2 ** (precision_bits + 2)
    values = torch.zeros((4, 8 * max_group_size), dtype=dtype)
    for row, positions in enumerate(((0, 1, 4, 5), (0, 1, 2, 3), (0, 2, 4, 6), (0, 4, 1, 5))):
        columns = [group * max_group_size for group in positions]
        values[row, columns] = torch.tensor([large, 1.0, -large, 1.0], dtype=dtype)
    actual = ordered_tile_reduce(values, max_group_size=max_group_size, combine_fn=torch.add)
    assert_bitwise(actual, torch.tensor([2, 2, 2, 0], dtype=dtype), "halving tree")


@pytest.mark.parametrize("block_size", _TILE_WIDTHS)
@pytest.mark.parametrize("leading_shape", ((), (2, 3), (0,)))
def test_fixed_tile_tree_accepts_power_of_two_widths(block_size, leading_shape):
    """Include strided vectors, arbitrary leading dimensions and empty batches."""
    storage = torch.ones((*leading_shape, block_size * 2), dtype=torch.float32)
    values = storage[..., ::2]
    actual = ordered_tile_reduce(values, **_TREE_CONFIG, combine_fn=torch.add)
    expected = torch.full(leading_shape, float(block_size), dtype=torch.float32)
    assert_bitwise(actual, expected, "power-of-two shape and sum")
    assert_bitwise(
        ordered_tile_reduce(torch.full_like(values, -0.0), **_TREE_CONFIG, combine_fn=torch.add),
        torch.full(leading_shape, -0.0, dtype=torch.float32),
        "no extra positive-zero operand inside a tile",
    )


@pytest.mark.parametrize("block_size", (4, 8, 16))
def test_small_tile_tree_preserves_sequential_groups(block_size):
    """A whole-tile reduction must not silently replace the eight-value folds."""
    values = torch.zeros(block_size, dtype=torch.float32)
    values[:4] = torch.tensor([1e20, 1.0, -1e20, 1.0])
    assert_bitwise(
        ordered_tile_reduce(values, **_TREE_CONFIG, combine_fn=torch.add),
        torch.tensor(1.0),
        "small sequential group",
    )


@pytest.mark.parametrize("block_size", _TILE_WIDTHS)
@pytest.mark.parametrize("max_group_size", (1, 8, 64))
def test_tile_tree_matches_explicit_pairs(block_size, max_group_size):
    """Cross-check a list of explicit operand pairs across widths, groups and batches."""
    generator = torch.Generator().manual_seed(939)
    storage = torch.randn((2, 3, 2 * block_size), generator=generator)
    storage *= torch.pow(2.0, torch.randint(-30, 30, storage.shape, generator=generator))
    values = storage[..., ::2]
    group_size = min(block_size, max_group_size)
    n_groups = block_size // group_size
    totals = []
    for group in range(n_groups):
        start = group * group_size
        total = values[..., start]
        for offset in range(1, group_size):
            total = total + values[..., start + offset]
        totals.append(total)
    while len(totals) > 1:
        halfway = len(totals) // 2
        totals = [
            left + right for left, right in zip(totals[:halfway], totals[halfway:], strict=True)
        ]
    expected = totals[0]
    assert_bitwise(
        ordered_tile_reduce(values, max_group_size=max_group_size, combine_fn=torch.add),
        expected,
        "explicit pair sequence",
    )


@pytest.mark.parametrize("block_size", (0, 3, 24, 1023))
def test_reference_rejects_invalid_tile_widths(block_size):
    with pytest.raises(ValueError, match="positive power of two"):
        ordered_tile_reduce(torch.ones((1, block_size)), **_TREE_CONFIG, combine_fn=torch.add)


def test_fixed_tile_reference_requires_array():
    with pytest.raises(ValueError, match="values with at least one dimension"):
        ordered_tile_reduce(torch.tensor(1.0), **_TREE_CONFIG, combine_fn=torch.add)


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
