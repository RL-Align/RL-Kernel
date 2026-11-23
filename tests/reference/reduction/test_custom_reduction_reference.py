# SPDX-License-Identifier: Apache-2.0
"""Public callback reductions checked against independent scalar arithmetic."""

from functools import reduce as scalar_reduce
from operator import and_, xor

import pytest
import torch

from rl_engine.reference import reduction as ordered
from rl_engine.reference.reduction.ordered_reduce import (
    ordered_reduce_merge,
    ordered_reduce_serial,
    reduce_add,
)
from tests.helpers.tensor_checks import assert_bitwise


def _modular_add(left, right):
    """A caller-owned operation with a neutral zero on residues modulo 257."""
    return (left + right) % 257


_OPERATIONS = (
    pytest.param(torch.bitwise_xor, xor, 0, torch.int64, id="integer-xor"),
    pytest.param(torch.bitwise_and, and_, -1, torch.int64, id="integer-and"),
    pytest.param(_modular_add, _modular_add, 0, torch.int64, id="modular-add"),
    pytest.param(torch.logical_xor, xor, False, torch.bool, id="boolean-xor"),
)


@pytest.mark.parametrize("combine,scalar,identity,dtype", _OPERATIONS)
@pytest.mark.parametrize("width,length", ((16, 0), (16, 11), (64, 33), (1024, 513)))
def test_custom_callback_covers_loaded_memory_and_partial_paths(
    combine, scalar, identity, dtype, width, length
):
    """The callback controls all combines; no sum/product dispatch can pass this matrix."""
    storage = ((torch.arange(4 * width) * 13 + 5) % 257).reshape(2, 2 * width).to(dtype)
    values = storage[:, ::2]
    before = storage.clone()
    expected = torch.tensor(
        [scalar_reduce(scalar, row[:length], identity) for row in values.tolist()], dtype=dtype
    )
    options = dict(combine_fn=combine, identity=identity)
    assert_bitwise(ordered.reduce(values, length=length, **options), expected)
    assert_bitwise(
        ordered.reduce(values, length=length, from_memory=True, block_size=32, **options), expected
    )

    # Producers use the same public helper on loaded tiles. Merge reads fixed
    # slots regardless of the order in which a producer might finish on a GPU.
    n_tiles = (length + 31) // 32
    partials = torch.empty((2, n_tiles), dtype=dtype)
    for tile_index in range(n_tiles - 1, -1, -1):
        start = tile_index * 32
        stop = min(start + 32, length)
        tile = torch.full((2, 32), identity, dtype=dtype)
        tile[:, : stop - start] = values[:, start:stop]
        partials[:, tile_index] = ordered.reduce(tile, **options)
    for unroll in (1, 3):
        assert_bitwise(
            ordered.reduce_partials(
                partials, combine_fn=combine, initial=identity, merge_unroll=unroll
            ),
            expected,
        )
    assert_bitwise(storage, before)


def test_custom_merge_applies_nonidentity_seed_only_once():
    """A nonzero seed distinguishes merging from injecting a value into every tile."""
    partials = torch.tensor([[250, 12, 255, 3], [200, 250, 9, 15]], dtype=torch.int64)
    seeds = torch.tensor([17, 29], dtype=torch.int64)
    before = partials.clone()
    for count in (0, 1, 4):
        expected = torch.tensor(
            [
                scalar_reduce(_modular_add, row[:count], seed)
                for row, seed in zip(partials.tolist(), seeds.tolist(), strict=True)
            ]
        )
        for unroll in (1, 3, 8):
            assert_bitwise(
                ordered.reduce_partials(
                    partials[:, :count], combine_fn=_modular_add, initial=seeds, merge_unroll=unroll
                ),
                expected,
            )
    assert_bitwise(partials, before)


def test_generic_callback_retains_the_declared_loaded_tree_and_memory_seed():
    """Use a caller function plus frozen float results; names must not select arithmetic."""

    def caller_add(left, right):
        return left + right

    values = torch.zeros(64)
    values[[0, 8, 16, 24]] = torch.tensor([1e20, 1, -1e20, 1])
    values[32:] = float("nan")
    options = dict(combine_fn=caller_add, identity=0)
    # The loaded tree cancels large terms before adding ones. Two memory
    # tiles of 16 values each round away their one before their totals merge.
    assert_bitwise(ordered.reduce(values, length=32, **options), torch.tensor(2.0))
    assert_bitwise(
        ordered.reduce(values, length=32, from_memory=True, block_size=16, **options),
        torch.tensor(0.0),
    )
    values = torch.tensor([-0.0])
    assert_bitwise(ordered.reduce(values, **options), values[0])
    assert_bitwise(ordered.reduce(values, from_memory=True, **options), torch.tensor(0.0))


def test_generic_merge_retains_left_association_and_storage_precision():
    """A new tree or implicit dtype conversion would change this frozen result."""

    def caller_add(left, right):
        return left + right

    partials = torch.tensor([[1e20, 1, -1e20, 1]], dtype=torch.float32)
    assert_bitwise(
        ordered.reduce_partials(partials, combine_fn=caller_add, initial=0), torch.tensor([1.0])
    )
    with pytest.raises(ValueError, match="partial storage must match compute dtype"):
        ordered.reduce_partials(
            partials, combine_fn=caller_add, initial=0, compute_dtype=torch.float64
        )
    for unroll in (0, -1):
        with pytest.raises(ValueError, match="merge unroll must be a positive integer"):
            ordered.reduce_partials(partials, combine_fn=caller_add, initial=0, merge_unroll=unroll)


def test_custom_callback_uses_explicit_compute_dtype_without_touching_input():
    values = torch.tensor([2**53 + 1, 4, 8, 0], dtype=torch.int64)
    expected = torch.tensor((2**53 + 1) ^ 4 ^ 8, dtype=torch.int64)
    assert_bitwise(ordered.reduce(values, combine_fn=torch.bitwise_xor, identity=0), expected)
    small = torch.tensor([255, 128, 1, 0], dtype=torch.uint8)
    actual = ordered.reduce(
        small, combine_fn=torch.bitwise_xor, identity=0, compute_dtype=torch.int64
    )
    assert_bitwise(actual, torch.tensor(255 ^ 128 ^ 1, dtype=torch.int64))


@pytest.mark.parametrize("count", (0, 3))
def test_custom_merge_propagates_initial_tensor_gradients(count):
    """Preserve the seed's graph through a cast, broadcasting and an empty fold."""
    partials = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)[:, :count]
    partials.requires_grad_()
    initial = torch.tensor(5.0, requires_grad=True)
    actual = ordered.reduce_partials(partials, combine_fn=torch.add, initial=initial)
    upstream = torch.tensor([2.0, -3.0], dtype=torch.float64)
    seed_gradient = torch.autograd.grad(actual, initial, upstream, retain_graph=True)[0]
    torch.testing.assert_close(seed_gradient, upstream.sum().float(), rtol=0, atol=0)
    if count:
        gradient = torch.autograd.grad(actual, partials, upstream)[0]
        torch.testing.assert_close(gradient, upstream[:, None].expand_as(partials), rtol=0, atol=0)
    with torch.no_grad():
        no_grad = ordered.reduce_partials(partials, combine_fn=torch.add, initial=initial)
    assert_bitwise(actual.detach(), no_grad)
    assert_bitwise(initial.detach(), torch.tensor(5.0))
    assert initial.grad is None


@pytest.mark.parametrize("from_memory", (False, True))
def test_custom_differentiable_callback_matches_independent_product_formula(from_memory):
    """(1+a)(1+b)-1 is associative with identity zero; its derivative is independent."""

    def caller_combine(left, right):
        return left + right + left * right

    values = torch.linspace(-0.1, 0.1, 32, dtype=torch.float64).reshape(2, 16).requires_grad_()
    actual = ordered.reduce(
        values,
        length=11,
        from_memory=from_memory,
        block_size=4,
        combine_fn=caller_combine,
        identity=0,
    )
    upstream = torch.tensor([0.5, -2.0], dtype=torch.float64)
    gradient = torch.autograd.grad(actual, values, upstream)[0]
    expected = torch.zeros_like(values)
    with torch.no_grad():
        factors = 1 + values[:, :11]
        for column in range(11):
            others = torch.cat((factors[:, :column], factors[:, column + 1 :]), dim=-1)
            expected[:, column] = upstream * others.prod(dim=-1)
        gold = factors.prod(dim=-1) - 1
    torch.testing.assert_close(actual.detach(), gold, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "dtype,seed", ((torch.float64, -0.0), (torch.float64, 1 + 2**-35), (torch.int64, 2**53 + 1))
)
def test_empty_stage_inputs_preserve_typed_seed_bits(dtype, seed):
    """No row/tile combines occur on empty inputs; scalar seeds avoid FP32 conversion."""
    empty = torch.empty((2, 0), dtype=dtype)
    expected = torch.full((2,), seed, dtype=dtype)
    serial = ordered_reduce_serial(
        empty,
        seed,
        block_size=16,
        max_group_size=8,
        combine_fn=reduce_add,
        identity=0,
        compute_dtype=dtype,
    )
    merged = ordered_reduce_merge(
        empty, seed, combine_fn=reduce_add, merge_unroll=3, compute_dtype=dtype
    )
    assert_bitwise(serial, expected)
    assert_bitwise(merged, expected)


@pytest.mark.parametrize("fold", ("serial", "partials"))
@pytest.mark.parametrize("seed_shape", ((), (3,), (2, 3)))
@pytest.mark.parametrize("requires_grad", (False, True))
def test_empty_folds_return_independent_seed_values(fold, seed_shape, requires_grad):
    """Empty results are writable without changing the seed or detaching its graph.

    Cover scalar, broadcast and shape-matched seeds in their compute dtype, so
    dtype conversion cannot hide the storage alias. Empty inputs contribute no
    autograd dependency even when they require gradients.
    """
    seed_value = 1 + 2**-35
    initial = torch.full(seed_shape, seed_value, dtype=torch.float64, requires_grad=requires_grad)
    initial_before = initial.detach().clone()
    values = torch.empty((2, 3, 0), dtype=torch.float64, requires_grad=True)
    if fold == "serial":
        actual = ordered_reduce_serial(
            values,
            initial,
            block_size=16,
            max_group_size=8,
            combine_fn=reduce_add,
            identity=0,
            compute_dtype=torch.float64,
        )
    else:
        actual = ordered.reduce_partials(values, combine_fn=reduce_add, initial=initial)

    expected = torch.full((2, 3), seed_value, dtype=torch.float64)
    assert_bitwise(actual, expected)
    assert actual.requires_grad == requires_grad
    actual[0, 0].add_(2)
    expected[0, 0] += 2
    assert_bitwise(actual, expected)
    assert_bitwise(initial, initial_before)

    if requires_grad:
        seed_grad, input_grad = torch.autograd.grad(
            actual, (initial, values), torch.ones_like(actual), allow_unused=True
        )
        # Each seed element is reused once, twice, or six times by broadcasting.
        expected_grad = torch.full_like(initial, actual.numel() // initial.numel())
        assert_bitwise(seed_grad, expected_grad)
        assert input_grad is None
