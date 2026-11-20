# SPDX-License-Identifier: Apache-2.0
"""Public eager references for configured embedded Triton reductions."""

import pytest
import torch

from rl_engine.reference.reduction import product as ordered_product
from rl_engine.reference.reduction import product_partials
from rl_engine.reference.reduction import sum as ordered_sum
from rl_engine.reference.reduction import sum_partials
from tests.helpers.tensor_checks import assert_bitwise

_OPERATIONS = (
    pytest.param(ordered_sum, torch.sum, 0.0, id="sum"),
    pytest.param(ordered_product, torch.prod, 1.0, id="product"),
)
_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


@pytest.mark.parametrize(
    "reduce,combine,identity",
    ((sum_partials, torch.add, 0), (product_partials, torch.mul, 1)),
)
@pytest.mark.parametrize("dtype", (*_DTYPES, torch.int64))
def test_public_partial_defaults_preserve_order_dtype_and_input(reduce, combine, identity, dtype):
    """One-argument calls retain the left fold, low-precision rounding and wide bits."""
    if reduce is sum_partials:
        magnitude = 2**54 if dtype == torch.int64 else 2 / torch.finfo(dtype).eps
        data = [magnitude, 1, -magnitude, 1, 2, -2, 4, -4, 5]
        four_result = 2 if dtype == torch.int64 else 1
    else:
        # Left-fold underflow yields zero; a lower-half/upper-half tree yields one.
        exponent = {torch.float16: 13, torch.bfloat16: 70, torch.float32: 80, torch.float64: 600}
        small = 2.0 ** -exponent[dtype] if dtype != torch.int64 else 2
        inverse = 1 / small if dtype != torch.int64 else 3
        data = [small, small, inverse, inverse, 2, 1, 1, 1, 1]
        four_result = 36 if dtype == torch.int64 else 0
    storage = torch.full((2, 18), identity, dtype=dtype)
    storage[:, ::2] = torch.tensor(data, dtype=dtype)
    if dtype == torch.float64:
        storage[1, 0] = 1 + 2**-35
    elif dtype == torch.int64:
        storage[1, 0] = 2**53 + 1
    before = storage.clone()
    for n_tiles in (0, 1, 4, 9):
        partials = storage[:, : 2 * n_tiles : 2]
        expected = torch.full((2,), identity, dtype=dtype)
        for tile in range(n_tiles):
            expected = combine(expected, partials[:, tile])
        if n_tiles == 4:
            assert expected[0].item() == four_result
        assert_bitwise(reduce(partials), expected)
        assert_bitwise(reduce(partials, compute_dtype=None), expected)
        for unroll in (1, 3, 16):
            assert_bitwise(reduce(partials, compute_dtype=dtype, merge_unroll=unroll), expected)
    assert_bitwise(storage, before)


@pytest.mark.parametrize("reduce", (sum_partials, product_partials))
def test_public_partial_helpers_reject_hidden_storage_casts_and_invalid_unroll(reduce):
    with pytest.raises(ValueError, match="partial storage must match"):
        reduce(torch.ones(3, dtype=torch.float64), compute_dtype=torch.float32)
    for unroll in (0, -1):
        with pytest.raises(ValueError, match="merge unroll must be a positive integer"):
            reduce(torch.empty(0), merge_unroll=unroll)


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize(
    "input_dtype,compute_dtype",
    (
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float64),
        (torch.float64, torch.float16),
        (torch.float64, torch.bfloat16),
    ),
)
def test_public_reference_matches_independent_exact_math(
    reduce, native, identity, input_dtype, compute_dtype
):
    """Use representable results; the oracle does not copy the fixed tree."""
    values = torch.tensor([[2.0, 0.5, -2.0], [0.5, 2.0, 4.0]], dtype=input_dtype)
    # Keep the last dimension strided, including an uneven tile tail.
    storage = torch.zeros((2, 6), dtype=input_dtype)
    storage[:, ::2] = values
    before = storage.clone()
    actual = reduce(storage[:, ::2], compute_dtype=compute_dtype, from_memory=True)
    expected = native(values.double(), dim=-1).to(compute_dtype)
    assert actual.shape == (2,) and actual.dtype == compute_dtype
    assert actual.device == values.device
    assert_bitwise(actual, expected, "independent native FP64 oracle")
    assert_bitwise(storage, before, "input storage unchanged")


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize("compute_dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((0,), (3, 0), (0, 1025), (2, 0, 1025)))
def test_empty_vectors_and_batches_match_public_contract(
    reduce, native, identity, compute_dtype, shape
):
    values = torch.empty(shape)
    actual = reduce(values, compute_dtype=compute_dtype, from_memory=True)
    expected = torch.full(shape[:-1], identity, dtype=compute_dtype)
    assert_bitwise(actual, expected)


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize("dtype", (*_DTYPES, torch.int8, torch.int64))
@pytest.mark.parametrize("length", (0, 3, 1025))
def test_public_reference_infers_input_dtype_without_widening(
    reduce, native, identity, dtype, length
):
    """Omission and explicit None retain storage precision, including empty results."""
    values = torch.full((2, length), identity, dtype=dtype)
    if length:
        values[:, :3] = torch.tensor([2, 1, 4], dtype=dtype)
        if dtype == torch.float64:
            values[1, 0] += 2**-35
        elif dtype == torch.int64:
            values[1, 0] = 2**53 + 1
    # These values are exact under either reduction tree; retain the input dtype
    # explicitly because native integer reductions otherwise promote to int64.
    expected = native(values, dim=-1, dtype=dtype)
    assert_bitwise(reduce(values, from_memory=True), expected)
    assert_bitwise(reduce(values, compute_dtype=None, from_memory=True), expected)
    assert_bitwise(reduce(values, compute_dtype=dtype, from_memory=True), expected)


@pytest.mark.parametrize(
    "positions,expected",
    (
        ((0, 1, 2, 3), 1.0),
        ((0, 8, 512, 520), 2.0),
        ((0, 256, 512, 768), 2.0),
        ((0, 1024, 2048, 3072), 1.0),
    ),
)
def test_public_sum_keeps_declared_tile_tree_and_ascending_merge(positions, expected):
    """Freeze numerical behavior independently of shared constants and reference helpers."""
    values = torch.zeros(3073)
    values[list(positions)] = torch.tensor([1e20, 1, -1e20, 1])
    assert_bitwise(
        ordered_sum(values, compute_dtype=torch.float32, from_memory=True), torch.tensor(expected)
    )


@pytest.mark.parametrize(
    "block_size,max_group_size,positions,expected",
    (
        (16, 8, (0, 8, 16, 24), 0.0),
        (32, 8, (0, 8, 16, 24), 2.0),
        (32, 8, (0, 1, 2, 3), 1.0),
        (32, 1, (0, 1, 2, 3), 2.0),
    ),
)
def test_public_sum_configuration_controls_rounding(
    block_size, max_group_size, positions, expected
):
    """Tile width and group size change rounding; expected bits are derived independently."""
    values = torch.zeros(64)
    values[list(positions)] = torch.tensor([1e20, 1, -1e20, 1])
    actual = ordered_sum(
        values,
        compute_dtype=torch.float32,
        block_size=block_size,
        max_group_size=max_group_size,
        from_memory=True,
    )
    assert_bitwise(actual, torch.tensor(expected))


@pytest.mark.parametrize("max_group_size,expected", ((8, float("inf")), (1, 1.0)))
def test_public_product_configuration_controls_intermediate_overflow(max_group_size, expected):
    """Sequential large factors overflow; the alternate tree first pairs reciprocal factors."""
    values = torch.ones(32)
    values[:4] = torch.tensor([2.0**80, 2.0**80, 2.0**-80, 2.0**-80])
    actual = ordered_product(
        values,
        compute_dtype=torch.float32,
        block_size=32,
        max_group_size=max_group_size,
        from_memory=True,
    )
    assert_bitwise(actual, torch.tensor(expected))


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize(
    "block_size,max_group_size",
    ((1, 8), (16, 1), (16, 4), (1024, 8), (2048, 16)),
)
def test_public_configurations_cover_empty_strided_vectors_and_tails(
    reduce, native, identity, block_size, max_group_size
):
    """Keep an independent exact-math oracle for both public configured operations."""
    for length in sorted({0, block_size - 1, block_size, 2 * block_size + 3}):
        storage = torch.ones((2, 2 * length), dtype=torch.float64)
        values = storage[:, ::2]
        if length:
            values[:, 0] = 2.0
        if length > 1:
            values[:, -1] = -0.5
        before = storage.clone()
        actual = reduce(
            values,
            compute_dtype=torch.float32,
            block_size=block_size,
            max_group_size=max_group_size,
            from_memory=True,
        )
        assert_bitwise(actual, native(values, dim=-1).float())
        assert_bitwise(storage, before, "input storage unchanged")


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize("parameter", ("block_size", "max_group_size"))
@pytest.mark.parametrize("size", (-1, 0, 3))
@pytest.mark.parametrize("shape", ((2, 17), (2, 0)))
def test_public_configuration_validation_includes_empty_inputs(
    reduce, native, identity, parameter, size, shape
):
    with pytest.raises(ValueError, match="positive power of two"):
        reduce(
            torch.ones(shape), compute_dtype=torch.float32, **{parameter: size}, from_memory=True
        )


@pytest.mark.parametrize(
    "compute_dtype,bits",
    ((torch.float16, 11), (torch.bfloat16, 8), (torch.float32, 24), (torch.float64, 53)),
)
def test_public_sum_rounds_each_step_in_requested_dtype(compute_dtype, bits):
    values = torch.zeros((2, 2049), dtype=torch.float64)
    values[0, :3] = torch.tensor([2**bits, 1, 1], dtype=torch.float64)
    values[1, [0, 1024, 2048]] = values[0, :3]
    assert_bitwise(
        ordered_sum(values, compute_dtype=compute_dtype, from_memory=True),
        torch.full((2,), 2**bits, dtype=compute_dtype),
    )
    # Inferring the input dtype must not accumulate in a wider dtype and cast
    # only at the end: that would retain the two unit terms.
    stored = values.to(compute_dtype)
    expected = torch.full((2,), 2**bits, dtype=compute_dtype)
    assert_bitwise(ordered_sum(stored, from_memory=True), expected)
    assert_bitwise(ordered_sum(stored, compute_dtype=None, from_memory=True), expected)


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize(
    "values,error,message",
    (
        (torch.tensor(1.0), ValueError, "at least one dimension"),
        (torch.eye(3).to_sparse(), ValueError, "strided tensor"),
    ),
)
def test_public_reference_validates_inputs_independently(
    reduce, native, identity, values, error, message
):
    with pytest.raises(error, match=message):
        reduce(values, compute_dtype=torch.float32, from_memory=True)


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize("dtype", (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64))
@pytest.mark.parametrize("length", (0, 17, 1025))
def test_integer_references_keep_requested_accumulator_dtype(
    reduce, native, identity, dtype, length
):
    values = torch.ones((2, length), dtype=dtype)
    if length:
        values[:, :3] = torch.tensor([2, 3, 5], dtype=dtype)
    actual = reduce(values, compute_dtype=dtype, from_memory=True)
    expected = native(values.to(torch.int64), dim=-1).to(dtype)
    assert_bitwise(actual, expected)


def test_boolean_count_explicitly_uses_integer_accumulation():
    values = torch.tensor([[True, False, True], [False, True, False]])
    assert_bitwise(
        ordered_sum(values, compute_dtype=torch.int32, from_memory=True),
        torch.tensor([2, 1], dtype=torch.int32),
    )


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("compute_dtype", (None, torch.float64))
@pytest.mark.parametrize("from_memory", (False, True))
def test_public_reference_supports_autograd_without_changing_forward_bits(
    reduce, native, identity, dtype, compute_dtype, from_memory
):
    """Check casts, strided input, an excluded suffix and a partially filled tile."""
    storage = torch.linspace(0.75, 1.25, 64, dtype=dtype).reshape(2, 32).requires_grad_()
    native_storage = storage.detach().clone().requires_grad_()
    before = storage.detach().clone()
    values = storage[:, ::2]
    options = dict(length=11, from_memory=from_memory, block_size=4, compute_dtype=compute_dtype)
    actual = reduce(values, **options)
    expected = native(native_storage[:, ::2][:, :11].to(compute_dtype or dtype), dim=-1)
    upstream = torch.tensor([0.75, -1.25], dtype=actual.dtype)
    gradient = torch.autograd.grad(actual, storage, upstream)[0]
    native_gradient = torch.autograd.grad(expected, native_storage, upstream)[0]
    torch.testing.assert_close(gradient, native_gradient)
    assert gradient.dtype == dtype
    assert torch.count_nonzero(gradient[:, 1::2]) == 0
    assert torch.count_nonzero(gradient[:, 22:]) == 0
    with torch.no_grad():
        no_grad = reduce(values, **options)
    with torch.inference_mode():
        inference = reduce(values, **options)
    assert_bitwise(actual.detach(), no_grad)
    assert_bitwise(actual.detach(), inference)
    assert_bitwise(storage.detach(), before)
    assert actual.requires_grad and not no_grad.requires_grad


@pytest.mark.parametrize(
    "reduce,native", ((sum_partials, torch.sum), (product_partials, torch.prod))
)
@pytest.mark.parametrize("unroll", (1, 3, 8))
def test_partial_helpers_propagate_gradients_through_unrolled_groups_and_tails(
    reduce, native, unroll
):
    partials = torch.linspace(0.75, 1.25, 10, dtype=torch.float64).reshape(2, 5).requires_grad_()
    native_partials = partials.detach().clone().requires_grad_()
    actual = reduce(partials, merge_unroll=unroll)
    expected = native(native_partials, dim=-1)
    upstream = torch.tensor([0.5, -2.0], dtype=torch.float64)
    gradient = torch.autograd.grad(actual, partials, upstream)[0]
    native_gradient = torch.autograd.grad(expected, native_partials, upstream)[0]
    torch.testing.assert_close(gradient, native_gradient)
    with torch.no_grad():
        assert_bitwise(actual.detach(), reduce(partials, merge_unroll=unroll))


@pytest.mark.parametrize("path", ("loaded", "memory", "partials"))
@pytest.mark.parametrize("zeros", (0, 1, 2))
def test_product_gradients_handle_zero_factors(path, zeros):
    values = torch.tensor([0.5, 2.0, -1.0, 4.0], dtype=torch.float64)
    values[:zeros] = 0
    values.requires_grad_()
    native_values = values.detach().clone().requires_grad_()
    if path == "partials":
        actual = product_partials(values, merge_unroll=3)
    else:
        actual = ordered_product(values, from_memory=path == "memory", block_size=2)
    gradient = torch.autograd.grad(actual, values)[0]
    expected = torch.autograd.grad(native_values.prod(), native_values)[0]
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
def test_empty_memory_fold_remains_constant_without_reading_differentiable_input(
    reduce, native, identity
):
    """Removing the guard must not add arithmetic or artificial graph dependencies."""
    values = torch.ones((2, 8), requires_grad=True)
    actual = reduce(values, length=0, from_memory=True)
    assert_bitwise(actual, torch.full((2,), identity))
    assert not actual.requires_grad


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
def test_public_reference_propagates_nan(reduce, native, identity):
    values = torch.ones((2, 1025))
    values[0, 0] = float("nan")
    values[1, -1] = float("nan")
    assert torch.isnan(reduce(values, compute_dtype=torch.float32, from_memory=True)).all()


@pytest.mark.parametrize("helper", (ordered_sum, ordered_product))
def test_tile_helpers_cast_before_arithmetic_and_do_not_insert_a_seed(helper):
    """A one-element tile retains negative zero and low FP64 bits."""
    values = torch.tensor([[-0.0], [1 + 2**-35]], dtype=torch.float64)
    assert_bitwise(helper(values), values[:, 0])
    assert_bitwise(helper(values, compute_dtype=None), values[:, 0])
    assert_bitwise(helper(values, compute_dtype=torch.float64), values[:, 0])
    assert_bitwise(helper(values, compute_dtype=torch.float32), values[:, 0].float())


@pytest.mark.parametrize("helper", (ordered_sum, ordered_product))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_tile_helpers_apply_requested_group_size(helper, dtype):
    if helper is ordered_sum:
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
                helper(values, max_group_size=group_size, **dtype_kwargs),
                torch.tensor(result, dtype=dtype),
            )


@pytest.mark.parametrize("helper", (ordered_sum, ordered_product))
def test_tile_helpers_infer_integer_precision_without_float_conversion(helper):
    values = torch.tensor([2**53 + 1, 1, 2, 4], dtype=torch.int64)
    expected = 2**53 + 8 if helper is ordered_sum else (2**53 + 1) * 8
    for dtype_kwargs in ({}, {"compute_dtype": None}, {"compute_dtype": torch.int64}):
        assert_bitwise(helper(values, **dtype_kwargs), torch.tensor(expected, dtype=torch.int64))
