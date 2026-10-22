# SPDX-License-Identifier: Apache-2.0
"""Public eager references for configured embedded Triton reductions."""

import pytest
import torch

from rl_engine.reference.reduction import product_partials, product_vector, sum_partials, sum_vector
from tests.helpers.tensor_checks import assert_bitwise

_OPERATIONS = (
    pytest.param(sum_vector, torch.sum, 0.0, id="sum"),
    pytest.param(product_vector, torch.prod, 1.0, id="product"),
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
@pytest.mark.parametrize("input_dtype", _DTYPES)
@pytest.mark.parametrize("compute_dtype", _DTYPES)
def test_public_reference_matches_independent_exact_math(
    reduce, native, identity, input_dtype, compute_dtype
):
    """Use representable results; the oracle does not copy the fixed tree."""
    values = torch.tensor([[2.0, 0.5, -2.0], [0.5, 2.0, 4.0]], dtype=input_dtype)
    # Keep the last dimension strided, including an uneven tile tail.
    storage = torch.zeros((2, 6), dtype=input_dtype)
    storage[:, ::2] = values
    before = storage.clone()
    actual = reduce(storage[:, ::2], compute_dtype=compute_dtype)
    expected = native(values.double(), dim=-1).to(compute_dtype)
    assert actual.shape == (2,) and actual.dtype == compute_dtype
    assert actual.device == values.device
    assert_bitwise(actual, expected, "independent native FP64 oracle")
    assert_bitwise(storage, before, "input storage unchanged")


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize("compute_dtype", _DTYPES)
@pytest.mark.parametrize("shape", ((0,), (3, 0), (0, 1025), (2, 0, 1025)))
def test_empty_vectors_and_batches_match_public_contract(
    reduce, native, identity, compute_dtype, shape
):
    values = torch.empty(shape)
    actual = reduce(values, compute_dtype=compute_dtype)
    expected = torch.full(shape[:-1], identity, dtype=compute_dtype)
    assert_bitwise(actual, expected)


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
def test_scalar_output_and_explicit_compute_precision(reduce, native, identity):
    values = torch.tensor([1 + 2**-35], dtype=torch.float64)
    assert_bitwise(reduce(values, compute_dtype=torch.float32), torch.tensor(1.0))
    assert_bitwise(reduce(values, compute_dtype=torch.float64), values[0])


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
    assert_bitwise(reduce(values), expected)
    assert_bitwise(reduce(values, compute_dtype=None), expected)
    assert_bitwise(reduce(values, compute_dtype=dtype), expected)


@pytest.mark.parametrize(
    "positions,expected",
    (
        ((0, 1, 2, 3), 1.0),
        ((0, 8, 128, 136), 2.0),
        ((0, 256, 512, 768), 2.0),
        ((0, 1024, 2048, 3072), 1.0),
    ),
)
def test_public_sum_keeps_declared_tile_tree_and_ascending_merge(positions, expected):
    """Freeze numerical behavior independently of shared constants and reference helpers."""
    values = torch.zeros(3073)
    values[list(positions)] = torch.tensor([1e20, 1, -1e20, 1])
    assert_bitwise(sum_vector(values, compute_dtype=torch.float32), torch.tensor(expected))


@pytest.mark.parametrize(
    "block_size,max_group_size,section_count,positions,expected",
    (
        (16, 8, 4, (0, 8, 16, 24), 0.0),
        (32, 8, 4, (0, 8, 16, 24), 2.0),
        (32, 8, 4, (0, 1, 2, 3), 1.0),
        (32, 1, 4, (0, 1, 2, 3), 2.0),
        (64, 8, 4, (0, 8, 16, 24), 0.0),
        (64, 8, 8, (0, 8, 16, 24), 2.0),
    ),
)
def test_public_sum_configuration_controls_rounding(
    block_size, max_group_size, section_count, positions, expected
):
    """Each adjacent pair changes one parameter and has independently derived expected bits."""
    values = torch.zeros(64)
    values[list(positions)] = torch.tensor([1e20, 1, -1e20, 1])
    actual = sum_vector(
        values,
        compute_dtype=torch.float32,
        block_size=block_size,
        max_group_size=max_group_size,
        section_count=section_count,
    )
    assert_bitwise(actual, torch.tensor(expected))


@pytest.mark.parametrize("max_group_size,expected", ((8, float("inf")), (1, 1.0)))
def test_public_product_configuration_controls_intermediate_overflow(max_group_size, expected):
    """Sequential large factors overflow; the alternate tree first pairs reciprocal factors."""
    values = torch.ones(32)
    values[:4] = torch.tensor([2.0**80, 2.0**80, 2.0**-80, 2.0**-80])
    actual = product_vector(
        values,
        compute_dtype=torch.float32,
        block_size=32,
        max_group_size=max_group_size,
        section_count=4,
    )
    assert_bitwise(actual, torch.tensor(expected))


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize(
    "block_size,max_group_size,section_count",
    ((1, 8, 4), (16, 1, 8), (16, 4, 1), (1024, 8, 4), (2048, 16, 8)),
)
def test_public_configurations_cover_empty_strided_vectors_and_tails(
    reduce, native, identity, block_size, max_group_size, section_count
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
            section_count=section_count,
        )
        assert_bitwise(actual, native(values, dim=-1).float())
        assert_bitwise(storage, before, "input storage unchanged")


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize("parameter", ("block_size", "max_group_size", "section_count"))
@pytest.mark.parametrize("size", (-1, 0, 3))
@pytest.mark.parametrize("shape", ((2, 17), (2, 0), (0, 17)))
def test_public_configuration_validation_includes_empty_inputs(
    reduce, native, identity, parameter, size, shape
):
    with pytest.raises(ValueError, match="positive power of two"):
        reduce(torch.ones(shape), compute_dtype=torch.float32, **{parameter: size})


@pytest.mark.parametrize(
    "compute_dtype,bits",
    ((torch.float16, 11), (torch.bfloat16, 8), (torch.float32, 24), (torch.float64, 53)),
)
def test_public_sum_rounds_each_step_in_requested_dtype(compute_dtype, bits):
    values = torch.zeros((2, 2049), dtype=torch.float64)
    values[0, :3] = torch.tensor([2**bits, 1, 1], dtype=torch.float64)
    values[1, [0, 1024, 2048]] = values[0, :3]
    assert_bitwise(
        sum_vector(values, compute_dtype=compute_dtype),
        torch.full((2,), 2**bits, dtype=compute_dtype),
    )
    # Inferring the input dtype must not accumulate in a wider dtype and cast
    # only at the end: that would retain the two unit terms.
    stored = values.to(compute_dtype)
    expected = torch.full((2,), 2**bits, dtype=compute_dtype)
    assert_bitwise(sum_vector(stored), expected)
    assert_bitwise(sum_vector(stored, compute_dtype=None), expected)


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
        reduce(values, compute_dtype=torch.float32)


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
@pytest.mark.parametrize("dtype", (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64))
@pytest.mark.parametrize("length", (0, 17, 1025))
def test_integer_references_keep_requested_accumulator_dtype(
    reduce, native, identity, dtype, length
):
    values = torch.ones((2, length), dtype=dtype)
    if length:
        values[:, :3] = torch.tensor([2, 3, 5], dtype=dtype)
    actual = reduce(values, compute_dtype=dtype)
    expected = native(values.to(torch.int64), dim=-1).to(dtype)
    assert_bitwise(actual, expected)


def test_boolean_count_explicitly_uses_integer_accumulation():
    values = torch.tensor([[True, False, True], [False, True, False]])
    assert_bitwise(
        sum_vector(values, compute_dtype=torch.int32), torch.tensor([2, 1], dtype=torch.int32)
    )


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
def test_public_reference_preserves_autograd_boundary(reduce, native, identity):
    values = torch.ones((2, 8), requires_grad=True)
    with pytest.raises(RuntimeError, match="without autograd"):
        reduce(values, compute_dtype=torch.float32)
    with torch.no_grad():
        actual = reduce(values, compute_dtype=torch.float32)
    with torch.inference_mode():
        inference = reduce(values, compute_dtype=torch.float32)
    assert_bitwise(actual, inference)
    assert not actual.requires_grad


@pytest.mark.parametrize("reduce,native,identity", _OPERATIONS)
def test_public_reference_propagates_nan(reduce, native, identity):
    values = torch.ones((2, 1025))
    values[0, 0] = float("nan")
    values[1, -1] = float("nan")
    assert torch.isnan(reduce(values, compute_dtype=torch.float32)).all()
