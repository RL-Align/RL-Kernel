# SPDX-License-Identifier: Apache-2.0
"""Embedded JIT helpers compose with surrounding arithmetic and match eager references."""

import pytest
import torch

from rl_engine.reference import reduction as torch_reduction
from rl_engine.reference.reduction.ordered_reduce import ordered_reduce_serial, ordered_tile_reduce
from tests.helpers.tensor_checks import assert_bitwise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm is required")


# Keep CPU collection independent of optional GPU frontends.
if torch.cuda.is_available():
    import triton.language as tl

    from tests.backends.shared.triton.reduction import _kernels as kernels

_OPERATIONS = (
    pytest.param("sum", torch.add, 0.0, id="sum"),
    pytest.param("product", torch.mul, 1.0, id="product"),
)
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64))
@pytest.mark.parametrize("length", (0, 17, 1025))
def test_embedded_vector_helpers_keep_integer_arithmetic(operation, dtype, length):
    """Defaults supply integer-safe identities; the caller selects accumulator width."""

    from rl_engine.backends.shared.triton import reduction as primitives

    kernel_dtype = {
        torch.uint8: tl.uint8,
        torch.int8: tl.int8,
        torch.int16: tl.int16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
    }[dtype]
    values = torch.ones((2, length), device="cuda", dtype=dtype)
    if length:
        values[:, :3] = torch.tensor([2, 3, 5], device="cuda", dtype=dtype)
        if dtype == torch.int64:
            values[0, 0] = 2**53 + 1
    native = torch.sum if operation == "sum" else torch.prod
    expected = (native(values.long(), dim=-1).flip(0) + 3).to(dtype)
    actual = torch.empty_like(expected)
    kernels.composed_vector_probe[(2,)](
        values,
        actual,
        LENGTH=length,
        N_ROWS=2,
        REDUCE_FN=(primitives.sum if operation == "sum" else primitives.product),
        COMPUTE_DTYPE=kernel_dtype,
        num_warps=4,
    )
    assert_bitwise(actual, expected)


def test_embedded_sum_counts_boolean_inputs_in_int32():

    from rl_engine.backends.shared.triton.reduction import sum as ordered_sum

    values = (torch.arange(1025, device="cuda") % 3 != 0).reshape(1, -1)
    actual = torch.empty((1,), device="cuda", dtype=torch.int32)
    kernels.composed_vector_probe[(1,)](
        values,
        actual,
        LENGTH=1025,
        N_ROWS=1,
        REDUCE_FN=ordered_sum,
        COMPUTE_DTYPE=tl.int32,
        num_warps=4,
    )
    assert_bitwise(actual, values.sum(-1, dtype=torch.int32) + 3)


def _launch_options(num_warps):
    options = {"num_warps": num_warps, "enable_fp_fusion": False}
    if torch.version.hip is not None:
        options["allow_flush_denorm"] = False
    return options


def _values(operation, dtype, length):
    generator = torch.Generator().manual_seed(419)
    if operation == "product":
        columns = torch.arange(length)
        values = torch.where(columns % 2 == 0, 2.0, 0.5).expand(3, -1).clone()
        if length:
            values[1, 0] *= -1
            values[2, 0] = 0
    else:
        values = torch.randn((3, length), generator=generator)
        values *= torch.pow(2.0, torch.randint(-5, 12, values.shape, generator=generator))
    return values.to(device="cuda", dtype=dtype)


def _whole_vector_reference(values, combine, identity, compute_dtype=torch.float32):
    return ordered_reduce_serial(
        values,
        block_size=1024,
        max_group_size=8,
        combine_fn=combine,
        identity=identity,
        initial=identity,
        compute_dtype=compute_dtype,
    )


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (*_DTYPES, torch.float64, torch.int64))
@pytest.mark.parametrize("length", (0, 3, 1025))
def test_vector_default_compute_dtype_matches_input_storage(operation, dtype, length):
    """Check inferred precision before storage, including per-add rounding and FP64 bits."""
    from rl_engine.backends.shared.triton import reduction

    identity = 0 if operation == "sum" else 1
    values = torch.full((2, length), identity, dtype=dtype, device="cuda")
    expected = torch.full((2,), identity, dtype=dtype, device="cuda")
    if length:
        values[:, :3] = torch.tensor([2, 1, 4], dtype=dtype, device="cuda")
        expected.fill_(7 if operation == "sum" else 8)
        if dtype == torch.int64:
            values[1, 0] = 2**53 + 1
            expected[1] = 2**53 + 6 if operation == "sum" else (2**53 + 1) * 4
        elif operation == "sum":
            magnitude = 2 / torch.finfo(dtype).eps
            values[0, :3] = torch.tensor([magnitude, 1, 1], dtype=dtype, device="cuda")
            expected[0] = magnitude
        if dtype == torch.float64:
            values[1, 0] += 2**-35
            expected[1] += 2**-35 if operation == "sum" else 2**-33
    outputs = [torch.empty_like(expected) for _ in range(3)]
    kernels.inferred_vector_dtype_probe[(2,)](
        values,
        *outputs,
        LENGTH=length,
        REDUCE_FN=(reduction.sum if operation == "sum" else reduction.product),
        **_launch_options(4),
    )
    for actual in outputs:
        assert_bitwise(actual, expected, f"{operation}: inferred {dtype}, length={length}")


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (*_DTYPES, torch.float64, torch.int64))
def test_public_partial_helpers_infer_precision_and_preserve_fold(operation, dtype):
    """The two-argument API must retain partial bits and the declared merge order."""
    from rl_engine.backends.shared.triton import reduction

    identity = 0 if operation == "sum" else 1
    if operation == "sum":
        magnitude = 2**54 if dtype == torch.int64 else 2 / torch.finfo(dtype).eps
        data = [magnitude, 1, -magnitude, 1, 2, -2, 4, -4, 5]
        combine, reduce_fn = torch.add, reduction.sum_partials
    else:
        exponent = {torch.float16: 13, torch.bfloat16: 70, torch.float32: 80, torch.float64: 600}
        small = 2.0 ** -exponent[dtype] if dtype != torch.int64 else 2
        inverse = 1 / small if dtype != torch.int64 else 3
        data = [small, small, inverse, inverse, 2, 1, 1, 1, 1]
        combine, reduce_fn = torch.mul, reduction.product_partials
    values = torch.tensor([data, [identity] * 9], dtype=dtype)
    if dtype == torch.float64:
        values[1, 0] = 1 + 2**-35
    elif dtype == torch.int64:
        values[1, 0] = 2**53 + 1
    else:
        values[1, 0] = -0.0
    for n_tiles in (0, 1, 4, 9):
        partials = values[:, :n_tiles].contiguous().to("cuda")
        before = partials.clone()
        expected = torch.full((2,), identity, dtype=dtype)
        for tile in range(n_tiles):
            expected = combine(expected, values[:, tile])
        outputs = [torch.empty((2,), device="cuda", dtype=dtype) for _ in range(3)]
        for unroll in (3, 16):
            kernels.public_partials_probe[(2,)](
                partials,
                *outputs,
                N_TILES=n_tiles,
                REDUCE_FN=reduce_fn,
                MERGE_UNROLL=unroll,
                **_launch_options(4),
            )
            for actual in outputs:
                assert_bitwise(actual, expected.to("cuda"), "default and configured partial merge")
        assert_bitwise(partials, before, "partial inputs unchanged")


@pytest.mark.parametrize("operation,combine,identity", _OPERATIONS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_embedded_helpers_also_match_independent_math(operation, combine, identity, dtype):
    """Retain an independent oracle instead of only comparing matching reduction trees."""

    from rl_engine.backends.shared.triton import reduction as primitives

    columns = torch.arange(1025, device="cuda")
    if operation == "product":
        values = torch.where(columns % 2 == 0, 2.0, 0.5).to(dtype)
        expected = values.double().prod().float()
    else:
        values = (columns % 8 - 5).to(dtype)
        expected = values.double().sum().float()
    actual = torch.empty((), device="cuda")
    kernels.composed_vector_probe[(1,)](
        values,
        actual,
        LENGTH=1025,
        N_ROWS=1,
        REDUCE_FN=(primitives.sum if operation == "sum" else primitives.product),
        COMPUTE_DTYPE=tl.float32,
        **_launch_options(4),
    )
    assert_bitwise(actual, expected + 3.0, "independent exact arithmetic cases")


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (*_DTYPES, torch.float64, torch.int64))
@pytest.mark.parametrize("block_size", (1, 4))
def test_tile_default_compute_dtype_preserves_input_bits_and_rounding(operation, dtype, block_size):
    """Omitted/None dtype must preserve seedless tiles and round in the input precision."""
    from rl_engine.backends.shared.triton import reduction

    identity = 0 if operation == "sum" else 1
    values = torch.full((2, block_size), identity, device="cuda", dtype=dtype)
    expected = torch.empty((2,), device="cuda", dtype=dtype)
    if block_size == 1:
        values[:, 0] = torch.tensor([-0.0, 1], device="cuda", dtype=dtype)
        expected.copy_(values[:, 0])
    else:
        values[:] = torch.tensor([2, 1, 2, 4], device="cuda", dtype=dtype)
        expected.fill_(9 if operation == "sum" else 16)
        if dtype != torch.int64:
            if operation == "sum":
                large = 2 / torch.finfo(dtype).eps
                values[0] = torch.tensor([large, 1, -large, 1], device="cuda", dtype=dtype)
                expected[0] = 1
            else:
                exponent = {
                    torch.float16: 10,
                    torch.bfloat16: 80,
                    torch.float32: 80,
                    torch.float64: 600,
                }[dtype]
                values[0] = torch.tensor(
                    [2.0**exponent] * 2 + [2.0**-exponent] * 2, device="cuda", dtype=dtype
                )
                expected[0] = float("inf")
    if dtype == torch.float64:
        values[1, 0] += 2**-35
        expected[1] += 2**-35 if operation == "sum" or block_size == 1 else 2**-32
    elif dtype == torch.int64:
        values[1, 0] = 2**53 + 1
        if block_size == 1:
            expected[1] = 2**53 + 1
        elif operation == "sum":
            expected[1] = 2**53 + 8
        else:
            expected[1] = (2**53 + 1) * 8
    outputs = [torch.empty_like(expected) for _ in range(3)]
    kernels.inferred_tile_dtype_probe[(2,)](
        values,
        *outputs,
        BLOCK_SIZE=block_size,
        REDUCE_FN=(reduction.sum if operation == "sum" else reduction.product),
        **_launch_options(4),
    )
    for actual in outputs:
        assert_bitwise(actual, expected, f"{operation}: inferred {dtype}, block_size={block_size}")


@pytest.mark.parametrize(
    "block_size,max_group_size,num_warps",
    ((1, 8, 4), (16, 1, 1), (16, 4, 2), (1024, 8, 4), (2048, 32, 8)),
)
@pytest.mark.parametrize("operation,combine,identity", _OPERATIONS)
def test_tile_helpers_accept_caller_selected_tree(
    block_size, max_group_size, num_warps, operation, combine, identity
):
    """Public tree overrides must reach the reduction, including small-tile limits."""

    from rl_engine.backends.shared.triton import reduction as primitives

    values = _values("sum", torch.float32, block_size)
    if operation == "sum":
        values[0].fill_(-0.0)
        values[1].zero_()
        pattern = torch.tensor([1e20, 1.0, -1e20, 1.0], device="cuda")
        values[1, : min(block_size, 4)] = pattern[: min(block_size, 4)]
    else:
        # Near-one factors exercise rounding without overflowing long products.
        values = 1.0 + values * 2**-20
        values[0].fill_(1.0)
        values[0, 0] = -0.0
    expected = (torch_reduction.sum if operation == "sum" else torch_reduction.product)(
        values,
        compute_dtype=torch.float32,
        max_group_size=max_group_size,
    )
    actual = torch.empty_like(expected)
    kernels.configured_tile_probe[(3,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        COMPUTE_DTYPE=tl.float32,
        REDUCE_FN=(primitives.sum if operation == "sum" else primitives.product),
        MAX_GROUP_SIZE=max_group_size,
        **_launch_options(num_warps),
    )
    assert_bitwise(actual, expected, f"embedded {operation}_tile tree parameters")


@pytest.mark.parametrize(
    "block_size,max_group_size",
    ((1, 8), (16, 1), (16, 4), (1024, 8), (2048, 16)),
)
@pytest.mark.parametrize("operation,combine,identity", _OPERATIONS)
@pytest.mark.parametrize("pipeline_stages", (None, 3))
def test_vector_helpers_accept_caller_selected_tiles_and_tree(
    block_size, max_group_size, operation, combine, identity, pipeline_stages
):
    """Loop scheduling preserves the configured tree on empty vectors, tiles and tails."""

    from rl_engine.backends.shared.triton import reduction as primitives

    reference_reduce = torch_reduction.sum if operation == "sum" else torch_reduction.product
    for length in sorted({0, block_size - 1, block_size, 2 * block_size + 3}):
        values = _values("sum", torch.float32, length)
        if length:
            if operation == "sum":
                values[0].fill_(-0.0)
                values[1].zero_()
                pattern = torch.tensor([1e20, 1.0, -1e20, 1.0], device="cuda")
                values[1, : min(length, 4)] = pattern[: min(length, 4)]
            else:
                values = 1.0 + values * 2**-20
                values[0].fill_(1.0)
                values[0, 0] = -0.0
        expected = reference_reduce(
            values,
            compute_dtype=torch.float32,
            block_size=block_size,
            max_group_size=max_group_size,
            from_memory=True,
        )
        before = values.clone()
        actual = torch.full_like(expected, float("nan"))
        kernels.configured_vector_probe[(3,)](
            values,
            actual,
            values.stride(0),
            LENGTH=length,
            COMPUTE_DTYPE=tl.float32,
            REDUCE_FN=(primitives.sum if operation == "sum" else primitives.product),
            BLOCK_SIZE=block_size,
            MAX_GROUP_SIZE=max_group_size,
            PIPELINE_STAGES=pipeline_stages,
            TILE_UNROLL=None,
            **_launch_options(4),
        )
        assert_bitwise(
            actual,
            expected,
            f"public {operation}_vector, length={length}, pipeline_stages={pipeline_stages}",
        )
        assert_bitwise(values, before, "input unchanged")


@pytest.mark.parametrize("dtype,initial", ((torch.float64, 1 + 2**-35), (torch.int64, 2**53 + 1)))
@pytest.mark.parametrize("count", (0, 1))
def test_merge_initial_preserves_fp64_and_large_integer_bits(dtype, initial, count):
    """Materialize constexpr seeds directly in compute precision, even with no partials."""

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add

    partials = torch.zeros((1, count), device="cuda", dtype=dtype)
    actual = torch.empty((1,), device="cuda", dtype=dtype)
    kernels.ordered_reduce_constexpr_seed_probe[(1,)](
        partials,
        actual,
        N_TILES=count,
        COMBINE_FN=reduce_add,
        INITIAL=initial,
        COMPUTE_DTYPE=tl.float64 if dtype == torch.float64 else tl.int64,
        MERGE_UNROLL=1,
    )
    assert_bitwise(actual, torch.tensor([initial], device="cuda", dtype=dtype))


@pytest.mark.parametrize("operation,combine,identity", _OPERATIONS)
@pytest.mark.parametrize("compute_dtype", (*_DTYPES, torch.float64))
@pytest.mark.parametrize("length", (0, 17, 1025))
def test_precision_is_exposed_by_embedded_helpers(
    operation, combine, identity, compute_dtype, length
):

    from rl_engine.backends.shared.triton import reduction as primitives

    kernel_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.float64: tl.float64,
    }[compute_dtype]
    # Nontrivial FP64 source bits catch a hidden conversion to FP32.
    columns = torch.arange(length, device="cuda", dtype=torch.float64)
    values = ((columns % 7 - 3) / 7 + 2**-35).expand(3, -1).contiguous()
    if operation == "product":
        values = torch.where(columns % 2 == 0, 1 + 2**-10, 1 - 2**-10).expand(3, -1).contiguous()
    actual = torch.empty((3,), device="cuda", dtype=compute_dtype)
    expected = _whole_vector_reference(values, combine, identity, compute_dtype)
    kernels.composed_vector_probe[(3,)](
        values,
        actual,
        LENGTH=length,
        N_ROWS=3,
        REDUCE_FN=(primitives.sum if operation == "sum" else primitives.product),
        COMPUTE_DTYPE=kernel_dtype,
        **_launch_options(4),
    )
    assert_bitwise(actual, expected.flip(0) + 3.0, "embedded vector precision")
    if not length:
        return
    # Map in the requested precision and keep products bounded after 2*x+1.
    if operation == "product":
        values = ((values - 1) / 2).contiguous()
    block_size = 1 << (length - 1).bit_length()
    mapped = values.to(compute_dtype) * 2.0 + 1.0
    mapped = torch.nn.functional.pad(mapped, (0, block_size - length), value=identity)
    expected_tile = ordered_tile_reduce(mapped, max_group_size=8, combine_fn=combine)
    kernels.composed_tile_probe[(3,)](
        values,
        actual,
        LENGTH=length,
        BLOCK_SIZE=block_size,
        REDUCE_FN=(primitives.sum if operation == "sum" else primitives.product),
        IDENTITY=identity,
        COMPUTE_DTYPE=kernel_dtype,
        **_launch_options(4),
    )
    assert_bitwise(actual, expected_tile + 3.0, "embedded tile precision")
