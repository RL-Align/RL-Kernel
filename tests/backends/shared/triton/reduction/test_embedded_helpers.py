# SPDX-License-Identifier: Apache-2.0
"""Embedded JIT helpers compose with surrounding arithmetic and match eager references."""

import pytest
import torch

from rl_engine.reference import reduction as torch_reduction
from rl_engine.reference.reduction.ordered_reduce import (
    ordered_reduce_merge,
    ordered_reduce_serial,
    ordered_tile_reduce,
)
from tests.helpers.tensor_checks import assert_bitwise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm is required")

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
    import triton.language as tl

    from rl_engine.backends.shared.triton import reduction as primitives
    from tests.helpers.ordered_reduce_kernels import composed_vector_probe

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
    composed_vector_probe[(2,)](
        values,
        actual,
        LENGTH=length,
        N_ROWS=2,
        REDUCE_FN=(primitives.sum_vector if operation == "sum" else primitives.product_vector),
        COMPUTE_DTYPE=kernel_dtype,
        num_warps=4,
    )
    assert_bitwise(actual, expected)


def test_embedded_sum_counts_boolean_inputs_in_int32():
    import triton.language as tl

    from rl_engine.backends.shared.triton.reduction import sum_vector
    from tests.helpers.ordered_reduce_kernels import composed_vector_probe

    values = (torch.arange(1025, device="cuda") % 3 != 0).reshape(1, -1)
    actual = torch.empty((1,), device="cuda", dtype=torch.int32)
    composed_vector_probe[(1,)](
        values,
        actual,
        LENGTH=1025,
        N_ROWS=1,
        REDUCE_FN=sum_vector,
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
        section_count=4,
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
    from tests.helpers.ordered_reduce_kernels import inferred_vector_dtype_probe

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
    inferred_vector_dtype_probe[(2,)](
        values,
        *outputs,
        LENGTH=length,
        REDUCE_FN=(reduction.sum_vector if operation == "sum" else reduction.product_vector),
        **_launch_options(4),
    )
    for actual in outputs:
        assert_bitwise(actual, expected, f"{operation}: inferred {dtype}, length={length}")


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.int64))
@pytest.mark.parametrize("merge_unroll", (1, 3, 4, 16))
def test_partial_merge_preserves_left_fold_seed_and_storage(dtype, merge_unroll):
    """A stored-partial merge must not regroup sums, lose dtype bits or mutate scratch."""
    import triton.language as tl

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import ordered_reduce_constexpr_seed_probe

    kernel_dtype = {
        torch.float32: tl.float32,
        torch.float64: tl.float64,
        torch.int64: tl.int64,
    }[dtype]
    magnitude = 2**25 if dtype == torch.float32 else 2**54
    # Float left fold yields 1 at length four; adjacent-pair reduction yields 0.
    source = torch.tensor(
        [
            [magnitude, 1, -magnitude, 1, 2, -2, 4, -4, 0.25],
            [-0.0] * 9,
            [3, -1, 5, 0, 8, -2, 7, 0, -4],
        ],
        dtype=dtype,
    )
    for n_tiles in (0, 1, 3, 4, 9):
        selected = source[:, :n_tiles].contiguous()
        expected = torch.zeros((3,), dtype=dtype)
        for tile in range(n_tiles):
            expected = expected + selected[:, tile]
        if n_tiles == 4:
            assert expected[0].item() == (2 if dtype == torch.int64 else 1)
        assert_bitwise(
            ordered_reduce_merge(
                selected,
                initial=0,
                combine_fn=torch.add,
                compute_dtype=dtype,
                merge_unroll=merge_unroll,
            ),
            expected,
            "same-named eager partial merge",
        )

        partials = selected.to("cuda")
        before = partials.clone()
        storage = torch.full((5,), -123, device="cuda", dtype=dtype)
        actual = storage[1:-1]
        ordered_reduce_constexpr_seed_probe[(3,)](
            partials,
            actual,
            N_TILES=n_tiles,
            INITIAL=0,
            COMBINE_FN=reduce_add,
            COMPUTE_DTYPE=kernel_dtype,
            MERGE_UNROLL=merge_unroll,
            **_launch_options(4),
        )
        assert_bitwise(actual, expected.to("cuda"), "ascending fold from typed positive zero")
        assert_bitwise(partials, before, "merge leaves producer partials unchanged")
        assert_bitwise(storage[[0, -1]], torch.full((2,), -123, device="cuda", dtype=dtype))


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (*_DTYPES, torch.float64, torch.int64))
def test_public_partial_helpers_infer_precision_and_preserve_fold(operation, dtype):
    """The two-argument API must retain partial bits and the declared merge order."""
    from rl_engine.backends.shared.triton import reduction
    from tests.helpers.ordered_reduce_kernels import public_partials_probe

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
            public_partials_probe[(2,)](
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
@pytest.mark.parametrize("length", (0, 1, 8, 16, 31, 32, 1023, 1024, 1025, 8193))
def test_default_embedded_helpers_match_reference(operation, combine, identity, dtype, length):
    """Use the default tree with explicit FP32 arithmetic, then continue computing."""
    import triton.language as tl

    from rl_engine.backends.shared.triton import reduction as primitives
    from rl_engine.reference import reduction as reference_package
    from tests.helpers.ordered_reduce_kernels import composed_vector_probe

    embedded_reduce = primitives.sum_vector if operation == "sum" else primitives.product_vector
    reference_reduce = (
        reference_package.sum_vector if operation == "sum" else reference_package.product_vector
    )
    values = _values(operation, dtype, length)
    before = values.clone()
    expected = _whole_vector_reference(values, combine, identity)
    # Guard both sides of the caller-allocated outputs.
    storage = torch.full((5,), -123.0, device="cuda")
    actual = storage[1:-1]
    assert_bitwise(
        reference_reduce(values, compute_dtype=torch.float32), expected, "default eager reference"
    )

    composed_vector_probe[(3,)](
        values,
        actual,
        LENGTH=length,
        N_ROWS=3,
        REDUCE_FN=embedded_reduce,
        COMPUTE_DTYPE=tl.float32,
        **_launch_options(4),
    )
    assert_bitwise(actual, expected.flip(0) + 3.0, "embedded return value and caller-owned pointer")
    assert_bitwise(storage[[0, -1]], torch.full((2,), -123.0, device="cuda"))
    assert_bitwise(values, before, "inputs unchanged")


@pytest.mark.parametrize("operation,combine,identity", _OPERATIONS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_embedded_helpers_also_match_independent_math(operation, combine, identity, dtype):
    """Retain an independent oracle instead of only comparing matching reduction trees."""
    import triton.language as tl

    from rl_engine.backends.shared.triton import reduction as primitives
    from tests.helpers.ordered_reduce_kernels import composed_vector_probe

    columns = torch.arange(1025, device="cuda")
    if operation == "product":
        values = torch.where(columns % 2 == 0, 2.0, 0.5).to(dtype)
        expected = values.double().prod().float()
    else:
        values = (columns % 8 - 5).to(dtype)
        expected = values.double().sum().float()
    actual = torch.empty((), device="cuda")
    composed_vector_probe[(1,)](
        values,
        actual,
        LENGTH=1025,
        N_ROWS=1,
        REDUCE_FN=(primitives.sum_vector if operation == "sum" else primitives.product_vector),
        COMPUTE_DTYPE=tl.float32,
        **_launch_options(4),
    )
    assert_bitwise(actual, expected + 3.0, "independent exact arithmetic cases")


def test_compute_precision_preserves_or_rounds_fp64_input_bits():
    """A non-FP32 source bit distinguishes computation precision from input storage."""
    import triton.language as tl

    from rl_engine.backends.shared.triton.reduction import sum_vector
    from tests.helpers.ordered_reduce_kernels import composed_vector_probe

    values = torch.tensor([1 + 2**-35], device="cuda", dtype=torch.float64)
    actual32 = torch.empty((), device="cuda")
    actual64 = torch.empty((), device="cuda", dtype=torch.float64)
    composed_vector_probe[(1,)](
        values,
        actual32,
        LENGTH=1,
        N_ROWS=1,
        REDUCE_FN=sum_vector,
        COMPUTE_DTYPE=tl.float32,
        **_launch_options(4),
    )
    composed_vector_probe[(1,)](
        values,
        actual64,
        LENGTH=1,
        N_ROWS=1,
        REDUCE_FN=sum_vector,
        COMPUTE_DTYPE=tl.float64,
        **_launch_options(4),
    )
    assert_bitwise(actual32, torch.tensor(4.0, device="cuda"))
    assert_bitwise(actual64, values[0] + 3.0)


@pytest.mark.parametrize("operation,combine,identity", _OPERATIONS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("block_size", (1, 4, 16, 32, 1024, 2048))
def test_loaded_tile_helpers_fuse_before_and_after_reduction(
    operation, combine, identity, dtype, block_size
):
    """An operator can map local values without writing a temporary tensor to HBM."""
    import triton.language as tl

    from rl_engine.backends.shared.triton import reduction as primitives
    from tests.helpers.ordered_reduce_kernels import composed_tile_probe

    length = max(1, block_size - 1)
    values = _values(operation, dtype, length)
    if operation == "product":
        # The probe applies 2*x+1, yielding alternating 2 and 1/2 without overflow.
        columns = torch.arange(length, device="cuda")
        values[:] = torch.where(columns % 2 == 0, 0.5, -0.25)
    mapped = values.float() * 2.0 + 1.0
    mapped = torch.nn.functional.pad(mapped, (0, block_size - length), value=identity)
    expected = (torch_reduction.sum_tile if operation == "sum" else torch_reduction.product_tile)(
        mapped, compute_dtype=torch.float32
    )
    actual = torch.empty((3,), device="cuda")
    composed_tile_probe[(3,)](
        values,
        actual,
        LENGTH=length,
        BLOCK_SIZE=block_size,
        REDUCE_FN=(primitives.sum_tile if operation == "sum" else primitives.product_tile),
        IDENTITY=identity,
        COMPUTE_DTYPE=tl.float32,
        **_launch_options(4),
    )
    assert_bitwise(actual, expected + 3.0, "fused tile map, reduction and subsequent arithmetic")


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("dtype", (*_DTYPES, torch.float64, torch.int64))
@pytest.mark.parametrize("block_size", (1, 4))
def test_tile_default_compute_dtype_preserves_input_bits_and_rounding(operation, dtype, block_size):
    """Omitted/None dtype must preserve seedless tiles and round in the input precision."""
    from rl_engine.backends.shared.triton import reduction
    from tests.helpers.ordered_reduce_kernels import inferred_tile_dtype_probe

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
    inferred_tile_dtype_probe[(2,)](
        values,
        *outputs,
        BLOCK_SIZE=block_size,
        REDUCE_FN=(reduction.sum_tile if operation == "sum" else reduction.product_tile),
        **_launch_options(4),
    )
    for actual in outputs:
        assert_bitwise(actual, expected, f"{operation}: inferred {dtype}, block_size={block_size}")


@pytest.mark.parametrize("block_size", (16, 1024))
@pytest.mark.parametrize("max_group_size", (1, 8, 32))
@pytest.mark.parametrize("section_count", (1, 4, 8))
@pytest.mark.parametrize("operation,combine,identity", _OPERATIONS)
def test_tile_helpers_accept_caller_selected_tree(
    block_size, max_group_size, section_count, operation, combine, identity
):
    """Public tree overrides must reach the reduction, including small-tile limits."""
    import triton.language as tl

    from rl_engine.backends.shared.triton import reduction as primitives
    from tests.helpers.ordered_reduce_kernels import configured_tile_probe

    values = _values("sum", torch.float32, block_size)
    if operation == "sum":
        values[0].fill_(-0.0)
        values[1].zero_()
        values[1, :4] = torch.tensor([1e20, 1.0, -1e20, 1.0], device="cuda")
    else:
        # Near-one factors exercise rounding without overflowing long products.
        values = 1.0 + values * 2**-20
        values[0].fill_(1.0)
        values[0, 0] = -0.0
    expected = (torch_reduction.sum_tile if operation == "sum" else torch_reduction.product_tile)(
        values,
        compute_dtype=torch.float32,
        max_group_size=max_group_size,
        section_count=section_count,
    )
    actual = torch.empty_like(expected)
    configured_tile_probe[(3,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        COMPUTE_DTYPE=tl.float32,
        REDUCE_FN=(primitives.sum_tile if operation == "sum" else primitives.product_tile),
        MAX_GROUP_SIZE=max_group_size,
        SECTION_COUNT=section_count,
        **_launch_options(4),
    )
    assert_bitwise(actual, expected, f"embedded {operation}_tile tree parameters")


@pytest.mark.parametrize("operation,combine,identity", _OPERATIONS)
def test_enclosing_kernel_warp_count_does_not_change_arithmetic(operation, combine, identity):
    import triton.language as tl

    from rl_engine.backends.shared.triton import reduction as primitives
    from tests.helpers.ordered_reduce_kernels import composed_vector_probe

    values = _values(operation, torch.float32, 1025)
    expected = _whole_vector_reference(values, combine, identity)
    actual = torch.empty((3,), device="cuda")
    for num_warps in (4, 8):
        composed_vector_probe[(3,)](
            values,
            actual,
            LENGTH=1025,
            N_ROWS=3,
            REDUCE_FN=(primitives.sum_vector if operation == "sum" else primitives.product_vector),
            COMPUTE_DTYPE=tl.float32,
            **_launch_options(num_warps),
        )
        assert_bitwise(actual, expected.flip(0) + 3.0, f"embedded helper with {num_warps} warps")


@pytest.mark.parametrize(
    "block_size,max_group_size,section_count",
    ((1, 8, 4), (16, 1, 8), (16, 4, 1), (1024, 8, 4), (2048, 16, 8)),
)
@pytest.mark.parametrize("operation,combine,identity", _OPERATIONS)
@pytest.mark.parametrize("pipeline_stages", (None, 3))
def test_vector_helpers_accept_caller_selected_tiles_and_tree(
    block_size, max_group_size, section_count, operation, combine, identity, pipeline_stages
):
    """Loop scheduling preserves the configured tree on empty vectors, tiles and tails."""
    import triton.language as tl

    from rl_engine.backends.shared.triton import reduction as primitives
    from tests.helpers.ordered_reduce_kernels import configured_vector_probe

    reference_reduce = (
        torch_reduction.sum_vector if operation == "sum" else torch_reduction.product_vector
    )
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
            section_count=section_count,
        )
        before = values.clone()
        actual = torch.full_like(expected, float("nan"))
        configured_vector_probe[(3,)](
            values,
            actual,
            LENGTH=length,
            COMPUTE_DTYPE=tl.float32,
            REDUCE_FN=(primitives.sum_vector if operation == "sum" else primitives.product_vector),
            BLOCK_SIZE=block_size,
            MAX_GROUP_SIZE=max_group_size,
            SECTION_COUNT=section_count,
            PIPELINE_STAGES=pipeline_stages,
            **_launch_options(4),
        )
        assert_bitwise(
            actual,
            expected,
            f"public {operation}_vector, length={length}, pipeline_stages={pipeline_stages}",
        )
        assert_bitwise(values, before, "input unchanged")


def test_fp64_merge_initial_is_not_first_rounded_to_fp32():
    """A constexpr Python float must be materialized directly in compute precision."""
    import triton.language as tl

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import ordered_reduce_constexpr_seed_probe

    partials = torch.zeros((1, 1), device="cuda", dtype=torch.float64)
    actual = torch.empty((1,), device="cuda", dtype=torch.float64)
    initial = 1 + 2**-35
    ordered_reduce_constexpr_seed_probe[(1,)](
        partials,
        actual,
        N_TILES=1,
        COMBINE_FN=reduce_add,
        INITIAL=initial,
        COMPUTE_DTYPE=tl.float64,
        MERGE_UNROLL=1,
    )
    assert_bitwise(actual, torch.tensor([initial], device="cuda", dtype=torch.float64))


@pytest.mark.parametrize("operation,combine,identity", _OPERATIONS)
@pytest.mark.parametrize("compute_dtype", (*_DTYPES, torch.float64))
@pytest.mark.parametrize("length", (0, 17, 1025))
def test_precision_is_exposed_by_embedded_helpers(
    operation, combine, identity, compute_dtype, length
):
    import triton.language as tl

    from rl_engine.backends.shared.triton import reduction as primitives
    from tests.helpers.ordered_reduce_kernels import composed_tile_probe, composed_vector_probe

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
    composed_vector_probe[(3,)](
        values,
        actual,
        LENGTH=length,
        N_ROWS=3,
        REDUCE_FN=(primitives.sum_vector if operation == "sum" else primitives.product_vector),
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
    expected_tile = ordered_tile_reduce(
        mapped, max_group_size=8, section_count=4, combine_fn=combine
    )
    composed_tile_probe[(3,)](
        values,
        actual,
        LENGTH=length,
        BLOCK_SIZE=block_size,
        REDUCE_FN=(primitives.sum_tile if operation == "sum" else primitives.product_tile),
        IDENTITY=identity,
        COMPUTE_DTYPE=kernel_dtype,
        **_launch_options(4),
    )
    assert_bitwise(actual, expected_tile + 3.0, "embedded tile precision")
