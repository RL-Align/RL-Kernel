# SPDX-License-Identifier: Apache-2.0
"""Fixed-order reduction conformance, independent of model/operator tests."""

import pytest
import torch

from rl_engine.reference.reduction.ordered_reduce import (
    ordered_reduce_merge,
    ordered_reduce_partial,
    ordered_reduce_serial,
    ordered_tile_reduce,
)
from tests.helpers.tensor_checks import assert_bitwise

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm is required")


# Keep CPU collection independent of optional GPU frontends.
if torch.cuda.is_available():
    import triton.language as tl

    from tests.backends.shared.triton.reduction import _kernels as kernels

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TREE_CONFIG = {"max_group_size": 8}
"""Eight-value sequential groups, then one tree over all group totals."""


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ),
)
@pytest.mark.parametrize("num_warps", (1, 2, 4))
@pytest.mark.parametrize(
    "group_size, row_padding, storage_offset",
    ((1, 0, 0), (2, 1, 1), (8, 0, 0), (8, 1, 0), (32, 8, 1)),
)
def test_group_selection_copies_original_bits(
    dtype, num_warps, group_size, row_padding, storage_offset
):
    """Copy exact bits under aligned, odd-stride and offset-pointer layouts.

    Every column must survive extraction. Raw floating payloads include negative
    zero, infinities, distinct NaNs, a low mantissa bit and the least subnormal.
    Integers cover all supported widths, including values beyond FP64's exact
    integer range. The result must keep both the input dtype and its raw bits.
    """

    floating_payloads = {
        torch.float16: (
            torch.uint16,
            [0x8000, 0x7C00, 0xFC00, 0x7E11, 0xFE22, 0, 0x3C01, 1],
        ),
        torch.bfloat16: (
            torch.uint16,
            [0x8000, 0x7F80, 0xFF80, 0x7FC1, 0xFFC2, 0, 0x3F81, 1],
        ),
        torch.float32: (
            torch.uint32,
            [
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x7FC01234,
                0xFFC05678,
                0,
                0x3F800001,
                1,
            ],
        ),
        torch.float64: (
            torch.uint64,
            [
                0x8000000000000000,
                0x7FF0000000000000,
                0xFFF0000000000000,
                0x7FF8000000001234,
                0xFFF8000000005678,
                0,
                0x3FF0000000000001,
                1,
            ],
        ),
    }
    if dtype.is_floating_point:
        bit_dtype, payload = floating_payloads[dtype]
        pattern = torch.tensor(payload, dtype=bit_dtype).view(dtype)
    elif dtype == torch.bool:
        pattern = torch.tensor([True, False, True, False, False, True, False, True])
    elif dtype == torch.int64:
        pattern = torch.tensor([2**60 + 1, -(2**60) + 3, -1, 0, 1, 7, 2**53 + 1, 31])
    else:
        info = torch.iinfo(dtype)
        pattern = torch.tensor([info.min, info.max, 0, 1, 2, 3, 17, 29], dtype=dtype)
    source = pattern.repeat((group_size + 7) // 8)[:group_size].repeat(8, 1)
    source[1::2] = source[1::2].flip(-1)
    row_stride = group_size + row_padding
    storage = torch.empty(8 * row_stride + storage_offset, dtype=dtype, device="cuda")
    values = torch.as_strided(storage, (8, group_size), (row_stride, 1), storage_offset)
    values.copy_(source)
    actual = torch.empty((8, group_size), dtype=dtype, device="cuda")
    kernels.group_column_probe[(1,)](
        values,
        actual,
        values.stride(0),
        GROUP_SIZE=group_size,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    assert_bitwise(actual, values, "group column extraction preserves the source bits")


@pytest.mark.parametrize(
    "n_tiles,merge_unroll",
    (
        (0, 3),
        (1, 1),
        (2, 3),
        (3, 3),
        (4, 4),
        (5, 4),
        (9, 8),
        (17, 16),
        (63, 1),
        (64, 3),
        (65, 2),
        (255, 1),
        (256, 4),
        (257, 3),
        (513, 8),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("operation", ("sum", "product"))
def test_prefetched_merge_preserves_rounding_seed_and_window_boundaries(
    n_tiles, dtype, operation, merge_unroll
):
    """Window reads change neither callback order nor tail/seed arithmetic.

    A separate scalar PyTorch fold supplies the expected bits. Caller-owned
    callbacks, cancellation, negative zero, nonidentity seeds and graph replay
    cover the prefetch threshold, full windows and incomplete final windows.
    """

    if operation == "sum":
        combine, reference_combine = kernels.caller_add, torch.add
        large = 2.0 / torch.finfo(dtype).eps
        pattern = [[large, 1.0, -large, 1.0], [-0.0] * 4, [1.0] * 4]
        seeds = torch.tensor([0.0, -0.0, large], dtype=dtype)
    else:
        combine, reference_combine = kernels.caller_multiply, torch.mul
        pattern = [[2.0, 0.5, 4.0, 0.25], [2.0, 0.5, -2.0, -0.5], [1.0] * 4]
        seeds = torch.tensor([3.0, -0.0, -2.0], dtype=dtype)
    cpu_partials = torch.tensor(pattern, dtype=dtype).repeat(1, (n_tiles + 3) // 4)
    cpu_partials = cpu_partials[:, :n_tiles].contiguous()
    expected = seeds.clone()
    for tile in range(n_tiles):
        expected = reference_combine(expected, cpu_partials[:, tile])
    partials, initials = cpu_partials.cuda(), seeds.cuda()
    actual = torch.empty(3, device="cuda", dtype=dtype)

    def call():
        kernels.ordered_reduce_merge_probe[(3,)](
            partials,
            initials,
            actual,
            N_TILES=n_tiles,
            MERGE_UNROLL=merge_unroll,
            COMBINE_FN=combine,
            COMPUTE_DTYPE=tl.float32 if dtype == torch.float32 else tl.float64,
            num_warps=4,
            enable_fp_fusion=False,
        )

    call()
    assert_bitwise(actual.cpu(), expected, "prefetched scalar fold")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    graph.replay()
    assert_bitwise(actual.cpu(), expected, "prefetched graph replay")
    assert_bitwise(partials.cpu(), cpu_partials, "prefetch preserves partials")
    assert_bitwise(initials.cpu(), seeds, "prefetch preserves seeds")


@pytest.mark.parametrize("merge_unroll", (0, -1))
def test_merge_rejects_nonpositive_unroll(merge_unroll):
    """Reject invalid loop steps before generating a merge kernel."""
    from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add

    partials = torch.ones((1, 1), device="cuda")
    initials = torch.zeros(1, device="cuda")
    actual = torch.empty_like(initials)
    with pytest.raises(CompilationError) as error_info:
        kernels.ordered_reduce_merge_probe[(1,)](
            partials,
            initials,
            actual,
            N_TILES=1,
            MERGE_UNROLL=merge_unroll,
            COMBINE_FN=reduce_add,
            num_warps=4,
        )
    # Nested JIT calls wrap the assertion with their own source locations.
    # Require the actual static assertion, not merely any compilation failure.
    error = error_info.value
    while error.__cause__ is not None:
        error = error.__cause__
    assert isinstance(error, CompileTimeAssertionFailure)
    assert error.error_message == "merge unroll must be a positive integer"


@pytest.mark.parametrize(
    "compute_dtype,bits",
    (
        (torch.float16, 11),
        (torch.bfloat16, 8),
        (torch.float32, 24),
        (torch.float64, 53),
    ),
)
def test_compute_dtype_rounds_each_add_in_tiles_and_merge(compute_dtype, bits):
    """Both schedules must round at each addition, not just when storing the output."""

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add

    kernel_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.float64: tl.float64,
    }[compute_dtype]
    values = torch.zeros((2, 2049), device="cuda", dtype=torch.float64)
    values[0, :3] = torch.tensor([2**bits, 1, 1], device="cuda", dtype=torch.float64)
    values[1, [0, 1024, 2048]] = values[0, :3]
    expected = torch.full((2,), 2**bits, device="cuda", dtype=compute_dtype)
    initials = torch.zeros(2, device="cuda", dtype=compute_dtype)
    actual = torch.empty_like(expected)
    partials = torch.empty((2, 3), device="cuda", dtype=compute_dtype)
    options = {"num_warps": 4, "enable_fp_fusion": False}
    if torch.version.hip is not None:
        options["allow_flush_denorm"] = False
    tile_options = dict(
        LENGTH=2049,
        BLOCK_SIZE=1024,
        MAX_GROUP_SIZE=8,
        COMBINE_FN=reduce_add,
        IDENTITY=0.0,
        COMPUTE_DTYPE=kernel_dtype,
    )
    for stages in (None, 3):
        kernels.ordered_reduce_serial_probe[(2,)](
            values,
            initials,
            actual,
            PIPELINE_STAGES=stages,
            **tile_options,
            **options,
        )
        assert_bitwise(actual, expected, f"per-add rounding, stages={stages}")
    kernels.ordered_reduce_partial_probe[(2, 3)](
        values, partials, values.stride(0), REVERSE_TILES=False, **tile_options, **options
    )
    for unroll in (1, 2, 3, 4, 8):
        kernels.ordered_reduce_merge_probe[(2,)](
            partials,
            initials,
            actual,
            N_TILES=3,
            MERGE_UNROLL=unroll,
            COMBINE_FN=reduce_add,
            COMPUTE_DTYPE=kernel_dtype,
            **options,
        )
        assert_bitwise(actual, expected, f"per-add rounding, merge unroll={unroll}")


@pytest.mark.parametrize("num_warps", (1, 4))
@pytest.mark.parametrize("block_size", (1, 8, 16, 32, 64, 1024, 2048, 8192))
def test_production_tile_tree_matches_pytorch_on_identical_fp32_values(num_warps, block_size):
    """Compare identical FP32 inputs across tile widths and enclosing warp counts."""
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add

    generator = torch.Generator().manual_seed(939)
    values = torch.randn((16, block_size), generator=generator)
    values *= torch.pow(2.0, torch.randint(-30, 30, values.shape, generator=generator))
    values[0].fill_(-0.0)
    if block_size >= 4:
        values[1].zero_()
        values[1, :4] = torch.tensor([1e20, 1.0, -1e20, 1.0])
    values = values.cuda()
    expected = ordered_tile_reduce(values, **_TREE_CONFIG, combine_fn=torch.add)
    actual = torch.empty(16, device="cuda", dtype=torch.float32)
    kernels.tile_reduce_probe[(16,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=_TREE_CONFIG["max_group_size"],
        num_warps=num_warps,
        enable_fp_fusion=False,
        COMBINE_FN=reduce_add,
    )
    assert_bitwise(actual, expected, "isolated production tile tree")


@pytest.mark.parametrize(
    "dtype,num_warps", ((torch.float32, 4), (torch.float16, 1), (torch.bfloat16, 8))
)
@pytest.mark.parametrize(
    "block_size,length",
    (
        (1, 1),
        (4, 17),
        (16, 48),
        (32, 128),
        (1024, 1025),
        (1024, 1031),
        (1024, 1032),
        (1024, 1033),
        (1024, 4095),
        (1024, 4096),
        (1024, 4097),
        (1024, 4104),
        (1024, 4105),
        (1024, 1536),
        (1024, 1537),
        (1024, 5121),
        (1024, 8192),
        (1024, 8193),
        (1024, 16384),
        (1024, 16385),
        (1024, 17408),
        (1024, 17409),
        (2048, 6147),
        (1024, 263169),
    ),
)
def test_serial_and_parallel_sums_share_all_rounding_points(dtype, block_size, length, num_warps):
    """Compare partials/totals across tails, pipelining and merging.

    Positive nonassociative values expose a changed addition tree. The source
    buffer already contains the values to reduce; no callback transforms them.
    Four/five and sixteen/seventeen-tile boundaries cover expanded and rolled
    loops. Tails cross group and half-tile boundaries. Parallel producers and
    the independent reference still compute complete padded tiles.
    """
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add

    generator = torch.Generator().manual_seed(415)
    values = torch.randn((4, length), generator=generator).to(device="cuda", dtype=dtype)
    values[0].fill_(1.0)
    values[1].zero_()
    values[1, 0] = 16384.0
    values[1, block_size::block_size] = 2**-10
    values[2].fill_(-0.0)
    n_tiles = (length + block_size - 1) // block_size
    expected_partials = torch.stack(
        [
            ordered_reduce_partial(
                values,
                tile,
                block_size=block_size,
                max_group_size=8,
                combine_fn=torch.add,
                identity=0.0,
                compute_dtype=torch.float32,
            )
            for tile in range(n_tiles)
        ],
        dim=-1,
    )
    expected_total = ordered_reduce_merge(
        expected_partials,
        0.0,
        combine_fn=torch.add,
        merge_unroll=1,
        compute_dtype=torch.float32,
    )
    math_options = {"enable_fp_fusion": False}
    common = dict(
        LENGTH=length,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=_TREE_CONFIG["max_group_size"],
        COMBINE_FN=reduce_add,
        IDENTITY=0.0,
        num_warps=num_warps,
        **math_options,
    )
    initials = torch.zeros(4, device="cuda")
    input_before = values.clone()
    for pipeline_stages in (None, 3):
        totals = torch.full_like(expected_total, float("nan"))
        kernels.ordered_reduce_serial_probe[(4,)](
            values, initials, totals, PIPELINE_STAGES=pipeline_stages, **common
        )
        assert_bitwise(totals, expected_total, f"serial stages={pipeline_stages}")

    storage = torch.full((4 * n_tiles + 2,), -123.0, device="cuda")
    partials = storage[1:-1].view(4, n_tiles)
    partials.fill_(float("nan"))
    kernels.ordered_reduce_partial_probe[(4, n_tiles)](
        values, partials, values.stride(0), REVERSE_TILES=False, **common
    )
    assert_bitwise(partials, expected_partials, "parallel tile results")
    partials_before = partials.clone()
    for merge_unroll in (1, 3):
        totals = torch.full_like(expected_total, float("nan"))
        kernels.ordered_reduce_merge_probe[(4,)](
            partials,
            initials,
            totals,
            N_TILES=n_tiles,
            MERGE_UNROLL=merge_unroll,
            COMBINE_FN=reduce_add,
            num_warps=num_warps,
            **math_options,
        )
        assert_bitwise(totals, expected_total, f"merge unroll={merge_unroll}")
    assert_bitwise(partials, partials_before, "merge preserves saved partials")
    assert_bitwise(values, input_before, "reduction preserves input")
    assert_bitwise(storage[[0, -1]], torch.full((2,), -123.0, device="cuda"), "scratch guards")


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("length,tile_unroll", ((1033, 1), (5121, 3), (16385, 8)))
def test_public_tile_unroll_override_preserves_the_declared_tree(operation, length, tile_unroll):
    """Explicit and automatic schedules retain every group, tail and seeded fold."""

    from rl_engine.backends.shared.triton import reduction
    from rl_engine.reference import reduction as reference

    generator = torch.Generator().manual_seed(1059)
    values = torch.randn((3, length), generator=generator, device="cpu")
    if operation == "sum":
        values[0, :4] = torch.tensor([1e20, 1.0, -1e20, 1.0])
        values[1].fill_(-0.0)
    else:
        values.fill_(1.0)
        values[0, ::2] = 2.0
        values[0, 1::2] = 0.5
        values[1, 0] = -0.0
        values[2, ::2] = -1.0
    expected = getattr(reference, operation)(values, from_memory=True)
    values = values.cuda()
    before = values.clone()
    actual = torch.empty(3, device="cuda")
    for warps in (1, 4):
        for pipeline in (None, 3):
            kernels.configured_vector_probe[(3,)](
                values,
                actual,
                values.stride(0),
                LENGTH=length,
                COMPUTE_DTYPE=tl.float32,
                BLOCK_SIZE=1024,
                MAX_GROUP_SIZE=8,
                REDUCE_FN=getattr(reduction, operation),
                TILE_UNROLL=tile_unroll,
                PIPELINE_STAGES=pipeline,
                num_warps=warps,
                enable_fp_fusion=False,
            )
            assert_bitwise(actual.cpu(), expected, f"{operation}: unroll={tile_unroll}")
    assert_bitwise(values, before, "tile-loop expansion preserves inputs")


@pytest.mark.parametrize("tile_unroll", (0, -1))
def test_public_tile_unroll_rejects_nonpositive_factors(tile_unroll):
    from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure

    from rl_engine.backends.shared.triton import reduction

    values = torch.ones((1, 1033), device="cuda")
    actual = torch.empty(1, device="cuda")
    with pytest.raises(CompilationError) as error_info:
        kernels.configured_vector_probe[(1,)](
            values,
            actual,
            values.stride(0),
            LENGTH=1033,
            COMPUTE_DTYPE=tl.float32,
            BLOCK_SIZE=1024,
            MAX_GROUP_SIZE=8,
            REDUCE_FN=reduction.sum,
            TILE_UNROLL=tile_unroll,
            PIPELINE_STAGES=None,
        )

    error = error_info.value
    while error.__cause__ is not None:
        error = error.__cause__
    assert isinstance(error, CompileTimeAssertionFailure)
    assert error.error_message == "tile unroll must be a positive integer"


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("operation,identity", (("sum", 0.0), ("product", 1.0)))
@pytest.mark.parametrize(
    "block_size,group_size,length",
    (
        (64, 1, 65),
        (64, 2, 66),
        (64, 8, 7),
        (64, 8, 8),
        (64, 8, 9),
        (64, 8, 15),
        (64, 8, 16),
        (64, 8, 17),
        (64, 8, 31),
        (64, 8, 32),
        (64, 8, 33),
        (64, 1, 79),
        (64, 1, 96),
        (64, 1, 97),
        (64, 8, 264),
        (64, 8, 265),
        (128, 32, 159),
        (128, 32, 160),
        (128, 32, 161),
    ),
)
def test_compact_tail_keeps_padding_tree_and_seed(
    dtype, operation, identity, block_size, group_size, length
):
    """A compact tail must match the full padded tree, including signed-zero combines.

    The oracle materializes every padding group; it does not use the optimized
    compact-tail path. Group and half-tile boundaries select both paths. Nonzero and
    negative-zero seeds must enter once, after the same per-tile computations.
    Caller-owned combiners ensure this is not an addition-only shortcut.
    Padding uses the normal sum/product identities; signed zeros come from
    input tensors and runtime seeds, not Triton negative-zero literals.
    """

    values = torch.ones((4, length), dtype=dtype, device="cuda")
    values[1].fill_(-0.0)
    if operation == "sum":
        combine, reference_combine = kernels.caller_add, torch.add
        values[0, :3] = torch.tensor([2.0 / torch.finfo(dtype).eps, 1.0, 1.0], device="cuda")
        values[2, 1::2] = -1.0
    else:
        combine, reference_combine = kernels.caller_multiply, torch.mul
        values[0, ::2] = 2.0
        values[0, 1::2] = 0.5
        values[2, 1::2] = -1.0
    initials = torch.tensor([3.0, -0.0, -2.0, 0.5], dtype=dtype, device="cuda")
    expected = ordered_reduce_serial(
        values,
        initials,
        block_size=block_size,
        max_group_size=group_size,
        combine_fn=reference_combine,
        identity=identity,
        compute_dtype=dtype,
    )
    actual = torch.empty_like(expected)
    for warps in (1, 4):
        for pipeline in (None, 3):
            kernels.ordered_reduce_serial_probe[(4,)](
                values,
                initials,
                actual,
                LENGTH=length,
                BLOCK_SIZE=block_size,
                MAX_GROUP_SIZE=group_size,
                COMBINE_FN=combine,
                IDENTITY=identity,
                COMPUTE_DTYPE=tl.float32 if dtype == torch.float32 else tl.float64,
                PIPELINE_STAGES=pipeline,
                num_warps=warps,
                enable_fp_fusion=False,
            )
            assert_bitwise(actual, expected, f"{operation}: tail group, w{warps}/p{pipeline}")


@pytest.mark.parametrize("dtype", (*_DTYPES, torch.float64, torch.int16, torch.int64))
@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("num_warps", (1, 4))
def test_two_entry_reduction_matches_custom_combiner_and_reference(dtype, operation, num_warps):
    """Built-in and caller-owned combiners retain the same typed tree.

    Compare raw bits to an independently expressed PyTorch tree. Negative
    zero detects added seeds, cancellation spans both sequential groups and tree
    levels, and int64 values must never pass through floating point.
    """
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
        reduce_add,
        reduce_multiply,
    )

    generator = torch.Generator().manual_seed(939)
    if operation == "sum":
        values = torch.randn((4, 1024), generator=generator).to(dtype)
        combine, custom, reference = reduce_add, kernels.caller_add, torch.add
        large = 2.0 / torch.finfo(dtype).eps if dtype.is_floating_point else 8192
        values[1].zero_()
        values[1, [0, 8, 128, 256, 512, 768]] = torch.tensor(
            [large, 1, -large, 1, 1, 1], dtype=dtype
        )
        if dtype == torch.int64:
            values[2, 0] = 2**53 + 1
    else:
        values = (1 + torch.randn((4, 1024), generator=generator) * 0.01).to(dtype)
        combine, custom, reference = reduce_multiply, kernels.caller_multiply, torch.mul
        if not dtype.is_floating_point:
            values.fill_(1)
            values[:, 0] = -1
        elif dtype == torch.float64:
            values[2, 0] = 1 + 2**-35
    values[0].fill_(-0.0)
    expected = ordered_tile_reduce(values, max_group_size=8, combine_fn=reference)
    inputs = values.cuda()
    for combiner in (combine, custom):
        actual = torch.empty((4,), dtype=dtype, device="cuda")
        kernels.tile_reduce_probe[(4,)](
            inputs,
            actual,
            BLOCK_SIZE=1024,
            MAX_GROUP_SIZE=8,
            COMBINE_FN=combiner,
            num_warps=num_warps,
            enable_fp_fusion=False,
        )
        assert_bitwise(actual.cpu(), expected, f"two-entry pairs: {operation}/{dtype}")
    assert_bitwise(inputs.cpu(), values, "pair reduction preserves inputs")


@pytest.mark.parametrize("dtype", (*_DTYPES, torch.float64))
@pytest.mark.parametrize("block_size,num_warps", ((64, 1), (1024, 4), (2048, 8)))
def test_group_tree_pairs_lower_and_upper_halves_first(dtype, block_size, num_warps):
    """Pin the pair order with cancellation across halves and a sequential group fold.

    The first row cancels the large terms before adding the ones, yielding 2.
    In the second row all four terms share a sequential group, yielding 1.
    """
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add

    large = 8 / torch.finfo(dtype).eps
    values = torch.zeros((2, block_size), dtype=dtype, device="cuda")
    pattern = torch.tensor([large, 1, -large, 1], dtype=dtype, device="cuda")
    values[0, [0, 8, block_size // 2, block_size // 2 + 8]] = pattern
    values[1, :4] = pattern
    actual = torch.empty((2,), dtype=dtype, device="cuda")
    kernels.tile_reduce_probe[(2,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=8,
        COMBINE_FN=reduce_add,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    assert_bitwise(actual, torch.tensor([2, 1], dtype=dtype, device="cuda"))
