# SPDX-License-Identifier: Apache-2.0
"""Fixed-order reduction conformance, independent of model/operator tests."""

import pytest
import torch

from rl_engine.reference.reduction.ordered_reduce import ordered_reduce, ordered_tile_reduce
from tests.helpers.tensor_checks import assert_bitwise

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_TILE_WIDTHS = tuple(1 << exponent for exponent in range(14))
_TREE_CONFIG = {"max_group_size": 8, "section_count": 4}
"""The established 8-value/four-section tree; passed explicitly to each reference."""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("merge_unroll", (1, 2, 3, 4, 8, 16))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("operation", ("sum", "product"))
def test_merge_unroll_preserves_left_fold_and_handles_tails(merge_unroll, dtype, operation):
    """Use order-sensitive partials, runtime seeds and empty/short/uneven merges.

    The oracle folds the given partials directly with PyTorch, independently of
    tile reduction. Cancellation detects reassociation; negative zero detects
    extra additions used to pad a tail. A nonidentity seed must enter only once.
    """
    import triton.language as tl

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
        reduce_add,
        reduce_multiply,
    )
    from tests.helpers.ordered_reduce_kernels import ordered_reduce_merge_probe

    kernel_dtype = tl.float32 if dtype == torch.float32 else tl.float64
    if operation == "sum":
        combine, reference_combine = reduce_add, torch.add
        large = 2.0 / torch.finfo(dtype).eps
        patterns = torch.tensor(
            [[large, 1.0, -large, 1.0], [1.0] * 4, [-0.0] * 4, [2.0, 3.0, 5.0, 7.0]],
            dtype=dtype,
        )
        seeds = torch.tensor([0.0, large, -0.0, 7.0], dtype=dtype)
    else:
        combine, reference_combine = reduce_multiply, torch.mul
        patterns = torch.tensor(
            [[2.0, 0.5, 4.0, 0.25], [1.0] * 4, [2.0, 0.5, -2.0, -0.5], [0.5] * 4],
            dtype=dtype,
        )
        seeds = torch.tensor([3.0, -2.0, -0.0, 7.0], dtype=dtype)
    initials = seeds.cuda()

    # Include only-tail, exact groups and groups plus a tail for every factor.
    tile_counts = sorted(
        {0, 1, merge_unroll - 1, merge_unroll, merge_unroll + 1, 2 * merge_unroll + 1}
    )
    for n_tiles in tile_counts:
        cpu_partials = patterns.repeat(1, (n_tiles + 3) // 4)[:, :n_tiles].contiguous()
        expected = seeds.clone()
        for tile in range(n_tiles):
            expected = reference_combine(expected, cpu_partials[:, tile])
        partials = cpu_partials.cuda()
        actual = torch.full((4,), float("nan"), device="cuda", dtype=dtype)
        ordered_reduce_merge_probe[(4,)](
            partials,
            initials,
            actual,
            N_TILES=n_tiles,
            MERGE_UNROLL=merge_unroll,
            COMBINE_FN=combine,
            COMPUTE_DTYPE=kernel_dtype,
            num_warps=4,
            enable_fp_fusion=False,
        )
        assert_bitwise(
            actual.cpu(), expected, f"{operation}: unroll={merge_unroll}, tiles={n_tiles}"
        )
        assert_bitwise(partials.cpu(), cpu_partials, "merge preserves partials")
    assert_bitwise(initials.cpu(), seeds, "merge preserves runtime seeds")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("merge_unroll", (0, -1))
def test_merge_rejects_nonpositive_unroll(merge_unroll):
    """Reject invalid loop steps before generating a merge kernel."""
    from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import ordered_reduce_merge_probe

    partials = torch.ones((1, 1), device="cuda")
    initials = torch.zeros(1, device="cuda")
    actual = torch.empty_like(initials)
    with pytest.raises(CompilationError) as error_info:
        ordered_reduce_merge_probe[(1,)](
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize(
    "dtype,operation",
    [
        (dtype, op)
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        for op in ("sum", "product")
    ]
    + [(torch.bool, "xor")],
)
@pytest.mark.parametrize("length", (17, 1025))
def test_integer_and_custom_boolean_reductions_across_schedules(dtype, operation, length):
    """Exercise typed seeds, padded tails and partial storage outside floating point."""
    import triton.language as tl

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
        reduce_add,
        reduce_multiply,
    )
    from tests.helpers.ordered_reduce_kernels import (
        boolean_xor,
        ordered_reduce_merge_probe,
        ordered_reduce_partial_probe,
        ordered_reduce_serial_probe,
    )

    kernel_dtype = {
        torch.uint8: tl.uint8,
        torch.int8: tl.int8,
        torch.int16: tl.int16,
        torch.int32: tl.int32,
        torch.int64: tl.int64,
        torch.bool: tl.int1,
    }[dtype]
    values = torch.ones((2, length), dtype=dtype, device="cuda")
    initials = torch.tensor([3, 7], dtype=dtype, device="cuda")
    if operation == "xor":
        combine, reference_combine, identity = boolean_xor, torch.bitwise_xor, False
        values[:, 1] = False
        initials = torch.tensor([True, False], device="cuda")
        math_expected = ((values.long().sum(-1) % 2) != 0) ^ initials
    else:
        values[:, :3] = torch.tensor([2, 3, 5], dtype=dtype, device="cuda")
        if dtype == torch.int64:
            values[0, 0] = 2**53 + 1
            initials[1] = 2**53 + 1
        if operation == "sum":
            combine, reference_combine, identity = reduce_add, torch.add, 0
            math_expected = (initials.long() + values.long().sum(-1)).to(dtype)
        else:
            combine, reference_combine, identity = reduce_multiply, torch.mul, 1
            math_expected = (initials.long() * values.long().prod(-1)).to(dtype)
    expected = ordered_reduce(
        values,
        block_size=32,
        **_TREE_CONFIG,
        combine_fn=reference_combine,
        identity=identity,
        initial=initials,
        compute_dtype=dtype,
    )
    assert_bitwise(expected.total, math_expected, "independent integer/boolean oracle")
    common = dict(
        LENGTH=length,
        BLOCK_SIZE=32,
        MAX_GROUP_SIZE=8,
        SECTION_COUNT=4,
        COMBINE_FN=combine,
        IDENTITY=identity,
        COMPUTE_DTYPE=kernel_dtype,
        num_warps=4,
    )
    actual = torch.empty_like(expected.total)
    for stages in (None, 3):
        ordered_reduce_serial_probe[(2,)](
            values, initials, actual, PIPELINE_STAGES=stages, **common
        )
        assert_bitwise(actual, expected.total, f"{operation}: serial stages={stages}")
    partials = torch.empty_like(expected.partials)
    n_tiles = partials.shape[-1]
    ordered_reduce_partial_probe[(2, n_tiles)](values, partials, **common)
    assert_bitwise(partials, expected.partials, "typed partials")
    for unroll in (1, 3, 4):
        ordered_reduce_merge_probe[(2,)](
            partials,
            initials,
            actual,
            N_TILES=n_tiles,
            MERGE_UNROLL=unroll,
            COMBINE_FN=combine,
            COMPUTE_DTYPE=kernel_dtype,
            num_warps=4,
        )
        assert_bitwise(actual, expected.total, f"{operation}: merge unroll={unroll}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("block_size,length", ((1, 5), (4, 3), (4, 17), (32, 65), (1024, 1025)))
@pytest.mark.parametrize("operation,section_count", (("sum", 4), ("product", 4), ("product", 8)))
@pytest.mark.parametrize("compute_dtype", (*_DTYPES, torch.float64))
def test_operations_with_nonidentity_runtime_seeds(
    dtype, block_size, length, operation, section_count, compute_dtype
):
    """Padding identity and final seed must agree across serial/parallel schedules."""
    import triton.language as tl

    kernel_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
        torch.float64: tl.float64,
    }[compute_dtype]
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
        reduce_add,
        reduce_multiply,
    )
    from tests.helpers.ordered_reduce_kernels import (
        ordered_reduce_merge_probe,
        ordered_reduce_partial_probe,
        ordered_reduce_serial_probe,
    )

    columns = torch.arange(length, device="cuda")
    values = (columns % 8 + 2).to(dtype).expand(4, -1).clone()
    if operation == "sum":
        combine, reference_combine, identity = reduce_add, torch.add, 0.0
        initials = torch.tensor([10.0, -3.0, 0.0, 2.0], device="cuda")
    else:
        combine, reference_combine, identity = reduce_multiply, torch.mul, 1.0
        values = torch.where(columns % 2 == 0, 2.0, 0.5).to(dtype).expand(4, -1).clone()
        initials = torch.tensor([3.0, 2.0, 0.5, -2.0], device="cuda")
    expected = ordered_reduce(
        values,
        block_size=block_size,
        combine_fn=reference_combine,
        identity=identity,
        initial=initials,
        max_group_size=8,
        section_count=section_count,
        compute_dtype=compute_dtype,
    )
    common = dict(
        LENGTH=length,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=8,
        SECTION_COUNT=section_count,
        COMBINE_FN=combine,
        IDENTITY=identity,
        COMPUTE_DTYPE=kernel_dtype,
        num_warps=4,
        enable_fp_fusion=False,
    )
    initials_before = initials.clone()
    for stages in (None, 3):
        actual = torch.empty_like(expected.total)
        ordered_reduce_serial_probe[(4,)](
            values, initials, actual, PIPELINE_STAGES=stages, **common
        )
        assert_bitwise(actual, expected.total, f"{operation}: serial stages={stages}")
    n_tiles = (length + block_size - 1) // block_size
    partials = torch.empty_like(expected.partials)
    ordered_reduce_partial_probe[(4, n_tiles)](values, partials, **common)
    assert_bitwise(partials, expected.partials, f"{operation}: unseeded partials")
    for unroll in (1, 3, 4):
        actual = torch.empty_like(expected.total)
        ordered_reduce_merge_probe[(4,)](
            partials,
            initials,
            actual,
            N_TILES=n_tiles,
            COMBINE_FN=combine,
            MERGE_UNROLL=unroll,
            COMPUTE_DTYPE=kernel_dtype,
            num_warps=4,
            enable_fp_fusion=False,
        )
        assert_bitwise(actual, expected.total, f"{operation}: merge unroll={unroll}")
    assert_bitwise(initials, initials_before, "caller-owned seed unchanged")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize(
    "compute_dtype,bits",
    ((torch.float16, 11), (torch.bfloat16, 8), (torch.float32, 24), (torch.float64, 53)),
)
def test_compute_dtype_rounds_each_add_in_tiles_and_merge(compute_dtype, bits):
    """Both schedules must round at each addition, not just when storing the output."""
    import triton.language as tl

    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import (
        ordered_reduce_merge_probe,
        ordered_reduce_partial_probe,
        ordered_reduce_serial_probe,
    )

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
        SECTION_COUNT=4,
        COMBINE_FN=reduce_add,
        IDENTITY=0.0,
        COMPUTE_DTYPE=kernel_dtype,
    )
    for stages in (None, 3):
        ordered_reduce_serial_probe[(2,)](
            values,
            initials,
            actual,
            PIPELINE_STAGES=stages,
            **tile_options,
            **options,
        )
        assert_bitwise(actual, expected, f"per-add rounding, stages={stages}")
    ordered_reduce_partial_probe[(2, 3)](values, partials, **tile_options, **options)
    for unroll in (1, 2, 3, 4, 8):
        ordered_reduce_merge_probe[(2,)](
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("block_size", (4, 32, 256))
@pytest.mark.parametrize("max_group_size", (1, 4, 16, 64))
@pytest.mark.parametrize("section_count", (1, 2, 4, 8, 16, 128))
@pytest.mark.parametrize("num_warps", (4, 8))
def test_explicit_group_and_section_sizes_match_reference(
    block_size, max_group_size, section_count, num_warps
):
    """Tree parameters agree across backends even when sections exceed groups."""
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import tile_reduce_probe

    generator = torch.Generator().manual_seed(415)
    values = torch.randn((3, block_size), generator=generator).cuda()
    values[0].fill_(-0.0)
    values[1].zero_()
    values[1, :4] = torch.tensor([1e20, 1.0, -1e20, 1.0], device="cuda")
    expected = ordered_tile_reduce(
        values, max_group_size=max_group_size, section_count=section_count, combine_fn=torch.add
    )
    actual = torch.empty_like(expected)
    tile_reduce_probe[(3,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=max_group_size,
        SECTION_COUNT=section_count,
        num_warps=num_warps,
        enable_fp_fusion=False,
        COMBINE_FN=reduce_add,
    )
    assert_bitwise(actual, expected, "explicit arithmetic parameters")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("num_warps", (4, 8))
@pytest.mark.parametrize("seed", (415, 438, 939))
@pytest.mark.parametrize("block_size", _TILE_WIDTHS)
def test_production_tile_tree_matches_pytorch_on_identical_fp32_values(num_warps, seed, block_size):
    """Isolate reduction from exp/tanh; changing scheduling cannot change pairs."""
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import tile_reduce_probe

    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((16, block_size), generator=generator)
    values *= torch.pow(2.0, torch.randint(-30, 30, values.shape, generator=generator))
    values[0].fill_(-0.0)
    if block_size >= 4:
        values[1].zero_()
        values[1, :4] = torch.tensor([1e20, 1.0, -1e20, 1.0])
    values = values.cuda()
    expected = ordered_tile_reduce(values, **_TREE_CONFIG, combine_fn=torch.add)
    actual = torch.empty(16, device="cuda", dtype=torch.float32)
    tile_reduce_probe[(16,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=_TREE_CONFIG["max_group_size"],
        SECTION_COUNT=_TREE_CONFIG["section_count"],
        num_warps=num_warps,
        enable_fp_fusion=False,
        COMBINE_FN=reduce_add,
    )
    assert_bitwise(actual, expected, "isolated production tile tree")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float64))
@pytest.mark.parametrize("block_size", (1, 2, 4, 8, 16, 32, 1024, 2048))
@pytest.mark.parametrize("num_warps", (4, 8))
@pytest.mark.parametrize("section_count", (1, 2, 4, 8, 128))
def test_tile_tree_accepts_other_floating_dtypes(dtype, block_size, num_warps, section_count):
    """Exercise direct, sequential and two-operand reduction paths in the caller's dtype."""
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import tile_reduce_probe

    generator = torch.Generator().manual_seed(415)
    values = torch.randn((4, block_size), generator=generator).to(device="cuda", dtype=dtype)
    values[0].fill_(-0.0)
    if block_size >= 4:
        # This finite value loses a +1 in each dtype, distinguishing addition trees.
        large = 2.0 / torch.finfo(dtype).eps
        values[1].zero_()
        values[1, :4] = torch.tensor([large, 1.0, -large, 1.0], device="cuda", dtype=dtype)
    expected = ordered_tile_reduce(
        values, max_group_size=8, section_count=section_count, combine_fn=torch.add
    )
    actual = torch.empty(4, device="cuda", dtype=dtype)
    tile_reduce_probe[(4,)](
        values,
        actual,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=_TREE_CONFIG["max_group_size"],
        SECTION_COUNT=section_count,
        num_warps=num_warps,
        enable_fp_fusion=False,
        COMBINE_FN=reduce_add,
    )
    assert_bitwise(actual, expected, "caller-selected dtype and addition tree")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "block_size,length",
    ((1, 1), (4, 17), (16, 48), (32, 128), (1024, 1025), (2048, 6147), (1024, 263169)),
)
@pytest.mark.parametrize("section_count", (1, 2, 4, 8, 128))
def test_serial_and_parallel_sums_share_all_rounding_points(
    dtype, block_size, length, section_count
):
    """Compare partials/totals across tails, pipelining and merging.

    Positive nonassociative values expose a changed addition tree. The source
    buffer already contains the values to reduce; no callback transforms them.
    """
    from rl_engine.backends.shared.triton.reduction.ordered_reduce import reduce_add
    from tests.helpers.ordered_reduce_kernels import (
        ordered_reduce_merge_probe,
        ordered_reduce_partial_probe,
        ordered_reduce_serial_probe,
    )

    generator = torch.Generator().manual_seed(415)
    values = torch.randn((4, length), generator=generator).to(device="cuda", dtype=dtype)
    values[0].fill_(1.0)
    values[1].zero_()
    values[1, 0] = 16384.0
    values[1, block_size::block_size] = 2**-10
    values[2].fill_(-0.0)
    expected = ordered_reduce(
        values,
        block_size=block_size,
        max_group_size=8,
        section_count=section_count,
        combine_fn=torch.add,
        identity=0.0,
        initial=0.0,
        compute_dtype=torch.float32,
    )
    n_tiles = (length + block_size - 1) // block_size
    math_options = {"enable_fp_fusion": False}
    common = dict(
        LENGTH=length,
        BLOCK_SIZE=block_size,
        MAX_GROUP_SIZE=_TREE_CONFIG["max_group_size"],
        SECTION_COUNT=section_count,
        COMBINE_FN=reduce_add,
        IDENTITY=0.0,
        num_warps=4,
        **math_options,
    )
    initials = torch.zeros(4, device="cuda")
    input_before = values.clone()
    for pipeline_stages in (None, 3):
        totals = torch.full_like(expected.total, float("nan"))
        ordered_reduce_serial_probe[(4,)](
            values, initials, totals, PIPELINE_STAGES=pipeline_stages, **common
        )
        assert_bitwise(totals, expected.total, f"serial stages={pipeline_stages}")

    storage = torch.full((4 * n_tiles + 2,), -123.0, device="cuda")
    partials = storage[1:-1].view(4, n_tiles)
    partials.fill_(float("nan"))
    ordered_reduce_partial_probe[(4, n_tiles)](values, partials, **common)
    assert_bitwise(partials, expected.partials, "parallel tile results")
    partials_before = partials.clone()
    for merge_unroll in (1, 2, 3, 4, 8):
        totals = torch.full_like(expected.total, float("nan"))
        ordered_reduce_merge_probe[(4,)](
            partials,
            initials,
            totals,
            N_TILES=n_tiles,
            MERGE_UNROLL=merge_unroll,
            COMBINE_FN=reduce_add,
            num_warps=4,
            **math_options,
        )
        assert_bitwise(totals, expected.total, f"merge unroll={merge_unroll}")
    assert_bitwise(partials, partials_before, "merge preserves saved partials")
    assert_bitwise(values, input_before, "reduction preserves input")
    assert_bitwise(storage[[0, -1]], torch.full((2,), -123.0, device="cuda"), "scratch guards")
