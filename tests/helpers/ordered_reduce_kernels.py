# SPDX-License-Identifier: Apache-2.0
"""Launch adapters for device-local reduction tests; no operator dependencies."""

import triton
import triton.language as tl

from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
    ordered_reduce_merge,
    ordered_reduce_partial,
    ordered_reduce_serial,
    ordered_tile_reduce,
)


@triton.jit
def tile_reduce_probe(
    values_ptr,
    sums_ptr,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
    COMBINE_FN: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    values = tl.load(values_ptr + tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    tile_sum = ordered_tile_reduce(
        values, MAX_GROUP_SIZE=MAX_GROUP_SIZE, SECTION_COUNT=SECTION_COUNT, COMBINE_FN=COMBINE_FN
    )
    tl.store(sums_ptr + tile, tile_sum)


@triton.jit
def configured_tile_probe(
    values_ptr,
    sums_ptr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    REDUCE_FN: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
):
    """Exercise the tile helper used by fused operators with an explicit tree."""
    row = tl.program_id(0).to(tl.int64)
    values = tl.load(values_ptr + row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    total = REDUCE_FN(
        values,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
    )
    tl.store(sums_ptr + row, total)


@triton.jit
def inferred_tile_dtype_probe(
    input_ptr,
    default_ptr,
    none_ptr,
    explicit_ptr,
    BLOCK_SIZE: tl.constexpr,
    REDUCE_FN: tl.constexpr,
):
    """Check inferred tile precision before output stores can cast the result."""
    row = tl.program_id(0).to(tl.int64)
    values = tl.load(input_ptr + row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    default = REDUCE_FN(values)
    inferred = REDUCE_FN(values, COMPUTE_DTYPE=None)
    explicit = REDUCE_FN(values, COMPUTE_DTYPE=values.dtype)
    tl.static_assert(default.dtype == values.dtype, "default dtype must match values")
    tl.static_assert(inferred.dtype == values.dtype, "None dtype must match values")
    tl.store(default_ptr + row, default)
    tl.store(none_ptr + row, inferred)
    tl.store(explicit_ptr + row, explicit)


@triton.jit
def boolean_xor(left, right):
    """Caller-defined boolean combiner; false is its neutral padding identity."""
    return left ^ right


@triton.jit
def ordered_reduce_serial_probe(
    input_ptr,
    initials_ptr,
    totals_ptr,
    LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = tl.float32,
):
    """Expose the full-vector reduction with a row-dependent initial value."""
    row = tl.program_id(0).to(tl.int64)
    total = ordered_reduce_serial(
        input_ptr + row * LENGTH,
        LENGTH,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
        PIPELINE_STAGES=PIPELINE_STAGES,
        COMBINE_FN=COMBINE_FN,
        IDENTITY=IDENTITY,
        initial=tl.load(initials_ptr + row),
        COMPUTE_DTYPE=COMPUTE_DTYPE,
    )
    tl.store(totals_ptr + row, total)


@triton.jit
def ordered_reduce_partial_probe(
    input_ptr,
    partials_ptr,
    LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = tl.float32,
):
    """Expose the parallel schedule's independently computed partials."""
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    n_tiles: tl.constexpr = (LENGTH + BLOCK_SIZE - 1) // BLOCK_SIZE
    partial = ordered_reduce_partial(
        input_ptr + row * LENGTH,
        LENGTH,
        tile,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
        COMBINE_FN=COMBINE_FN,
        IDENTITY=IDENTITY,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
    )
    tl.store(partials_ptr + row * n_tiles + tile, partial)


@triton.jit
def ordered_reduce_merge_probe(
    partials_ptr,
    initials_ptr,
    totals_ptr,
    N_TILES: tl.constexpr,
    MERGE_UNROLL: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = tl.float32,
):
    """Expose the ordered merge without any subsequent logarithm."""
    row = tl.program_id(0).to(tl.int64)
    total = ordered_reduce_merge(
        partials_ptr + row * N_TILES,
        N_TILES,
        MERGE_UNROLL=MERGE_UNROLL,
        COMBINE_FN=COMBINE_FN,
        initial=tl.load(initials_ptr + row),
        COMPUTE_DTYPE=COMPUTE_DTYPE,
    )
    tl.store(totals_ptr + row, total)


@triton.jit
def ordered_reduce_constexpr_seed_probe(
    partials_ptr,
    totals_ptr,
    N_TILES: tl.constexpr,
    INITIAL: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    MERGE_UNROLL: tl.constexpr,
):
    """Check that an embedded merge materializes a Python seed in the requested dtype."""
    row = tl.program_id(0).to(tl.int64)
    total = ordered_reduce_merge(
        partials_ptr + row * N_TILES,
        N_TILES,
        initial=INITIAL,
        COMBINE_FN=COMBINE_FN,
        MERGE_UNROLL=MERGE_UNROLL,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
    )
    tl.store(totals_ptr + row, total)


@triton.jit
def configured_vector_probe(
    input_ptr,
    output_ptr,
    LENGTH: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    REDUCE_FN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """Exercise a public vector helper with explicit arithmetic and loop scheduling."""
    row = tl.program_id(0).to(tl.int64)
    total = REDUCE_FN(
        input_ptr + row * LENGTH,
        LENGTH,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
        PIPELINE_STAGES=PIPELINE_STAGES,
    )
    tl.store(output_ptr + row, total)


@triton.jit
def inferred_vector_dtype_probe(
    input_ptr,
    default_ptr,
    none_ptr,
    explicit_ptr,
    LENGTH: tl.constexpr,
    REDUCE_FN: tl.constexpr,
):
    """Compare omitted, None and explicit input-dtype calls before output storage."""
    row = tl.program_id(0).to(tl.int64)
    vector_ptr = input_ptr + row * LENGTH
    default = REDUCE_FN(vector_ptr, LENGTH)
    inferred = REDUCE_FN(vector_ptr, LENGTH, COMPUTE_DTYPE=None)
    explicit = REDUCE_FN(vector_ptr, LENGTH, COMPUTE_DTYPE=input_ptr.dtype.element_ty)
    tl.static_assert(default.dtype == input_ptr.dtype.element_ty, "default dtype must match input")
    tl.static_assert(inferred.dtype == input_ptr.dtype.element_ty, "None dtype must match input")
    tl.store(default_ptr + row, default)
    tl.store(none_ptr + row, inferred)
    tl.store(explicit_ptr + row, explicit)


@triton.jit
def public_partials_probe(
    partials_ptr,
    default_ptr,
    none_ptr,
    explicit_ptr,
    N_TILES: tl.constexpr,
    REDUCE_FN: tl.constexpr,
    MERGE_UNROLL: tl.constexpr,
):
    """Check simple partial calls and optional overrides without implicit output casts."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = partials_ptr + row * N_TILES
    default = REDUCE_FN(row_ptr, N_TILES)
    inferred = REDUCE_FN(row_ptr, N_TILES, COMPUTE_DTYPE=None, MERGE_UNROLL=MERGE_UNROLL)
    explicit = REDUCE_FN(
        row_ptr,
        N_TILES,
        COMPUTE_DTYPE=partials_ptr.dtype.element_ty,
        MERGE_UNROLL=MERGE_UNROLL,
    )
    tl.static_assert(default.dtype == partials_ptr.dtype.element_ty, "default must retain dtype")
    tl.static_assert(inferred.dtype == partials_ptr.dtype.element_ty, "None must retain dtype")
    tl.static_assert(explicit.dtype == partials_ptr.dtype.element_ty, "explicit must retain dtype")
    tl.store(default_ptr + row, default)
    tl.store(none_ptr + row, inferred)
    tl.store(explicit_ptr + row, explicit)


@triton.jit
def composed_vector_probe(
    input_ptr,
    output_ptr,
    LENGTH: tl.constexpr,
    N_ROWS: tl.constexpr,
    REDUCE_FN: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Call an exported vector helper on a caller-selected row, then keep computing."""
    row = tl.program_id(0).to(tl.int64)
    source_row = N_ROWS - 1 - row
    total = REDUCE_FN(input_ptr + source_row * LENGTH, LENGTH, COMPUTE_DTYPE=COMPUTE_DTYPE)
    tl.store(output_ptr + row, total + 3)


@triton.jit
def composed_tile_probe(
    input_ptr,
    output_ptr,
    LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REDUCE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Map a loaded tile, mask padding, reduce it, then use the returned scalar."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + row * LENGTH + cols, mask=cols < LENGTH, other=0.0)
    mapped = values.to(COMPUTE_DTYPE) * 2.0 + 1.0
    mapped = tl.where(cols < LENGTH, mapped, IDENTITY)
    total = REDUCE_FN(mapped, COMPUTE_DTYPE=COMPUTE_DTYPE)
    tl.store(output_ptr + row, total + 3.0)
