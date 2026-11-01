# SPDX-License-Identifier: Apache-2.0
"""Small JIT callers shared by this backend's GPU and compiler tests.

Imported lazily so CPU reference tests never need a Triton installation. These
adapters exercise the real helpers; they contain no second reduction algorithm.
"""

import triton
import triton.language as tl

from rl_engine.backends.shared.triton.reduction import reduce as public_reduce
from rl_engine.backends.shared.triton.reduction import reduce_partials as public_reduce_partials
from rl_engine.backends.shared.triton.reduction.ordered_reduce import (
    _split_group_column,
    ordered_reduce_merge,
    ordered_reduce_partial,
    ordered_reduce_serial,
    ordered_tile_reduce,
)


@triton.jit
def group_column_probe(values_ptr, output_ptr, row_stride, GROUP_SIZE: tl.constexpr):
    """Copy every group column through the internal extraction helper."""
    rows = tl.arange(0, 8)
    cols = tl.arange(0, GROUP_SIZE)
    values = tl.load(values_ptr + rows[:, None] * row_stride + cols[None, :])
    for column in tl.static_range(GROUP_SIZE):
        selected = _split_group_column(values, COLUMN=column)
        tl.static_assert(selected.dtype == values.dtype, "extraction must not convert values")
        tl.store(output_ptr + rows * GROUP_SIZE + column, selected)


@triton.jit
def tile_reduce_probe(
    values_ptr,
    sums_ptr,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    COMBINE_FN: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    values = tl.load(values_ptr + tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    tile_sum = ordered_tile_reduce(values, MAX_GROUP_SIZE=MAX_GROUP_SIZE, COMBINE_FN=COMBINE_FN)
    tl.store(sums_ptr + tile, tile_sum)


@triton.jit
def configured_tile_probe(
    values_ptr,
    sums_ptr,
    BLOCK_SIZE: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    REDUCE_FN: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
):
    """Exercise the tile helper used by fused operators with an explicit tree."""
    row = tl.program_id(0).to(tl.int64)
    values = tl.load(values_ptr + row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    total = REDUCE_FN(
        values,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
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
    """Caller-defined XOR for boolean/integer values; false/zero is neutral."""
    return left ^ right


@triton.jit
def caller_add(left, right):
    """A distinct caller-owned JIT function exercises generic callback forwarding."""
    return left + right


@triton.jit
def caller_multiply(left, right):
    """Match multiplication while exercising generic callback forwarding."""
    return left * right


@triton.jit
def ordered_reduce_serial_probe(
    input_ptr,
    initials_ptr,
    totals_ptr,
    LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = tl.float32,
    TILE_UNROLL: tl.constexpr = None,
):
    """Expose the full-vector reduction with a row-dependent initial value."""
    row = tl.program_id(0).to(tl.int64)
    total = ordered_reduce_serial(
        input_ptr + row * LENGTH,
        LENGTH,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        PIPELINE_STAGES=PIPELINE_STAGES,
        TILE_UNROLL=TILE_UNROLL,
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
    row_stride,
    LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
    REVERSE_TILES: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = tl.float32,
):
    """Write fixed tile slots from contiguous or padded rows, in either producer order."""
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    n_tiles: tl.constexpr = (LENGTH + BLOCK_SIZE - 1) // BLOCK_SIZE
    if REVERSE_TILES:
        tile = n_tiles - 1 - tile
    partial = ordered_reduce_partial(
        input_ptr + row * row_stride,
        LENGTH,
        tile,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
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
    """Merge fixed row slots with a loaded seed, without extra output arithmetic."""
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
    row_stride,
    LENGTH: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    REDUCE_FN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    TILE_UNROLL: tl.constexpr,
):
    """Exercise a public vector helper with explicit arithmetic and loop scheduling."""
    row = tl.program_id(0).to(tl.int64)
    total = REDUCE_FN(
        input_ptr + row * row_stride,
        LENGTH,
        COMPUTE_DTYPE=COMPUTE_DTYPE,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        PIPELINE_STAGES=PIPELINE_STAGES,
        TILE_UNROLL=TILE_UNROLL,
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


@triton.jit
def input_dispatch_probe(
    input_ptr,
    loaded_output_ptr,
    memory_output_ptr,
    WIDTH: tl.constexpr,
    LENGTH: tl.constexpr,
    REDUCE_FN: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    POINTER_BLOCK_SIZE: tl.constexpr,
):
    """One public JIT function receives loaded values and a pointer in the same kernel."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = input_ptr + row * WIDTH
    values = tl.load(row_ptr + tl.arange(0, WIDTH))
    loaded = REDUCE_FN(
        values, length=LENGTH, COMPUTE_DTYPE=COMPUTE_DTYPE, BLOCK_SIZE=POINTER_BLOCK_SIZE
    )
    memory = REDUCE_FN(
        row_ptr, length=LENGTH, COMPUTE_DTYPE=COMPUTE_DTYPE, BLOCK_SIZE=POINTER_BLOCK_SIZE
    )
    tl.static_assert(loaded.dtype == loaded_output_ptr.dtype.element_ty, "loaded result dtype")
    tl.static_assert(memory.dtype == memory_output_ptr.dtype.element_ty, "pointer result dtype")
    tl.store(loaded_output_ptr + row, loaded)
    tl.store(memory_output_ptr + row, memory)


@triton.jit
def invalid_input_probe(
    input_ptr,
    output_ptr,
    MODE: tl.constexpr,
    LENGTH: tl.constexpr,
    REDUCE_FN: tl.constexpr,
):
    """Exercise public argument errors during JIT compilation, before any launch."""
    if MODE == "loaded":
        input = tl.load(input_ptr + tl.arange(0, 16))
    elif MODE == "pointer_vector":
        input = input_ptr + tl.arange(0, 16)
    else:
        input = input_ptr
    total = REDUCE_FN(input, length=LENGTH)
    tl.store(output_ptr, total)


@triton.jit
def modular_add(left, right):
    """Caller-owned combine operation, distinct from the convenience operations."""
    return (left + right) % 257


@triton.jit
def custom_input_probe(
    input_ptr,
    loaded_output_ptr,
    memory_output_ptr,
    WIDTH: tl.constexpr,
    LENGTH: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
):
    """Use a custom callback through both public input forms in the same kernel."""
    row = tl.program_id(0).to(tl.int64)
    row_ptr = input_ptr + row * WIDTH
    values = tl.load(row_ptr + tl.arange(0, WIDTH))
    loaded = public_reduce(values, length=LENGTH, COMBINE_FN=COMBINE_FN, IDENTITY=IDENTITY)
    memory = public_reduce(
        row_ptr, length=LENGTH, COMBINE_FN=COMBINE_FN, IDENTITY=IDENTITY, BLOCK_SIZE=32
    )
    tl.store(loaded_output_ptr + row, loaded)
    tl.store(memory_output_ptr + row, memory)


@triton.jit
def custom_partial_probe(
    input_ptr,
    partials_ptr,
    WIDTH: tl.constexpr,
    LENGTH: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
):
    """Parallel producers use public loaded-value reduction and canonical output slots."""
    row = tl.program_id(0).to(tl.int64)
    n_tiles: tl.constexpr = (LENGTH + 31) // 32
    tile = n_tiles - 1 - tl.program_id(1).to(tl.int64)
    columns = tile * 32 + tl.arange(0, 32)
    values = tl.load(input_ptr + row * WIDTH + columns, mask=columns < LENGTH, other=IDENTITY)
    partial = public_reduce(values, COMBINE_FN=COMBINE_FN, IDENTITY=IDENTITY)
    tl.store(partials_ptr + row * n_tiles + tile, partial)


@triton.jit
def custom_merge_probe(
    partials_ptr,
    initials_ptr,
    output_ptr,
    N_TILES: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    MERGE_UNROLL: tl.constexpr,
):
    """Expose a caller-owned operation and runtime seed through the public merge."""
    row = tl.program_id(0).to(tl.int64)
    total = public_reduce_partials(
        partials_ptr + row * N_TILES,
        N_TILES,
        COMBINE_FN=COMBINE_FN,
        initial=tl.load(initials_ptr + row),
        MERGE_UNROLL=MERGE_UNROLL,
    )
    tl.store(output_ptr + row, total)
