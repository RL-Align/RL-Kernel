# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared Gluon implementation of the fixed-order reduction arithmetic.

Adjacent groups fold left to right; group totals halve lower/upper pairs to one.
Memory vectors use fixed arithmetic tiles followed by an ascending seeded fold.
Execution batches only overlap independent tile work. They never retile the
input, insert extra seeds, or replace the final fold with a different tree.

WARP_SIZE describes the target's warp/wave width, not an arithmetic group.
Pass the compilation target's width for AMD or other layouts; the default 32
retains the measured NVIDIA H100 layout. Batch size, vectorization and the
enclosing num_warps are independent execution settings. Changing them never
changes the prescribed operand pairs. GPU correctness has been validated on H100
and A40; the reported performance measurements are from H100.

Callers own device selection, streams and math flags; use enable_fp_fusion=False
when comparing with the fixed-order reference. This module must be called from
@gluon.jit, not from ordinary @triton.jit. Importing the ordinary reduction
package does not load this optional frontend.
"""

from types import SimpleNamespace

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from rl_engine.contracts.operators.ordered_reduction import MAX_GROUP_SIZE, TILE_SIZE

__all__ = ["sum", "product", "sum_partials", "product_partials", "reduce", "reduce_partials"]


@gluon.jit
def reduce_add(left, right):
    """Add a prescribed operand pair; zero is the padding identity."""
    return left + right


@gluon.jit
def reduce_multiply(left, right):
    """Multiply a prescribed operand pair; one is the padding identity."""
    return left * right


@triton.constexpr_function
def _reduction_callback(combine_fn):
    """Unwrap a compile-time JIT callback for gl.reduce, without method binding."""
    return SimpleNamespace(combine=combine_fn)


@gluon.jit
def _group_column(grouped, COLUMN: gl.constexpr):
    """Extract one column from [tiles, groups, columns], preserving original bits."""
    TILES: gl.constexpr = grouped.shape[0]
    GROUPS: gl.constexpr = grouped.shape[1]
    COLUMNS: gl.constexpr = grouped.shape[2]
    for bit in gl.static_range(int(COLUMNS).bit_length() - 1):
        pairs = gl.reshape(grouped, (TILES, GROUPS, COLUMNS >> (bit + 1), 2))
        even, odd = gl.split(pairs)
        if COLUMN & (1 << bit):
            grouped = odd
        else:
            grouped = even
    return gl.reshape(grouped, (TILES, GROUPS))


@gluon.jit
def _reduce_tiles(
    values, MAX_GROUP_SIZE: gl.constexpr, COMBINE_FN: gl.constexpr, WARP_SIZE: gl.constexpr
):
    """Reduce each independent row of [tiles, width] using the shared tile tree.

    Put adjacent group members in the same thread's registers. Threads carry
    distinct group totals and warps distribute independent tiles. Only layout
    changes: every arithmetic pair is identical to ordered_tile_reduce in the
    default backend. Return one unseeded result per tile.
    """
    TILES: gl.constexpr = values.shape[0]
    WIDTH: gl.constexpr = values.shape[1]
    GROUP: gl.constexpr = min(WIDTH, MAX_GROUP_SIZE)
    GROUPS: gl.constexpr = WIDTH // GROUP
    grouped = gl.reshape(values, (TILES, GROUPS, GROUP))
    grouped = gl.convert_layout(
        grouped,
        gl.BlockedLayout([1, 1, GROUP], [1, WARP_SIZE, 1], [gl.num_warps(), 1, 1], [2, 1, 0]),
    )
    subtotal = _group_column(grouped, COLUMN=0)
    for column in gl.static_range(1, GROUP):
        subtotal = COMBINE_FN(subtotal, _group_column(grouped, COLUMN=column))
    for shift in gl.static_range(int(GROUPS).bit_length() - 2, -1, -1):
        pairs = gl.reshape(subtotal, (TILES, 2, 1 << shift))
        subtotal = gl.reduce(pairs, axis=1, combine_fn=_reduction_callback(COMBINE_FN).combine)
    return gl.reshape(subtotal, (TILES,))


@gluon.jit
def _fold_partials(
    partials, total, COUNT: gl.constexpr, COMBINE_FN: gl.constexpr, WARP_SIZE: gl.constexpr
):
    """Consume exactly COUNT partials in index order, carrying the existing seed.

    Replicate the small partial vector into registers before its serial fold.
    Masked slots are never combined, including their neutral values. This
    preserves seed placement, signed-zero behavior and batch-boundary order.
    """
    WIDTH: gl.constexpr = partials.shape[0]
    partials = gl.convert_layout(
        partials, gl.BlockedLayout([WIDTH], [WARP_SIZE], [gl.num_warps()], [0])
    )
    grouped = gl.reshape(partials, (1, 1, WIDTH))
    for tile in gl.static_range(COUNT):
        value = gl.reshape(_group_column(grouped, COLUMN=tile), ())
        total = COMBINE_FN(total, value)
    return total


@gluon.jit
def _load_batch(
    input_ptr,
    length: gl.constexpr,
    first_tile,
    BLOCK_SIZE: gl.constexpr,
    BATCH: gl.constexpr,
    MAX_GROUP_SIZE: gl.constexpr,
    COMBINE_FN: gl.constexpr,
    IDENTITY: gl.constexpr,
    COMPUTE_DTYPE: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    WARP_SIZE: gl.constexpr,
):
    """Load independent fixed-width tiles; never derive arithmetic width from length."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, ELEMENTS_PER_THREAD], [1, WARP_SIZE], [gl.num_warps(), 1], [1, 0]
    )
    tiles = gl.arange(0, BATCH, layout=gl.SliceLayout(1, layout))
    columns = gl.arange(0, BLOCK_SIZE, layout=gl.SliceLayout(0, layout))
    offsets = (first_tile + tiles[:, None]).to(gl.int64) * BLOCK_SIZE + columns[None, :]
    valid = offsets < length
    values = gl.load(input_ptr + offsets, valid, other=0).to(COMPUTE_DTYPE)
    values = gl.where(valid, values, gl.full((), IDENTITY, COMPUTE_DTYPE))
    return _reduce_tiles(values, MAX_GROUP_SIZE, COMBINE_FN, WARP_SIZE)


@gluon.jit
def reduce(
    input,
    COMBINE_FN: gl.constexpr,
    IDENTITY: gl.constexpr,
    length: gl.constexpr = None,
    COMPUTE_DTYPE: gl.constexpr = None,
    BLOCK_SIZE: gl.constexpr = TILE_SIZE,
    MAX_GROUP_SIZE: gl.constexpr = MAX_GROUP_SIZE,
    TILES_PER_BATCH: gl.constexpr = 32,
    ELEMENTS_PER_THREAD: gl.constexpr = 8,
    WARP_SIZE: gl.constexpr = 32,
):
    """Reduce a loaded Gluon vector or contiguous memory prefix to one scalar.

    Arithmetic arguments match shared Triton reduce: COMBINE_FN is a pure,
    shape/dtype-preserving @gluon.jit binary function; IDENTITY is its neutral
    value. The operation must be mathematically associative and commutative on
    the supported domain; subtraction/division are not substitutes. The
    callback owns arithmetic casts. Input type chooses loaded-vector or
    scalar-pointer handling. None compute precision retains input dtype.
    Length is compile-time and required for pointers. For loaded input it
    optionally masks a prefix without changing the vector width.

    Loaded values use the unseeded fixed tree. Pointer input pads fixed
    BLOCK_SIZE tiles, then folds their totals left to right from IDENTITY.
    Zero-length memory input returns IDENTITY. Tree sizes must be positive
    powers of two and stay identical to the default backend when comparing bits.

    Execution options, independent of arithmetic:
        TILES_PER_BATCH: Positive power of two, at most 64. Process at most this
            many independent tiles together; default 32. Longer vectors use
            successive batches and carry the same outer accumulator.
        ELEMENTS_PER_THREAD: Positive power of two controlling the initial
            memory layout, default 8. It does not change group membership.
        WARP_SIZE: Hardware threads per warp/wave, default 32 for compatibility
            with the measured H100 configuration. The caller must match the
            compiler target (for example, 64 for a wave64 AMD target). This is
            a hardware property, not a tuning parameter or arithmetic width.

    Enclosing kernels choose num_warps and own launches/math flags. These
    helpers do not allocate or store. They cannot be selected dynamically
    inside an ordinary @triton.jit kernel; host dispatch selects the enclosing
    Gluon kernel. Custom callbacks must use Gluon-compatible operations.
    """
    gl.static_assert(
        WARP_SIZE > 0 and (WARP_SIZE & (WARP_SIZE - 1)) == 0,
        "warp size must be a positive power of two matching the compilation target",
    )
    gl.static_assert(
        MAX_GROUP_SIZE > 0 and (MAX_GROUP_SIZE & (MAX_GROUP_SIZE - 1)) == 0,
        "maximum group size must be a positive power of two",
    )
    input_dtype: gl.constexpr = input.dtype.element_ty if input.dtype.is_ptr() else input.dtype
    compute_dtype: gl.constexpr = input_dtype if COMPUTE_DTYPE is None else COMPUTE_DTYPE
    if input.dtype.is_ptr():
        gl.static_assert(len(input.shape) == 0, "input must be a scalar pointer")
        gl.static_assert(length is not None, "pointer input requires length")
        gl.static_assert(length >= 0, "vector length must be nonnegative")
        gl.static_assert(
            BLOCK_SIZE > 0 and (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
            "block size must be a positive power of two",
        )
        gl.static_assert(
            TILES_PER_BATCH > 0
            and TILES_PER_BATCH <= 64
            and (TILES_PER_BATCH & (TILES_PER_BATCH - 1)) == 0,
            "execution batch must be a power of two no greater than 64",
        )
        gl.static_assert(
            ELEMENTS_PER_THREAD > 0 and (ELEMENTS_PER_THREAD & (ELEMENTS_PER_THREAD - 1)) == 0,
            "elements per thread must be a positive power of two",
        )
        n_tiles: gl.constexpr = triton.cdiv(length, BLOCK_SIZE)
        BATCH: gl.constexpr = min(TILES_PER_BATCH, triton.next_power_of_2(max(1, n_tiles)))
        n_full_batches: gl.constexpr = n_tiles // BATCH
        n_tail_tiles: gl.constexpr = n_tiles % BATCH
        total = gl.full((), IDENTITY, compute_dtype)
        for batch in range(n_full_batches):
            partials = _load_batch(
                input,
                length,
                batch * BATCH,
                BLOCK_SIZE,
                BATCH,
                MAX_GROUP_SIZE,
                COMBINE_FN,
                IDENTITY,
                compute_dtype,
                ELEMENTS_PER_THREAD,
                WARP_SIZE,
            )
            total = _fold_partials(partials, total, BATCH, COMBINE_FN, WARP_SIZE)
        if n_tail_tiles > 0:
            first_tail_tile: gl.constexpr = n_full_batches * BATCH
            # Keep the batch layout; fold only the remaining valid tile results below.
            partials = _load_batch(
                input,
                length,
                first_tail_tile,
                BLOCK_SIZE,
                BATCH,
                MAX_GROUP_SIZE,
                COMBINE_FN,
                IDENTITY,
                compute_dtype,
                ELEMENTS_PER_THREAD,
                WARP_SIZE,
            )
            total = _fold_partials(partials, total, n_tail_tiles, COMBINE_FN, WARP_SIZE)
        return total
    else:
        gl.static_assert(len(input.shape) == 1, "loaded input must be a vector")
        WIDTH: gl.constexpr = input.shape[0]
        gl.static_assert(
            WIDTH > 0 and (WIDTH & (WIDTH - 1)) == 0, "loaded width must be a positive power of two"
        )
        values = input.to(compute_dtype)
        if length is not None:
            gl.static_assert(
                0 <= length and length <= WIDTH,
                "loaded length must be between zero and the vector width",
            )
            columns = gl.arange(0, WIDTH, layout=values.type.layout)
            values = gl.where(columns < length, values, gl.full((), IDENTITY, compute_dtype))
        return gl.reshape(
            _reduce_tiles(gl.reshape(values, (1, WIDTH)), MAX_GROUP_SIZE, COMBINE_FN, WARP_SIZE),
            (),
        )


@gluon.jit
def sum(
    input,
    length: gl.constexpr = None,
    COMPUTE_DTYPE: gl.constexpr = None,
    BLOCK_SIZE: gl.constexpr = TILE_SIZE,
    MAX_GROUP_SIZE: gl.constexpr = MAX_GROUP_SIZE,
    TILES_PER_BATCH: gl.constexpr = 32,
    ELEMENTS_PER_THREAD: gl.constexpr = 8,
    WARP_SIZE: gl.constexpr = 32,
):
    """Fixed-order sum: sum(values) or sum(ptr, length=n).

    All arguments follow reduce. Default precision retains input dtype. Batch
    and memory-layout settings affect execution only; block/group sizes define
    arithmetic. Use from @gluon.jit with the caller's compiler math settings.
    """
    return reduce(
        input,
        reduce_add,
        0,
        length,
        COMPUTE_DTYPE,
        BLOCK_SIZE,
        MAX_GROUP_SIZE,
        TILES_PER_BATCH,
        ELEMENTS_PER_THREAD,
        WARP_SIZE,
    )


@gluon.jit
def product(
    input,
    length: gl.constexpr = None,
    COMPUTE_DTYPE: gl.constexpr = None,
    BLOCK_SIZE: gl.constexpr = TILE_SIZE,
    MAX_GROUP_SIZE: gl.constexpr = MAX_GROUP_SIZE,
    TILES_PER_BATCH: gl.constexpr = 32,
    ELEMENTS_PER_THREAD: gl.constexpr = 8,
    WARP_SIZE: gl.constexpr = 32,
):
    """Fixed-order product, with the same input/configuration rules as sum.

    One pads masked lanes and seeds memory-vector accumulation. Loaded vectors
    have no additional seed. Arithmetic precision is explicit or input-derived.
    """
    return reduce(
        input,
        reduce_multiply,
        1,
        length,
        COMPUTE_DTYPE,
        BLOCK_SIZE,
        MAX_GROUP_SIZE,
        TILES_PER_BATCH,
        ELEMENTS_PER_THREAD,
        WARP_SIZE,
    )


@gluon.jit
def reduce_partials(
    partials_ptr,
    n_tiles: gl.constexpr,
    COMBINE_FN: gl.constexpr,
    initial,
    COMPUTE_DTYPE: gl.constexpr = None,
    WARP_SIZE: gl.constexpr = 32,
):
    """Fold stored unseeded partials in ascending order from initial, exactly once.

    Pointer/count/callback/seed semantics match the default backend. Compute
    dtype must equal partial storage dtype. Loads are prefetched in bounded
    windows; the fold carries its accumulator across windows. An empty buffer
    returns initial. The caller must order producers before this helper.
    WARP_SIZE has the same target-matching requirement as reduce.
    """
    gl.static_assert(
        WARP_SIZE > 0 and (WARP_SIZE & (WARP_SIZE - 1)) == 0,
        "warp size must be a positive power of two matching the compilation target",
    )
    gl.static_assert(
        partials_ptr.dtype.is_ptr() and len(partials_ptr.shape) == 0,
        "partials must be a scalar pointer",
    )
    gl.static_assert(n_tiles >= 0, "partial count must be nonnegative")
    dtype: gl.constexpr = partials_ptr.dtype.element_ty if COMPUTE_DTYPE is None else COMPUTE_DTYPE
    gl.static_assert(
        dtype == partials_ptr.dtype.element_ty, "compute dtype must match partial storage dtype"
    )
    WIDTH: gl.constexpr = min(256, triton.next_power_of_2(max(1, n_tiles)))
    columns = gl.arange(0, WIDTH, layout=gl.BlockedLayout([1], [WARP_SIZE], [gl.num_warps()], [0]))
    # Construct constants directly in compute precision; cast could first round
    # a Python float to FP32. full also accepts a runtime scalar tensor seed.
    total = gl.full((), initial, dtype)
    for window in range(n_tiles // WIDTH):
        # Widen before multiplying: the element offset can exceed int32.
        values = gl.load(partials_ptr + gl.cast(window, gl.int64) * WIDTH + columns)
        total = _fold_partials(values, total, WIDTH, COMBINE_FN, WARP_SIZE)
    if n_tiles % WIDTH:
        values = gl.load(
            partials_ptr + (n_tiles // WIDTH) * WIDTH + columns, columns < n_tiles % WIDTH, other=0
        )
        total = _fold_partials(values, total, n_tiles % WIDTH, COMBINE_FN, WARP_SIZE)
    return total


@gluon.jit
def sum_partials(
    partials_ptr,
    n_tiles: gl.constexpr,
    COMPUTE_DTYPE: gl.constexpr = None,
    WARP_SIZE: gl.constexpr = 32,
):
    """Sum stored tile results as (((0 + p0) + p1) + ...), without another tree."""
    return reduce_partials(partials_ptr, n_tiles, reduce_add, 0, COMPUTE_DTYPE, WARP_SIZE)


@gluon.jit
def product_partials(
    partials_ptr,
    n_tiles: gl.constexpr,
    COMPUTE_DTYPE: gl.constexpr = None,
    WARP_SIZE: gl.constexpr = 32,
):
    """Multiply stored tile results as (((1 * p0) * p1) * ...), without another tree."""
    return reduce_partials(partials_ptr, n_tiles, reduce_multiply, 1, COMPUTE_DTYPE, WARP_SIZE)
