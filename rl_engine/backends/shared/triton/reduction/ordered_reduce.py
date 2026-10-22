# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared implementation of tile and whole-vector sum/product JIT helpers.

Ordinary callers import sum_tile/product_tile for computed values or
sum_vector/product_vector for a pointer and length. The generic tree, partial,
serial and merge functions below share the arithmetic across these entry points
and custom fused/parallel operators; callers need not assemble them for simple use.
sum_partials/product_partials merge results already stored by parallel producers,
with the operation, seed and precision defaults supplied by the helper.

The operation, padding identity and final seed are distinct explicit arguments.
The tile tree and ascending partial fold are shared by all schedules. Common
operations below cover sum and product; callers may supply a pure JIT
combiner with matching shape/dtype and a valid identity. This scalar-result API
is not a multi-field argmax/mean/variance reducer.

Vector helpers cast loaded values to COMPUTE_DTYPE and keep partials in that
dtype. sum_vector/product_vector default to the input pointer's element type;
internal stages receive a resolved dtype explicitly.
There is no dtype whitelist: the chosen backend must support the
loads, casts and combiner for that type. Callers choose an accumulator wide
enough for their values; integer reductions do not automatically widen it.
Callers own allocation, launch/device/stream ordering, arithmetic config and
math compiler flags. Elementwise transforms belong to the caller, which can
pass locally computed values to ordered_tile_reduce without storing them first.
Helpers do not communicate between GPUs or select policy.
Fixed order is not exact arithmetic or a cross-hardware reproducibility promise.
Independent accuracy and GPU conformance tests remain necessary.
"""

import triton
import triton.language as tl

from rl_engine.contracts.operators.ordered_reduction import MAX_GROUP_SIZE, SECTION_COUNT, TILE_SIZE


@triton.jit
def reduce_add(left, right):
    """Add two operands; zero is the padding identity."""
    return left + right


@triton.jit
def reduce_multiply(left, right):
    """Multiply two operands; one is the padding identity."""
    return left * right


@triton.jit
def ordered_tile_reduce(
    values,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
    COMBINE_FN: tl.constexpr,
):
    """Combine one already-computed Triton vector into a scalar in a fixed order.

    Call this helper inside a Triton kernel after loading or computing a tile.
    It reduces exactly the supplied values: reduce_add gives their sum, while
    reduce_multiply gives their product. The grouping parameters determine
    which operands combine at every step, independently of tile scheduling.
    Loading, elementwise transforms and masking belong to the caller; this
    function neither reads pointers nor launches another kernel.

    Args:
        values: One-dimensional Triton tensor with shape [BLOCK_SIZE], where
            BLOCK_SIZE is a positive power of two. These are the actual values
            to reduce, not a pointer. Any padded lanes must already contain the
            operation's neutral value, such as zero for sum or one for product.
        MAX_GROUP_SIZE: Compile-time maximum number of adjacent elements in
            each initial group; must be a positive power of two. The actual
            group size is min(BLOCK_SIZE, MAX_GROUP_SIZE). Each group combines
            left to right. These are arithmetic groups, not GPU thread groups.
        SECTION_COUNT: Compile-time maximum number of equal sections of group
            totals; must be a positive power of two. The actual section count
            is min(N_GROUPS, SECTION_COUNT), so no empty sections are inserted.
            Each section is reduced first, then its total joins the section tree.
        COMBINE_FN: Compile-time-selected, pure @triton.jit binary function,
            called as COMBINE_FN(left, right). It must return the same shape
            and dtype as its operands. Choose a mathematically associative,
            commutative operation; floating-point rounding may still depend
            on the prescribed order. The combiner owns any arithmetic casts.

    Returns:
        A scalar Triton tensor with shape () and the same dtype as values,
        ready for further computation or storage by the enclosing kernel.

    Reduction order:
        1. Split values into adjacent groups and fold each group from its first
           element, without inserting an initial value.
        2. Divide group totals into min(N_GROUPS, SECTION_COUNT) equal sections.
           Within each section, repeatedly combine corresponding entries from
           its lower and upper halves until one section total remains.
        3. Repeatedly combine the lower and upper halves of the section totals.
           Four sections retain combine(combine(s0, s2), combine(s1, s3)).
           A single entry at either stage passes through without a combine.

    Every tree step calls COMBINE_FN directly on explicit left/right operands.
    No seed, dtype conversion or merge across separate tiles is added here.
    The same BLOCK_SIZE, MAX_GROUP_SIZE and SECTION_COUNT determine the same
    operand tree. Bitwise conformance also requires identical inputs, combiner,
    dtype and rounding behavior. Changing the arithmetic configuration can
    change results; fixed order alone does not promise equal bits across
    backends or compiler versions.

    Example:
        tile_sum = ordered_tile_reduce(
            exponentials,
            MAX_GROUP_SIZE=8,
            SECTION_COUNT=4,
            COMBINE_FN=reduce_add,
        )
    """
    tl.static_assert(len(values.shape) == 1, "ordered tile reduction requires a vector")
    BLOCK_SIZE: tl.constexpr = values.shape[0]
    tl.static_assert(
        BLOCK_SIZE > 0 and (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
        "ordered tile width must be a positive power of two",
    )
    tl.static_assert(
        MAX_GROUP_SIZE > 0 and (MAX_GROUP_SIZE & (MAX_GROUP_SIZE - 1)) == 0,
        "maximum group size must be a positive power of two",
    )
    tl.static_assert(
        SECTION_COUNT > 0 and (SECTION_COUNT & (SECTION_COUNT - 1)) == 0,
        "section count must be a positive power of two",
    )
    GROUP_SIZE: tl.constexpr = min(BLOCK_SIZE, MAX_GROUP_SIZE)
    N_GROUPS: tl.constexpr = BLOCK_SIZE // GROUP_SIZE
    groups = tl.arange(0, N_GROUPS) * GROUP_SIZE
    subtotal = tl.gather(values, groups, axis=0)
    for offset in tl.static_range(1, GROUP_SIZE):
        subtotal = COMBINE_FN(subtotal, tl.gather(values, groups + offset, axis=0))
    N_SECTIONS: tl.constexpr = min(N_GROUPS, SECTION_COUNT)
    SECTION_SIZE: tl.constexpr = N_GROUPS // N_SECTIONS
    # Split each section into lower/upper halves, then combine matching entries.
    # split needs the two-entry axis last; permute only rearranges the operands.
    # Call the JIT combiner directly: tl.reduce does not unwrap a constexpr
    # function argument in Triton 3.7.1.
    for shift in tl.static_range(SECTION_SIZE.bit_length() - 2, -1, -1):
        pairs = tl.reshape(subtotal, (N_SECTIONS, 2, 1 << shift))
        left, right = tl.split(tl.permute(pairs, (0, 2, 1)))
        subtotal = COMBINE_FN(left, right)
    subtotal = tl.reshape(subtotal, (N_SECTIONS,))
    # Merge section totals by halving distance: four sections pair 0+2, 1+3.
    for shift in tl.static_range(N_SECTIONS.bit_length() - 2, -1, -1):
        pairs = tl.reshape(subtotal, (2, 1 << shift))
        left, right = tl.split(tl.permute(pairs, (1, 0)))
        subtotal = COMBINE_FN(left, right)
    return tl.reshape(subtotal, ())


@triton.jit
def ordered_reduce_partial(
    input_ptr,
    length: tl.constexpr,
    tile_index,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Load and reduce one tile of a contiguous vector in COMPUTE_DTYPE.

    input_ptr addresses the vector's beginning; tile_index must be nonnegative.
    Cast loaded values to the
    requested dtype, then fill padded lanes with IDENTITY in that dtype.
    IDENTITY must be neutral for COMBINE_FN on the supported value domain.
    No user seed enters partials. Both schedules use this exact operation.
    To fuse a transform, compute it in the caller and use ordered_tile_reduce.
    """
    offsets = tl.cast(tile_index, tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < length
    values = tl.load(input_ptr + offsets, mask=mask, other=0).to(COMPUTE_DTYPE)
    padding = tl.full((), IDENTITY, COMPUTE_DTYPE)
    values = tl.where(mask, values, padding)
    return ordered_tile_reduce(
        values,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
        COMBINE_FN=COMBINE_FN,
    )


@triton.jit
def ordered_reduce_serial(
    input_ptr,
    length: tl.constexpr,
    initial,
    BLOCK_SIZE: tl.constexpr,
    MAX_GROUP_SIZE: tl.constexpr,
    SECTION_COUNT: tl.constexpr,
    COMBINE_FN: tl.constexpr,
    IDENTITY: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """Reduce a contiguous vector in one program, one tile at a time.

    Fold tile results left to right, starting with initial converted to COMPUTE_DTYPE.
    initial is applied once, not once per tile, and may be a runtime scalar.
    IDENTITY is only for padding; it can differ from initial. The matching
    parallel merge receives the same initial and COMBINE_FN. PIPELINE_STAGES
    changes scheduling, not combine pairs. Length zero returns initial.
    No global partial buffer is needed. Elementwise transforms are not applied.
    """
    tl.static_assert(length >= 0, "vector length must be nonnegative")
    n_tiles: tl.constexpr = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Construct constants directly in the chosen dtype; a scalar cast can first
    # materialize a Python float as FP32 and lose bits before conversion to FP64.
    total = tl.full((), initial, COMPUTE_DTYPE)
    for tile in tl.range(0, n_tiles, num_stages=PIPELINE_STAGES):
        partial = ordered_reduce_partial(
            input_ptr,
            length,
            tile,
            BLOCK_SIZE=BLOCK_SIZE,
            MAX_GROUP_SIZE=MAX_GROUP_SIZE,
            SECTION_COUNT=SECTION_COUNT,
            COMBINE_FN=COMBINE_FN,
            IDENTITY=IDENTITY,
            COMPUTE_DTYPE=COMPUTE_DTYPE,
        )
        total = COMBINE_FN(total, partial)
    return total


@triton.jit
def ordered_reduce_merge(
    partials_ptr,
    n_tiles: tl.constexpr,
    initial,
    COMBINE_FN: tl.constexpr,
    MERGE_UNROLL: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
):
    """Fold unseeded COMPUTE_DTYPE partials in ascending tile order, starting at initial.

    A parallel producer stores tile k at partials_ptr+k. Its completion order
    does not select the reduction order. Apply initial once, before tile zero,
    exactly as the serial schedule does. MERGE_UNROLL is any positive integer:
    it expands that many load/combine steps per loop iteration at compile time.
    It need not be a power of two. Combines remain left-associated, including
    incomplete tails; changing the unroll factor does not change their order.
    This is not a balanced tree over partials. No padding/identity is needed
    in this stage. Zero partials return initial without a load or combine.

    The caller must complete all producers before reading partials, using the
    same stream or an explicit dependency. This helper does not synchronize.
    Use this fold for stored partials; sum_vector would introduce a new tile tree.
    """
    tl.static_assert(n_tiles >= 0, "partial count must be nonnegative")
    total = tl.full((), initial, COMPUTE_DTYPE)
    tl.static_assert(
        partials_ptr.dtype.element_ty == COMPUTE_DTYPE, "partial storage must match compute dtype"
    )
    tl.static_assert(MERGE_UNROLL > 0, "merge unroll must be a positive integer")
    full_tiles: tl.constexpr = (n_tiles // MERGE_UNROLL) * MERGE_UNROLL
    for tile in range(0, full_tiles, MERGE_UNROLL):
        for offset in tl.static_range(MERGE_UNROLL):
            partial = tl.load(partials_ptr + tile + offset)
            total = COMBINE_FN(total, partial)
    for tile in tl.static_range(full_tiles, n_tiles):
        total = COMBINE_FN(total, tl.load(partials_ptr + tile))
    return total


@triton.jit
def sum_tile(
    values,
    COMPUTE_DTYPE: tl.constexpr = None,
    MAX_GROUP_SIZE: tl.constexpr = MAX_GROUP_SIZE,
    SECTION_COUNT: tl.constexpr = SECTION_COUNT,
):
    """Sum a loaded power-of-two vector, using its dtype unless overridden.

    Args:
        values: Already-loaded one-dimensional vector with power-of-two length.
            Mask invalid lanes with zero before calling this helper.
        COMPUTE_DTYPE: Optional arithmetic dtype; None uses values.dtype without
            widening. An explicit dtype converts values before summing.
        MAX_GROUP_SIZE: Maximum adjacent group size, defaulting to 8. Each group
            sums left to right. Must be a positive power of two.
        SECTION_COUNT: Maximum number of sections of group totals, defaulting to 4.
            Must be a positive power of two. See ordered_tile_reduce for the tree.

    Call inside another @triton.jit function. Omitted tree parameters retain the
    default 8/4 order; explicit values select a different fixed tree and can change
    rounding. Keep them identical across paths that require bitwise consistency.
    No whole-vector seed or inter-tile fold is added. Use sum_vector to load and
    reduce a complete, possibly much longer vector.
    """
    compute_dtype: tl.constexpr = values.dtype if COMPUTE_DTYPE is None else COMPUTE_DTYPE
    return ordered_tile_reduce(
        values.to(compute_dtype),
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
        COMBINE_FN=reduce_add,
    )


@triton.jit
def product_tile(
    values,
    COMPUTE_DTYPE: tl.constexpr = None,
    MAX_GROUP_SIZE: tl.constexpr = MAX_GROUP_SIZE,
    SECTION_COUNT: tl.constexpr = SECTION_COUNT,
):
    """Multiply a loaded power-of-two vector, using its dtype unless overridden.

    Args:
        values: Already-loaded one-dimensional vector with power-of-two length.
            Mask invalid lanes with one before calling this helper.
        COMPUTE_DTYPE: Optional arithmetic dtype; None uses values.dtype without
            widening. An explicit dtype converts values before multiplying.
        MAX_GROUP_SIZE: Maximum adjacent group size, defaulting to 8. Each group
            multiplies left to right. Must be a positive power of two.
        SECTION_COUNT: Maximum number of sections of group totals, defaulting to 4.
            Must be a positive power of two. See ordered_tile_reduce for the tree.

    Omitted tree parameters retain the 8/4 tree. Changing either can change rounding;
    keep them identical across paths requiring bitwise consistency. No final
    seed or inter-tile fold is added.
    """
    compute_dtype: tl.constexpr = values.dtype if COMPUTE_DTYPE is None else COMPUTE_DTYPE
    return ordered_tile_reduce(
        values.to(compute_dtype),
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
        COMBINE_FN=reduce_multiply,
    )


@triton.jit
def sum_vector(
    input_ptr,
    length: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = None,
    BLOCK_SIZE: tl.constexpr = TILE_SIZE,
    MAX_GROUP_SIZE: tl.constexpr = MAX_GROUP_SIZE,
    SECTION_COUNT: tl.constexpr = SECTION_COUNT,
    PIPELINE_STAGES: tl.constexpr = None,
):
    """Return a complete vector's sum, using its storage dtype unless overridden.

    Args:
        input_ptr: Start of a contiguous vector in device memory.
        length: Nonnegative number of elements, known at compile time.
        COMPUTE_DTYPE: Dtype for input conversion, additions and output.
            None (the default) uses input_ptr.dtype.element_ty, without widening.
        BLOCK_SIZE: Elements loaded per tile, defaulting to 1024. Must be a
            positive power of two; the last tile is padded with zero. This is
            the arithmetic tile width, not a limit on the complete vector length.
        MAX_GROUP_SIZE: Maximum adjacent group size inside each tile, defaulting
            to 8. Must be a positive power of two.
        SECTION_COUNT: Maximum sections of group totals per tile, defaulting to
            4. Must be a positive power of two. See ordered_tile_reduce for the tree.
        PIPELINE_STAGES: Optional software-pipeline hint for the tile loop;
            None leaves the loop unhinted. Passed to tl.range(num_stages=...),
            not the enclosing kernel's launch options. This changes scheduling,
            not the order of additions.

    Omitted arithmetic parameters retain the 1024/8/4 order. Changing them can
    change rounding; keep them identical across paths requiring bitwise consistency.
    Tile totals are always added left to right, starting from a zero seed.
    All tiles are processed within the calling program. The enclosing kernel's
    grid and num_warps control program and thread parallelism; this function
    does not distribute a single vector across multiple programs.
    Length zero returns zero. There is no program_id, output store, allocation
    or child kernel launch: the caller chooses which vector this pointer names
    and can immediately use the result in further fused computation.
    Select an integer COMPUTE_DTYPE wide enough to avoid overflow; for example,
    tl.int64 for wide sums or tl.int32 to count boolean inputs. No implicit
    accumulator widening is applied by this helper.
    """
    compute_dtype: tl.constexpr = (
        input_ptr.dtype.element_ty if COMPUTE_DTYPE is None else COMPUTE_DTYPE
    )
    return ordered_reduce_serial(
        input_ptr,
        length,
        initial=0,
        COMBINE_FN=reduce_add,
        IDENTITY=0,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
        PIPELINE_STAGES=PIPELINE_STAGES,
    )


@triton.jit
def product_vector(
    input_ptr,
    length: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = None,
    BLOCK_SIZE: tl.constexpr = TILE_SIZE,
    MAX_GROUP_SIZE: tl.constexpr = MAX_GROUP_SIZE,
    SECTION_COUNT: tl.constexpr = SECTION_COUNT,
    PIPELINE_STAGES: tl.constexpr = None,
):
    """Return a vector's product, using its storage dtype unless overridden.

    Args:
        input_ptr: Start of a contiguous vector in device memory.
        length: Nonnegative number of elements, known at compile time.
        COMPUTE_DTYPE: Dtype for input conversion, multiplications and output.
            None (the default) uses input_ptr.dtype.element_ty, without widening.
        BLOCK_SIZE: Elements loaded per tile, defaulting to 1024. Must be a
            positive power of two; the last tile is padded with one. The complete
            vector may contain any nonnegative number of elements.
        MAX_GROUP_SIZE: Maximum adjacent group size inside each tile, defaulting
            to 8. Must be a positive power of two.
        SECTION_COUNT: Maximum sections of group totals per tile, defaulting to
            4. Must be a positive power of two. See ordered_tile_reduce for the tree.
        PIPELINE_STAGES: Optional software-pipeline hint for the tile loop;
            None leaves the loop unhinted. Passed to tl.range(num_stages=...),
            not the enclosing kernel's launch options. This changes scheduling,
            not the order of multiplications.

    Omitted parameters retain the 1024/8/4 order. Changing them can change rounding;
    keep them identical across paths requiring bitwise consistency. Tile products
    are multiplied left to right from a seed of one; empty returns one.
    All tiles execute within the calling
    program. The caller owns the pointer, grid, warp count and output storage,
    as with sum_vector. Fixed order does not prevent overflow or underflow.
    """
    compute_dtype: tl.constexpr = (
        input_ptr.dtype.element_ty if COMPUTE_DTYPE is None else COMPUTE_DTYPE
    )
    return ordered_reduce_serial(
        input_ptr,
        length,
        initial=1,
        COMBINE_FN=reduce_multiply,
        IDENTITY=1,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_GROUP_SIZE=MAX_GROUP_SIZE,
        SECTION_COUNT=SECTION_COUNT,
        PIPELINE_STAGES=PIPELINE_STAGES,
    )


@triton.jit
def sum_partials(
    partials_ptr,
    n_tiles: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = None,
    MERGE_UNROLL: tl.constexpr = 1,
):
    """Merge stored tile sums in order; only the pointer and count are required.

    Args:
        partials_ptr: Contiguous tile sums in ascending tile order. Each entry
            is an unseeded result computed by a producer.
        n_tiles: Nonnegative number of partials, known at compile time. It need
            not be a power of two; zero partials return positive zero.
        COMPUTE_DTYPE: None uses the partial storage dtype. An explicit dtype
            must match storage, so merging introduces no extra conversion.
        MERGE_UNROLL: Optional positive loop-unroll factor, defaulting to 1.
            It changes scheduling, not the addition order.

    Return (((0 + partial[0]) + partial[1]) + ...) in the partial storage dtype.
    This is the same fold used by sum_vector after reducing each tile, without
    applying another tile tree to the partials. The caller must complete all
    producers before this function runs; no kernel launch or synchronization
    occurs here.
    """
    compute_dtype: tl.constexpr = (
        partials_ptr.dtype.element_ty if COMPUTE_DTYPE is None else COMPUTE_DTYPE
    )
    return ordered_reduce_merge(
        partials_ptr,
        n_tiles,
        initial=0,
        COMBINE_FN=reduce_add,
        MERGE_UNROLL=MERGE_UNROLL,
        COMPUTE_DTYPE=compute_dtype,
    )


@triton.jit
def product_partials(
    partials_ptr,
    n_tiles: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr = None,
    MERGE_UNROLL: tl.constexpr = 1,
):
    """Merge stored tile products in order; only the pointer and count are required.

    Args:
        partials_ptr: Contiguous, unseeded tile products in ascending tile order.
        n_tiles: Nonnegative compile-time count; zero partials return one.
        COMPUTE_DTYPE: None uses the partial storage dtype. An explicit dtype
            must match storage, preserving the producer's arithmetic precision.
        MERGE_UNROLL: Optional positive loop-unroll factor, defaulting to 1;
            it does not change the multiplication order.

    Return (((1 * partial[0]) * partial[1]) * ...), matching product_vector's
    fold over tile results. Producers must finish before this function reads
    their results, as with sum_partials. No new tile tree or dtype cast is added.
    """
    compute_dtype: tl.constexpr = (
        partials_ptr.dtype.element_ty if COMPUTE_DTYPE is None else COMPUTE_DTYPE
    )
    return ordered_reduce_merge(
        partials_ptr,
        n_tiles,
        initial=1,
        COMBINE_FN=reduce_multiply,
        MERGE_UNROLL=MERGE_UNROLL,
        COMPUTE_DTYPE=compute_dtype,
    )
