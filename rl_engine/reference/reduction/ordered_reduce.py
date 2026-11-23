# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Eager PyTorch translations of the device-local Triton reduction helpers.

Callers import reduce/reduce_partials for custom binary functions, or the
sum/product convenience helpers from the reduction package. The individual stages
support custom operator integration and diagnostics; callers need not compose
them for simple use.

Functions follow the same names, order, combine pairs and cast points as
backends/shared/triton/reduction/ordered_reduce.py. Python parameter names are
lowercase counterparts of Triton's constexpr names, e.g. block_size/BLOCK_SIZE.
Tensors replace pointers; from_memory=True explicitly selects pointer semantics.
Without it, sum/product model loaded values. length selects a prefix, defaulting
to the last-dimension width; n_tiles comes from the partial tensor's last dimension.
Leading dimensions batch independent vectors. PyTorch needs no pipeline_stages:
that Triton option changes scheduling, not arithmetic. merge_unroll is retained
so its complete groups and tail can be compared directly with the JIT helper.

This module imports no Triton kernels. Run eagerly; compilation can transform
arithmetic. These helpers check fixed-path conformance, not independent numerical
accuracy. Differentiable operations participate in ordinary PyTorch autograd;
this does not prescribe a custom Triton operator's backward order. Operators
requiring a fixed backward reduction must define and validate it separately.
"""

from typing import Protocol

import torch

from rl_engine.contracts.operators.ordered_reduction import MAX_GROUP_SIZE, TILE_SIZE


class _Combine(Protocol):
    """Pure binary operation preserving the input shape and dtype."""

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Combine corresponding values with explicit left/right argument order."""
        ...


def _validate_public_inputs(values: torch.Tensor) -> None:
    """Check PyTorch layout and rank; JIT helpers receive device values."""
    if values.layout != torch.strided:
        raise ValueError("ordered reductions require a strided tensor")
    if values.ndim == 0:
        raise ValueError("ordered reductions require at least one dimension")


def reduce_add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Add two operands; zero is the padding identity."""
    return left + right


def reduce_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply two operands; one is the padding identity."""
    return left * right


def ordered_tile_reduce(
    values: torch.Tensor, *, max_group_size: int, combine_fn: _Combine
) -> torch.Tensor:
    """Apply the declared binary-combine tree to [..., 2**n] values.

    First fold up to max_group_size adjacent elements sequentially, starting
    with element 0 rather than a seed. max_group_size must be a positive power
    of two. Repeatedly combine corresponding entries from the lower and upper
    halves of all group totals until one value remains. Eight group totals pair
    0+4, 1+5, 2+6, 3+7 first, then 0+2, 1+3, then 0+1. One group returns directly.
    Widths below max_group_size form one shorter sequential group.
    No seed or dtype conversion is inserted in this tile tree. The caller
    supplies values in the required compute dtype; combine_fn must preserve
    their shape and dtype. All combine pairs are explicit.

    Lowercase sizes correspond to Triton's constexpr values. PyTorch additionally
    accepts leading batch dimensions; only the last dimension participates in
    the tree. Flat index_select keeps an independent expression of the same
    operands that Triton extracts along the short axis of a grouped view.
    The two slices select the same halves as Triton's two-entry axis reduction,
    followed by one direct combine_fn call for every combiner. Keep these
    explicit slices as an independent expression of the operand pairs.
    Neither implementation delegates the tree to a native reduction.
    """
    if values.ndim == 0:
        raise ValueError("ordered_tile_reduce requires values with at least one dimension")
    block_size = values.shape[-1]
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("ordered tile width must be a positive power of two")
    if max_group_size <= 0 or (max_group_size & (max_group_size - 1)) != 0:
        raise ValueError("maximum group size must be a positive power of two")
    group_size = min(block_size, max_group_size)
    n_groups = block_size // group_size
    groups = torch.arange(0, n_groups, device=values.device) * group_size
    subtotal = values.index_select(-1, groups)
    for offset in range(1, group_size):
        subtotal = combine_fn(subtotal, values.index_select(-1, groups + offset))
    # Pair the lower/upper halves of all group totals, retaining batch dimensions.
    for shift in range(n_groups.bit_length() - 2, -1, -1):
        subtotal = subtotal.reshape(*values.shape[:-1], 2, 1 << shift)
        subtotal = combine_fn(subtotal[..., 0, :], subtotal[..., 1, :])
    return subtotal.reshape(values.shape[:-1])


def ordered_reduce_partial(
    values: torch.Tensor,
    tile_index: int,
    *,
    block_size: int,
    max_group_size: int,
    combine_fn: _Combine,
    identity: bool | int | float,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Read and reduce tile tile_index of [..., length] in compute_dtype.

    This corresponds to Triton's ordered_reduce_partial. Slice the tile instead
    of loading pointer offsets, convert values, then pad missing lanes with the
    typed identity. No whole-vector seed is added. Returns one partial per leading
    index. Strided tensors are accepted; length comes from values.shape[-1].
    """
    if values.ndim == 0:
        raise ValueError("ordered_reduce_partial requires at least one dimension")
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a positive power of two")
    if tile_index < 0:
        raise ValueError("tile_index must be nonnegative")
    start = tile_index * block_size
    tile = values[..., start : start + block_size].to(compute_dtype)
    valid_length = tile.shape[-1]
    if valid_length < block_size:
        padding = torch.full(
            (*tile.shape[:-1], block_size - valid_length),
            identity,
            device=values.device,
            dtype=compute_dtype,
        )
        tile = torch.cat((tile, padding), dim=-1)
    return ordered_tile_reduce(
        tile,
        max_group_size=max_group_size,
        combine_fn=combine_fn,
    )


def ordered_reduce_serial(
    values: torch.Tensor,
    initial: bool | int | float | torch.Tensor,
    *,
    block_size: int,
    max_group_size: int,
    combine_fn: _Combine,
    identity: bool | int | float,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Reduce [..., length] one tile at a time, folding partials from left to right.

    Mirrors Triton's ordered_reduce_serial. Construct initial directly in
    compute_dtype and apply it once, before tile zero. It can be a scalar or a
    tensor broadcastable to the leading dimensions; it is never mutated. Empty
    vectors return an independent copy of the broadcast initial value, retaining
    any autograd connection to it. PyTorch has no pipeline_stages parameter.
    """
    if values.ndim == 0:
        raise ValueError("ordered_reduce_serial requires at least one dimension")
    # Keep invalid configs invalid even when the loop has no tiles to evaluate.
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a positive power of two")
    if max_group_size <= 0 or (max_group_size & (max_group_size - 1)) != 0:
        raise ValueError("maximum group size must be a positive power of two")
    length = values.shape[-1]
    n_tiles = (length + block_size - 1) // block_size
    total = torch.as_tensor(initial, device=values.device, dtype=compute_dtype)
    # Materialize the broadcast so even an empty fold owns its result storage.
    total = torch.broadcast_to(total, values.shape[:-1]).clone()
    for tile in range(0, n_tiles):
        partial = ordered_reduce_partial(
            values,
            tile,
            block_size=block_size,
            max_group_size=max_group_size,
            combine_fn=combine_fn,
            identity=identity,
            compute_dtype=compute_dtype,
        )
        total = combine_fn(total, partial)
    return total


def ordered_reduce_merge(
    partials: torch.Tensor,
    initial: bool | int | float | torch.Tensor,
    *,
    combine_fn: _Combine,
    merge_unroll: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Fold stored [..., n_tiles] partials in ascending order, starting at initial.

    Mirrors Triton's ordered_reduce_merge arithmetic, including its scalar groups and
    incomplete tail. Storage must already have compute_dtype; no hidden cast
    changes partials. Every combine is left-associated. Empty partials return
    an independent copy of the broadcast initial value, retaining its autograd
    connection. Leading dimensions batch independent merges; inputs are not mutated.
    """
    if partials.ndim == 0:
        raise ValueError("ordered_reduce_merge requires at least one dimension")
    total = torch.as_tensor(initial, device=partials.device, dtype=compute_dtype)
    # clone preserves the seed's graph while preventing aliasing on empty input.
    total = torch.broadcast_to(total, partials.shape[:-1]).clone()
    if partials.dtype != compute_dtype:
        raise ValueError("partial storage must match compute dtype")
    if merge_unroll <= 0:
        raise ValueError("merge unroll must be a positive integer")
    n_tiles = partials.shape[-1]
    full_tiles = (n_tiles // merge_unroll) * merge_unroll
    for tile in range(0, full_tiles, merge_unroll):
        for offset in range(merge_unroll):
            partial = partials[..., tile + offset]
            total = combine_fn(total, partial)
    for tile in range(full_tiles, n_tiles):
        total = combine_fn(total, partials[..., tile])
    return total


def reduce(
    values: torch.Tensor,
    length: int | None = None,
    *,
    combine_fn: _Combine,
    identity: bool | int | float,
    from_memory: bool = False,
    compute_dtype: torch.dtype | None = None,
    block_size: int = TILE_SIZE,
    max_group_size: int = MAX_GROUP_SIZE,
) -> torch.Tensor:
    """Eager fixed-order reduction with a caller-owned PyTorch binary function.

    This is the public implementation used by sum/product, matching Triton's
    reduce. combine_fn(left, right) must be pure, preserve tensor shape/dtype and
    be mathematically associative and commutative on the supported domain.
    identity is its neutral value, supplied explicitly rather than inferred from
    the function. Padding/empty prefixes use that value in the compute dtype.
    For example, use combine_fn=torch.bitwise_xor, identity=0 for integer XOR.
    This scalar-state interface does not implement tuple-state reductions.

    values is a strided tensor [..., width]; leading dimensions batch independent
    reductions. The default models loaded values: a positive power-of-two width
    with no extra seed. Optional length masks the suffix without resizing the
    tree. from_memory=True models a JIT scalar pointer: select the prefix, tile
    it and fold partials from identity. Empty memory vectors are allowed.
    length=None uses the full last dimension; other lengths must lie in [0, width].
    The reference needs this mode because Torch tensors cannot distinguish JIT
    numeric values from storage addressed by a pointer. length never selects it.

    compute_dtype=None keeps input dtype; explicit precision casts inputs before
    combining. block_size defaults to 1024 and only affects the memory path;
    max_group_size defaults to 8. Both sizes are positive powers of
    two. Inputs are never mutated. Run eagerly; differentiable operations retain
    PyTorch's autograd graph. The fixed-order guarantee covers forward arithmetic;
    backward order must be specified and validated separately. A same-tree
    reference is separate from independent accuracy validation. An empty memory
    fold returns the constant identity without a dependency on values.
    """
    _validate_public_inputs(values)
    width = values.shape[-1]
    if length is not None and not 0 <= length <= width:
        raise ValueError("length must be between zero and the input width")
    compute_dtype = values.dtype if compute_dtype is None else compute_dtype
    if from_memory:
        selected = values if length is None else values[..., :length]
        return ordered_reduce_serial(
            selected,
            initial=identity,
            block_size=block_size,
            max_group_size=max_group_size,
            combine_fn=combine_fn,
            identity=identity,
            compute_dtype=compute_dtype,
        )
    else:
        selected = values.to(compute_dtype)
        if length is not None:
            selected = torch.where(
                torch.arange(width, device=values.device) < length,
                selected,
                torch.full((), identity, dtype=compute_dtype, device=values.device),
            )
        return ordered_tile_reduce(
            selected,
            max_group_size=max_group_size,
            combine_fn=combine_fn,
        )


def sum(
    values: torch.Tensor,
    length: int | None = None,
    *,
    from_memory: bool = False,
    compute_dtype: torch.dtype | None = None,
    block_size: int = TILE_SIZE,
    max_group_size: int = MAX_GROUP_SIZE,
) -> torch.Tensor:
    """Eager reference for Triton's sum, with explicit loaded/storage semantics.

    Args:
        values: CPU/GPU strided tensor [..., width]; leading dimensions batch
            independent reductions. The default models already-loaded values,
            whose last dimension must be a positive power of two.
        length: Optional prefix length between zero and width. None means the
            complete last dimension. This never selects the input mode.
        from_memory: False models Triton's loaded-vector input: mask excluded
            suffix lanes with zero, retain width and use the seedless tree.
            True models a scalar pointer: slice to length, process fixed-width
            tiles and fold from positive zero. Storage can have any width,
            including zero. This reference-only flag is needed because a Torch
            tensor does not carry Triton's pointer-versus-numeric distinction.
        compute_dtype: None retains values.dtype without widening. An explicit
            dtype converts inputs before arithmetic, not just the final result.
        block_size: Positive power-of-two tile width for from_memory=True only,
            default 1024. Loaded values always retain their supplied width.
        max_group_size: Positive power-of-two adjacent group size, default 8.

    Returns a tensor shaped [...] in compute_dtype on the input device.
    An empty prefix returns zero; a loaded vector must still have positive width.
    Inputs are never mutated. Run eagerly; floating-point arithmetic supports
    ordinary PyTorch autograd, without prescribing a Triton backward order.
    An empty memory fold returns a constant with no dependency on values.
    Same-tree checks require the matching mode, width, parameters and precision.
    They are separate from independent native/FP64 accuracy validation.
    """
    return reduce(
        values,
        length,
        from_memory=from_memory,
        combine_fn=reduce_add,
        identity=0,
        compute_dtype=compute_dtype,
        block_size=block_size,
        max_group_size=max_group_size,
    )


def product(
    values: torch.Tensor,
    length: int | None = None,
    *,
    from_memory: bool = False,
    compute_dtype: torch.dtype | None = None,
    block_size: int = TILE_SIZE,
    max_group_size: int = MAX_GROUP_SIZE,
) -> torch.Tensor:
    """Eager reference for Triton's product; use one for padding and the memory seed.

    The input, length, from_memory and precision rules match sum. By default,
    reduce a loaded power-of-two last dimension without inserting a seed;
    length masks suffix lanes with one without shrinking its tree.
    from_memory=True models the pointer path: process the selected prefix in
    block_size tiles, pad the final tile with one and fold products left to right
    starting from one. Empty prefixes return one. The result retains leading
    dimensions and the resolved compute dtype, without modifying inputs.

    block_size defaults to 1024 and only affects the memory path; max_group_size
    defaults to 8 for either path. compute_dtype=None uses
    input precision without widening. Fixed order does not prevent rounding,
    underflow or overflow; independent accuracy checks are still needed.
    """
    return reduce(
        values,
        length,
        from_memory=from_memory,
        combine_fn=reduce_multiply,
        identity=1,
        compute_dtype=compute_dtype,
        block_size=block_size,
        max_group_size=max_group_size,
    )


def reduce_partials(
    partials: torch.Tensor,
    *,
    combine_fn: _Combine,
    initial: bool | int | float | torch.Tensor,
    compute_dtype: torch.dtype | None = None,
    merge_unroll: int = 1,
) -> torch.Tensor:
    """Merge stored [..., n_tiles] partials with a caller-owned binary function.

    Mirrors Triton's reduce_partials: fold ascending slots starting with initial,
    without introducing another tree. combine_fn has reduce's purity/shape/dtype
    requirements. initial is required and applied once, including for empty
    partials; it may be a scalar or tensor broadcastable over leading dimensions.
    Use the producers' callback and initial=identity to match reduce's memory
    path. The initial value is separate from producer padding, which is absent
    from a merge. It can be a nonidentity seed when the caller needs one.

    compute_dtype=None uses storage precision. An explicit dtype must match
    storage; no cast silently changes partials. Positive merge_unroll changes
    only loop expansion. Inputs are preserved. Differentiable partials and tensor
    seeds retain ordinary PyTorch autograd connections; backward reduction order
    is not prescribed. An empty fold returns an independent copy of the broadcast
    seed without reading partials, so only a differentiable seed contributes an
    autograd connection. Writing to that result does not modify the seed.
    """
    _validate_public_inputs(partials)
    compute_dtype = partials.dtype if compute_dtype is None else compute_dtype
    return ordered_reduce_merge(
        partials,
        initial=initial,
        combine_fn=combine_fn,
        merge_unroll=merge_unroll,
        compute_dtype=compute_dtype,
    )


def sum_partials(
    partials: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    merge_unroll: int = 1,
) -> torch.Tensor:
    """Merge [..., n_tiles] sums in order, with only the tensor required.

    Mirrors Triton's sum_partials. Start from positive zero and add unseeded
    partials left to right; an empty last dimension returns zero. No additional
    tile tree is applied. Leading dimensions batch independent merges and
    strided tensors are accepted.

    compute_dtype=None uses partials.dtype. An explicit dtype must match storage,
    so the merge cannot silently change the producer's precision. merge_unroll
    accepts any positive integer and preserves the fold, including tail entries.
    Inputs are unchanged; differentiable arithmetic retains PyTorch's autograd
    graph. An empty fold returns a constant with no dependency on partials.
    """
    return reduce_partials(
        partials,
        combine_fn=reduce_add,
        initial=0,
        compute_dtype=compute_dtype,
        merge_unroll=merge_unroll,
    )


def product_partials(
    partials: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    merge_unroll: int = 1,
) -> torch.Tensor:
    """Merge [..., n_tiles] products in order, with only the tensor required.

    Mirrors Triton's product_partials. Start from one and multiply unseeded
    partials left to right; an empty last dimension returns one. No new tile
    tree or dtype conversion is added. compute_dtype=None uses partials.dtype;
    an explicit dtype must match storage. merge_unroll defaults to 1 and accepts
    any positive integer without changing order. The input and autograd rules
    are the same as sum_partials.
    """
    return reduce_partials(
        partials,
        combine_fn=reduce_multiply,
        initial=1,
        compute_dtype=compute_dtype,
        merge_unroll=merge_unroll,
    )
