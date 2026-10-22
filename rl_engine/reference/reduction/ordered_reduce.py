# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Eager PyTorch translations of the device-local Triton reduction helpers.

Ordinary callers import sum/product helpers for tiles, vectors or stored partials
from the reduction package. The individual stages here support custom operator
integration and diagnostics; callers need not compose them for simple use.

Functions follow the same names, order, combine pairs and cast points as
backends/shared/triton/reduction/ordered_reduce.py. Python parameter names are
lowercase counterparts of Triton's constexpr names, e.g. block_size/BLOCK_SIZE.
Tensors replace pointers; length and n_tiles come from their last dimension.
Leading dimensions batch independent vectors. PyTorch needs no pipeline_stages:
that Triton option changes scheduling, not arithmetic. merge_unroll is retained
so its complete groups and tail can be compared directly with the JIT helper.

This module imports no Triton kernels. Run eagerly; compilation can transform
arithmetic. These helpers check fixed-path conformance, not independent numerical
accuracy, and do not define a custom Triton operator's backward order. Public
sum/product helpers require no_grad when their inputs require gradients.

The final ordered_reduce collector is PyTorch-only diagnostic support: it returns
both partials and their merged total using the same partial/merge helpers. It is
not an additional reduction algorithm or a performance baseline.
"""

from dataclasses import dataclass
from typing import Protocol

import torch

from rl_engine.contracts.operators.ordered_reduction import MAX_GROUP_SIZE, SECTION_COUNT, TILE_SIZE


class _Combine(Protocol):
    """Pure binary operation preserving the input shape and dtype."""

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Combine corresponding values with explicit left/right argument order."""
        ...


def _validate_public_inputs(values: torch.Tensor) -> None:
    """Check PyTorch layout/autograd constraints; JIT helpers receive device values."""
    if values.layout != torch.strided:
        raise ValueError("ordered reductions require a strided tensor")
    if values.ndim == 0:
        raise ValueError("ordered reductions require at least one dimension")
    if torch.is_grad_enabled() and values.requires_grad:
        raise RuntimeError(
            "ordered references validate forward arithmetic without autograd; "
            "evaluate under torch.no_grad()"
        )


def reduce_add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Add two operands; zero is the padding identity."""
    return left + right


def reduce_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply two operands; one is the padding identity."""
    return left * right


def ordered_tile_reduce(
    values: torch.Tensor, *, max_group_size: int, section_count: int, combine_fn: _Combine
) -> torch.Tensor:
    """Apply the declared binary-combine tree to [..., 2**n] values.

    First fold up to max_group_size adjacent elements sequentially, starting
    with element 0 rather than a seed. Both max_group_size and section_count
    must be positive powers of two. Divide the group totals into
    min(n_groups, section_count) equal sections, then repeatedly combine each
    section's lower and upper halves. Merge section totals in the same way.
    Four sections retain (s0+s2)+(s1+s3) for addition. One group returns directly.
    Widths below max_group_size form one shorter sequential group.
    No seed or dtype conversion is inserted in this tile tree. The caller
    supplies values in the required compute dtype; combine_fn must preserve
    their shape and dtype. All combine pairs are explicit.

    Steps and local names mirror Triton's ordered_tile_reduce. Lowercase sizes
    correspond to its constexpr values. PyTorch additionally accepts leading
    batch dimensions; only the last dimension participates in the tree.
    index_select corresponds to tl.gather. The two slices select the same
    halves as Triton's permute/split, followed by one direct combine_fn call.
    Neither implementation delegates the tree to a native reduction.
    """
    if values.ndim == 0:
        raise ValueError("ordered_tile_reduce requires values with at least one dimension")
    block_size = values.shape[-1]
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("ordered tile width must be a positive power of two")
    if max_group_size <= 0 or (max_group_size & (max_group_size - 1)) != 0:
        raise ValueError("maximum group size must be a positive power of two")
    if section_count <= 0 or (section_count & (section_count - 1)) != 0:
        raise ValueError("section count must be a positive power of two")
    group_size = min(block_size, max_group_size)
    n_groups = block_size // group_size
    groups = torch.arange(0, n_groups, device=values.device) * group_size
    subtotal = values.index_select(-1, groups)
    for offset in range(1, group_size):
        subtotal = combine_fn(subtotal, values.index_select(-1, groups + offset))
    n_sections = min(n_groups, section_count)
    section_size = n_groups // n_sections
    # Split each section into lower/upper halves, then combine matching entries.
    for shift in range(section_size.bit_length() - 2, -1, -1):
        subtotal = subtotal.reshape(*values.shape[:-1], n_sections, 2, 1 << shift)
        subtotal = combine_fn(subtotal[..., 0, :], subtotal[..., 1, :])
    subtotal = subtotal.reshape(*values.shape[:-1], n_sections)
    # Merge section totals by halving distance: four sections pair 0+2, 1+3.
    for shift in range(n_sections.bit_length() - 2, -1, -1):
        subtotal = subtotal.reshape(*values.shape[:-1], 2, 1 << shift)
        subtotal = combine_fn(subtotal[..., 0, :], subtotal[..., 1, :])
    return subtotal.reshape(values.shape[:-1])


def ordered_reduce_partial(
    values: torch.Tensor,
    tile_index: int,
    *,
    block_size: int,
    max_group_size: int,
    section_count: int,
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
        section_count=section_count,
        combine_fn=combine_fn,
    )


def ordered_reduce_serial(
    values: torch.Tensor,
    initial: bool | int | float | torch.Tensor,
    *,
    block_size: int,
    max_group_size: int,
    section_count: int,
    combine_fn: _Combine,
    identity: bool | int | float,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Reduce [..., length] one tile at a time, folding partials from left to right.

    Mirrors Triton's ordered_reduce_serial. Construct initial directly in
    compute_dtype and apply it once, before tile zero. It can be a scalar or a
    tensor broadcastable to the leading dimensions; it is never mutated. Empty
    vectors return initial. PyTorch has no pipeline_stages scheduling parameter.
    """
    if values.ndim == 0:
        raise ValueError("ordered_reduce_serial requires at least one dimension")
    # Keep invalid configs invalid even when the loop has no tiles to evaluate.
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a positive power of two")
    if max_group_size <= 0 or (max_group_size & (max_group_size - 1)) != 0:
        raise ValueError("maximum group size must be a positive power of two")
    if section_count <= 0 or (section_count & (section_count - 1)) != 0:
        raise ValueError("section count must be a positive power of two")
    length = values.shape[-1]
    n_tiles = (length + block_size - 1) // block_size
    total = torch.as_tensor(initial, device=values.device, dtype=compute_dtype)
    total = torch.broadcast_to(total, values.shape[:-1])
    for tile in range(0, n_tiles):
        partial = ordered_reduce_partial(
            values,
            tile,
            block_size=block_size,
            max_group_size=max_group_size,
            section_count=section_count,
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

    Mirrors Triton's ordered_reduce_merge, including its unrolled groups and
    incomplete tail. Storage must already have compute_dtype; no hidden cast
    changes partials. Every combine is left-associated. Empty partials return
    initial. Leading dimensions batch independent merges; inputs are not mutated.
    """
    if partials.ndim == 0:
        raise ValueError("ordered_reduce_merge requires at least one dimension")
    total = torch.as_tensor(initial, device=partials.device, dtype=compute_dtype)
    total = torch.broadcast_to(total, partials.shape[:-1])
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


def sum_tile(
    values: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    max_group_size: int = MAX_GROUP_SIZE,
    section_count: int = SECTION_COUNT,
) -> torch.Tensor:
    """Sum [..., 2**n] values, using their dtype unless overridden.

    Mirrors Triton's sum_tile. compute_dtype=None uses values.dtype without
    widening; an explicit dtype converts values before the tree.
    Invalid lanes must already contain zero. The
    default group/section sizes are 8/4; changing either changes the tree.
    No whole-vector seed or fold across separate tiles is added here.
    """
    _validate_public_inputs(values)
    compute_dtype = values.dtype if compute_dtype is None else compute_dtype
    return ordered_tile_reduce(
        values.to(compute_dtype),
        max_group_size=max_group_size,
        section_count=section_count,
        combine_fn=reduce_add,
    )


def product_tile(
    values: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    max_group_size: int = MAX_GROUP_SIZE,
    section_count: int = SECTION_COUNT,
) -> torch.Tensor:
    """Multiply [..., 2**n] values, using their dtype unless overridden.

    Mirrors Triton's product_tile. compute_dtype=None uses values.dtype without
    widening; an explicit dtype converts values before the tree.
    Invalid lanes must already contain one.
    max_group_size/section_count default to 8/4, as in sum_tile. No whole-vector
    seed or inter-tile fold is inserted; conversion happens before the tree.
    """
    _validate_public_inputs(values)
    compute_dtype = values.dtype if compute_dtype is None else compute_dtype
    return ordered_tile_reduce(
        values.to(compute_dtype),
        max_group_size=max_group_size,
        section_count=section_count,
        combine_fn=reduce_multiply,
    )


def sum_vector(
    values: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    block_size: int = TILE_SIZE,
    max_group_size: int = MAX_GROUP_SIZE,
    section_count: int = SECTION_COUNT,
) -> torch.Tensor:
    """Eager fixed-order reference for the Triton sum_vector helper.

    Args:
        values: CPU or GPU tensor shaped [..., length]. Noncontiguous inputs
            are accepted; each leading index identifies an independent vector.
        compute_dtype: Dtype for input conversion, additions, partials and output.
            None (the default) uses values.dtype without widening.
            Use an integer dtype to count boolean inputs. PyTorch
            must support the chosen casts and operations on the input device.
        block_size: Positive power-of-two tile width, defaulting to 1024;
            this does not limit the complete vector length.
            Corresponds to Triton's BLOCK_SIZE. Pad the final tile with zero,
            then fold tile totals in ascending order, starting from zero once.
        max_group_size: Positive power-of-two maximum adjacent group size,
            defaulting to 8. Within each tile, accumulate each group sequentially.
        section_count: Positive power-of-two maximum number of sections,
            defaulting to 4. Reduce group totals within sections, then section
            totals, with the explicit lower-half/upper-half tree.

    Returns:
        A tensor shaped [...] on the input device, in the resolved compute dtype.
        An empty vector returns zero. Inputs are never modified.

    This is a fixed-path reference, not torch.sum's own reduction algorithm or
    an accuracy oracle. Compare on the same device and with the same precision.
    All three sizes must match the Triton path; changing them can change rounding.
    PyTorch dtype support does not establish support in a Triton backend.
    This reference does not define a Triton autograd formula or backward order;
    evaluate under no_grad when inputs require gradients.
    """
    _validate_public_inputs(values)
    compute_dtype = values.dtype if compute_dtype is None else compute_dtype
    return ordered_reduce_serial(
        values,
        initial=0,
        combine_fn=reduce_add,
        identity=0,
        compute_dtype=compute_dtype,
        block_size=block_size,
        max_group_size=max_group_size,
        section_count=section_count,
    )


def product_vector(
    values: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    block_size: int = TILE_SIZE,
    max_group_size: int = MAX_GROUP_SIZE,
    section_count: int = SECTION_COUNT,
) -> torch.Tensor:
    """Eager fixed-order reference for the Triton product_vector helper.

    Args:
        values: CPU or GPU tensor shaped [..., length]; strides are accepted.
        compute_dtype: Dtype for conversion, multiplication and output.
            None (the default) uses values.dtype without widening.
        block_size: Positive power-of-two tile width, defaulting to 1024;
            the complete vector may have any nonnegative length. Corresponds
            to Triton's BLOCK_SIZE. Pad with one and fold tile
            products in ascending order, starting from one once.
        max_group_size: Positive power-of-two maximum adjacent group size,
            defaulting to 8; multiply each group's values sequentially.
        section_count: Positive power-of-two maximum section count, defaulting
            to 4; use the same lower-half/upper-half tree as sum_vector.

    Returns:
        A tensor shaped [...] on the input device, in the resolved compute dtype.
        An empty vector returns one. Inputs are never modified.

    The same configuration, dtype and forward-only boundary as sum_vector
    apply. Fixed order does not prevent rounding, overflow or underflow.
    """
    _validate_public_inputs(values)
    compute_dtype = values.dtype if compute_dtype is None else compute_dtype
    return ordered_reduce_serial(
        values,
        initial=1,
        combine_fn=reduce_multiply,
        identity=1,
        compute_dtype=compute_dtype,
        block_size=block_size,
        max_group_size=max_group_size,
        section_count=section_count,
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
    Inputs are unchanged; use no_grad when they require gradients.
    """
    _validate_public_inputs(partials)
    compute_dtype = partials.dtype if compute_dtype is None else compute_dtype
    return ordered_reduce_merge(
        partials,
        initial=0,
        combine_fn=reduce_add,
        merge_unroll=merge_unroll,
        compute_dtype=compute_dtype,
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
    _validate_public_inputs(partials)
    compute_dtype = partials.dtype if compute_dtype is None else compute_dtype
    return ordered_reduce_merge(
        partials,
        initial=1,
        combine_fn=reduce_multiply,
        merge_unroll=merge_unroll,
        compute_dtype=compute_dtype,
    )


@dataclass(frozen=True)
class OrderedReduceResult:
    """PyTorch-only diagnostic trace of the partial and merge stages."""

    partials: torch.Tensor
    """compute_dtype [..., ceil(length / block_size)] totals in ascending tile order."""

    total: torch.Tensor
    """compute_dtype [...] left fold of partials, starting with initial once."""


def ordered_reduce(
    values: torch.Tensor,
    *,
    block_size: int,
    max_group_size: int,
    section_count: int,
    combine_fn: _Combine,
    identity: bool | int | float,
    initial: bool | int | float | torch.Tensor,
    compute_dtype: torch.dtype,
) -> OrderedReduceResult:
    """Collect inspectable partials and total for operator-stage diagnostics.

    This PyTorch-only collector composes ordered_reduce_partial and
    ordered_reduce_merge; it does not define another tree. Use the matching
    stage helpers or sum_vector/product_vector when only a total is needed.
    Leading dimensions batch independent vectors; the last dimension must be
    nonempty. All arithmetic parameters are required, with the same meanings
    as in the stage helpers. initial never enters the unseeded partials.
    """
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("ordered_reduce requires values with a nonempty last dimension")
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        raise ValueError("block_size must be a positive power of two")
    length = values.shape[-1]
    n_tiles = (length + block_size - 1) // block_size
    partials = []
    for tile in range(n_tiles):
        partials.append(
            ordered_reduce_partial(
                values,
                tile,
                block_size=block_size,
                max_group_size=max_group_size,
                section_count=section_count,
                combine_fn=combine_fn,
                identity=identity,
                compute_dtype=compute_dtype,
            )
        )
    partials = torch.stack(partials, dim=-1)
    total = ordered_reduce_merge(
        partials,
        initial,
        combine_fn=combine_fn,
        merge_unroll=1,
        compute_dtype=compute_dtype,
    )
    return OrderedReduceResult(partials=partials, total=total)
