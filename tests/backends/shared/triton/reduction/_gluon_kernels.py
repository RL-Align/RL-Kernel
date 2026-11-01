# SPDX-License-Identifier: Apache-2.0
"""Suite-local Gluon callers, imported only after checking frontend availability."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from rl_engine.backends.shared.triton.reduction import gluon as ordered


@gluon.jit
def xor_pair(left, right):
    """Caller-owned operation, with no built-in-operation lookup in the helper."""
    return left ^ right


@gluon.jit
def reduction_probe(
    x,
    out,
    stride,
    LENGTH: gl.constexpr,
    WIDTH: gl.constexpr,
    BLOCK: gl.constexpr,
    GROUP: gl.constexpr,
    BATCH: gl.constexpr,
    ELEMENTS: gl.constexpr,
    OP: gl.constexpr,
    LOADED: gl.constexpr,
    COMPUTE: gl.constexpr,
    WARP_SIZE: gl.constexpr,
):
    """Reduce one logical row per program through the public loaded/pointer APIs."""
    row = gl.program_id(0).to(gl.int64)
    input = x + row * stride
    if LOADED:
        cols = gl.arange(0, WIDTH, layout=gl.BlockedLayout([1], [WARP_SIZE], [gl.num_warps()], [0]))
        input = gl.load(input + cols)
    if OP == "xor":
        value = ordered.reduce(
            input,
            COMBINE_FN=xor_pair,
            IDENTITY=0,
            length=LENGTH,
            COMPUTE_DTYPE=COMPUTE,
            BLOCK_SIZE=BLOCK,
            MAX_GROUP_SIZE=GROUP,
            TILES_PER_BATCH=BATCH,
            ELEMENTS_PER_THREAD=ELEMENTS,
            WARP_SIZE=WARP_SIZE,
        )
    elif OP == "sum":
        value = ordered.sum(input, LENGTH, COMPUTE, BLOCK, GROUP, BATCH, ELEMENTS, WARP_SIZE)
    else:
        value = ordered.product(input, LENGTH, COMPUTE, BLOCK, GROUP, BATCH, ELEMENTS, WARP_SIZE)
    gl.static_assert(value.dtype == out.dtype.element_ty, "result precision must precede storage")
    gl.store(out + row, value)


@gluon.jit
def partial_probe(
    x,
    partials,
    stride,
    LENGTH: gl.constexpr,
    BLOCK: gl.constexpr,
    GROUP: gl.constexpr,
    ELEMENTS: gl.constexpr,
    WARP_SIZE: gl.constexpr,
):
    """Embed the public loaded helper in independent, masked tile producers."""
    row = gl.program_id(0).to(gl.int64)
    tile = gl.program_id(1).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([ELEMENTS], [WARP_SIZE], [gl.num_warps()], [0])
    columns = tile * BLOCK + gl.arange(0, BLOCK, layout=layout)
    values = gl.load(x + row * stride + columns, columns < LENGTH, other=0).to(gl.float32)
    total = ordered.sum(values, MAX_GROUP_SIZE=GROUP, WARP_SIZE=WARP_SIZE)
    gl.store(partials + row * gl.cdiv(LENGTH, BLOCK) + tile, total)


@gluon.jit
def merge_probe(x, out, stride, COUNT: gl.constexpr, OP: gl.constexpr, WARP_SIZE: gl.constexpr):
    """Exercise convenience merges, which always use their operation's identity."""
    gl.static_assert(OP == "sum" or OP == "product", "expected sum or product")
    row = gl.program_id(0).to(gl.int64)
    if OP == "sum":
        value = ordered.sum_partials(x + row * stride, COUNT, WARP_SIZE=WARP_SIZE)
    else:
        value = ordered.product_partials(x + row * stride, COUNT, WARP_SIZE=WARP_SIZE)
    gl.store(out + row, value)


@gluon.jit
def seeded_merge_probe(
    x,
    out,
    initial_ptr,
    stride,
    COUNT: gl.constexpr,
    COMBINE_FN: gl.constexpr,
    WARP_SIZE: gl.constexpr,
):
    """Pass a distinct runtime seed per row through the generic merge API."""
    row = gl.program_id(0).to(gl.int64)
    initial = gl.load(initial_ptr + row)
    value = ordered.reduce_partials(
        x + row * stride, COUNT, COMBINE_FN, initial, WARP_SIZE=WARP_SIZE
    )
    gl.store(out + row, value)


@gluon.jit
def constexpr_seeded_merge_probe(
    x,
    out,
    stride,
    COUNT: gl.constexpr,
    INITIAL: gl.constexpr,
    WARP_SIZE: gl.constexpr,
):
    """Pass a Python float seed without first materializing it as a device tensor."""
    row = gl.program_id(0).to(gl.int64)
    value = ordered.reduce_partials(
        x + row * stride, COUNT, ordered.reduce_add, INITIAL, WARP_SIZE=WARP_SIZE
    )
    gl.store(out + row, value)
