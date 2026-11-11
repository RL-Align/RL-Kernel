# SPDX-License-Identifier: Apache-2.0
"""Optional shared Gluon callers; the host supplies the compilation target's warp width."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from rl_engine.backends.shared.triton.reduction import gluon as ordered


@gluon.jit
def row_sum(
    x,
    out,
    stride,
    LENGTH: gl.constexpr,
    BLOCK: gl.constexpr,
    GROUP: gl.constexpr,
    BATCH: gl.constexpr,
    ELEMENTS: gl.constexpr,
    WARP_SIZE: gl.constexpr,
):
    """Sum each contiguous row with the same arithmetic tree as the default backend."""
    row = gl.program_id(0).to(gl.int64)
    total = ordered.sum(
        x + row * stride,
        length=LENGTH,
        COMPUTE_DTYPE=gl.float32,
        BLOCK_SIZE=BLOCK,
        MAX_GROUP_SIZE=GROUP,
        TILES_PER_BATCH=BATCH,
        ELEMENTS_PER_THREAD=ELEMENTS,
        WARP_SIZE=WARP_SIZE,
    )
    gl.store(out + row, total)


@gluon.jit
def partial_sum(
    x,
    partials,
    stride,
    LENGTH: gl.constexpr,
    BLOCK: gl.constexpr,
    GROUP: gl.constexpr,
    ELEMENTS: gl.constexpr,
    WARP_SIZE: gl.constexpr,
):
    """Compute one fixed arithmetic tile per program, without seeding its result.

    Long rows can occupy many SMs. Each program stores its result by logical
    tile index; completion order never determines the subsequent addition order.
    The public loaded-vector helper remains embedded inside this kernel.
    """
    row = gl.program_id(0).to(gl.int64)
    tile = gl.program_id(1).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([ELEMENTS], [WARP_SIZE], [gl.num_warps()], [0])
    columns = tile * BLOCK + gl.arange(0, BLOCK, layout=layout)
    values = gl.load(x + row * stride + columns, columns < LENGTH, other=0).to(gl.float32)
    total = ordered.sum(values, MAX_GROUP_SIZE=GROUP, WARP_SIZE=WARP_SIZE)
    gl.store(partials + row * gl.cdiv(LENGTH, BLOCK) + tile, total)


@gluon.jit
def merge_sum(partials, out, COUNT: gl.constexpr, WARP_SIZE: gl.constexpr):
    """Consume stored tile results in the same order as the per-row kernel.

    The host launches this after partial_sum on the same stream. No second
    arithmetic tree, extra seed, or atomic addition is introduced here.
    """
    row = gl.program_id(0).to(gl.int64)
    total = ordered.sum_partials(partials + row * COUNT, COUNT, WARP_SIZE=WARP_SIZE)
    gl.store(out + row, total)
