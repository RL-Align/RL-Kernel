# SPDX-License-Identifier: Apache-2.0
"""P6 T05 NVIDIA kernel; host gates live in rl_engine.p6.combine."""

import triton
import triton.language as tl


@triton.jit
def _add_rn(a, b):
    # Opaque PTX prevents reassociation and preserves signed zero/subnormal results.
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _profile_error(value):
    bits = value.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    nonfinite = bits >= 0x7F800000
    subnormal = (bits != 0) & (bits < 0x00800000)
    return tl.where(nonfinite, 4, tl.where(subnormal, 2, 0))


@triton.jit
def combine_kernel(
    Rows,
    Lookup,
    Shared,
    Residual,
    Output,
    Status,
    Canonical,
    Partials,
    Routed,
    AfterShared,
    Precast,
    T: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    DEBUG: tl.constexpr,
):
    token = tl.program_id(0)
    tile = tl.program_id(1)
    h = tile * BLOCK + tl.arange(0, BLOCK)
    active = h < H
    offset = token * H + h
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    seen = tl.full((), False, tl.int1)
    errors = tl.full((BLOCK,), 0, tl.int32)
    for slot in tl.static_range(6):
        row = tl.load(Lookup + token * 6 + slot)
        valid = row >= 0
        value = tl.load(Rows + tl.maximum(row, 0) * H + h, mask=active & valid, other=0.0).to(
            tl.float32
        )
        errors |= _profile_error(value)
        # Starting with +0 + first row would erase an initial -0.
        next_acc = tl.where(seen, _add_rn(acc, value), value)
        acc = tl.where(valid, next_acc, acc)
        seen = seen | valid
        errors |= _profile_error(acc)
        if DEBUG:
            tl.store(Canonical + (token * 6 + slot) * H + h, value, mask=active)
            tl.store(Partials + (slot * T + token) * H + h, acc, mask=active)
    shared = tl.load(Shared + offset, mask=active, other=0.0).to(tl.float32)
    residual = tl.load(Residual + offset, mask=active, other=0.0).to(tl.float32)
    errors |= _profile_error(shared) | _profile_error(residual)
    after_shared = _add_rn(acc, shared)
    precast = _add_rn(after_shared, residual)
    errors |= _profile_error(after_shared) | _profile_error(precast)
    output = precast.to(tl.bfloat16, fp_downcast_rounding="rtne")
    errors |= _profile_error(output.to(tl.float32))
    tl.store(Output + offset, output, mask=active)
    if DEBUG:
        tl.store(Routed + offset, acc, mask=active)
        tl.store(AfterShared + offset, after_shared, mask=active)
        tl.store(Precast + offset, precast, mask=active)
    # Diagnostic reduction only; no row reduction or global atomics.
    status = tl.max(tl.where(active, errors, 0), axis=0)
    tl.store(Status + token * tl.cdiv(H, BLOCK) + tile, status)
