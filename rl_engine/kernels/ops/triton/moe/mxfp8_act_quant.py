# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton MXFP8 activation quantization (P5-1).

Bit-exact re-implementation of ``rl_engine.moe.mx_format.mx_quantize(x, "e4m3")``
(numeric profile ``oracle-fp32-serial-v1``):

- Block = 32 elements along the last dim; the amax reduction is row-local and
  never crosses a row (so it never crosses a rank).
- Shared scale is E8M0: ``code = clamp(floor(log2(amax)) - 8, -127, 127) + 127``
  with ``amax == 0`` mapped to code 127 (scale 1.0). The amax is reduced as an
  integer max over ``|x|`` bit patterns and ``floor(log2)`` is read straight off
  the exponent field after clamping to FLT_MIN — the exact integer equivalent
  of the oracle's ``frexp`` path, and immune to flush-to-zero.
- Elements are ``x / 2**(code - 127)`` in round-to-nearest FP32, clamped to
  +/-448 and cast to E4M3 with RNE (``cvt.rn.satfinite.e4m3x2.f32``).
- Non-finite input is rejected (fail-closed), matching ``_check_finite``.

Determinism: amax is an order-independent max over a fixed 32-element window,
every element is then transformed independently, so the output bytes do not
depend on block size, grid shape, or how many rows are in flight.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from rl_engine.kernels.ops.moe_common import (
    finalize_act_quant,
    validate_act_quant_input,
    validate_ste_grad,
)
from rl_engine.moe.mx_format import E4M3_MAX, E8M0_BIAS, EMAX_ELEM, MX_BLOCK, MXTensor

# Triton can only close over constexpr globals; values come from the contract.
_TL_E4M3_MAX = tl.constexpr(E4M3_MAX)
_TL_E8M0_BIAS = tl.constexpr(E8M0_BIAS)
_TL_EMAX_E4M3 = tl.constexpr(EMAX_ELEM["e4m3"])
_TL_ABS_MASK = tl.constexpr(0x7FFFFFFF)
_TL_EXP_MASK = tl.constexpr(0x7F800000)
_TL_FLT_MIN_BITS = tl.constexpr(0x00800000)  # 2**-126, smallest FP32 normal
_TL_SUBNORMAL_2POW_M127 = tl.constexpr(0x00400000)

# Above this many MX blocks (~8M elements) the kernel is DRAM bound; below it,
# kernel-launch latency dominates. See mxfp8_act_quant_fwd_triton().
_LARGE_INPUT_BLOCKS = 1 << 18


@triton.jit
def _mxfp8_act_quant_fwd_kernel(
    x_ptr,
    codes_ptr,
    scales_ptr,
    flag_ptr,
    n_blocks,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    MX_BLOCK: tl.constexpr,
):
    # int64 offsets: tl.program_id / tl.arange are int32 and would wrap past
    # 2**31 elements.
    pid = tl.program_id(0).to(tl.int64)
    block_ids = pid * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM).to(tl.int64)
    block_mask = block_ids < n_blocks
    lanes = tl.arange(0, MX_BLOCK).to(tl.int64)
    offs = block_ids[:, None] * MX_BLOCK + lanes[None, :]
    mask = block_mask[:, None]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # |x| as a bit pattern: for non-negative floats the unsigned pattern is
    # monotone in the value, so amax is an integer max (no flush-to-zero).
    abs_bits = x.to(tl.uint32, bitcast=True) & _TL_ABS_MASK
    amax_bits = tl.max(abs_bits, axis=1)

    # Fail-closed: inf/NaN patterns are >= 0x7F800000, so the tile's largest
    # |x| pattern tells whether any element is non-finite.
    if tl.max(amax_bits, axis=0) >= _TL_EXP_MASK:
        tl.atomic_max(flag_ptr, 1)

    # floor(log2(amax)) from the exponent field (amax clamped to FLT_MIN first,
    # so the value is always normal and the field is exact).
    exp_field = tl.maximum(amax_bits, _TL_FLT_MIN_BITS) >> 23
    shared_exp = exp_field.to(tl.int32) - _TL_E8M0_BIAS - _TL_EMAX_E4M3
    shared_exp = tl.minimum(tl.maximum(shared_exp, -_TL_E8M0_BIAS), _TL_E8M0_BIAS)
    code = tl.where(amax_bits == 0, _TL_E8M0_BIAS, shared_exp + _TL_E8M0_BIAS)

    # scale = 2**(code - 127), built from bits so no libm rounding and no
    # flush-to-zero can touch it (code 0 is the subnormal 2**-127).
    scale_bits = tl.where(code > 0, code.to(tl.uint32) << 23, _TL_SUBNORMAL_2POW_M127)
    scale = scale_bits.to(tl.float32, bitcast=True)

    scaled = tl.fdiv(x, scale[:, None], ieee_rounding=True)
    scaled = tl.minimum(tl.maximum(scaled, -_TL_E4M3_MAX), _TL_E4M3_MAX)
    out = scaled.to(tl.float8e4nv).to(tl.uint8, bitcast=True)

    tl.store(codes_ptr + offs, out, mask=mask)
    tl.store(scales_ptr + block_ids, code.to(tl.uint8), mask=block_mask)


@triton.jit
def _ste_copy_kernel(dy_ptr, dx_ptr, n_elements, BLOCK: tl.constexpr):
    """Pure copy (dX = dY); sized so the STE runs at copy bandwidth."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    tl.store(dx_ptr + offs, tl.load(dy_ptr + offs, mask=mask), mask=mask)


def mxfp8_act_quant_fwd_triton(x: Tensor, check_finite: bool = True) -> MXTensor:
    """BF16/FP16/FP32 ``[..., K]`` -> MXFP8 (E4M3 codes + block-32 E8M0 scales).

    ``check_finite`` reads back the kernel's fail-closed flag and therefore
    costs one device sync per call; it is only turned off for throughput
    measurement (the providers always keep it on).
    """
    x = validate_act_quant_input(x, "triton")
    shape = tuple(x.shape)
    n_blocks = x.numel() // MX_BLOCK
    codes = torch.empty(shape, dtype=torch.uint8, device=x.device)
    scales = torch.empty((*shape[:-1], shape[-1] // MX_BLOCK), dtype=torch.uint8, device=x.device)
    # The flag is only read back when the fail-closed check is on, so skip the
    # memset otherwise.
    flag = (
        torch.zeros(1, dtype=torch.int32, device=x.device)
        if check_finite
        else torch.empty(1, dtype=torch.int32, device=x.device)
    )
    if n_blocks == 0:
        return MXTensor(codes=codes, scales=scales, elem_format="e4m3", shape=shape)

    # One program per 32 MX blocks (a [32, 32] tile). Large inputs are pure
    # DRAM streaming and run best with a single warp holding the whole tile
    # (32 elements per thread, no cross-lane reduction); small inputs are
    # launch-latency bound, where more warps shave the tail. Tuned on H100;
    # the emitted bytes are identical for every configuration.
    blocks_per_program = 32
    num_warps = 1 if n_blocks >= _LARGE_INPUT_BLOCKS else 4
    grid = (triton.cdiv(n_blocks, blocks_per_program),)
    _mxfp8_act_quant_fwd_kernel[grid](
        x,
        codes,
        scales,
        flag,
        n_blocks,
        BLOCKS_PER_PROGRAM=blocks_per_program,
        MX_BLOCK=MX_BLOCK,
        num_warps=num_warps,
    )
    return finalize_act_quant(codes, scales, flag, check_finite, shape)


def mxfp8_act_quant_bwd_triton(dy: Tensor) -> Tensor:
    """Straight-through estimator: ``dX = dY`` for any floating dtype (contiguous, same shape)."""
    dy_c = validate_ste_grad(dy, "triton")
    dx = torch.empty_like(dy_c)
    n = dy_c.numel()
    if n == 0:
        return dx
    block = 4096
    _ste_copy_kernel[(triton.cdiv(n, block),)](dy_c, dx, n, BLOCK=block, num_warps=8)
    return dx
