# SPDX-License-Identifier: Apache-2.0
"""Task 7: fuse post mixing, its BF16 boundary and the next controller projection."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rounded(x):
    # Match the BF16 round-to-nearest-even boundary even in the CPU interpreter.
    u = x.to(tl.uint32, bitcast=True)
    hi = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    # Keep NaNs quiet rather than rounding a small payload down to infinity.
    hi = tl.where((u & 0x7FFFFFFF) > 0x7F800000, (u & 0xFFFF0000) | 0x00400000, hi)
    return hi.to(tl.float32, bitcast=True)


@triton.jit
def _pairs(v, W: tl.constexpr):
    # At stage i lane 0 holds the leading 2**(i+1) values in pair order.
    # Keep a fixed vector type through the loop to avoid recursive JIT calls.
    lane = tl.arange(0, W)
    for i in tl.static_range(tl.constexpr(W.bit_length() - 1)):
        v = v + tl.gather(v, tl.minimum(lane + (1 << i), W - 1), 0)
    return tl.sum(tl.gather(v, tl.full((1,), 0, tl.int32), 0), 0)


@triton.jit
def _post_project(Y, X, P, B, W, RES, LIN, SS, C: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    k = tl.program_id(1)
    j = tl.arange(0, 4)
    c = tl.arange(0, C)
    dest = j[:, None] * C + c[None, :]
    a = tl.load(X + t * 4 * C + c).to(tl.float32)[None, :] * tl.load(B + t * 16 + j)[:, None]
    b = (
        tl.load(X + t * 4 * C + C + c).to(tl.float32)[None, :]
        * tl.load(B + t * 16 + 4 + j)[:, None]
    )
    d = (
        tl.load(X + t * 4 * C + 2 * C + c).to(tl.float32)[None, :]
        * tl.load(B + t * 16 + 8 + j)[:, None]
    )
    e = (
        tl.load(X + t * 4 * C + 3 * C + c).to(tl.float32)[None, :]
        * tl.load(B + t * 16 + 12 + j)[:, None]
    )
    added = tl.load(P + t * 4 + j)[:, None] * tl.load(Y + t * C + c).to(tl.float32)[None, :]
    z = _rounded(((a + b) + (d + e)) + added)
    if k == 24:
        tl.store(RES + t * 4 * C + dest, z.to(tl.bfloat16))
        tl.store(SS + t, _pairs(tl.reshape(z * z, (4 * C,)), 4 * C))
    else:
        w = _rounded(tl.load(W + k * 4 * C + dest))
        tl.store(LIN + t * 24 + k, _pairs(tl.reshape(z * w, (4 * C,)), 4 * C))


def post_project(y, x, p, b, w):
    t, _, c = x.shape
    if not 1 <= c <= 4096 or c & (c - 1):
        raise ValueError("fused mHC requires power-of-two hidden size <=4096")
    res = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    lin = w.new_empty((t, 24))
    ss = w.new_empty((t,))
    if t:
        _post_project[(t, 25)](
            *(v.contiguous() for v in (y, x, p, b, w)),
            res,
            lin,
            ss,
            c,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return res, lin, ss
