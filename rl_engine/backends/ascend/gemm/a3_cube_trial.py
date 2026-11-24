# SPDX-License-Identifier: Apache-2.0
"""Opt-in A3 deterministic BF16 Qwen3-8B gate/up projection (64x128 output tiles)."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _thirtytwo(
    X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr
):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    group = tl.program_id(2)
    k0 = group * 1024 + tl.arange(0, 32)
    left1 = tl.full((BM, BN), 0, tl.float32)
    left2 = tl.full((BM, BN), 0, tl.float32)
    left4 = tl.full((BM, BN), 0, tl.float32)
    left8 = tl.full((BM, BN), 0, tl.float32)
    left16 = tl.full((BM, BN), 0, tl.float32)
    for i in tl.static_range(32):
        k = k0 + i * 32
        a = tl.load(X + r[:, None] * K + k[None, :], r[:, None] < M, other=0)
        b = tl.load(W + c[:, None] * K + k[None, :], c[:, None] < N, other=0)
        v = tl.dot(a, tl.trans(b), out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
        if i % 2 == 0:
            left1 = v
        else:
            v = (left1 + v).to(tl.bfloat16).to(tl.float32)
            if i % 4 == 1:
                left2 = v
            else:
                v = (left2 + v).to(tl.bfloat16).to(tl.float32)
                if i % 8 == 3:
                    left4 = v
                else:
                    root = (left4 + v).to(tl.bfloat16)
                    if i % 16 == 7:
                        left8 = root.to(tl.float32)
                    else:
                        final = (left8 + root.to(tl.float32)).to(tl.bfloat16)
                        if i == 15:
                            left16 = final.to(tl.float32)
                        else:
                            final = (left16 + final.to(tl.float32)).to(tl.bfloat16)
                            tl.store(
                                Y + (group * M + r[:, None]) * N + c[None, :],
                                final,
                                (r[:, None] < M) & (c[None, :] < N),
                            )


@triton.jit
def _pair(a, b):
    return (a.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _four(P, offset, size, mask):
    a = tl.load(P + offset, mask, other=0)
    b = tl.load(P + size + offset, mask, other=0)
    c = tl.load(P + 2 * size + offset, mask, other=0)
    d = tl.load(P + 3 * size + offset, mask, other=0)
    return _pair(_pair(a, b), _pair(c, d))


@triton.jit
def _merge(P, Y, size: tl.constexpr, COUNT: tl.constexpr, B: tl.constexpr):
    offset = tl.program_id(0) * B + tl.arange(0, B)
    mask = offset < size
    a = _four(P, offset, size, mask)
    if COUNT >= 8:
        b = _four(P + 4 * size, offset, size, mask)
    if COUNT == 4:
        tl.store(Y + offset, a, mask)
    elif COUNT == 8:
        tl.store(Y + offset, _pair(a, b), mask)
    else:
        c = _four(P + 8 * size, offset, size, mask)
        d = _four(P + 12 * size, offset, size, mask)
        tl.store(Y + offset, _pair(_pair(a, b), _pair(c, d)), mask)


def merge(parts):
    assert parts.shape[0] in (2, 4, 8) and parts.is_contiguous()
    if parts.shape[0] == 2:
        return parts[0] + parts[1]
    out = torch.empty_like(parts[0])
    _merge[(triton.cdiv(out.numel(), 1024),)](parts, out, out.numel(), parts.shape[0], 1024)
    return out


def project(x, w, bm, bn):
    m, k = x.shape
    n = w.shape[0]
    assert k % 1024 == 0
    count = k // 1024
    out = torch.empty((count, m, n), device=x.device, dtype=x.dtype)
    _thirtytwo[(triton.cdiv(m, bm), triton.cdiv(n, bn), count)](x, w, out, m, n, k, bm, bn)
    return out


def gate_up_linear(x, weight):
    # Limit intermediate storage to the measured 128-row tile batch (12 MiB).
    if os.getenv("RLK_A3_FFN_TRACE") == "1":
        print(f"RLK_A3_CUBE pid={os.getpid()} rows={x.shape[0]} K=4096 N=12288", flush=True)
    if x.shape[0] <= 128:
        return merge(project(x, weight, 64, 128))
    output = x.new_empty((x.shape[0], weight.shape[0]))
    for start in range(0, x.shape[0], 128):
        stop = min(start + 128, x.shape[0])
        output[start:stop].copy_(merge(project(x[start:stop], weight, 64, 128)))
    return output
