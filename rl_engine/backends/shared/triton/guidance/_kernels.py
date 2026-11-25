# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

BLOCK = 256


@triton.jit
def _combine(V, T, Image, W, n, st, si, MIX_IMAGE: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(V + k, k < n, 0).to(tl.float32)
    t = tl.load(T + k, k < n, 0).to(tl.float32)
    w = t + st * (v - t)
    if MIX_IMAGE:
        i = tl.load(Image + k, k < n, 0).to(tl.float32)
        w = i + si * (w - i)
    tl.store(W + k, w, k < n)


@triton.jit
def _reduce(X, Y, Out, width, chunks, multiplier, PRODUCT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    row = pid // chunks
    lane = tl.arange(0, BLOCK)
    col = (pid % chunks) * BLOCK + lane
    idx = row * width + col
    values = tl.load(X + idx, col < width, 0).to(tl.float32)
    if PRODUCT:
        other = tl.load(Y + idx, col < width, 0).to(tl.float32)
        values = (values * multiplier) * other
    # 将相邻叶子配对，保留与 PyTorch 参考相同的加法顺序。
    for level in tl.static_range(8):
        left, right = tl.split(tl.reshape(values, (BLOCK >> (level + 1), 2)))
        values = left + right
    tl.store(Out + pid + tl.arange(0, 1), values)


def tree_product(x, y, rows, multiplier=1.0):
    width = x.numel() // rows
    leaves = True
    while True:
        chunks = triton.cdiv(width, BLOCK)
        out = torch.empty((rows, chunks), dtype=torch.float32, device=x.device)
        _reduce[(rows * chunks,)](
            x, y, out, width, chunks, multiplier, leaves, BLOCK, enable_fp_fusion=False
        )
        if chunks == 1:
            return out
        x = out
        width = chunks
        leaves = False


@triton.jit
def _scale(A2, B2, row, valid, minimum, eps):
    a = tl.sqrt_rn(tl.load(A2 + row, valid, 0))
    b = tl.sqrt_rn(tl.load(B2 + row, valid, 0))
    denom = b + eps
    ratio = tl.div_rn(a, denom)
    scale = tl.clamp(ratio, minimum, 1.0, propagate_nan=tl.PropagateNan.ALL)
    return a, b, denom, ratio, scale


@triton.jit
def _output(
    W, Image, A2, B2, Out, n, width, si, minimum, eps, POST_IMAGE: tl.constexpr, BLOCK: tl.constexpr
):
    k = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = k < n
    _, _, _, _, scale = _scale(A2, B2, k // width, valid, minimum, eps)
    out = tl.load(W + k, valid, 0) * scale
    if POST_IMAGE:
        i = tl.load(Image + k, valid, 0).to(tl.float32)
        out = i + si * (out - i)
    tl.store(Out + k, out, valid)


@triton.jit
def _backward(
    G,
    V,
    W,
    A2,
    B2,
    Dot,
    DV,
    DT,
    DI,
    n,
    width,
    st,
    si,
    minimum,
    eps,
    HAS_IMAGE: tl.constexpr,
    POST_IMAGE: tl.constexpr,
    NEED_V: tl.constexpr,
    NEED_T: tl.constexpr,
    NEED_I: tl.constexpr,
    BLOCK: tl.constexpr,
):
    k = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = k < n
    row = k // width
    g = tl.load(G + k, valid, 0).to(tl.float32)
    a, b, denom, ratio, scale = _scale(A2, B2, row, valid, minimum, eps)
    dot = tl.load(Dot + row, valid, 0)
    factor = tl.div_rn(tl.where((ratio > minimum) & (ratio < 1.0), dot, 0.0), denom)
    mixed_unit = tl.div_rn(tl.load(W + k, valid, 0), tl.where(b == 0, 1.0, b))
    g_scaled = g
    if POST_IMAGE:
        g_scaled = si * g
    dw = g_scaled * scale - (factor * scale) * mixed_unit
    du = dw
    if HAS_IMAGE and not POST_IMAGE:
        du = si * dw
    if NEED_V:
        v = tl.load(V + k, valid, 0).to(tl.float32)
        dv = factor * tl.div_rn(v, tl.where(a == 0, 1.0, a)) + st * du
        tl.store(DV + k, dv, valid)
    if NEED_T:
        tl.store(DT + k, du - st * du, valid)
    if NEED_I:
        if POST_IMAGE:
            di = g - g_scaled
        elif HAS_IMAGE:
            di = dw - du
        else:
            di = tl.full((BLOCK,), 0, tl.float32)
        tl.store(DI + k, di, valid)


def forward(v, t, image, rows, st, si, post_image, minimum, eps):
    n = v.numel()
    grid = (triton.cdiv(n, BLOCK),)
    mixed = torch.empty_like(v, dtype=torch.float32)
    out = torch.empty_like(v)
    has_image = image is not None and si > 1.0
    img_ptr = image if has_image else v
    _combine[grid](
        v, t, img_ptr, mixed, n, st, si, has_image and not post_image, BLOCK, enable_fp_fusion=False
    )
    a2 = tree_product(v, v, rows)
    b2 = tree_product(mixed, mixed, rows)
    _output[grid](
        mixed,
        img_ptr,
        a2,
        b2,
        out,
        n,
        n // rows,
        si,
        minimum,
        eps,
        has_image and post_image,
        BLOCK,
        enable_fp_fusion=False,
    )
    return out, mixed, a2, b2


def backward(grad, v, mixed, a2, b2, st, si, post_image, has_image, minimum, eps, needs):
    rows = a2.numel()
    post_image = post_image and has_image
    dot = tree_product(grad, mixed, rows, si if post_image else 1.0)
    grads = tuple(torch.empty_like(v) if needed else None for needed in needs)
    pointers = tuple(tensor if tensor is not None else v for tensor in grads)
    _backward[(triton.cdiv(v.numel(), BLOCK),)](
        grad,
        v,
        mixed,
        a2,
        b2,
        dot,
        *pointers,
        v.numel(),
        v.numel() // rows,
        st,
        si,
        minimum,
        eps,
        has_image,
        post_image,
        *needs,
        BLOCK,
        enable_fp_fusion=False,
    )
    return grads
