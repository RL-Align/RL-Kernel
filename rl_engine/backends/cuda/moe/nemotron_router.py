# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Nemotron Nano single-device router with provisional RFC #434 ordering.

SM90 CUDA only. Forward and per-token gradients preserve a fixed FP32 tree;
weight gradients use the complete logical token order. No atomic reductions.
"""

import torch
import triton
import triton.language as tl

CONTRACT_VERSION = "nemotron-router-sm90-v3"
# Bound every 32-bit index product, including the six-copy packed payload.
MAX_TOKENS = 65536


@triton.jit
def _route_from_scores(s, B, IDS, W, D, t):
    e = tl.arange(0, 128)
    lane = tl.arange(0, 8)
    choices = s + tl.load(B + e)
    choices = tl.where(choices == choices, choices, -float("inf"))
    ids = tl.full((8,), 128, tl.int32)
    for slot in range(6):
        _, winner = tl.max(choices, 0, return_indices=True, return_indices_tie_break_left=True)
        ids = tl.where(lane == slot, winner, ids)
        choices = tl.where(e == winner, -float("inf"), choices)
    # Only eight lanes need sorting/gathering after six deterministic argmaxes.
    ids = tl.sort(ids, descending=False)
    values = tl.gather(s, tl.minimum(ids, 127), 0)
    denom = tl.full((), 0.0, tl.float32)
    for slot in range(6):
        denom = denom + tl.sum(tl.where(lane == slot, values, 0.0), 0)
    denom = denom + 1.0e-20
    tl.store(D + t, denom)
    tl.store(IDS + t * 6 + lane, ids, lane < 6)
    tl.store(W + t * 6 + lane, (values / denom) * 2.5, lane < 6)


@triton.jit
def _route_fwd(S, B, IDS, W, D):
    t = tl.program_id(0)
    e = tl.arange(0, 128)
    _route_from_scores(tl.load(S + t * 128 + e), B, IDS, W, D, t)


@triton.jit
def _merge_and_route(P, B, SCORES, IDS, W, D, T: tl.constexpr):
    t = tl.program_id(0)
    e = tl.arange(0, 128)
    segment = tl.arange(0, 32)
    partials = tl.load(
        P + segment[:, None] * T * 128 + t * 128 + e[None, :], segment[:, None] < 21, 0
    )
    logits = tl.sum(partials, 0)
    scores = 1.0 / (1.0 + tl.exp(-logits))
    tl.store(SCORES + t * 128 + e, scores)
    _route_from_scores(scores, B, IDS, W, D, t)


@triton.jit
def _route_bwd(S, IDS, D, G, DS, SIGMOID: tl.constexpr = False):
    t = tl.program_id(0)
    e = tl.arange(0, 128)
    denom = tl.load(D + t)
    dot = tl.full((), 0.0, tl.float32)
    for slot in range(6):
        idx = tl.load(IDS + t * 6 + slot)
        score = tl.load(S + t * 128 + idx)
        grad = tl.load(G + t * 6 + slot)
        dot = dot + grad * score
    result = tl.full((128,), 0.0, tl.float32)
    for slot in range(6):
        idx = tl.load(IDS + t * 6 + slot)
        grad = tl.load(G + t * 6 + slot)
        ds = (grad / denom - (dot / denom) / denom) * 2.5
        result = tl.where(e == idx, ds, result)
    if SIGMOID:
        scores = tl.load(S + t * 128 + e)
        result = (result * (1.0 - scores)) * scores
    tl.store(DS + t * 128 + e, result)


class _RouteScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, bias):
        n = scores.shape[0]
        ids = torch.empty((n, 6), device=scores.device, dtype=torch.int64)
        weights = torch.empty((n, 6), device=scores.device, dtype=torch.float32)
        denom = torch.empty(n, device=scores.device, dtype=torch.float32)
        if n:
            _route_fwd[(n,)](scores, bias, ids, weights, denom, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(scores, ids, denom)
        ctx.mark_non_differentiable(ids)
        return ids, weights

    @staticmethod
    def backward(ctx, grad_ids, grad_weights):
        scores, ids, denom = ctx.saved_tensors
        ds = torch.empty_like(scores)
        if scores.shape[0]:
            _route_bwd[(scores.shape[0],)](
                scores,
                ids,
                denom,
                grad_weights.contiguous(),
                ds,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return ds, None


def route_scores_cuda(scores, correction_bias):
    """Checked experimental interface. Validation syncs; benchmark separately.

    Non-finite inputs fail closed, including overflow of corrected scores.
    This primitive implements only the fixed Nano 128-expert/top-6 contract.
    """
    if not scores.is_cuda or scores.dtype != torch.float32:
        raise ValueError("FP32 CUDA scores required")
    if scores.ndim != 2 or scores.shape[1] != 128 or not scores.is_contiguous():
        raise ValueError("contiguous [T,128] scores required")
    if (
        correction_bias.shape != (128,)
        or correction_bias.dtype != torch.float32
        or correction_bias.device != scores.device
        or not correction_bias.is_contiguous()
    ):
        raise ValueError("contiguous FP32 bias[128] on input device required")
    if (
        not torch.isfinite(scores).all()
        or not torch.isfinite(correction_bias).all()
        or not torch.isfinite(scores + correction_bias).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError("finite sigmoid scores and non-overflowing corrected scores required")
    return _RouteScores.apply(scores, correction_bias)


@triton.jit
def _backward_partials(
    A, B, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, CH: tl.constexpr, BN: tl.constexpr
):
    # Fixed output tiles and ordered IEEE FP32 products within each K segment.
    m = tl.program_id(0) * 64 + tl.arange(0, 64)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, 16)
    acc = tl.full((64, BN), 0, tl.float32)
    for block in range(tl.cdiv(tl.minimum(K, CH), 16)):
        kk = tl.program_id(2) * CH + block * 16 + k
        a = tl.load(A + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0).to(
            tl.float32
        )
        b = tl.load(B + kk[:, None] * N + n[None, :], (kk[:, None] < K) & (n[None, :] < N), 0).to(
            tl.float32
        )
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(
        P + tl.program_id(2) * M * N + m[:, None] * N + n[None, :],
        acc,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def _backward_merge(P, Y, M: tl.constexpr, N: tl.constexpr, S: tl.constexpr, R: tl.constexpr):
    i = tl.program_id(0) * 128 + tl.arange(0, 128)
    s = tl.arange(0, R)
    values = tl.load(
        P + s[:, None] * M * N + i[None, :], (s[:, None] < S) & (i[None, :] < M * N), 0
    )
    tl.store(Y + i, tl.sum(values, 0), i < M * N)


def _mm(a, b, chunk, out_dtype=torch.float32):
    a, b = a.contiguous(), b.contiguous()
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    if not m or not n:
        return out
    if not k:
        return out.zero_()
    segments = triton.cdiv(k, chunk)
    partials = (
        out
        if segments == 1
        else torch.empty((segments, m, n), device=a.device, dtype=torch.float32)
    )
    # Narrow dX tiles reduce per-thread accumulator pressure without changing K order.
    bn = 64
    _backward_partials[(triton.cdiv(m, 64), triton.cdiv(n, bn), segments)](
        a,
        b,
        partials,
        m,
        n,
        k,
        chunk,
        bn,
        num_warps=4,
        # A single stage reduces BF16 weight-gradient resource pressure.
        # Keep the K blocks, output tiles and reduction tree unchanged.
        num_stages=1 if b.dtype == torch.bfloat16 else 3,
        enable_fp_fusion=False,
    )
    if segments > 1:
        _backward_merge[(triton.cdiv(m * n, 128),)](
            partials,
            out,
            m,
            n,
            segments,
            triton.next_power_of_2(segments),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _project_partials(X, WT, PARTIALS, T: tl.constexpr):
    # Fixed 32x64 output tiles and 21 independent 128-element K segments.
    # WT is contiguous [2688,128]; each segment visits four 32-wide blocks.
    m = tl.program_id(0) * 32 + tl.arange(0, 32)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 32)
    segment = tl.program_id(2)
    acc = tl.full((32, 64), 0.0, tl.float32)
    for block in range(4):
        kk = segment * 128 + block * 32 + k
        x = tl.load(X + m[:, None] * 2688 + kk[None, :], m[:, None] < T, 0)
        w = tl.load(WT + kk[:, None] * 128 + n[None, :])
        acc = tl.dot(x.to(tl.float32), w, acc, input_precision="ieee")
    tl.store(
        PARTIALS + segment * T * 128 + m[:, None] * 128 + n[None, :],
        acc,
        m[:, None] < T,
    )


@triton.jit
def _project_merge(PARTIALS, OUT, T: tl.constexpr):
    # A fixed 32-leaf FP32 tree: segment leaves 21..31 are exact zero.
    n = tl.program_id(0) * 128 + tl.arange(0, 128)
    segment = tl.arange(0, 32)
    partials = tl.load(
        PARTIALS + segment[:, None] * T * 128 + n[None, :],
        segment[:, None] < 21,
        0,
    )
    tl.store(OUT + n, tl.sum(partials, 0))


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        out = torch.empty((x.shape[0], 128), device=x.device, dtype=torch.float32)
        if x.shape[0]:
            transposed_weight = weight.t().contiguous()
            partials = torch.empty((21, x.shape[0], 128), device=x.device, dtype=torch.float32)
            _project_partials[(triton.cdiv(x.shape[0], 32), 2, 21)](
                x,
                transposed_weight,
                partials,
                x.shape[0],
                num_warps=2,
                num_stages=3,
                enable_fp_fusion=False,
            )
            _project_merge[(x.shape[0],)](
                partials, out, x.shape[0], num_warps=4, enable_fp_fusion=False
            )
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        dx = _mm(grad, weight, 128).to(x.dtype) if ctx.needs_input_grad[0] else None
        dw = _mm(grad.t(), x, 512).to(weight.dtype) if ctx.needs_input_grad[1] else None
        return dx, dw


def _project_route_forward(x, weight, bias):
    """Shared executable forward; qualification reads the actual fused scores."""
    n = x.shape[0]
    scores = torch.empty((n, 128), device=x.device, dtype=torch.float32)
    ids = torch.empty((n, 6), device=x.device, dtype=torch.int64)
    weights = torch.empty((n, 6), device=x.device, dtype=torch.float32)
    denom = torch.empty(n, device=x.device, dtype=torch.float32)
    if n:
        wt = weight.t().contiguous()
        partials = torch.empty((21, n, 128), device=x.device, dtype=torch.float32)
        _project_partials[(triton.cdiv(n, 32), 2, 21)](
            x, wt, partials, n, num_warps=2, num_stages=3, enable_fp_fusion=False
        )
        _merge_and_route[(n,)](
            partials, bias, scores, ids, weights, denom, n, num_warps=4, enable_fp_fusion=False
        )
    return ids, weights, scores, denom


class _ProjectRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ids, weights, scores, denom = _project_route_forward(x, weight, bias)
        ctx.save_for_backward(x, weight, scores, ids, denom)
        ctx.mark_non_differentiable(ids)
        return ids, weights

    @staticmethod
    def backward(ctx, grad_ids, grad_weights):
        x, weight, scores, ids, denom = ctx.saved_tensors
        grad = torch.empty_like(scores)
        if x.shape[0]:
            _route_bwd[(x.shape[0],)](
                scores,
                ids,
                denom,
                grad_weights.contiguous(),
                grad,
                True,
                num_warps=4,
                enable_fp_fusion=False,
            )
        dx = _mm(grad, weight, 128).to(x.dtype) if ctx.needs_input_grad[0] else None
        dw = _mm(grad.t(), x, 512).to(weight.dtype) if ctx.needs_input_grad[1] else None
        return dx, dw, None


@triton.jit
def _dispatch_counts(IDS, Counts, T: tl.constexpr, BLOCKS: tl.constexpr):
    block = tl.program_id(0)
    r = block * 256 + tl.arange(0, 256)
    ids = tl.load(IDS + r, r < T * 6, 0).to(tl.int32)
    # Invalid tail lanes must be excluded explicitly from the histogram.
    counts = tl.histogram(ids, 128, mask=r < T * 6)
    tl.store(Counts + tl.arange(0, 128) * BLOCKS + block, counts)


@triton.jit
def _integer_maximum(a, b):
    return tl.maximum(a, b)


@triton.jit
def _dispatch_map(IDS, Prefix, Offsets, Perm, Inverse, T: tl.constexpr, BLOCKS: tl.constexpr):
    block = tl.program_id(0)
    lane = tl.arange(0, 256)
    r = block * 256 + lane
    ids = tl.load(IDS + r, r < T * 6, 128).to(tl.int32)
    # Unique integer keys sort by expert, then original route position. The
    # sentinel expert 128 sorts tail lanes last and never writes an output.
    key = tl.sort(ids * 256 + lane, descending=False)
    expert = key // 256
    original = block * 256 + key % 256
    previous = tl.gather(expert, tl.maximum(lane - 1, 0), 0)
    starts = tl.where((lane == 0) | (expert != previous), lane, 0)
    group_start = tl.associative_scan(starts, 0, _integer_maximum)
    before = tl.load(Prefix + expert * BLOCKS + block - 1, (expert < 128) & (block > 0), 0)
    # Prefix counts of earlier blocks plus local stable rank recover the
    # global expert-major / route-major position without floating atomics.
    dest = tl.load(Offsets + expert, expert < 128, 0) + before + lane - group_start
    tl.store(Perm + dest, original, expert < 128)
    tl.store(Inverse + original, dest, expert < 128)


@triton.jit
def _pack(X, Inverse, Y, H: tl.constexpr):
    token = tl.program_id(0)
    h = tl.program_id(1) * 1024 + tl.arange(0, 1024)
    # Read a token once and reuse its exact payload for all six destinations.
    # Inverse is a bijection, so CTAs never race on an output address.
    value = tl.load(X + token * H + h, h < H, 0)
    for slot in tl.static_range(6):
        dest = tl.load(Inverse + token * 6 + slot)
        tl.store(Y + dest * H + h, value, h < H)


@triton.jit
def _unpack_grad(G, Inverse, DX, H: tl.constexpr, Route=None, HAS_ROUTE: tl.constexpr = False):
    token = tl.program_id(0)
    h = tl.program_id(1) * 1024 + tl.arange(0, 1024)
    total = tl.full((1024,), 0, tl.float32)
    for slot in range(6):
        r = tl.load(Inverse + token * 6 + slot)
        total = total + tl.load(G + r * H + h, h < H, 0).to(tl.float32)
    if HAS_ROUTE:
        # Preserve separate branch rounding before the autograd-equivalent sum.
        total = total.to(DX.dtype.element_ty).to(tl.float32)
        total = total + tl.load(Route + token * H + h, h < H, 0).to(tl.float32)
    tl.store(DX + token * H + h, total, h < H)


def _dispatch_forward(x, ids):
    t, h = x.shape
    r = t * 6
    blocks = triton.cdiv(r, 256)
    perm = torch.empty(r, device=x.device, dtype=torch.int64)
    inverse = torch.empty_like(perm)
    offsets = torch.zeros(129, device=x.device, dtype=torch.int64)
    packed = torch.empty((r, h), device=x.device, dtype=x.dtype)
    if t:
        counts = torch.empty((128, blocks), device=x.device, dtype=torch.int64)
        _dispatch_counts[(blocks,)](ids, counts, t, blocks)
        prefix = counts.cumsum(1)
        offsets[1:] = prefix[:, -1].cumsum(0)
        _dispatch_map[(blocks,)](ids, prefix, offsets, perm, inverse, t, blocks)
        _pack[(t, triton.cdiv(h, 1024))](x, inverse, packed, h)
    return perm, offsets, packed, inverse


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ids):
        t, h = x.shape
        perm, offsets, packed, inverse = _dispatch_forward(x, ids)
        ctx.save_for_backward(inverse)
        ctx.shape = (t, h)
        ctx.mark_non_differentiable(perm, offsets)
        return perm, offsets, packed

    @staticmethod
    def backward(ctx, gperm, goffsets, grad):
        (inverse,) = ctx.saved_tensors
        t, h = ctx.shape
        dx = torch.empty((t, h), device=grad.device, dtype=grad.dtype)
        if t:
            _unpack_grad[(t, triton.cdiv(h, 1024))](
                grad.contiguous(), inverse, dx, h, enable_fp_fusion=False
            )
        return dx, None


class _RouterDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ids, weights, scores, denom = _project_route_forward(x, weight, bias)
        perm, offsets, packed, inverse = _dispatch_forward(x, ids)
        ctx.save_for_backward(x, weight, scores, ids, denom, inverse)
        ctx.mark_non_differentiable(ids, perm, offsets)
        ctx.set_materialize_grads(False)
        return ids, weights, perm, offsets, packed

    @staticmethod
    def backward(ctx, grad_ids, grad_weights, grad_perm, grad_offsets, grad_packed):
        x, weight, scores, ids, denom, inverse = ctx.saved_tensors
        dx = dw = None
        if grad_weights is not None:
            grad = torch.empty_like(scores)
            if x.shape[0]:
                _route_bwd[(x.shape[0],)](
                    scores,
                    ids,
                    denom,
                    grad_weights.contiguous(),
                    grad,
                    True,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            if ctx.needs_input_grad[0]:
                dx = _mm(grad, weight, 128, out_dtype=x.dtype)
            if ctx.needs_input_grad[1]:
                dw = _mm(grad.t(), x, 512)
        if grad_packed is not None and ctx.needs_input_grad[0]:
            combined = torch.empty_like(x)
            if x.shape[0]:
                _unpack_grad[(x.shape[0], triton.cdiv(x.shape[1], 1024))](
                    grad_packed.contiguous(),
                    inverse,
                    combined,
                    x.shape[1],
                    Route=dx,
                    HAS_ROUTE=dx is not None,
                    enable_fp_fusion=False,
                )
            dx = combined
        return dx, dw, None


def nemotron_router_cuda(x, weight, correction_bias):
    """Single-device Nano router with a provisional deterministic ordering ABI.

    Inputs must be finite. Shapes/dtypes are validated without device sync.
    Weight gradients use increasing logical token order; externally accumulated
    microbatch weight gradients are not promised bitwise equivalence.
    """
    if not x.is_cuda or x.ndim != 2 or x.shape[1] != 2688:
        raise ValueError("CUDA x[T,2688] required")
    if x.shape[0] > MAX_TOKENS:
        raise ValueError(f"at most {MAX_TOKENS} tokens supported by this provider")
    if x.dtype not in (torch.bfloat16, torch.float32) or not x.is_contiguous():
        raise ValueError("contiguous BF16 or FP32 input required")
    if (
        weight.shape != (128, 2688)
        or weight.dtype != torch.float32
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError("contiguous FP32 weight[128,2688] on input device required")
    if (
        correction_bias.shape != (128,)
        or correction_bias.dtype != torch.float32
        or correction_bias.device != x.device
        or not correction_bias.is_contiguous()
    ):
        raise ValueError("contiguous FP32 bias[128] on input device required")
    if torch.version.hip is not None or torch.cuda.get_device_capability(x.device) != (
        9,
        0,
    ):
        raise ValueError("this provider is qualified only for NVIDIA SM90")
    return _RouterDispatch.apply(x, weight, correction_bias)


class NemotronRouterOp:
    """Registry entry point for the provisional single-device Nano router."""

    __call__ = staticmethod(nemotron_router_cuda)
