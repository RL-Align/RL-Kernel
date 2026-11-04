# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared Triton provider for the Nemotron grouped expert MLP (CUDA and ROCm).

One source for both platforms: the CUDA column falls back to it when the SM90
extension is absent, and it is the only ROCm implementation of
`moe_experts_combine`.  Pinned schedule, no autotuning, no split-K, no atomics:
`BLOCK_M=128`, `BLOCK_N=64`, `BLOCK_K=32` (one FP32 leaf), 4 warps, 2 stages and
`enable_fp_fusion=False` on every launch, because a per-shape configuration
choice would itself break batch invariance.

Arithmetic, matching `nemotron-experts-sm90-v1` where Triton can express it:

* K is cut into 32-wide leaves; each leaf is one FP32 `tl.dot` rounded once to
  BF16, the same leaf width and the same single rounding as the native kernel
  (the in-leaf accumulation order is the backend's own, exactly as `mma.sync`
  and a scalar MAC differ inside one BF16 product);
* leaves fold in ascending order. The native provider merges them with the
  canonical midpoint tree; Triton's frontend cannot build a trace-time stack
  (kernels may not call plain-Python helpers, and `tl.static_range` forbids the
  list operations the stack algorithm needs), so the association differs. Folding
  is still fixed, atomic-free and batch invariant, and each provider is
  qualified against FP64 on its own branch;
* everything else is identical to the native provider: the relu-square leaf
  rounding, the FP32 down-projection, the ascending-slot FP32 combine, the single
  exit cast, and `dWeights` accumulating in FP32.

Cross-provider bitwise equality is not claimed.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - installations without Triton
    _TRITON_AVAILABLE = False

PROVIDER_ID = "rlkernel.nemotron_experts.triton.v1"

BLOCK_M = 128
BLOCK_N = 64
LEAF = 32
NUM_WARPS = 4
NUM_STAGES = 2
# Partial sums accumulate in FP32 and round to BF16 once per window of leaves.
GROUP_LEAVES = 8
_ELEMENT_BLOCK = 1024

if _TRITON_AVAILABLE:

    @triton.jit
    def _resolve_segment(
        OFFSETS, tile, rows, BM: tl.constexpr, NSEG: tl.constexpr, NSEG_P2: tl.constexpr
    ):
        """Map a row-tile index to (expert, first row, segment end, tile count)."""
        experts = tl.arange(0, NSEG_P2)
        known = experts < NSEG
        begin = tl.load(OFFSETS + experts, known, 0)
        end = tl.load(OFFSETS + experts + 1, known, 0)
        tiles = (end - begin + BM - 1) // BM
        prefix = tl.cumsum(tiles, 0)
        total = tl.sum(tiles, 0)
        expert = tl.minimum(tl.min(tl.where(prefix > tile, experts, NSEG_P2)), NSEG - 1)
        before = tl.sum(tl.where(experts < expert, tiles, 0), 0)
        first = tl.sum(tl.where(experts == expert, begin, 0), 0)
        last = tl.sum(tl.where(experts == expert, end, 0), 0)
        return expert, first + (tile - before) * BM, last, total

    @triton.jit
    def _invert_kernel(INVERSE, PERMUTATION, routes, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = index < routes
        # The router permutation is a bijection, so this scatter cannot collide.
        target = tl.load(PERMUTATION + index, mask, 0)
        tl.store(INVERSE + target, index.to(tl.int64), mask)

    @triton.jit
    def _grouped_rows_kernel(
        A,
        W,
        C,
        OFFSETS,
        rows,
        N,
        K,
        RELU2_A: tl.constexpr,
        OUT_FP32: tl.constexpr,
        NSEG: tl.constexpr,
        NSEG_P2: tl.constexpr,
        SINGLE: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        """Grouped GEMM with the weights stacked as [experts, N, K]."""
        tile = tl.program_id(0)
        col = tl.program_id(1) * BN
        if SINGLE:
            total = (rows + BM - 1) // BM
            if tile >= total:
                return
            expert = 0
            first = tile * BM
            last = rows
        else:
            expert, first, last, total = _resolve_segment(OFFSETS, tile, rows, BM, NSEG, NSEG_P2)
            if tile >= total:
                return
        rm = tl.arange(0, BM)
        rn = col + tl.arange(0, BN)
        row_mask = first + rm < last
        col_mask = rn < N
        weight_base = W + (expert * N + rn)[:, None] * K
        total_acc = tl.zeros((BM, BN), tl.bfloat16)
        window = tl.zeros((BM, BN), tl.float32)
        for leaf in tl.range(0, K // BK):
            koff = leaf * BK + tl.arange(0, BK)
            a = tl.load(A + (first + rm)[:, None] * K + koff[None, :], row_mask[:, None], 0.0)
            b = tl.load(weight_base + koff[None, :], col_mask[:, None], 0.0)
            if RELU2_A:
                gate = tl.maximum(a.to(tl.float32), 0.0)
                a = (gate * gate).to(tl.bfloat16)
            # One BF16 rounding per 32-wide leaf, then FP32 partial sums that
            # round once per window instead of once per leaf.
            window += tl.dot(a, tl.trans(b)).to(tl.bfloat16).to(tl.float32)
            flush = (leaf + 1) % GROUP == 0
            total_acc = tl.where(
                flush, (total_acc + window.to(tl.bfloat16)).to(tl.bfloat16), total_acc
            )
            window = tl.where(flush, 0.0, window)
        total_acc = (total_acc + window.to(tl.bfloat16)).to(tl.bfloat16)
        mask = row_mask[:, None] & col_mask[None, :]
        out = C + (first + rm)[:, None] * N + rn[None, :]
        if OUT_FP32:
            tl.store(out, total_acc.to(tl.float32), mask)
        else:
            tl.store(out, total_acc, mask)

    @triton.jit
    def _grouped_kn_kernel(
        A,
        W,
        C,
        AUX,
        OFFSETS,
        rows,
        N,
        K,
        EPI_RELU2_GRAD: tl.constexpr,
        NSEG: tl.constexpr,
        NSEG_P2: tl.constexpr,
        SINGLE: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        """Grouped GEMM with the weights stacked as [experts, K, N] (backward)."""
        tile = tl.program_id(0)
        col = tl.program_id(1) * BN
        if SINGLE:
            total = (rows + BM - 1) // BM
            if tile >= total:
                return
            expert = 0
            first = tile * BM
            last = rows
        else:
            expert, first, last, total = _resolve_segment(OFFSETS, tile, rows, BM, NSEG, NSEG_P2)
            if tile >= total:
                return
        rm = tl.arange(0, BM)
        rn = col + tl.arange(0, BN)
        row_mask = first + rm < last
        col_mask = rn < N
        weight_base = W + (expert * K + tl.arange(0, BK))[:, None] * N
        total_acc = tl.zeros((BM, BN), tl.bfloat16)
        window = tl.zeros((BM, BN), tl.float32)
        for leaf in tl.range(0, K // BK):
            koff = leaf * BK
            a = tl.load(
                A + (first + rm)[:, None] * K + (koff + tl.arange(0, BK))[None, :],
                row_mask[:, None],
                0.0,
            )
            b = tl.load(weight_base + koff * N + rn[None, :], col_mask[None, :], 0.0)
            window += tl.dot(a, b).to(tl.bfloat16).to(tl.float32)
            flush = (leaf + 1) % GROUP == 0
            total_acc = tl.where(
                flush, (total_acc + window.to(tl.bfloat16)).to(tl.bfloat16), total_acc
            )
            window = tl.where(flush, 0.0, window)
        total_acc = (total_acc + window.to(tl.bfloat16)).to(tl.bfloat16)
        mask = row_mask[:, None] & col_mask[None, :]
        value = total_acc.to(tl.float32)
        if EPI_RELU2_GRAD:
            up = tl.load(AUX + (first + rm)[:, None] * N + rn[None, :], mask, 0.0).to(tl.float32)
            value = value * (2.0 * tl.maximum(up, 0.0))
        tl.store(C + (first + rm)[:, None] * N + rn[None, :], value.to(tl.bfloat16), mask)

    @triton.jit
    def _wgrad_kernel(
        A,
        B,
        C,
        OFFSETS,
        rows,
        M,
        N,
        RELU2_B: tl.constexpr,
        NSEG: tl.constexpr,
        SINGLE: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        """dW[e, m, n] = sum over the expert's packed rows of A[r, m] * B[r, n]."""
        expert = tl.program_id(2)
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        if SINGLE:
            first = 0
            last = rows
        else:
            first = tl.load(OFFSETS + expert)
            last = tl.load(OFFSETS + expert + 1)
        m_mask = rm < M
        n_mask = rn < N
        keep = m_mask[:, None] & n_mask[None, :]
        out = C + (expert * M + rm)[:, None] * N + rn[None, :]
        # Every expert owns a tile, so an empty segment writes exact zeros.
        total = tl.zeros((BM, BN), tl.bfloat16)
        window = tl.zeros((BM, BN), tl.float32)
        a_base = A + rm[None, :]
        b_base = B + rn[None, :]
        for leaf in tl.range(0, tl.cdiv(last - first, BK)):
            rows_at = first + leaf * BK + tl.arange(0, BK)
            row_mask = rows_at < last
            a = tl.load(a_base + rows_at[:, None] * M, row_mask[:, None] & m_mask[None, :], 0.0)
            b = tl.load(b_base + rows_at[:, None] * N, row_mask[:, None] & n_mask[None, :], 0.0)
            if RELU2_B:
                gate = tl.maximum(b.to(tl.float32), 0.0)
                b = (gate * gate).to(tl.bfloat16)
            window += tl.dot(tl.trans(a), b).to(tl.bfloat16).to(tl.float32)
            flush = (leaf + 1) % GROUP == 0
            total = tl.where(flush, (total + window.to(tl.bfloat16)).to(tl.bfloat16), total)
            window = tl.where(flush, 0.0, window)
        total = (total + window.to(tl.bfloat16)).to(tl.bfloat16)
        tl.store(out, total, keep)

    @triton.jit
    def _combine_kernel(
        DOWN, INVERSE, WEIGHTS, SHARED, OUT, width, HAS_SHARED: tl.constexpr, BLOCK: tl.constexpr
    ):
        """Combine, shared add and the operator's only exit cast in one pass."""
        token = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < width
        base = token * 6
        offset = token * width + cols
        total = tl.zeros((BLOCK,), tl.float32)
        for slot in tl.static_range(6):
            row = tl.load(INVERSE + base + slot)
            weight = tl.load(WEIGHTS + base + slot)
            total += weight * tl.load(DOWN + row * width + cols, mask, 0.0)
        if HAS_SHARED:
            total += tl.load(SHARED + offset, mask, 0.0)
        tl.store(OUT + offset, total.to(tl.bfloat16), mask)

    @triton.jit
    def _dy_kernel(DY, GRAD, PERMUTATION, WEIGHTS, width, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < width
        original = tl.load(PERMUTATION + row)
        weight = tl.load(WEIGHTS + original)
        value = tl.load(GRAD + (original // 6) * width + cols, mask, 0.0).to(tl.float32)
        tl.store(DY + row * width + cols, (weight * value).to(tl.bfloat16), mask)

    @triton.jit
    def _dweights_kernel(DOWN, GRAD, INVERSE, OUT, width, BK: tl.constexpr):
        """FP32 route dot product; one program per route, slots come from INV."""
        route = tl.program_id(0)
        row = tl.load(INVERSE + route)
        token = route // 6
        cols = tl.arange(0, BK)
        down_base = DOWN + row * width
        grad_base = GRAD + token * width
        total = tl.zeros((BK,), tl.float32)
        for leaf in tl.range(0, width // BK):
            offset = leaf * BK + cols
            total += tl.load(down_base + offset) * tl.load(grad_base + offset).to(tl.float32)
        tl.store(OUT + route, tl.sum(total, 0))


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


class TritonNemotronExpertsOp:
    """Registry entry point for the shared Triton grouped expert operator."""

    provider_id = PROVIDER_ID

    def __init__(self) -> None:
        if not _TRITON_AVAILABLE:
            raise RuntimeError("triton is not installed; the shared experts provider needs it")
        if torch.version.hip is None and not torch.cuda.is_available():
            raise RuntimeError("the shared experts provider needs a CUDA or ROCm device")

    def __call__(
        self,
        packed: torch.Tensor,
        x: torch.Tensor | None,
        weights: torch.Tensor,
        permutation: torch.Tensor,
        offsets: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        w_sh1: torch.Tensor | None = None,
        w_sh2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _ExpertsCombine.apply(
            packed, x, weights, permutation, offsets, w_up, w_down, w_sh1, w_sh2
        )


def _rows_gemm(
    source,
    weight,
    out,
    aux,
    offsets,
    rows,
    reduction,
    width,
    experts,
    *,
    transpose_weights: bool,
    relu2_source: bool = False,
    out_fp32: bool = False,
    epi_relu2_grad: bool = False,
):
    """One grouped GEMM; `offsets=None` means the single segment [0, rows)."""
    single = offsets is None
    kernel = _grouped_kn_kernel if transpose_weights else _grouped_rows_kernel
    extra = (
        {"EPI_RELU2_GRAD": epi_relu2_grad}
        if transpose_weights
        else {"RELU2_A": relu2_source, "OUT_FP32": out_fp32}
    )
    grid = (
        triton.cdiv(rows, BLOCK_M) + 1 if single else triton.cdiv(rows, BLOCK_M) + experts,
        triton.cdiv(width, BLOCK_N),
    )
    # The backward kernels take the relu-square source for the derivative
    # epilogue; the forward kernels have no such operand.
    operands = (
        (source, weight, out, aux if aux is not None else out, offsets, rows)
        if transpose_weights
        else (source, weight, out, offsets, rows)
    )
    kernel[grid](
        *operands,
        N=width,
        K=reduction,
        NSEG=1 if single else experts,
        NSEG_P2=1 if single else _next_power_of_two(experts),
        SINGLE=single,
        BM=BLOCK_M,
        BN=BLOCK_N,
        BK=LEAF,
        GROUP=GROUP_LEAVES,
        **extra,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        enable_fp_fusion=False,
    )


def _check_leaf_alignment(**widths):
    """Refuse a reduction width the leaf loop would truncate.

    `_rows_gemm` and `_dweights_kernel` step the reduction in whole `LEAF`-wide
    leaves without a tail mask (the masked `_wgrad_kernel` covers the one place
    where the reduction length is arbitrary). The operator's validator already
    requires 64-aligned widths, so this only rejects a hand-built provider call -
    but silently dropping the tail leaf would be a wrong answer, not a slow one.
    """
    for name, value in widths.items():
        if value % LEAF:
            raise ValueError(
                f"nemotron experts (triton): {name}={value} is not a multiple of the "
                f"{LEAF}-wide leaf"
            )


def _forward(packed, x, weights, permutation, offsets, w_up, w_down, w_sh1, w_sh2):
    rows, width = packed.shape
    tokens = rows // 6
    experts, intermediate = w_up.shape[0], w_up.shape[1]
    device = packed.device
    dtype = packed.dtype
    _check_leaf_alignment(width=width, intermediate=intermediate)
    if w_sh1 is not None:
        _check_leaf_alignment(shared_width=w_sh1.shape[0])
    up = torch.empty((rows, intermediate), device=device, dtype=dtype)
    down = torch.empty((rows, width), device=device, dtype=torch.float32)
    inverse = torch.empty((rows,), device=device, dtype=torch.int64)
    if tokens == 0:
        return torch.empty((0, width), device=device, dtype=dtype), up, down, inverse, None, None
    _invert_kernel[(triton.cdiv(rows, 256),)](inverse, permutation, rows, 256)
    _rows_gemm(
        packed,
        w_up,
        up,
        None,
        offsets,
        rows,
        width,
        intermediate,
        experts,
        transpose_weights=False,
    )
    _rows_gemm(
        up,
        w_down,
        down,
        None,
        offsets,
        rows,
        intermediate,
        width,
        experts,
        transpose_weights=False,
        relu2_source=True,
        out_fp32=True,
    )
    shared_up = shared_down = None
    if w_sh1 is not None:
        shared_width = w_sh1.shape[0]
        shared_up = torch.empty((tokens, shared_width), device=device, dtype=dtype)
        shared_down = torch.empty((tokens, width), device=device, dtype=torch.float32)
        _rows_gemm(
            x,
            w_sh1,
            shared_up,
            None,
            None,
            tokens,
            width,
            shared_width,
            1,
            transpose_weights=False,
        )
        _rows_gemm(
            shared_up,
            w_sh2,
            shared_down,
            None,
            None,
            tokens,
            shared_width,
            width,
            1,
            transpose_weights=False,
            relu2_source=True,
            out_fp32=True,
        )
    out = torch.empty((tokens, width), device=device, dtype=dtype)
    # Combine, shared add and the single exit cast in one pass over the tokens.
    _combine_kernel[(tokens, triton.cdiv(width, _ELEMENT_BLOCK))](
        down,
        inverse,
        weights,
        shared_down if shared_down is not None else out,
        out,
        width,
        shared_down is not None,
        _ELEMENT_BLOCK,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out, up, down, inverse, shared_up, shared_down


def _backward(
    grad_out,
    packed,
    x,
    weights,
    permutation,
    offsets,
    up,
    down,
    inverse,
    w_up,
    w_down,
    w_sh1,
    w_sh2,
    shared_up,
    need_packed,
    need_x,
    need_weights,
    need_weight_grad,
    need_shared_weight_grad,
):
    rows, width = packed.shape
    tokens = rows // 6
    experts, intermediate = w_up.shape[0], w_up.shape[1]
    device = packed.device
    dtype = packed.dtype
    _check_leaf_alignment(width=width, intermediate=intermediate)
    if w_sh1 is not None:
        _check_leaf_alignment(shared_width=w_sh1.shape[0])
    if tokens == 0:
        # An expert-parallel rank may hold no tokens. Every returned gradient
        # still has to be shaped like its input: an empty-row payload gets an
        # empty payload gradient, and a weight gradient is exactly zero (not
        # absent), so an optimizer step and a DDP all-reduce stay well defined.
        shared_grad = need_shared_weight_grad and w_sh1 is not None and w_sh2 is not None
        return [
            torch.zeros((rows, width), device=device, dtype=dtype) if need_packed else None,
            torch.zeros((tokens, width), device=device, dtype=dtype) if need_x else None,
            (
                torch.zeros((tokens, 6), device=device, dtype=torch.float32)
                if need_weights
                else None
            ),
            torch.zeros_like(w_up) if need_weight_grad else None,
            torch.zeros_like(w_down) if need_weight_grad else None,
            torch.zeros_like(w_sh1) if shared_grad else None,
            torch.zeros_like(w_sh2) if shared_grad else None,
        ]
    dy = torch.empty((rows, width), device=device, dtype=dtype)
    _dy_kernel[(rows, triton.cdiv(width, _ELEMENT_BLOCK))](
        dy,
        grad_out,
        permutation,
        weights,
        width,
        _ELEMENT_BLOCK,
        num_warps=4,
        enable_fp_fusion=False,
    )
    d_weights = None
    if need_weights:
        d_weights = torch.empty((tokens, 6), device=device, dtype=torch.float32)
        _dweights_kernel[(rows,)](
            down, grad_out, inverse, d_weights, width, LEAF, num_warps=1, enable_fp_fusion=False
        )
    du = None
    if need_packed or need_weight_grad:
        du = torch.empty((rows, intermediate), device=device, dtype=dtype)
        _rows_gemm(
            dy,
            w_down,
            du,
            up,
            offsets,
            rows,
            width,
            intermediate,
            experts,
            transpose_weights=True,
            epi_relu2_grad=True,
        )
    d_packed = None
    if need_packed:
        d_packed = torch.empty((rows, width), device=device, dtype=dtype)
        _rows_gemm(
            du,
            w_up,
            d_packed,
            None,
            offsets,
            rows,
            intermediate,
            width,
            experts,
            transpose_weights=True,
        )
    d_w_up = d_w_down = None
    if need_weight_grad:
        d_w_up = torch.empty_like(w_up)
        d_w_down = torch.empty_like(w_down)
        _wgrad_kernel[(triton.cdiv(intermediate, BLOCK_M), triton.cdiv(width, BLOCK_N), experts)](
            du,
            packed,
            d_w_up,
            offsets,
            rows,
            intermediate,
            width,
            False,
            experts,
            False,
            BLOCK_M,
            BLOCK_N,
            LEAF,
            GROUP_LEAVES,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            enable_fp_fusion=False,
        )
        _wgrad_kernel[(triton.cdiv(width, BLOCK_M), triton.cdiv(intermediate, BLOCK_N), experts)](
            dy,
            up,
            d_w_down,
            offsets,
            rows,
            width,
            intermediate,
            True,
            experts,
            False,
            BLOCK_M,
            BLOCK_N,
            LEAF,
            GROUP_LEAVES,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            enable_fp_fusion=False,
        )
    d_x = d_w_sh1 = d_w_sh2 = None
    if w_sh1 is not None:
        shared_width = w_sh1.shape[0]
        d_shared_up = None
        if need_x or need_shared_weight_grad:
            d_shared_up = torch.empty((tokens, shared_width), device=device, dtype=dtype)
            _rows_gemm(
                grad_out,
                w_sh2,
                d_shared_up,
                shared_up,
                None,
                tokens,
                width,
                shared_width,
                1,
                transpose_weights=True,
                epi_relu2_grad=True,
            )
        if need_x:
            d_x = torch.empty((tokens, width), device=device, dtype=dtype)
            _rows_gemm(
                d_shared_up,
                w_sh1,
                d_x,
                None,
                None,
                tokens,
                shared_width,
                width,
                1,
                transpose_weights=True,
            )
        if need_shared_weight_grad:
            # Same shape as the inputs: `_wgrad_kernel` writes `[M, N]` from the
            # group offset, which is byte-identical for a 2-D allocation.
            d_w_sh1 = torch.empty((shared_width, width), device=device, dtype=dtype)
            d_w_sh2 = torch.empty((width, shared_width), device=device, dtype=dtype)
            _wgrad_kernel[(triton.cdiv(shared_width, BLOCK_M), triton.cdiv(width, BLOCK_N), 1)](
                d_shared_up,
                x,
                d_w_sh1,
                None,
                tokens,
                shared_width,
                width,
                False,
                1,
                True,
                BLOCK_M,
                BLOCK_N,
                LEAF,
                GROUP_LEAVES,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
                enable_fp_fusion=False,
            )
            _wgrad_kernel[(triton.cdiv(width, BLOCK_M), triton.cdiv(shared_width, BLOCK_N), 1)](
                grad_out,
                shared_up,
                d_w_sh2,
                None,
                tokens,
                width,
                shared_width,
                True,
                1,
                True,
                BLOCK_M,
                BLOCK_N,
                LEAF,
                GROUP_LEAVES,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
                enable_fp_fusion=False,
            )
    return [d_packed, d_x, d_weights, d_w_up, d_w_down, d_w_sh1, d_w_sh2]


def _optional(gradient):
    if gradient is None:
        return None
    return gradient if gradient.numel() else None


class _ExpertsCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, x, weights, permutation, offsets, w_up, w_down, w_sh1, w_sh2):
        out, up, down, inverse, shared_up, shared_down = _forward(
            packed, x, weights, permutation, offsets, w_up, w_down, w_sh1, w_sh2
        )
        ctx.save_for_backward(
            packed,
            x,
            weights,
            permutation,
            offsets,
            up,
            down,
            inverse,
            w_up,
            w_down,
            w_sh1,
            w_sh2,
            shared_up,
            shared_down,
        )
        ctx.mark_non_differentiable(permutation, offsets)
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            packed,
            x,
            weights,
            permutation,
            offsets,
            up,
            down,
            inverse,
            w_up,
            w_down,
            w_sh1,
            w_sh2,
            shared_up,
            shared_down,
        ) = ctx.saved_tensors
        needs = ctx.needs_input_grad
        if grad_out is None:
            return (None,) * 9
        shared = w_sh1 is not None
        need_shared_weight_grad = shared and (bool(needs[7]) or bool(needs[8]))
        need_any = (
            bool(needs[0])
            or (shared and bool(needs[1]))
            or bool(needs[2])
            or bool(needs[5])
            or bool(needs[6])
            or need_shared_weight_grad
        )
        if not need_any:
            return (None,) * 9
        gradients = _backward(
            grad_out.contiguous(),
            packed,
            x,
            weights,
            permutation,
            offsets,
            up,
            down,
            inverse,
            w_up,
            w_down,
            w_sh1,
            w_sh2,
            shared_up,
            bool(needs[0]),
            shared and bool(needs[1]),
            bool(needs[2]),
            bool(needs[5]) or bool(needs[6]),
            need_shared_weight_grad,
        )
        d_packed, d_x, d_weights, d_w_up, d_w_down, d_w_sh1, d_w_sh2 = gradients
        return (
            _optional(d_packed),
            _optional(d_x),
            _optional(d_weights),
            None,
            None,
            _optional(d_w_up),
            _optional(d_w_down),
            _optional(d_w_sh1),
            _optional(d_w_sh2),
        )
