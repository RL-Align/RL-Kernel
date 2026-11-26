# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton backend for the Qwen-Image attn-out bias GEMM (contract
``attn-out-bias-gemm-tree-v1``).

Design (see docs/operators/attn-out-bias-gemm.md):

- Forward reduces over K and ``dx`` over N. Each leaf accumulates up to
  32 products with ``tl.fma`` in ascending index order. K=N=3072 in the
  Qwen-Image projection.
- Each program evaluates the full mid-split tree for its output tile.
  It retains one left tile per active level and stores only the final tile.
- Each ``dW`` program owns an output tile and accumulates one FP32 FMA per
  row in ascending order. The tile accumulator size is independent of S.
- ``finalize_tree_output`` adds optional bias in fp32 after the complete
  tree and casts once to the output dtype.
"""

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.reference.gemm.attn_out_bias_gemm import (
    ATTN_OUT_BIAS_GEMM_CONTRACT_VERSION,
    AttnOutBiasGemmBackend,
    NativeAttnOutBiasGemmOp,
    finalize_tree_output,
    num_leaves,
)
from rl_engine.utils.logger import logger

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = torch.version.cuda is not None
except ImportError:  # pragma: no cover - Triton is an optional backend
    _TRITON_AVAILABLE = False

_BLOCK_M = 32
_BLOCK_N = 64
_NUM_WARPS = 4

if _TRITON_AVAILABLE:

    @triton.jit
    def _tree_tile(
        a_rows,
        b_rows,
        m_mask,
        n_mask,
        R,
        lo,
        hi,
        DEPTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        result = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
        if DEPTH == 0:
            # After floor(log2(leaves)) splits each range has one or two leaves.
            for leaf in range(lo, hi):
                k0 = tl.cast(leaf, tl.int64) * 32
                acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
                for k in range(k0, tl.minimum(k0 + 32, R)):
                    a = tl.load(a_rows + k, mask=m_mask, other=0.0)
                    b = tl.load(b_rows + k, mask=n_mask, other=0.0)
                    acc = tl.fma(a[:, None], b[None, :], acc)
                if leaf == lo:
                    result = acc
                else:
                    result += acc
        else:
            mid = lo + (hi - lo) // 2
            for child in range(2):
                part = _tree_tile(
                    a_rows,
                    b_rows,
                    m_mask,
                    n_mask,
                    R,
                    tl.where(child == 0, lo, mid),
                    tl.where(child == 0, mid, hi),
                    DEPTH - 1,
                    BLOCK_M,
                    BLOCK_N,
                )
                if child == 0:
                    result = part
                else:
                    result += part
        return result

    @triton.jit
    def _tree_gemm_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        P,
        Q,
        R,
        leaves,
        DEPTH: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1).to(tl.int64) + tl.program_id(2).to(tl.int64) * tl.num_programs(1)
        m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tile = _tree_tile(
            a_ptr + m * R,
            b_ptr + n * R,
            m < P,
            n < Q,
            R,
            0,
            leaves,
            DEPTH,
            BLOCK_M,
            BLOCK_N,
        )
        tl.store(
            out_ptr + m[:, None] * Q + n[None, :],
            tile,
            mask=(m[:, None] < P) & (n[None, :] < Q),
        )

    @triton.jit
    def _dw_left_fold_kernel(
        g_ptr,
        x_ptr,
        dw_ptr,
        S,
        N,
        K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """dW[n, k] = sum over rows s (ascending) of dY[s, n] * x[s, k].

        One program owns a [BLOCK_N, BLOCK_K] tile of dW and walks the rows
        in ascending order, one correctly-rounded FMA per row (left fold).
        """
        pid_n = tl.program_id(0).to(tl.int64)
        pid_k = tl.program_id(1).to(tl.int64) + tl.program_id(2).to(tl.int64) * tl.num_programs(1)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask = k_offs < K
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for s in range(0, S):
            row = tl.cast(s, tl.int64)
            g_vec = tl.load(g_ptr + row * N + n_offs, mask=n_mask, other=0.0)
            x_vec = tl.load(x_ptr + row * K + k_offs, mask=k_mask, other=0.0)
            acc = tl.fma(g_vec[:, None], x_vec[None, :], acc)
        dst = dw_ptr + n_offs[:, None] * K + k_offs[None, :]
        tl.store(dst, acc, mask=n_mask[:, None] & k_mask[None, :])


def _require_triton() -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("NVIDIA CUDA Triton is not available for attn_out_bias_gemm")


def _tile_grid(row_tiles: int, col_tiles: int) -> tuple[int, int, int]:
    # CUDA limits grid.y to 65535. Spread wider outputs across grid.z.
    slabs = triton.cdiv(col_tiles, 65535)
    return row_tiles, triton.cdiv(col_tiles, slabs), slabs


def triton_tree_gemm(a_f: torch.Tensor, b_f: torch.Tensor) -> torch.Tensor:
    """Compute the frozen mid-split tree in each output tile on the GPU."""
    _require_triton()
    if a_f.dim() != 2 or b_f.dim() != 2 or a_f.size(1) != b_f.size(1):
        raise ValueError(
            f"expected 2-D fp32 inputs sharing the reduction dim, got "
            f"{tuple(a_f.shape)} and {tuple(b_f.shape)}"
        )
    if not a_f.is_cuda or not b_f.is_cuda:
        raise ValueError("triton_tree_gemm requires CUDA tensors")
    a_c = a_f.resolve_neg().contiguous()
    b_c = b_f.resolve_neg().contiguous()
    if a_c.size(1) == 0 or a_c.size(0) == 0 or b_c.size(0) == 0:
        return torch.zeros(a_c.size(0), b_c.size(0), device=a_c.device, dtype=torch.float32)
    leaves = num_leaves(a_c.size(1))

    rows, reduction = a_c.shape
    cols = b_c.size(0)
    out = torch.empty((rows, cols), dtype=torch.float32, device=a_c.device)
    grid = _tile_grid(triton.cdiv(rows, _BLOCK_M), triton.cdiv(cols, _BLOCK_N))
    _tree_gemm_kernel[grid](
        a_c,
        b_c,
        out,
        rows,
        cols,
        reduction,
        leaves,
        DEPTH=leaves.bit_length() - 1,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
    )
    return out


def triton_dw_left_fold(g_f: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
    """``dW = dY.T @ x`` by the ascending-row left fold (contract c-prime)."""
    _require_triton()
    if g_f.dim() != 2 or x_f.dim() != 2 or g_f.size(0) != x_f.size(0):
        raise ValueError(
            f"expected [S, N] and [S, K] fp32 inputs, got "
            f"{tuple(g_f.shape)} and {tuple(x_f.shape)}"
        )
    g_c = g_f.resolve_neg().contiguous()
    x_c = x_f.resolve_neg().contiguous()
    rows, out_n = g_c.shape
    in_k = x_c.shape[1]
    dw = torch.empty(out_n, in_k, device=g_c.device, dtype=torch.float32)
    if dw.numel() == 0:
        return dw
    grid = _tile_grid(triton.cdiv(out_n, _BLOCK_N), triton.cdiv(in_k, _BLOCK_N))
    _dw_left_fold_kernel[grid](
        g_c,
        x_c,
        dw,
        rows,
        out_n,
        in_k,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_N,
        num_warps=_NUM_WARPS,
    )
    return dw


class _AttnOutBiasGemmTritonFunction(torch.autograd.Function):
    """Triton forward/backward under the same contract as the reference."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1))
        tree_sum = triton_tree_gemm(x2d.float(), weight.float())
        out = finalize_tree_output(tree_sum, None if bias is None else bias.float(), x.dtype)
        ctx.save_for_backward(x2d, weight, bias)
        ctx.lead_shape = x.shape[:-1]
        return out.reshape(*ctx.lead_shape, weight.size(0))

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        from rl_engine.ops.autograd.backward_runtime import record_backward
        from rl_engine.ops.autograd.vjp_fp32 import row_local_bias_fp32

        x2d, weight, bias = ctx.saved_tensors
        g_f = grad_output.reshape(-1, grad_output.size(-1)).float()
        grad_x = None
        grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = triton_tree_gemm(g_f, weight.t().contiguous().float())
            grad_x = grad_x.reshape(*ctx.lead_shape, x2d.size(-1)).to(x2d.dtype)
        if ctx.needs_input_grad[1]:
            grad_w = triton_dw_left_fold(g_f, x2d.float()).to(weight.dtype)
        grad_b: Optional[torch.Tensor] = None
        if bias is not None and ctx.needs_input_grad[2]:
            grad_b = row_local_bias_fp32(grad_output).to(bias.dtype)
        record_backward(
            "attn_out_bias_gemm",
            kernel_id=ATTN_OUT_BIAS_GEMM_CONTRACT_VERSION,
            impl="triton_tile_tree_fma",
            family="triton",
            node_metadata=ctx.metadata,
        )
        return grad_x, grad_w, grad_b


class TritonAttnOutBiasGemmOp(AttnOutBiasGemmBackend):
    """Tile-local forward and dx trees, with an ascending-row dW fold."""

    backend_id = "triton"
    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "triton_tile_tree_fma"

    def __init__(self) -> None:
        _require_triton()
        logger.info("TritonAttnOutBiasGemmOp ready (per-tile tree).")

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(x, weight, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not x.is_cuda or x.device != weight.device:
            raise ValueError("TritonAttnOutBiasGemmOp requires CUDA tensors on one device")
        NativeAttnOutBiasGemmOp._check_inputs(x, weight, bias)
        return _AttnOutBiasGemmTritonFunction.apply(x, weight, bias)
