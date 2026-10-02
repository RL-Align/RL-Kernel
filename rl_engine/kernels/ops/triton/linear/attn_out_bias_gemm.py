# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton backend for the Qwen-Image attn-out bias GEMM (contract
``attn-out-bias-gemm-tree-v1``).

Design (see docs/operators/attn-out-bias-gemm.md):

- The forward and ``dx`` reduce over a fixed 3072 dim under the mid-split
  tree. A leaf kernel computes 32-step ascending-k fp32 chains for an output
  tile; leaves are evaluated in chunks and combined on the host by the SAME
  tree code the FP32 CPU reference uses (``tree_from_leaf_evaluator``).
  Chunked depth-first evaluation bounds peak memory to about
  ``(CHUNK + log2(leaves))`` output tiles instead of materialising all 96
  partial planes at once (~2 GB at the largest acceptance tier instead of
  8.1 GB, and bitwise-identical by construction: the tree is the same).
- ``dW`` reduces over the row dim S with the ascending-row LEFT FOLD
  (contract c-prime): a dedicated kernel accumulates one separate
  multiply-add per row into a single [N, K] fp32 accumulator -- no partials,
  memory independent of S.
- The multiply-add discipline is FMA (fused, one rounding per multiply-add)
  on every device backend, via explicit ``libdevice.fma_rn``. On bf16 inputs
  FMA and separate mul-add are provably bit-identical (products are exact in
  fp32), so the byte-for-byte bar against the torch reference holds on the
  working dtype; on fp32 inputs the backends are bit-identical to each other
  and compared to the torch reference (which cannot express elementwise FMA)
  with a tight declared tolerance. ``tl.sum`` (unspecified tree order) and
  ``tl.dot``/mma (unspecified internal accumulation tree) remain forbidden.
- Bias is added once in fp32 after the complete tree and the output is cast
  once (RNE) -- via the shared ``finalize_tree_output`` host helper.
"""

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.kernels.ops.pytorch.linear.attn_out_bias_gemm import (
    finalize_tree_output,
    num_leaves,
    tree_from_leaf_evaluator,
)
from rl_engine.utils.logger import logger

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - Triton is an optional backend
    _TRITON_AVAILABLE = False

_BLOCK_M = 32
_BLOCK_N = 64
_NUM_WARPS = 4
_LEAF_CHUNK = 12  # leaves per kernel launch; bounds partial-plane memory

if _TRITON_AVAILABLE:

    @triton.jit
    def _tree_gemm_leaf_kernel(
        a_ptr,
        b_ptr,
        partials_ptr,
        P,
        Q,
        R,
        leaf_lo,
        leaf_hi,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Ascending-k chains for leaves ``[leaf_lo, leaf_hi)`` of one tile.

        ``a`` is [P, R] and ``b`` is [Q, R] row-major fp32; partials are
        written as contiguous [leaf_hi - leaf_lo, P, Q] planes.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = m_offs < P
        n_mask = n_offs < Q

        for leaf in range(leaf_lo, leaf_hi):
            k0 = leaf * 32
            k1 = tl.minimum(k0 + 32, R)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(k0, k1):
                a_vec = tl.load(a_ptr + m_offs * R + k, mask=m_mask, other=0.0)
                b_vec = tl.load(b_ptr + n_offs * R + k, mask=n_mask, other=0.0)
                # fma_rn is fma.rn.f32: exact product, one rounding per
                # multiply-add (contract v3.4: FMA is the uniform device
                # discipline; on bf16 inputs FMA and separate mul-add are
                # provably bit-identical, which is what the harness checks).
                acc = libdevice.fma_rn(a_vec[:, None], b_vec[None, :], acc)
            plane = (leaf - leaf_lo) * (P * Q)
            dst = partials_ptr + plane + m_offs[:, None] * Q + n_offs[None, :]
            tl.store(dst, acc, mask=m_mask[:, None] & n_mask[None, :])

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
        in ascending order, one separate multiply-add per row (left fold).
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        n_mask = n_offs < N
        k_mask = k_offs < K
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for s in range(0, S):
            g_vec = tl.load(g_ptr + s * N + n_offs, mask=n_mask, other=0.0)
            x_vec = tl.load(x_ptr + s * K + k_offs, mask=k_mask, other=0.0)
            acc = libdevice.fma_rn(g_vec[:, None], x_vec[None, :], acc)
        dst = dw_ptr + n_offs[:, None] * K + k_offs[None, :]
        tl.store(dst, acc, mask=n_mask[:, None] & k_mask[None, :])


def _require_triton() -> None:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for attn_out_bias_gemm")


def _launch_leaf_chunk(
    a_c: torch.Tensor, b_c: torch.Tensor, leaf_lo: int, leaf_hi: int
) -> torch.Tensor:
    rows_p, reduction = a_c.shape
    cols_q = b_c.shape[0]
    partials = torch.empty(
        leaf_hi - leaf_lo, rows_p, cols_q, device=a_c.device, dtype=torch.float32
    )
    grid = (triton.cdiv(rows_p, _BLOCK_M), triton.cdiv(cols_q, _BLOCK_N))
    _tree_gemm_leaf_kernel[grid](
        a_c,
        b_c,
        partials,
        rows_p,
        cols_q,
        reduction,
        leaf_lo,
        leaf_hi,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
    )
    return partials


def triton_tree_gemm(a_f: torch.Tensor, b_f: torch.Tensor) -> torch.Tensor:
    """``a_f @ b_f.T`` under the frozen mid-split tree (chunked DFS combine).

    Both inputs must be 2-D fp32 CUDA tensors sharing the reduction dim; the
    tree reduction runs over that dim. Leaf chunks are combined through the
    shared reference tree code, so the result is bitwise-identical to the
    CPU reference by construction.
    """
    _require_triton()
    if a_f.dim() != 2 or b_f.dim() != 2 or a_f.size(1) != b_f.size(1):
        raise ValueError(
            f"expected 2-D fp32 inputs sharing the reduction dim, got "
            f"{tuple(a_f.shape)} and {tuple(b_f.shape)}"
        )
    if not a_f.is_cuda or not b_f.is_cuda:
        raise ValueError("triton_tree_gemm requires CUDA tensors")
    a_c = a_f.contiguous()
    b_c = b_f.contiguous()
    leaves = num_leaves(a_c.size(1))

    def solve(lo: int, hi: int) -> torch.Tensor:
        if hi - lo <= _LEAF_CHUNK:
            planes = _launch_leaf_chunk(a_c, b_c, lo, hi)
            # planes are indexed locally 0..(hi-lo-1); the evaluator passes
            # indices in that same local range. The explicit del bounds the
            # partial-plane lifetime to the combine (closure cells otherwise
            # keep the chunk alive up the DFS spine).
            result = tree_from_leaf_evaluator(lambda index: planes[index], hi - lo)
            del planes
            return result
        mid = lo + (hi - lo) // 2
        left = solve(lo, mid)
        right = solve(mid, hi)
        result = left + right
        del left, right
        return result

    return solve(0, leaves)


def triton_dw_left_fold(g_f: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
    """``dW = dY.T @ x`` by the ascending-row left fold (contract c-prime)."""
    _require_triton()
    if g_f.dim() != 2 or x_f.dim() != 2 or g_f.size(0) != x_f.size(0):
        raise ValueError(
            f"expected [S, N] and [S, K] fp32 inputs, got "
            f"{tuple(g_f.shape)} and {tuple(x_f.shape)}"
        )
    g_c = g_f.contiguous()
    x_c = x_f.contiguous()
    rows, out_n = g_c.shape
    in_k = x_c.shape[1]
    dw = torch.empty(out_n, in_k, device=g_c.device, dtype=torch.float32)
    grid = (triton.cdiv(out_n, _BLOCK_N), triton.cdiv(in_k, _BLOCK_N))
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
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        from rl_engine.kernels.ops.backward_runtime import record_backward
        from rl_engine.kernels.ops.vjp_fp32 import row_local_bias_fp32

        x2d, weight, bias = ctx.saved_tensors
        g_f = grad_output.reshape(-1, grad_output.size(-1)).float()
        grad_x = triton_tree_gemm(g_f, weight.t().contiguous().float())
        grad_w = triton_dw_left_fold(g_f, x2d.float())
        grad_x = grad_x.reshape(*ctx.lead_shape, x2d.size(-1)).to(x2d.dtype)
        grad_w = grad_w.to(weight.dtype)
        grad_b: Optional[torch.Tensor] = None
        if bias is not None:
            grad_b = row_local_bias_fp32(grad_output).to(bias.dtype)
        record_backward(
            "attn_out_bias_gemm",
            kernel_id="attn-out-bias-gemm-tree-v1",
            impl="triton_leaf_chunks_shared_tree_combine",
            family="triton",
        )
        return grad_x, grad_w, grad_b


class TritonAttnOutBiasGemmOp:
    """Triton backend: leaf chunks on device, shared tree combine on host."""

    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "triton_leaf_chunks_shared_tree_combine"

    def __init__(self) -> None:
        _require_triton()
        logger.info("TritonAttnOutBiasGemmOp ready (leaf chunks + shared tree combine).")

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
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"supported dtypes are bf16/fp32 (contract scope), got {x.dtype}")
        return _AttnOutBiasGemmTritonFunction.apply(x, weight, bias)
