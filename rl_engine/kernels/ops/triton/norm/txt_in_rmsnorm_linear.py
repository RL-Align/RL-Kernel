# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton backend for the Qwen-Image txt_in RMSNorm->Linear op (contract
``txt-in-rmsnorm-linear-v1``).

Design (see docs/operators/txt-in-rmsnorm-linear.md):

- The four tree reductions run on device: the forward/backward GEMMs and the
  dW fold reuse the attn_out leaf-chunk machinery (``triton_tree_gemm`` /
  ``triton_dw_left_fold`` -- generic over the reduction dim, same mid-split
  tree), and the two per-row reductions (sumsq, dot) get dedicated leaf
  kernels whose 32-step ascending FMA chains feed the SAME host-side mid-split
  combine (``tree_from_leaf_evaluator``). dgamma runs a dedicated ascending-row
  left-fold kernel. Bitwise-identical to the CPU reference by construction.
- Every isolated single-op step of the frozen chain (rstd three-step, xhat/u
  seam, t1/t2/t3, dx mul) stays in torch elementwise ops: each is ONE
  correctly-rounded FP32 op (add/mul/sub/div/fp64-sqrt-then-round), so CPU and
  CUDA produce identical bits without a kernel.
- One correctly-rounded FP32 FMA discipline on BOTH sides: these kernels use
  explicit ``libdevice.fma_rn`` and the reference uses ``torch.addcmul``
  (oracle-verified correctly-rounded FMA). ``tl.sum`` (unspecified tree order)
  and ``tl.dot``/mma remain forbidden.
- The RMSNorm->Linear seam is FP32 with no intermediate cast; one RNE cast at
  the output boundary via the shared ``finalize_tree_output``.
"""

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.kernels.ops.pytorch.linear.attn_out_bias_gemm import (
    finalize_tree_output,
    num_leaves,
    tree_from_leaf_evaluator,
)
from rl_engine.kernels.ops.pytorch.norm.txt_in_rmsnorm_linear import (
    DIVISOR_F32,
    NativeTxtInRMSNormLinearOp,
    three_step_rstd,
    true_div_rn,
)
from rl_engine.kernels.ops.triton.linear.attn_out_bias_gemm import (
    _require_triton,
    triton_dw_left_fold,
    triton_tree_gemm,
)
from rl_engine.utils.logger import logger

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - Triton is an optional backend
    _TRITON_AVAILABLE = False

_BLOCK_S = 32
_BLOCK_H = 256
_NUM_WARPS = 4
_LEAF_CHUNK = 28  # leaves per launch; bounds partial memory (28 * S floats)

if _TRITON_AVAILABLE:

    @triton.jit
    def _row_sumsq_leaf_kernel(
        x_ptr,
        partials_ptr,
        S,
        H,
        leaf_lo,
        leaf_hi,
        BLOCK_S: tl.constexpr,
    ):
        """Ascending-h chains of ``fma(x, x, acc)`` for leaves
        ``[leaf_lo, leaf_hi)``; partials are ``[leaf_hi - leaf_lo, S]``."""

        pid_s = tl.program_id(0)
        s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        for leaf in range(leaf_lo, leaf_hi):
            h0 = leaf * 32
            h1 = tl.minimum(h0 + 32, H)
            acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
            for h in range(h0, h1):
                v = tl.load(x_ptr + s_offs * H + h, mask=s_mask, other=0.0)
                acc = libdevice.fma_rn(v, v, acc)
            dst = partials_ptr + (leaf - leaf_lo) * S + s_offs
            tl.store(dst, acc, mask=s_mask)

    @triton.jit
    def _row_dot_leaf_kernel(
        a_ptr,
        b_ptr,
        partials_ptr,
        S,
        H,
        leaf_lo,
        leaf_hi,
        BLOCK_S: tl.constexpr,
    ):
        """Ascending-h chains of ``fma(a, b, acc)`` per row; same partial
        layout as ``_row_sumsq_leaf_kernel`` (never ``p = a*b`` then sum)."""

        pid_s = tl.program_id(0)
        s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        for leaf in range(leaf_lo, leaf_hi):
            h0 = leaf * 32
            h1 = tl.minimum(h0 + 32, H)
            acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
            for h in range(h0, h1):
                a_v = tl.load(a_ptr + s_offs * H + h, mask=s_mask, other=0.0)
                b_v = tl.load(b_ptr + s_offs * H + h, mask=s_mask, other=0.0)
                acc = libdevice.fma_rn(a_v, b_v, acc)
            dst = partials_ptr + (leaf - leaf_lo) * S + s_offs
            tl.store(dst, acc, mask=s_mask)

    @triton.jit
    def _dgamma_left_fold_kernel(
        du_ptr,
        xhat_ptr,
        out_ptr,
        S,
        H,
        BLOCK_H: tl.constexpr,
    ):
        """dgamma[h] = fold over rows s (ascending) of ``fma(du[s,h], xhat[s,h])``.

        One program owns a BLOCK_H tile of dgamma and walks the rows in
        ascending order, one correctly-rounded FMA per row (left fold) --
        the same discipline as the reference's ``addcmul`` row loop.
        """

        pid_h = tl.program_id(0)
        h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for s in range(0, S):
            du_v = tl.load(du_ptr + s * H + h_offs, mask=h_mask, other=0.0)
            xhat_v = tl.load(xhat_ptr + s * H + h_offs, mask=h_mask, other=0.0)
            acc = libdevice.fma_rn(du_v, xhat_v, acc)
        tl.store(out_ptr + h_offs, acc, mask=h_mask)


def _combine_leaf_chunks(launch_chunk, rows: int, leaves: int) -> torch.Tensor:
    """Chunked depth-first mid-split combine over per-row leaf partials."""

    def solve(lo: int, hi: int) -> torch.Tensor:
        if hi - lo <= _LEAF_CHUNK:
            planes = launch_chunk(lo, hi)
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


def triton_row_sumsq(x_f: torch.Tensor) -> torch.Tensor:
    """``sumsq[s] = TreeH(x_f[s, :]^2)`` under the frozen 112-leaf tree."""

    _require_triton()
    if x_f.dim() != 2:
        raise ValueError(f"expected a 2-D fp32 CUDA tensor, got {tuple(x_f.shape)}")
    if not x_f.is_cuda:
        raise ValueError("triton_row_sumsq requires CUDA tensors")
    x_c = x_f.contiguous()
    rows, hidden = x_c.shape

    def launch_chunk(lo: int, hi: int) -> torch.Tensor:
        partials = torch.empty(hi - lo, rows, device=x_c.device, dtype=torch.float32)
        grid = (triton.cdiv(rows, _BLOCK_S),)
        _row_sumsq_leaf_kernel[grid](
            x_c, partials, rows, hidden, lo, hi, BLOCK_S=_BLOCK_S, num_warps=_NUM_WARPS
        )
        return partials

    return _combine_leaf_chunks(launch_chunk, rows, num_leaves(hidden))


def triton_row_dot(a_f: torch.Tensor, b_f: torch.Tensor) -> torch.Tensor:
    """``dot[s] = TreeH(FMA(a_f[s,:], b_f[s,:]))`` under the frozen tree."""

    _require_triton()
    if a_f.shape != b_f.shape or a_f.dim() != 2:
        raise ValueError(
            f"expected matching 2-D fp32 CUDA tensors, got "
            f"{tuple(a_f.shape)} and {tuple(b_f.shape)}"
        )
    if not a_f.is_cuda or b_f.device != a_f.device:
        raise ValueError("triton_row_dot requires CUDA tensors on one device")
    a_c = a_f.contiguous()
    b_c = b_f.contiguous()
    rows, hidden = a_c.shape

    def launch_chunk(lo: int, hi: int) -> torch.Tensor:
        partials = torch.empty(hi - lo, rows, device=a_c.device, dtype=torch.float32)
        grid = (triton.cdiv(rows, _BLOCK_S),)
        _row_dot_leaf_kernel[grid](
            a_c, b_c, partials, rows, hidden, lo, hi, BLOCK_S=_BLOCK_S, num_warps=_NUM_WARPS
        )
        return partials

    return _combine_leaf_chunks(launch_chunk, rows, num_leaves(hidden))


def triton_dgamma_fold(du: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    """``dgamma = fold_s FMA(du[s], xhat[s])`` by the ascending-row left fold."""

    _require_triton()
    if du.shape != xhat.shape or du.dim() != 2:
        raise ValueError(
            f"expected matching 2-D fp32 CUDA tensors, got "
            f"{tuple(du.shape)} and {tuple(xhat.shape)}"
        )
    du_c = du.contiguous()
    xhat_c = xhat.contiguous()
    rows, hidden = du_c.shape
    dgamma = torch.empty(hidden, device=du_c.device, dtype=torch.float32)
    grid = (triton.cdiv(hidden, _BLOCK_H),)
    _dgamma_left_fold_kernel[grid](
        du_c, xhat_c, dgamma, rows, hidden, BLOCK_H=_BLOCK_H, num_warps=_NUM_WARPS
    )
    return dgamma


def triton_norm_stats(x_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen norm statistics: device sumsq tree + torch three-step rstd."""

    sumsq = triton_row_sumsq(x_f)
    var = true_div_rn(sumsq, DIVISOR_F32)
    rstd = three_step_rstd(var)
    xhat = x_f * rstd.unsqueeze(1)
    return xhat, rstd


class _TxtInRMSNormLinearTritonFunction(torch.autograd.Function):
    """Triton forward/backward under the same contract as the reference."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1)).float()
        gamma_f = norm_weight.float()
        w_f = weight.float()
        b_f = None if bias is None else bias.float()

        xhat, rstd = triton_norm_stats(x2d)
        z = xhat * gamma_f.unsqueeze(0)  # frozen FP32 seam -- no cast
        tree_sum = triton_tree_gemm(z, w_f)
        out = finalize_tree_output(tree_sum, b_f, x.dtype)
        ctx.save_for_backward(x2d, gamma_f, w_f, xhat, rstd, z)
        ctx.lead_shape = x.shape[:-1]
        ctx.has_bias = bias is not None
        return out.reshape(*ctx.lead_shape, weight.size(0))

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        from rl_engine.kernels.ops.backward_runtime import record_backward
        from rl_engine.kernels.ops.vjp_fp32 import row_local_bias_fp32

        x2d, gamma_f, w_f, xhat, rstd, z = ctx.saved_tensors
        lead_shape = ctx.lead_shape
        rows, hidden = x2d.shape
        g = grad_output.reshape(-1, grad_output.size(-1)).float().contiguous()

        dz = triton_tree_gemm(g, w_f.t().contiguous())
        du = dz  # plain identity (frozen seam -- no cast, no STE)

        dxhat = du * gamma_f.unsqueeze(0)
        dot = triton_row_dot(dxhat.contiguous(), xhat)
        t1 = true_div_rn(dot, DIVISOR_F32)
        t2 = xhat * t1.unsqueeze(1)
        t3 = dxhat - t2  # isolated ops -- FMS contraction is forbidden
        dx = rstd.unsqueeze(1) * t3

        dgamma = triton_dgamma_fold(du, xhat)
        dw = triton_dw_left_fold(g, z)
        db = row_local_bias_fp32(g) if ctx.has_bias else None

        record_backward(
            "txt_in_rmsnorm_linear",
            kernel_id="txt-in-rmsnorm-linear-v1",
            impl="triton_leaf_chunks_shared_tree_combine",
            family="triton",
        )
        grad_x = dx.reshape(*lead_shape, hidden).to(grad_output.dtype)
        grad_gamma = dgamma.to(grad_output.dtype)
        grad_w = dw.to(grad_output.dtype)
        grad_b = None if db is None else db.to(grad_output.dtype)
        return grad_x, grad_gamma, grad_w, grad_b


class TritonTxtInRMSNormLinearOp:
    """Triton backend: device leaf chunks, shared tree combine, torch seam."""

    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "triton_leaf_chunks_shared_tree_combine"
    contract_version = "txt-in-rmsnorm-linear-v1"

    def __init__(self) -> None:
        _require_triton()
        logger.info("TritonTxtInRMSNormLinearOp ready (leaf chunks + shared combine).")

    def __call__(
        self,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(x, norm_weight, weight, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not x.is_cuda:
            raise ValueError("TritonTxtInRMSNormLinearOp requires CUDA tensors")
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"supported dtypes are bf16/fp32 (contract scope), got {x.dtype}")
        # one source of truth for the fail-closed shape/dtype/device checks
        NativeTxtInRMSNormLinearOp._check_inputs(x, norm_weight, weight, bias)
        return _TxtInRMSNormLinearTritonFunction.apply(x, norm_weight, weight, bias)
