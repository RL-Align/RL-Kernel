# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA backend for the Qwen-Image txt_in RMSNorm->Linear op (contract
``txt-in-rmsnorm-linear-v1``).

Wraps the correctness-anchor kernels in ``csrc/cuda/norm/txt_in_rmsnorm_linear.cu``:

- norm stats: one thread per row evaluates the full frozen sumsq tree, rstd
  runs the pinned three-step form (``div.rn`` / ``add.rn`` / ``sqrt.rn`` /
  ``div.rn`` with the eps bit pattern ``0x358637BD``), and xhat/z are two
  isolated rn muls -- the frozen FP32 seam, no intermediate cast.
- forward GEMM and dW reuse the attn-out entries (same frozen tree; the
  forward adds bias once in fp32 and performs the single RNE cast in-kernel).
- dx runs the frozen chain (dz tree via ``W.T``, dxhat mul, dot tree,
  ``t1 = dot/3584`` true division, isolated ``t2/t3`` -- no FMS contraction);
  dgamma runs the ascending-row left-fold kernel; ``db`` uses the shared
  ``row_local_bias_fp32`` pure-add fold.
- One correctly-rounded FP32 FMA discipline on BOTH sides (kernels use
  ``__fmaf_rn``; the reference uses ``torch.addcmul``, an oracle-verified
  correctly-rounded FMA), so this backend matches the reference byte for
  byte in BOTH dtypes -- no tolerance path.
"""

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.norm.txt_in_rmsnorm_linear import NativeTxtInRMSNormLinearOp
from rl_engine.utils.logger import logger

_REQUIRED_SYMBOLS = (
    "txt_in_norm_stats_cuda",
    "txt_in_row_tree_reduce_cuda",
    "txt_in_dx_cuda",
    "txt_in_dgamma_fold_cuda",
    "attn_out_bias_gemm_cuda_forward",
    "attn_out_tree_gemm_cuda",
    "attn_out_dw_left_fold_cuda",
)


def _cuda_backend_available() -> bool:
    return _EXT_AVAILABLE and all(hasattr(_C, name) for name in _REQUIRED_SYMBOLS)


class _TxtInRMSNormLinearCudaFunction(torch.autograd.Function):
    """CUDA forward/backward under the same contract as the reference."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x2d = x.reshape(-1, x.size(-1)).float().contiguous()
        gamma_f = norm_weight.float().contiguous()
        w_f = weight.float().contiguous()
        b_f = None if bias is None else bias.float().contiguous()

        xhat, z, rstd = _C.txt_in_norm_stats_cuda(x2d, gamma_f)
        out = _C.attn_out_bias_gemm_cuda_forward(z, w_f, b_f, x.dtype == torch.bfloat16)
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
        # autograd may hand us an expanded (non-contiguous) upstream gradient;
        # materialise it before the C++ contiguous checks
        g = grad_output.reshape(-1, grad_output.size(-1)).float().contiguous()

        # dz = tree(g, W.T) over the N dim; du := dz (frozen seam -- no cast)
        dz = _C.attn_out_tree_gemm_cuda(g, w_f.t().contiguous())
        dx = _C.txt_in_dx_cuda(dz, gamma_f, xhat, rstd)
        dgamma = _C.txt_in_dgamma_fold_cuda(dz, xhat)
        dw = _C.attn_out_dw_left_fold_cuda(g, z)
        db = row_local_bias_fp32(g) if ctx.has_bias else None

        record_backward(
            "txt_in_rmsnorm_linear",
            kernel_id="txt-in-rmsnorm-linear-v1",
            impl="cuda_per_thread_tree_fma",
            family="cuda",
        )
        grad_x = dx.reshape(*lead_shape, hidden).to(grad_output.dtype)
        grad_gamma = dgamma.to(grad_output.dtype)
        grad_w = dw.to(grad_output.dtype)
        grad_b = None if db is None else db.to(grad_output.dtype)
        return grad_x, grad_gamma, grad_w, grad_b


class CudaTxtInRMSNormLinearOp:
    """CUDA backend: per-thread row trees, attn-out GEMM reuse, left folds."""

    op_class = "reduction"
    is_batch_invariant = True
    backward_impl = "cuda_per_thread_tree_fma"
    contract_version = "txt-in-rmsnorm-linear-v1"

    def __init__(self) -> None:
        if not _cuda_backend_available():
            raise RuntimeError(
                "txt_in_rmsnorm_linear CUDA symbols are not compiled into the extension. "
                "Rebuild with the CUDA extension enabled: 'pip install -e .' on a CUDA "
                "host (csrc/cuda/norm/txt_in_rmsnorm_linear.cu is part of the default "
                "cuda_sources list); refusing non-strict fallback."
            )
        logger.info("CudaTxtInRMSNormLinearOp ready (per-thread row trees, FMA chains).")

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
            raise ValueError("CudaTxtInRMSNormLinearOp requires CUDA tensors")
        if x.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"supported dtypes are bf16/fp32 (contract scope), got {x.dtype}")
        # one source of truth for the fail-closed shape/dtype/device checks
        NativeTxtInRMSNormLinearOp._check_inputs(x, norm_weight, weight, bias)
        return _TxtInRMSNormLinearCudaFunction.apply(x, norm_weight, weight, bias)
