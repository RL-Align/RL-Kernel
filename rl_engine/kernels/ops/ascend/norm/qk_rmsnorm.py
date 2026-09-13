# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Any

import torch

from rl_engine.utils.logger import logger

_C_npu: Any = None
try:
    from rl_engine import _C_npu

    _NPU_EXT_AVAILABLE = True
except ImportError:  # pragma: no cover - Ascend extension not built
    _NPU_EXT_AVAILABLE = False


def _ascend_supported(x: torch.Tensor) -> bool:
    """Whether the Ascend C forward can run this input directly.

    NPU tensors only, fp32/bf16/fp16 only (mirrors the CUDA kernel's gate).
    """
    return x.device.type == "npu" and x.dtype in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    )


def _fallback_op():
    """Portable op for inputs the Ascend forward cannot take.

    Triton rejects non-CUDA devices, so on NPU the only fallback is native.
    """
    from rl_engine.kernels.ops.pytorch.norm.qk_rmsnorm import NativeQkRmsNormOp

    return NativeQkRmsNormOp()


def _qk_rms_norm_backward(
    x_2d: torch.Tensor,
    rstd: torch.Tensor,
    grad_out_2d: torch.Tensor,
) -> torch.Tensor:
    """Parameter-free per-head RMSNorm VJP in fp32, reusing the saved rstd.

    With y = x * rstd and s = sum(dy * x, dim=-1):
        dx = rstd * dy - x * rstd^3 * s / D
    """
    dy_f = grad_out_2d.float()
    x_f = x_2d.float()
    rstd_f = rstd.float()

    head_dim = x_2d.size(-1)
    s = (dy_f * x_f).sum(dim=-1, keepdim=True)
    dx = rstd_f.unsqueeze(-1) * dy_f - x_f * (rstd_f.pow(3) / head_dim) * s
    return dx.to(x_2d.dtype)


class _QkRmsNormAscendFunction(torch.autograd.Function):
    # Autograd wrapper: reference-formula rstd + Ascend C fused scale/cast
    # forward, and the PyTorch-formula backward reusing the forward-saved
    # rstd (same fp32 VJP as the PyTorch reference, like the CUDA op).

    @staticmethod
    def forward(ctx, x, eps):
        lead_shape = x.shape[:-1]
        head_dim = x.size(-1)

        x_2d = x.reshape(-1, head_dim).contiguous()

        # rstd is computed with the exact torch ops of the PyTorch reference
        # (rl_engine/kernels/ops/pytorch/norm/qk_rmsnorm.py): fp32 mean of
        # squares + torch.rsqrt. The Ascend C kernel then only performs the
        # elementwise y = x * rstd scale and the round-to-nearest-even cast,
        # which are order-free IEEE ops — this makes the fused output bitwise
        # identical to NativeQkRmsNormOp instead of approximating its
        # sum-of-squares/rsqrt arithmetic in-kernel.
        x_f = x_2d.float()
        var = x_f.pow(2).mean(dim=-1)
        rstd = torch.rsqrt(var + eps).contiguous()

        y = _C_npu.qk_rmsnorm_ascend(x_2d, rstd)

        ctx.save_for_backward(x_2d, rstd)
        ctx.lead_shape = lead_shape
        ctx.head_dim = head_dim
        return y.reshape(lead_shape + (head_dim,))

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, rstd = ctx.saved_tensors

        grad_out_2d = grad_output.reshape(-1, ctx.head_dim).contiguous()
        dx = _qk_rms_norm_backward(x_2d, rstd, grad_out_2d)

        dx = dx.reshape(ctx.lead_shape + (ctx.head_dim,))
        return dx, None


class QkRmsNormAscendOp:
    # Ascend C batch-invariant Qwen-Image per-head QK RMSNorm (forward kernel).

    def __init__(self) -> None:
        if not _NPU_EXT_AVAILABLE or not hasattr(_C_npu, "qk_rmsnorm_ascend"):
            raise RuntimeError(
                "qk_rmsnorm_ascend is not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host: 'pip install -e .'"
            )
        logger.info("Successfully linked to precompiled _C_npu.qk_rmsnorm_ascend kernel.")

    def __call__(self, x: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        return self.forward(x, eps=eps)

    def forward(self, x: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
        if not _ascend_supported(x):
            return _fallback_op()(x, eps=eps)

        return _QkRmsNormAscendFunction.apply(x, eps)


def qk_rmsnorm_ascend(x: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    return QkRmsNormAscendOp()(x, eps=eps)
