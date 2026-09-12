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
    from rl_engine.kernels.ops.pytorch.norm.rms_norm import NativeRMSNormOp

    return NativeRMSNormOp()


def _fixed_row_sum(values: torch.Tensor) -> torch.Tensor:
    """Sum the last dimension with an explicit adjacent-pair FP32 tree.

    A fixed reduction width passed to torch.sum is insufficient on NPU:
    dispatch can also depend on the number of rows. Each step here is an
    elementwise add; the pairs depend only on the hidden dimension. Carry
    an odd final element unchanged rather than dropping or duplicating it.
    """
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("row reduction requires a non-empty last dimension")
    partial = values.float()
    while partial.shape[-1] > 1:
        paired = (partial.shape[-1] // 2) * 2
        reduced = partial[..., :paired:2] + partial[..., 1:paired:2]
        if paired != partial.shape[-1]:
            reduced = torch.cat((reduced, partial[..., -1:]), dim=-1)
        partial = reduced
    return partial[..., 0]


def _rms_norm_backward_rows(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    grad_out_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm dx and unreduced FP32 dweight rows using forward-saved rstd.

    With y = x * rstd * w and s = sum(dy * w * x, dim=-1):
        dx = rstd * (dy * w) - x * rstd^3 * s / H
        dweight_rows = dy * x * rstd

    Both ordinary and canonical backward use this row-local computation.
    Parameter gradients are reduced by the caller, after all logical rows
    are available in the canonical case.
    """
    dy_f = grad_out_2d.float()
    x_f = x_2d.float()
    w_f = weight.float()
    rstd_f = rstd.float()

    dyw = dy_f * w_f
    s = _fixed_row_sum(dyw * x_f)
    hidden = x_2d.size(-1)
    dx = rstd_f.unsqueeze(-1) * dyw - x_f * (rstd_f.pow(3) / hidden).unsqueeze(-1) * s.unsqueeze(-1)
    rows = dy_f * x_f * rstd_f.unsqueeze(-1)
    return dx.to(x_2d.dtype), rows


def _rms_norm_backward(
    x_2d: torch.Tensor,
    weight: torch.Tensor,
    rstd: torch.Tensor,
    grad_out_2d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from rl_engine.kernels.ops.vjp_fp32 import reduce_rows_fp32

    dx, rows = _rms_norm_backward_rows(x_2d, weight, rstd, grad_out_2d)
    return dx, reduce_rows_fp32(rows).to(weight.dtype)


def _fixed_rstd(x32: torch.Tensor, eps: float) -> torch.Tensor:
    """Shape-invariant per-row rstd.

    torch mean/sum select shape-dependent reduction kernels on NPU and flip
    single-ULP results between batch layouts (e.g. [1,7,H] vs [1,20,H]),
    which breaks the chunked-vs-full model invariance. The rowwise FP32
    GEMM reduces each output row in one fixed per-row order regardless of
    the batch layout, so the sum of squares -- and hence the rstd -- is
    bitwise identical for every layout.
    """
    from rl_engine.kernels.ops.pytorch.norm.rms_norm import shape_invariant_rstd

    return shape_invariant_rstd(x32, float(eps)).contiguous()

class _RMSNormAscendFunction(torch.autograd.Function):
    # Autograd wrapper: reference-formula rstd + Ascend C fused scale/cast
    # forward, and the PyTorch-formula backward reusing the forward-saved
    # rstd (same fp32 VJP as the PyTorch reference, like the CUDA op).

    @staticmethod
    def forward(ctx, x, weight, eps):
        lead_shape = x.shape[:-1]
        hidden = x.size(-1)

        x_2d = x.reshape(-1, hidden).contiguous()

        # rstd is computed with the exact torch ops of the PyTorch reference
        # (rl_engine/kernels/ops/pytorch/norm/rms_norm.py): fp32 mean of
        # squares + torch.rsqrt. The Ascend C kernel then only performs the
        # elementwise y = x * rstd * w scale and the round-to-nearest-even
        # cast, which are order-free IEEE ops — this makes the fused output
        # bitwise identical to NativeRMSNormOp instead of approximating its
        # sum-of-squares/rsqrt arithmetic in-kernel.
        x_f = x_2d.float()
        rstd = _fixed_rstd(x_f, float(eps))

        y = _C_npu.rmsnorm_ascend(x_2d, weight, rstd)

        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.eps = eps
        ctx.lead_shape = lead_shape
        return y.reshape(lead_shape + (hidden,))

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, weight, rstd = ctx.saved_tensors
        hidden = x_2d.size(-1)

        grad_out_2d = grad_output.reshape(-1, hidden).contiguous()
        dx, dw = _rms_norm_backward(x_2d, weight, rstd, grad_out_2d)

        dx = dx.reshape(ctx.lead_shape + (hidden,))
        return dx, dw, None


class RMSNormAscendOp:
    # Ascend C batch-invariant RMSNorm (forward kernel).

    def __init__(self) -> None:
        if not _NPU_EXT_AVAILABLE or not hasattr(_C_npu, "rmsnorm_ascend"):
            raise RuntimeError(
                "rmsnorm_ascend is not compiled into the extension. Rebuild with "
                "KERNEL_ALIGN_FORCE_ASCEND=1 on an Ascend NPU host: 'pip install -e .'"
            )
        logger.info("Successfully linked to precompiled _C_npu.rmsnorm_ascend kernel.")

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return self.forward(x, weight, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if weight.dim() != 1 or weight.shape[0] != x.shape[-1]:
            raise ValueError(
                f"weight must be 1-D of size x.shape[-1]={x.shape[-1]}, "
                f"got tuple(weight.shape)={tuple(weight.shape)}"
            )

        if not _ascend_supported(x) or weight.dtype != x.dtype:
            return _fallback_op()(x, weight, eps=eps)

        return _RMSNormAscendFunction.apply(x, weight, eps)

    def parameter_vjp_contributions_fp32(
        self, *, x: torch.Tensor, weight: torch.Tensor, grad_output: torch.Tensor, eps: float = 1e-6
    ) -> dict[str, torch.Tensor]:
        """Canonical row-fold parameter contribution (the CUDA twin).

        dweight = sum_rows grad * x * rstd: each row's FP32 contribution is
        returned separately, and the C4 harness accumulates the per-row
        contributions in FP32 across call spans, so chunked / padded /
        permuted / singleton-aggregated layouts sum the same row
        contributions in the same order and produce a bitwise-identical
        weight gradient.
        """
        del weight
        x32 = x.float()
        rstd = _fixed_rstd(x32, float(eps))
        rows = grad_output.float() * x32 * rstd.unsqueeze(-1)
        return {"weight": rows}


def rmsnorm_ascend(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    return RMSNormAscendOp()(x, weight, eps=eps)
