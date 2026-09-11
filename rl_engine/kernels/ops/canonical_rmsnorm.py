# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Logical-row canonical RMSNorm parameter VJP for WS1."""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.canonical_backward import active_session
from rl_engine.kernels.ops.triton.rmsnorm_triton import (
    rmsnorm_triton_backward_rows,
    rmsnorm_triton_forward_with_rstd,
)
from rl_engine.kernels.ops.vjp_fp32 import reduce_rows_fp32


class _CanonicalCudaRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, logical_keys, parameter_id):
        session = active_session()
        if session is None:
            raise RuntimeError("canonical RMSNorm requires an active backward session")
        y, rstd = _C.rmsnorm_forward(x.contiguous(), weight.contiguous(), float(eps))
        ctx.save_for_backward(x, weight, rstd)
        ctx.session = session
        ctx.parameter_id = str(parameter_id)
        ctx.slot = session.register(ctx.parameter_id, logical_keys)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, rstd = ctx.saved_tensors
        dy = grad_out.contiguous()
        dx = _C.rmsnorm_backward_dx(dy, x, weight, rstd)
        dw = None
        if ctx.needs_input_grad[1]:
            rows = dy.float() * x.float() * rstd.float().unsqueeze(-1)
            dw = ctx.session.submit_rows(
                ctx.parameter_id,
                ctx.slot,
                rows,
                lambda ordered: reduce_rows_fp32(ordered).to(weight.dtype),
            )
        return dx, dw, None, None, None


def canonical_cuda_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    logical_keys: torch.Tensor,
    parameter_id: str,
) -> torch.Tensor:
    return _CanonicalCudaRMSNorm.apply(x, weight, eps, logical_keys, parameter_id)


__all__ = ["canonical_cuda_rmsnorm", "canonical_ascend_rmsnorm", "canonical_row_rmsnorm"]


class _CanonicalAscendRMSNorm(torch.autograd.Function):
    """Ascend twin of _CanonicalCudaRMSNorm.

    Forward reuses the Ascend op's split: the reference FP32 rstd, then the
    Ascend C kernel for the elementwise scale/cast. Backward folds the weight
    gradient over logical rows through the session, so dw depends only on the
    logical row identity and not on how rows were batched.
    """

    @staticmethod
    def forward(ctx, x, weight, eps, logical_keys, parameter_id):
        from rl_engine.kernels.ops.ascend.norm.rmsnorm import _C_npu

        session = active_session()
        if session is None:
            raise RuntimeError("canonical RMSNorm requires an active backward session")
        x_c = x.contiguous()
        weight_c = weight.contiguous()
        var = x_c.float().pow(2).mean(dim=-1)
        rstd = torch.rsqrt(var + float(eps)).contiguous()
        y = _C_npu.rmsnorm_ascend(x_c, weight_c, rstd)
        ctx.save_for_backward(x_c, weight_c, rstd)
        ctx.session = session
        ctx.parameter_id = str(parameter_id)
        ctx.slot = session.register(ctx.parameter_id, logical_keys)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, rstd = ctx.saved_tensors
        dy = grad_out.contiguous()
        # Same FP32 VJP the Ascend op uses, kept row-wise so the weight
        # gradient can be folded in canonical logical-row order.
        dy_f = dy.float()
        x_f = x.float()
        rstd_f = rstd.float()
        dyw = dy_f * weight.float()
        hidden = x.size(-1)
        s = (dyw * x_f).sum(dim=-1)
        dx = (
            rstd_f.unsqueeze(-1) * dyw
            - x_f * (rstd_f.pow(3) / hidden).unsqueeze(-1) * s.unsqueeze(-1)
        ).to(x.dtype)
        dw = None
        if ctx.needs_input_grad[1]:
            rows = dy_f * x_f * rstd_f.unsqueeze(-1)
            dw = ctx.session.submit_rows(
                ctx.parameter_id,
                ctx.slot,
                rows,
                lambda ordered: reduce_rows_fp32(ordered).to(weight.dtype),
            )
        record_backward(
            "rms_norm",
            kernel_id=(
                "csrc/ascend/rmsnorm_ascend.asc:rmsnorm_ascend_forward"
                "+rl_engine.kernels.ops.vjp_fp32.reduce_rows_fp32"
            ),
            impl="ascend_rmsnorm_canonical_rowfold",
            family="ascend",
        )
        return dx, dw, None, None, None


def canonical_ascend_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    logical_keys: torch.Tensor,
    parameter_id: str,
) -> torch.Tensor:
    return _CanonicalAscendRMSNorm.apply(x, weight, eps, logical_keys, parameter_id)


class _CanonicalRowRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, logical_keys, parameter_id, forward_op):
        session = active_session()
        if session is None:
            raise RuntimeError("canonical RMSNorm requires an active backward session")
        del forward_op
        with torch.no_grad():
            y, rstd = rmsnorm_triton_forward_with_rstd(x, weight, float(eps))
        ctx.save_for_backward(x, weight, rstd)
        ctx.session = session
        ctx.parameter_id = str(parameter_id)
        ctx.slot = session.register(ctx.parameter_id, logical_keys)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, rstd = ctx.saved_tensors
        dx, rows = rmsnorm_triton_backward_rows(grad_out.contiguous(), x, weight, rstd)
        dw = ctx.session.submit_rows(
            ctx.parameter_id,
            ctx.slot,
            rows,
            lambda ordered: reduce_rows_fp32(ordered).to(weight.dtype),
        )
        record_backward(
            "rms_norm",
            kernel_id=(
                "rl_engine.kernels.ops.triton.rmsnorm_triton._rmsnorm_bwd_dx_kernel"
                "+rl_engine.kernels.ops.vjp_fp32.reduce_rows_fp32"
            ),
            impl="triton_rmsnorm_canonical_rowfold",
            family="triton",
        )
        return dx, dw, None, None, None, None


def canonical_row_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    logical_keys: torch.Tensor,
    parameter_id: str,
    forward_op,
) -> torch.Tensor:
    return _CanonicalRowRMSNorm.apply(x, weight, eps, logical_keys, parameter_id, forward_op)
