# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA H3 final AdaLN output (RFC #420 ``final_adaln_out``)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.backends.extension import _C
from rl_engine.backends.cuda.model_specific.minimax_h3.adaln_row_gather import _segment_tiles
from rl_engine.backends.cuda.model_specific.minimax_h3.det_linear import (
    det_linear_available,
    det_linear_backward_input,
    det_linear_backward_weight,
    det_linear_forward,
    silu_backward_fp32,
)
from rl_engine.backends.cuda.model_specific.minimax_h3.rmsnorm import h3_rmsnorm_available
from rl_engine.reference.minimax_h3.final_adaln_out import validate_h3_final_adaln_out
from rl_engine.reference.minimax_h3.rmsnorm import H3_NORM_EPS

KERNEL_ID = (
    "torch.silu[fp32]+cast[weight_dtype]+rl_engine._C.h3_det_linear_forward"
    "+rl_engine._C.h3_rmsnorm_forward[modulated]"
)
BACKWARD_IMPL = "fp32_table_grad_into_h3_det_linear_backward"
BACKWARD_KERNEL_ID = (
    "rl_engine._C.h3_rmsnorm_backward+rl_engine._C.h3_det_linear_backward_weight"
    "+rl_engine._C.h3_det_linear_backward_input"
    "+rl_engine.backends.cuda.model_specific.minimax_h3.det_linear.silu_backward_fp32"
)


class _H3FinalAdaLNOutCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, norm_weight, temb, weight, bias, timestep_indices, eps):
        hidden = x.shape[-1]
        temb = temb.contiguous()
        act = F.silu(temb).to(weight.dtype)  # declared boundary: one cast after the FP32 SiLU
        (table,) = det_linear_forward(act, weight, bias)
        shift, scale = table.chunk(2, dim=-1)
        x2 = x.contiguous().view(-1, hidden)
        y, rstd = _C.h3_rmsnorm_forward(
            x2, norm_weight.contiguous(), float(eps), shift, scale, timestep_indices
        )
        ctx.save_for_backward(x2, norm_weight, rstd, temb, act, weight, table, timestep_indices)
        ctx.x_shape = x.shape
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, grad):
        x2, norm_weight, rstd, temb, act, weight, table, timestep_indices = ctx.saved_tensors
        shift, scale = table.chunk(2, dim=-1)
        positions = timestep_indices.repeat(x2.shape[0] // timestep_indices.shape[0])
        tiles = _segment_tiles(positions, table.shape[0])
        dx, d_norm_weight, d_shift, d_scale = _C.h3_rmsnorm_backward(
            grad.contiguous().view_as(x2),
            x2,
            norm_weight.contiguous(),
            rstd,
            shift,
            scale,
            timestep_indices,
            *tiles,
        )
        # The table gradient stays FP32 into the projection backward.
        d_table = torch.cat([d_shift, d_scale], dim=1)
        dw, db = det_linear_backward_weight(d_table, act, weight.dtype)
        d_temb = silu_backward_fp32(det_linear_backward_input(d_table, weight, torch.float32), temb)
        record_backward(
            "final_adaln_out", kernel_id=BACKWARD_KERNEL_ID, impl=BACKWARD_IMPL, family="cuda"
        )
        return dx.view(ctx.x_shape), d_norm_weight, d_temb, dw, db, None, None


class H3FinalAdaLNOutCudaOp:
    """CUDA candidate: tensor-core shift/scale projection + fused norm/modulation.

    The projection is ``adaln_projection_3mod``'s deterministic GEMV
    (``h3-det-linear-bf16-mma-v1``) on ``norm_out.linear``; the norm and
    modulation are ``h3_rmsnorm``'s kernel indexed by ``timestep_indices``.
    One autograd node, so the table gradient reaches the projection in FP32.
    """

    op_class = "reduction"
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL

    def __init__(self) -> None:
        if not (det_linear_available() and h3_rmsnorm_available()):
            raise RuntimeError("rl_engine._C lacks the h3_det_linear_* / h3_rmsnorm_* ops")

    def __call__(self, x, norm_weight, temb, weight, bias, timestep_indices, eps=H3_NORM_EPS):
        return self.forward(x, norm_weight, temb, weight, bias, timestep_indices, eps)

    def forward(self, x, norm_weight, temb, weight, bias, timestep_indices, eps=H3_NORM_EPS):
        validate_h3_final_adaln_out(x, norm_weight, temb, weight, bias, timestep_indices, eps)
        if weight.dtype != x.dtype:
            raise TypeError("norm_out.linear must share the activations' dtype")
        if not x.is_cuda:
            raise ValueError("H3FinalAdaLNOutCudaOp needs CUDA tensors")
        return _H3FinalAdaLNOutCuda.apply(
            x, norm_weight, temb, weight, bias, timestep_indices.contiguous(), eps
        )

    def forward_fp32(self, x, norm_weight, temb, weight, bias, timestep_indices, eps=H3_NORM_EPS):
        return self.forward(x, norm_weight, temb, weight, bias, timestep_indices, eps).float()
