# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA H3 FP32 timestep MLP (RFC #420 ``timestep_mlp_fp32``)."""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.cuda.h3.det_linear import (
    ACT_NONE,
    ACT_SILU,
    CONTRACT,
    det_linear_available,
    det_linear_backward_input,
    det_linear_backward_weight,
    det_linear_forward,
    silu_backward_fp32,
)
from rl_engine.kernels.ops.pytorch.h3.timestep_mlp import validate_h3_timestep_mlp

KERNEL_ID = "rl_engine._C.h3_det_linear_forward[silu]+rl_engine._C.h3_det_linear_forward"
BACKWARD_IMPL = "h3_det_linear_v1_chunked_dinput_ascending_row_dweight"
BACKWARD_KERNEL_ID = (
    "rl_engine._C.h3_det_linear_backward_weight"
    "+rl_engine._C.h3_det_linear_backward_input"
    "+rl_engine.kernels.ops.cuda.h3.det_linear.silu_backward_fp32"
)


class _H3TimestepMLPCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, b1, w2, b2):
        x = x.contiguous()
        hidden, pre = det_linear_forward(x, w1, b1, activation=ACT_SILU, save_pre_activation=True)
        (out,) = det_linear_forward(hidden, w2, b2, activation=ACT_NONE)
        ctx.save_for_backward(x, w1, w2, pre, hidden)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2, pre, hidden = ctx.saved_tensors
        need_x, need_w1, need_b1, need_w2, need_b2 = ctx.needs_input_grad
        grad_out = grad_out.float().contiguous()
        dw2 = db2 = dw1 = db1 = dx = None
        if need_w2 or need_b2:
            dw2, db2 = det_linear_backward_weight(grad_out, hidden, torch.float32)
        d_pre = None
        if need_x or need_w1 or need_b1:
            d_hidden = det_linear_backward_input(grad_out, w2, torch.float32)
            d_pre = silu_backward_fp32(d_hidden, pre).contiguous()
        if need_w1 or need_b1:
            dw1, db1 = det_linear_backward_weight(d_pre, x, torch.float32)
        if need_x:
            dx = det_linear_backward_input(d_pre, w1, torch.float32)
        record_backward(
            "timestep_mlp_fp32", kernel_id=BACKWARD_KERNEL_ID, impl=BACKWARD_IMPL, family="cuda"
        )
        return (
            dx,
            dw1 if need_w1 else None,
            db1 if need_b1 else None,
            dw2 if need_w2 else None,
            db2 if need_b2 else None,
        )


class H3TimestepMLPCudaOp:
    """CUDA candidate: two warp-per-column GEMVs, SiLU fused into the first.

    The weights dominate the traffic (5.5 MB + 57.8 MB FP32), the rows are the
    handful of distinct timesteps, and every output element follows contract
    ``h3-det-linear-v1``, so a timestep's ``temb`` bytes do not depend on which
    other timesteps share the call.
    """

    op_class = "reduction"
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL
    contract = CONTRACT

    def __init__(self) -> None:
        if not det_linear_available():
            raise RuntimeError(
                "rl_engine._C lacks h3_det_linear_*; rebuild with csrc/cuda/h3/det_linear.cu"
            )

    def __call__(self, x, w1, b1, w2, b2):
        return self.forward(x, w1, b1, w2, b2)

    def forward(self, x, w1, b1, w2, b2) -> torch.Tensor:
        validate_h3_timestep_mlp(x, w1, b1, w2, b2)
        if not x.is_cuda:
            raise ValueError("H3TimestepMLPCudaOp needs CUDA tensors")
        return _H3TimestepMLPCuda.apply(x, w1, b1, w2, b2)

    def forward_fp32(self, x, w1, b1, w2, b2) -> torch.Tensor:
        return self.forward(x, w1, b1, w2, b2)
