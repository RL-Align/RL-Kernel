# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA H3 RMSNorm with fused AdaLN modulation (RFC #420 ``h3_rmsnorm``)."""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.cuda.h3.adaln_row_gather import _segment_tiles
from rl_engine.kernels.ops.pytorch.h3.rmsnorm import (
    H3_NORM_EPS,
    validate_h3_modulation,
    validate_h3_rmsnorm,
)

KERNEL_ID = "rl_engine._C.h3_rmsnorm_forward"
BACKWARD_IMPL = "row_local_dx_tiled_dweight_sorted_table_grads"
BACKWARD_KERNEL_ID = "torch.argsort[stable]+rl_engine._C.h3_rmsnorm_backward"


def h3_rmsnorm_available() -> bool:
    return bool(
        _EXT_AVAILABLE and hasattr(_C, "h3_rmsnorm_forward") and hasattr(_C, "h3_rmsnorm_backward")
    )


class _H3RMSNormCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, shift, scale, index, eps):
        hidden = x.shape[-1]
        x2 = x.contiguous().view(-1, hidden)
        y, rstd = _C.h3_rmsnorm_forward(x2, weight.contiguous(), float(eps), shift, scale, index)
        ctx.save_for_backward(x2, weight, rstd, shift, scale, index)
        ctx.x_shape = x.shape
        ctx.modulated = index is not None
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, grad):
        x2, weight, rstd, shift, scale, index = ctx.saved_tensors
        g2 = grad.contiguous().view_as(x2)
        if not ctx.modulated:
            dx, dweight = _C.h3_rmsnorm_backward(g2, x2, weight.contiguous(), rstd)
            d_shift = d_scale = None
        else:
            positions = index.repeat(x2.shape[0] // index.shape[0])
            tiles = _segment_tiles(positions, shift.shape[0])
            dx, dweight, d_shift, d_scale = _C.h3_rmsnorm_backward(
                g2, x2, weight.contiguous(), rstd, shift, scale, index, *tiles
            )
            d_shift, d_scale = d_shift.to(shift.dtype), d_scale.to(scale.dtype)
        record_backward(
            "h3_rmsnorm", kernel_id=BACKWARD_KERNEL_ID, impl=BACKWARD_IMPL, family="cuda"
        )
        return dx.view(ctx.x_shape), dweight, d_shift, d_scale, None, None


class H3RMSNormCudaOp:
    """CUDA candidate.

    The statistics replay PyTorch's own RMSNorm reduction order, so the plain
    norm is bitwise equal to ``nn.RMSNorm``; the modulated form gathers its
    ``shift``/``scale`` rows in the kernel (no ``(S, H)`` intermediates) and
    rounds each step where the eager expression does, so it is bitwise equal to
    diffusers' ``n * (1.0 + scale[i]) + shift[i]`` as well. Rows are
    independent of batch size and position.
    """

    op_class = "reduction"
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL

    def __init__(self) -> None:
        if not h3_rmsnorm_available():
            raise RuntimeError(
                "rl_engine._C lacks h3_rmsnorm_*; rebuild with csrc/cuda/h3/rmsnorm_modulate.cu"
            )

    def __call__(self, x, weight, eps: float = H3_NORM_EPS):
        return self.forward(x, weight, eps)

    def forward(self, x, weight, eps: float = H3_NORM_EPS) -> torch.Tensor:
        validate_h3_rmsnorm(x, weight, eps)
        if not x.is_cuda:
            raise ValueError("H3RMSNormCudaOp needs CUDA tensors")
        return _H3RMSNormCuda.apply(x, weight, None, None, None, eps)

    def forward_modulated(
        self, x, weight, shift, scale, index, eps: float = H3_NORM_EPS, *, check_range=True
    ) -> torch.Tensor:
        validate_h3_rmsnorm(x, weight, eps)
        validate_h3_modulation(x, shift, scale, index, check_range=check_range)
        if not x.is_cuda:
            raise ValueError("H3RMSNormCudaOp needs CUDA tensors")
        if shift.stride(1) != 1 or scale.stride(1) != 1 or shift.stride(0) != scale.stride(0):
            shift, scale = shift.contiguous(), scale.contiguous()
        return _H3RMSNormCuda.apply(x, weight, shift, scale, index.contiguous(), eps)

    def forward_fp32(self, x, weight, eps: float = H3_NORM_EPS) -> torch.Tensor:
        return self.forward(x, weight, eps).float()
