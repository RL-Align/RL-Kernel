# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA per-head Q/K RMSNorm (RFC #420 ``h3_qk_rmsnorm_d128``).

The kernel is the merged ``h3_rmsnorm`` one (#487), with no modulation: it replays PyTorch's own
RMSNorm reduction, so the result is bitwise ``nn.RMSNorm``, and rows do not depend on the batch
size, position or other rows. This row pins the H3 shape to it: 56 heads (or a TP rank's share)
of 128 entries, one weight shared by the heads, ``eps`` 1e-5.
"""

from __future__ import annotations

import torch

from rl_engine.backends.cuda.model_specific.minimax_h3.rmsnorm import (
    KERNEL_ID,
    h3_rmsnorm_available,
)
from rl_engine.backends.extension import _C
from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.reference.minimax_h3.qk_rmsnorm import H3_QK_NORM_EPS, validate_h3_qk_rmsnorm

BACKWARD_IMPL = "row_local_dx_tiled_dweight"
BACKWARD_KERNEL_ID = "rl_engine._C.h3_rmsnorm_backward"


class _H3QKRMSNormCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x2 = x.contiguous().view(-1, x.shape[-1])
        y, rstd = _C.h3_rmsnorm_forward(x2, weight.contiguous(), float(eps), None, None, None)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.x_shape = x.shape
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, grad):
        x2, weight, rstd = ctx.saved_tensors
        dx, dweight = _C.h3_rmsnorm_backward(
            grad.contiguous().view_as(x2), x2, weight.contiguous(), rstd
        )
        record_backward(
            "h3_qk_rmsnorm_d128", kernel_id=BACKWARD_KERNEL_ID, impl=BACKWARD_IMPL, family="cuda"
        )
        return dx.view(ctx.x_shape), dweight, None


class H3QKRMSNormCudaOp:
    """CUDA candidate: bitwise ``nn.RMSNorm(128)`` per head; rows are independent."""

    op_class = "reduction"
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL

    def __init__(self) -> None:
        if not h3_rmsnorm_available():
            raise RuntimeError(
                "rl_engine._C lacks h3_rmsnorm_*; rebuild with csrc/cuda/h3/rmsnorm_modulate.cu"
            )

    def __call__(self, x, weight, eps: float = H3_QK_NORM_EPS):
        return self.forward(x, weight, eps)

    def forward(self, x, weight, eps: float = H3_QK_NORM_EPS) -> torch.Tensor:
        validate_h3_qk_rmsnorm(x, weight, eps)
        if not x.is_cuda:
            raise ValueError("H3QKRMSNormCudaOp needs CUDA tensors")
        return _H3QKRMSNormCuda.apply(x, weight, eps)

    def forward_fp32(self, x, weight, eps: float = H3_QK_NORM_EPS) -> torch.Tensor:
        return self.forward(x, weight, eps).float()
