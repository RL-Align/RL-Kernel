# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA H3 gated residual (RFC #420 ``adaln_gate_residual``)."""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.cuda.h3.adaln_row_gather import _segment_tiles
from rl_engine.kernels.ops.pytorch.h3.gate_residual import validate_h3_gate_residual

KERNEL_ID = "rl_engine._C.h3_gate_residual_forward"
BACKWARD_IMPL = "exact_dy_sorted_segment_dgate"
BACKWARD_KERNEL_ID = "torch.argsort[stable]+rl_engine._C.h3_gate_residual_backward"


def h3_gate_residual_available() -> bool:
    return bool(
        _EXT_AVAILABLE
        and hasattr(_C, "h3_gate_residual_forward")
        and hasattr(_C, "h3_gate_residual_backward")
    )


class _H3GateResidualCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, y, gate, index):
        hidden = residual.shape[-1]
        res2 = residual.contiguous().view(-1, hidden)
        y2 = y.contiguous().view(-1, hidden)
        out = _C.h3_gate_residual_forward(res2, y2, gate, index)
        ctx.save_for_backward(y2, gate, index)
        ctx.shape = residual.shape
        return out.view(residual.shape)

    @staticmethod
    def backward(ctx, grad):
        y2, gate, index = ctx.saved_tensors
        g2 = grad.contiguous().view_as(y2)
        positions = index.repeat(y2.shape[0] // index.shape[0])
        tiles = _segment_tiles(positions, gate.shape[0])
        dy, dgate = _C.h3_gate_residual_backward(g2, y2, gate, index, *tiles)
        record_backward(
            "adaln_gate_residual", kernel_id=BACKWARD_KERNEL_ID, impl=BACKWARD_IMPL, family="cuda"
        )
        return grad, dy.view(ctx.shape), dgate, None


class H3GateResidualCudaOp:
    """CUDA candidate: ``residual + gate[index] * y`` in one elementwise pass.

    The gate row is gathered in the kernel from the AdaLN table view, and the
    two roundings match the eager expression, so the output is bitwise equal
    to diffusers. ``dy`` is bitwise equal to the eager VJP; ``dgate`` is a
    deterministic FP32 segment sum instead of ``index_select``'s BF16 atomics.
    """

    op_class = "reduction"
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL

    def __init__(self) -> None:
        if not h3_gate_residual_available():
            raise RuntimeError(
                "rl_engine._C lacks h3_gate_residual_*; rebuild with csrc/cuda/h3/gate_residual.cu"
            )

    def __call__(self, residual, y, gate, index):
        return self.forward(residual, y, gate, index)

    def forward(self, residual, y, gate, index, *, check_range: bool = True):
        validate_h3_gate_residual(residual, y, gate, index, check_range=check_range)
        if not residual.is_cuda:
            raise ValueError("H3GateResidualCudaOp needs CUDA tensors")
        if gate.stride(1) != 1:
            gate = gate.contiguous()
        return _H3GateResidualCuda.apply(residual, y, gate, index.contiguous())

    def forward_fp32(self, residual, y, gate, index):
        return self.forward(residual, y, gate, index).float()
