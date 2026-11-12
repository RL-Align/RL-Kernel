# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Ascend C SwiGLU, with FP32 math and fused forward/backward kernels."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

_C_npu: Any = None
try:
    from rl_engine import _C_npu
except ImportError:  # pragma: no cover - extension requires CANN + torch_npu
    pass

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_inputs(gate: Tensor, up: Tensor) -> None:
    if gate.device.type != "npu" or up.device.type != "npu":
        raise RuntimeError("SwiGLUAscendOp requires NPU tensors.")
    if gate.device != up.device:
        raise RuntimeError("gate and up must be on the same NPU device.")
    if gate.shape != up.shape:
        raise ValueError("gate and up must share shape.")
    for name, value in (("gate", gate), ("up", up)):
        if value.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {value.dtype}.")
    if gate.dtype != up.dtype:
        raise TypeError("gate and up must share dtype.")


class _SwiGLUAscendFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        gate_c, up_c = gate.contiguous(), up.contiguous()
        result = _C_npu.swiglu_forward(gate_c, up_c)
        ctx.save_for_backward(gate_c, up_c)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: Tensor):
        gate, up = ctx.saved_tensors
        d_gate = d_up = None
        if any(ctx.needs_input_grad):
            grads = _C_npu.swiglu_backward(grad_out.contiguous(), gate, up)
            if ctx.needs_input_grad[0]:
                d_gate = grads[0]
            if ctx.needs_input_grad[1]:
                d_up = grads[1]
        return d_gate, d_up


class SwiGLUAscendOp:
    """``(gate * sigmoid(gate)) * up`` on NPU, with first-order autograd.

    Inputs share shape, dtype and device. Arbitrary shapes, empty tensors and
    strided views are supported; the native kernels receive contiguous tensors.
    """

    op_class = "elementwise"

    def __init__(self) -> None:
        if _C_npu is None or not all(
            hasattr(_C_npu, name) for name in ("swiglu_forward", "swiglu_backward")
        ):
            raise RuntimeError(
                "Ascend C SwiGLU kernels are not compiled into rl_engine._C_npu. "
                "Rebuild on an Ascend host with CANN and torch_npu: "
                "KERNEL_ALIGN_FORCE_ASCEND=1 pip install --no-build-isolation -e ."
            )

    def __call__(self, gate: Tensor, up: Tensor) -> Tensor:
        return self.forward(gate, up)

    def forward(self, gate: Tensor, up: Tensor) -> Tensor:
        """Compute in FP32 and return the input dtype."""
        _validate_inputs(gate, up)
        return _SwiGLUAscendFunction.apply(gate, up)

    def forward_fp32(self, gate: Tensor, up: Tensor) -> Tensor:
        """Compute and return FP32, preserving gradients to the original inputs."""
        _validate_inputs(gate, up)
        return _SwiGLUAscendFunction.apply(gate.float(), up.float())
