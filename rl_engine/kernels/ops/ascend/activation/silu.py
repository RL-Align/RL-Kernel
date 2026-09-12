# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Ascend C SiLU, with FP32 math and fused forward/backward kernels."""

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


def _validate_inputs(x: Tensor) -> None:
    if x.device.type != "npu":
        raise RuntimeError("SiLUAscendOp requires NPU tensors.")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x must have dtype fp16, bf16, or fp32, got {x.dtype}.")


class _SiLUAscendFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        x_c = x.contiguous()
        result = _C_npu.silu_forward(x_c)
        ctx.save_for_backward(x_c)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: Tensor):
        (x,) = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None
        return _C_npu.silu_backward(grad_out.contiguous(), x)


class SiLUAscendOp:
    """``x * sigmoid(x)`` on NPU, with first-order autograd.

    Shape-agnostic elementwise op: every element is evaluated by the same fixed
    FP32 expression regardless of tensor size or launched block count, so the
    result is batch-invariant. Empty tensors and strided views are supported;
    the native kernels receive contiguous tensors.
    """

    op_class = "elementwise"

    def __init__(self) -> None:
        if _C_npu is None or not all(
            hasattr(_C_npu, name) for name in ("silu_forward", "silu_backward")
        ):
            raise RuntimeError(
                "Ascend C SiLU kernels are not compiled into rl_engine._C_npu. "
                "Rebuild on an Ascend host with CANN and torch_npu: "
                "KERNEL_ALIGN_FORCE_ASCEND=1 pip install --no-build-isolation -e ."
            )

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """Compute in FP32 and return the input dtype."""
        _validate_inputs(x)
        return _SiLUAscendFunction.apply(x)

    def forward_fp32(self, x: Tensor) -> Tensor:
        """Compute and return FP32, preserving gradients to the original input."""
        _validate_inputs(x)
        return _SiLUAscendFunction.apply(x.float())
