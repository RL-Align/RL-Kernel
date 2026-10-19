# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA SiLU / SwiGLU ops matching NativeSiLUOp / NativeSwiGLUOp (WS1 ground truth).

Math is performed in fp32 inside the CUDA kernels and rounded back to the input
dtype on store — the same dual-path contract as the PyTorch references:

  silu(x)      = x * sigmoid(x)
  swiglu(g, u) = silu(g) * u

Element-wise and row-independent, so Axis-A batch invariance holds bitwise.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_dtype(x: Tensor, name: str) -> None:
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {x.dtype}.")


def _require_cuda_activation() -> None:
    if not _EXT_AVAILABLE or _C is None:
        raise RuntimeError("CUDA activation kernels require the compiled rl_engine._C extension.")
    if not hasattr(_C, "silu_forward") or not hasattr(_C, "swiglu_forward"):
        raise RuntimeError(
            "CUDA activation symbols (silu_forward / swiglu_forward) are not compiled "
            "into _C. Rebuild the extension with csrc/cuda/activation/activation.cu."
        )


class _SiLUCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        x_c = x.contiguous()
        y = _C.silu_forward(x_c)
        ctx.save_for_backward(x_c)
        return y

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (x,) = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            dx = _C.silu_backward(grad_out.contiguous(), x)
        return dx


class _SwiGLUCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        y = _C.swiglu_forward(gate_c, up_c)
        ctx.save_for_backward(gate_c, up_c)
        return y

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        gate, up = ctx.saved_tensors
        d_gate = d_up = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grads = _C.swiglu_backward(grad_out.contiguous(), gate, up)
            if ctx.needs_input_grad[0]:
                d_gate = grads[0]
            if ctx.needs_input_grad[1]:
                d_up = grads[1]
        return d_gate, d_up


class SiLUCudaOp:
    """CUDA SiLU: ``out = x * sigmoid(x)``, math in fp32."""

    op_class = "elementwise"

    def __init__(self) -> None:
        _require_cuda_activation()
        logger.info("Successfully linked to precompiled _C.silu_forward kernel.")

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        if x.device.type != "cuda":
            raise RuntimeError(f"SiLUCudaOp requires a CUDA tensor, got device '{x.device}'.")
        _validate_dtype(x, "x")
        return _SiLUCudaFunction.apply(x)

    def forward_fp32(self, x: Tensor) -> Tensor:
        """Ground-truth path: force fp32 input so the kernel stores fp32 output."""
        if x.device.type != "cuda":
            raise RuntimeError(f"SiLUCudaOp requires a CUDA tensor, got device '{x.device}'.")
        _validate_dtype(x, "x")
        return _SiLUCudaFunction.apply(x.float())


class SwiGLUCudaOp:
    """CUDA SwiGLU: ``out = silu(gate) * up``, math in fp32."""

    op_class = "elementwise"

    def __init__(self) -> None:
        _require_cuda_activation()
        logger.info("Successfully linked to precompiled _C.swiglu_forward kernel.")

    def __call__(self, gate: Tensor, up: Tensor) -> Tensor:
        return self.forward(gate, up)

    def forward(self, gate: Tensor, up: Tensor) -> Tensor:
        if gate.device.type != "cuda" or up.device.type != "cuda":
            raise RuntimeError(
                f"SwiGLUCudaOp requires CUDA tensors, got gate='{gate.device}', up='{up.device}'."
            )
        if gate.device != up.device:
            raise RuntimeError(
                f"gate and up must be on the same CUDA device, got "
                f"'{gate.device}' and '{up.device}'."
            )
        if gate.shape != up.shape:
            raise ValueError(
                f"gate and up must share shape, got tuple(gate.shape)="
                f"{tuple(gate.shape)} vs tuple(up.shape)={tuple(up.shape)}"
            )
        _validate_dtype(gate, "gate")
        _validate_dtype(up, "up")
        if gate.dtype != up.dtype:
            raise TypeError(f"gate and up must share dtype, got {gate.dtype} and {up.dtype}.")
        return _SwiGLUCudaFunction.apply(gate, up)

    def forward_fp32(self, gate: Tensor, up: Tensor) -> Tensor:
        if gate.device.type != "cuda" or up.device.type != "cuda":
            raise RuntimeError(
                f"SwiGLUCudaOp requires CUDA tensors, got gate='{gate.device}', up='{up.device}'."
            )
        if gate.device != up.device:
            raise RuntimeError(
                f"gate and up must be on the same CUDA device, got "
                f"'{gate.device}' and '{up.device}'."
            )
        if gate.shape != up.shape:
            raise ValueError(
                f"gate and up must share shape, got tuple(gate.shape)="
                f"{tuple(gate.shape)} vs tuple(up.shape)={tuple(up.shape)}"
            )
        _validate_dtype(gate, "gate")
        _validate_dtype(up, "up")
        if gate.dtype != up.dtype:
            raise TypeError(f"gate and up must share dtype, got {gate.dtype} and {up.dtype}.")
        return _SwiGLUCudaFunction.apply(gate.float(), up.float())
