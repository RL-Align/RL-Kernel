# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton kernels and autograd for ``c * tanh(x / c)`` with FP32 arithmetic.

``c`` is the softcap, passed at runtime. The caller supplies contiguous
FP16/BF16/FP32 input and a contiguous FP32 forward output with the same shape.
Backward also requires contiguous gradient buffers and casts its FP32 result
to the destination buffer's dtype. The launchers prepare these buffers.
The autograd wrapper saves the input and the softcap for backward
recomputation.
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_SUPPORTED_DEVICES = ("cuda", "hip", "xpu", "musa")

DEFAULT_SOFTCAP = 30.0


@triton.jit
def _final_logit_softcap_fwd_kernel(x_ptr, y_ptr, n_elements, softcap, BLOCK: tl.constexpr):
    # Widen before multiplying so element offsets can cross the int32 boundary.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scaled_x = tl.div_rn(x, softcap)
    # Triton maps libdevice to the active CUDA or HIP backend.
    t = libdevice.tanh(scaled_x)
    y = softcap * t

    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _final_logit_softcap_bwd_kernel(
    dy_ptr, x_ptr, dx_ptr, n_elements, softcap, BLOCK: tl.constexpr
):
    # Widen before multiplying, as in the forward kernel.
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    grad_y = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    scaled_x = tl.div_rn(x, softcap)
    t = libdevice.tanh(scaled_x)
    # f(x) = c*tanh(x/c) => f'(x) = 1 - tanh(x/c)^2. The c in the outer
    # factor and the 1/c from the chain rule cancel, so no softcap appears.
    grad_x = grad_y * (1.0 - t * t)

    tl.store(dx_ptr + offs, grad_x.to(dx_ptr.dtype.element_ty), mask=mask)


_BLOCK = 1024


def _launch_final_logit_softcap_fwd(x: Tensor, softcap: float) -> Tensor:
    n = x.numel()
    x_c = x.contiguous()
    y = torch.empty_like(x_c, dtype=torch.float32)
    if n == 0:
        return y

    grid = (triton.cdiv(n, _BLOCK),)
    _final_logit_softcap_fwd_kernel[grid](x_c, y, n, softcap, BLOCK=_BLOCK)
    return y


def _launch_final_logit_softcap_bwd(dy: Tensor, x: Tensor, softcap: float) -> Tensor:
    # Both pointers are indexed by x.numel(), so a shorter upstream gradient
    # would be read past its end with the mask still set. Autograd rejects a
    # mismatched grad_output before this point, but direct callers bypass that.
    if dy.shape != x.shape:
        raise ValueError(
            f"dy must match x, got tuple(dy.shape)={tuple(dy.shape)} "
            f"vs tuple(x.shape)={tuple(x.shape)}."
        )
    n = x.numel()
    dy_c = dy.contiguous()
    x_c = x.contiguous()
    dx = torch.empty_like(x_c)
    if n == 0:
        return dx

    grid = (triton.cdiv(n, _BLOCK),)
    _final_logit_softcap_bwd_kernel[grid](dy_c, x_c, dx, n, softcap, BLOCK=_BLOCK)
    return dx


class _FinalLogitSoftcapTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, softcap: float) -> Tensor:
        x_c = x.contiguous()
        y = _launch_final_logit_softcap_fwd(x_c, softcap)
        ctx.save_for_backward(x_c)
        ctx.softcap = softcap
        return y

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (x,) = ctx.saved_tensors
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = _launch_final_logit_softcap_bwd(grad_out, x, ctx.softcap)
        return grad_x, None


class TritonFinalLogitSoftcapOp:
    """Element-wise ``c * tanh(x / c)`` with FP32 math and output."""

    op_class = "elementwise"

    def __init__(self, softcap: float = DEFAULT_SOFTCAP) -> None:
        softcap = float(softcap)
        if not math.isfinite(softcap) or softcap <= 0.0:
            raise ValueError(f"softcap must be finite and positive, got {softcap}.")
        self.softcap = softcap

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        if x.device.type not in _SUPPORTED_DEVICES:
            raise RuntimeError(
                f"TritonFinalLogitSoftcapOp requires a GPU tensor, supported devices are "
                f"{_SUPPORTED_DEVICES}, got device '{x.device}'."
            )
        if x.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"x must have dtype {_SUPPORTED_DTYPES}, got {x.dtype}.")
        return _FinalLogitSoftcapTritonFunction.apply(x, self.softcap)
