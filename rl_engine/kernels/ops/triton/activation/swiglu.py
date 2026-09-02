# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton SiLU / SwiGLU ops matching NativeSiLUOp / NativeSwiGLUOp (WS1 ground truth).

Math is performed in fp32 inside the Triton kernels and rounded back to the input
dtype on store — the same dual-path contract as the PyTorch references:

  silu(x)      = x * sigmoid(x)
  swiglu(g, u) = silu(g) * u

Element-wise and row-independent, so Axis-A batch invariance holds bitwise.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_dtype(x: Tensor, name: str) -> None:
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {x.dtype}.")


@triton.jit
def _silu_fwd_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _silu_bwd_kernel(dy_ptr, x_ptr, dx_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-x))
    # silu'(x) = s * (1 + x * (1 - s))
    dx = dy * s * (1.0 + x * (1.0 - s))
    tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_fwd_kernel(gate_ptr, up_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g))
    y = (g * s) * u
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    dy_ptr, gate_ptr, up_ptr, d_gate_ptr, d_up_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g))
    silu_g = g * s
    d_up = dy * silu_g
    d_gate = dy * u * s * (1.0 + g * (1.0 - s))
    tl.store(d_up_ptr + offs, d_up.to(d_up_ptr.dtype.element_ty), mask=mask)
    tl.store(d_gate_ptr + offs, d_gate.to(d_gate_ptr.dtype.element_ty), mask=mask)


_BLOCK = 1024


def _launch_silu_fwd(x: Tensor) -> Tensor:
    x_c = x.contiguous()
    y = torch.empty_like(x_c)
    n = x_c.numel()
    if n == 0:
        return y
    grid = (triton.cdiv(n, _BLOCK),)
    _silu_fwd_kernel[grid](x_c, y, n, BLOCK=_BLOCK)
    return y


def _launch_silu_bwd(dy: Tensor, x: Tensor) -> Tensor:
    dy_c = dy.contiguous()
    x_c = x.contiguous()
    dx = torch.empty_like(x_c)
    n = x_c.numel()
    if n == 0:
        return dx
    grid = (triton.cdiv(n, _BLOCK),)
    _silu_bwd_kernel[grid](dy_c, x_c, dx, n, BLOCK=_BLOCK)
    return dx


def _launch_swiglu_fwd(gate: Tensor, up: Tensor) -> Tensor:
    gate_c = gate.contiguous()
    up_c = up.contiguous()
    y = torch.empty_like(gate_c)
    n = gate_c.numel()
    if n == 0:
        return y
    grid = (triton.cdiv(n, _BLOCK),)
    _swiglu_fwd_kernel[grid](gate_c, up_c, y, n, BLOCK=_BLOCK)
    return y


def _launch_swiglu_bwd(dy: Tensor, gate: Tensor, up: Tensor) -> tuple[Tensor, Tensor]:
    dy_c = dy.contiguous()
    gate_c = gate.contiguous()
    up_c = up.contiguous()
    d_gate = torch.empty_like(gate_c)
    d_up = torch.empty_like(up_c)
    n = gate_c.numel()
    if n == 0:
        return d_gate, d_up
    grid = (triton.cdiv(n, _BLOCK),)
    _swiglu_bwd_kernel[grid](dy_c, gate_c, up_c, d_gate, d_up, n, BLOCK=_BLOCK)
    return d_gate, d_up


class _SiLUTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        x_c = x.contiguous()
        y = _launch_silu_fwd(x_c)
        ctx.save_for_backward(x_c)
        return y

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        (x,) = ctx.saved_tensors
        dx = None
        if ctx.needs_input_grad[0]:
            dx = _launch_silu_bwd(grad_out, x)
        return dx


class _SwiGLUTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor) -> Tensor:
        gate_c = gate.contiguous()
        up_c = up.contiguous()
        y = _launch_swiglu_fwd(gate_c, up_c)
        ctx.save_for_backward(gate_c, up_c)
        return y

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        gate, up = ctx.saved_tensors
        d_gate = d_up = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            dg, du = _launch_swiglu_bwd(grad_out, gate, up)
            if ctx.needs_input_grad[0]:
                d_gate = dg
            if ctx.needs_input_grad[1]:
                d_up = du
        return d_gate, d_up


class TritonSiLUOp:
    """Triton SiLU: ``out = x * sigmoid(x)``, math in fp32."""

    op_class = "elementwise"

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        if x.device.type not in ("cuda", "hip", "xpu", "musa"):
            raise RuntimeError(f"TritonSiLUOp requires a GPU tensor, got device '{x.device}'.")
        _validate_dtype(x, "x")
        return _SiLUTritonFunction.apply(x)

    def forward_fp32(self, x: Tensor) -> Tensor:
        if x.device.type not in ("cuda", "hip", "xpu", "musa"):
            raise RuntimeError(f"TritonSiLUOp requires a GPU tensor, got device '{x.device}'.")
        _validate_dtype(x, "x")
        return _SiLUTritonFunction.apply(x.float())


class TritonSwiGLUOp:
    """Triton SwiGLU: ``out = silu(gate) * up``, math in fp32."""

    op_class = "elementwise"

    def __call__(self, gate: Tensor, up: Tensor) -> Tensor:
        return self.forward(gate, up)

    def forward(self, gate: Tensor, up: Tensor) -> Tensor:
        if gate.device.type not in ("cuda", "hip", "xpu", "musa") or up.device.type not in (
            "cuda",
            "hip",
            "xpu",
            "musa",
        ):
            raise RuntimeError(
                f"TritonSwiGLUOp requires GPU tensors, got gate='{gate.device}', up='{up.device}'."
            )
        if gate.device != up.device:
            raise RuntimeError(
                f"gate and up must be on the same GPU device, got "
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
        return _SwiGLUTritonFunction.apply(gate, up)

    def forward_fp32(self, gate: Tensor, up: Tensor) -> Tensor:
        if gate.device.type not in ("cuda", "hip", "xpu", "musa") or up.device.type not in (
            "cuda",
            "hip",
            "xpu",
            "musa",
        ):
            raise RuntimeError(
                f"TritonSwiGLUOp requires GPU tensors, got gate='{gate.device}', up='{up.device}'."
            )
        if gate.device != up.device:
            raise RuntimeError(
                f"gate and up must be on the same GPU device, got "
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
        return _SwiGLUTritonFunction.apply(gate.float(), up.float())
