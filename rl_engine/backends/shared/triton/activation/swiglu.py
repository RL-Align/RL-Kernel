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

from typing import overload

import torch
import triton
import triton.language as tl
from torch import Tensor

from rl_engine.contracts.operators.activation import SwiGLUPackedOrder

_SUPPORTED_DEVICES = ("cuda", "hip", "xpu", "musa")
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_dtype(x: Tensor, name: str) -> None:
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype in {_SUPPORTED_DTYPES}, got {x.dtype}.")


def _validate_swiglu_separate_inputs(gate: Tensor, up: Tensor) -> None:
    """Validate two operands with matching shape, dtype and supported GPU device."""
    if gate.device.type not in _SUPPORTED_DEVICES or up.device.type not in _SUPPORTED_DEVICES:
        raise RuntimeError(
            f"TritonSwiGLUOp requires GPU tensors, got gate='{gate.device}', up='{up.device}'."
        )
    if gate.device != up.device:
        raise RuntimeError(
            f"gate and up must be on the same GPU device, got '{gate.device}' and '{up.device}'."
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


def _validate_swiglu_packed_input(
    projected: Tensor, packed_order: SwiGLUPackedOrder | None
) -> None:
    """Validate one packed projection and the explicit order of its equal halves.

    The last dimension must be positive and even; leading dimensions may be
    empty. The dtype and GPU device must be supported by the packed kernels.
    """
    if not isinstance(packed_order, SwiGLUPackedOrder):
        raise ValueError(f"packed_order must be one of {tuple(SwiGLUPackedOrder)}")
    if projected.ndim == 0:
        raise ValueError("packed input must have at least one dimension")
    width = projected.shape[-1]
    if width == 0 or width % 2:
        raise ValueError(f"packed input must have a positive, even last dimension, got {width}")
    _validate_dtype(projected, "packed input")
    if projected.device.type not in _SUPPORTED_DEVICES:
        raise RuntimeError(
            f"packed input must use a device in {_SUPPORTED_DEVICES}, got {projected.device}"
        )


# Every kernel widens pid to int64 before scaling by BLOCK: in int32 the flat
# offset wraps negative past 2**31 elements(#640).
@triton.jit
def _silu_fwd_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.div_rn(1.0, 1.0 + tl.exp(-x))
    y = x * s
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _silu_bwd_kernel(dy_ptr, x_ptr, dx_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.div_rn(1.0, 1.0 + tl.exp(-x))
    # silu'(x) = s * (1 + x * (1 - s))
    dx = dy * s * (1.0 + x * (1.0 - s))
    tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_fwd_kernel(gate_ptr, up_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.div_rn(1.0, 1.0 + tl.exp(-g))
    y = (g * s) * u
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(dy_ptr, gate_ptr, silu_grad_ptr, d_up_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.div_rn(1.0, 1.0 + tl.exp(-g))
    silu_g = g * s
    d_up = dy * silu_g
    silu_grad = s * (1.0 + g * (1.0 - s))
    tl.store(d_up_ptr + offs, d_up.to(d_up_ptr.dtype.element_ty), mask=mask)
    tl.store(silu_grad_ptr + offs, silu_grad, mask=mask)


@triton.jit
def _swiglu_dgate_kernel(
    dy_ptr, up_ptr, silu_grad_ptr, d_gate_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    silu_grad = tl.load(silu_grad_ptr + offs, mask=mask, other=0.0)
    # The native HIP source evaluates (dy * up) * silu_grad.  Persisting the
    # latter as FP32 prevents LLVM from reassociating the expression across
    # the function boundary and changing a BF16 tie at rare elements.
    d_gate = (dy * u) * silu_grad
    tl.store(d_gate_ptr + offs, d_gate.to(d_gate_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_packed_fwd_kernel(
    projected_ptr, y_ptr, WIDTH: tl.constexpr, GATE_FIRST: tl.constexpr, BLOCK: tl.constexpr
):
    """Read both halves without copying the noncontiguous chunk views."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < WIDTH
    offsets = row * WIDTH + cols
    packed_offsets = row * (2 * WIDTH) + cols
    gate_offsets = packed_offsets + (0 if GATE_FIRST else WIDTH)
    up_offsets = packed_offsets + (WIDTH if GATE_FIRST else 0)
    g = tl.load(projected_ptr + gate_offsets, mask, other=0).to(tl.float32)
    u = tl.load(projected_ptr + up_offsets, mask, other=0).to(tl.float32)
    s = tl.div_rn(1.0, 1.0 + tl.exp(-g))
    y = (g * s) * u
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask)


@triton.jit
def _swiglu_packed_bwd_up_kernel(
    dy_ptr,
    projected_ptr,
    silu_grad_ptr,
    dx_ptr,
    WIDTH: tl.constexpr,
    GATE_FIRST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write the up gradient directly into the packed output and save FP32 SiLU'."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < WIDTH
    offsets = row * WIDTH + cols
    packed_offsets = row * (2 * WIDTH) + cols
    gate_offsets = packed_offsets + (0 if GATE_FIRST else WIDTH)
    up_offsets = packed_offsets + (WIDTH if GATE_FIRST else 0)
    dy = tl.load(dy_ptr + offsets, mask, other=0).to(tl.float32)
    g = tl.load(projected_ptr + gate_offsets, mask, other=0).to(tl.float32)
    s = tl.div_rn(1.0, 1.0 + tl.exp(-g))
    silu_g = g * s
    d_up = dy * silu_g
    silu_grad = s * (1.0 + g * (1.0 - s))
    tl.store(dx_ptr + up_offsets, d_up.to(dx_ptr.dtype.element_ty), mask)
    tl.store(silu_grad_ptr + offsets, silu_grad, mask)


@triton.jit
def _swiglu_packed_bwd_gate_kernel(
    dy_ptr,
    projected_ptr,
    silu_grad_ptr,
    dx_ptr,
    WIDTH: tl.constexpr,
    GATE_FIRST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Finish the gate gradient while preserving the shared SwiGLU rounding order."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < WIDTH
    offsets = row * WIDTH + cols
    packed_offsets = row * (2 * WIDTH) + cols
    gate_offsets = packed_offsets + (0 if GATE_FIRST else WIDTH)
    up_offsets = packed_offsets + (WIDTH if GATE_FIRST else 0)
    dy = tl.load(dy_ptr + offsets, mask, other=0).to(tl.float32)
    u = tl.load(projected_ptr + up_offsets, mask, other=0).to(tl.float32)
    silu_grad = tl.load(silu_grad_ptr + offsets, mask, other=0)
    # Keep the same FP32 workspace boundary as the shared SwiGLU. Fusing the
    # stages could let the compiler reassociate this expression at BF16 ties.
    d_gate = (dy * u) * silu_grad
    tl.store(dx_ptr + gate_offsets, d_gate.to(dx_ptr.dtype.element_ty), mask)


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
    silu_grad = torch.empty_like(gate_c, dtype=torch.float32)
    n = gate_c.numel()
    if n == 0:
        return d_gate, d_up
    grid = (triton.cdiv(n, _BLOCK),)
    _swiglu_bwd_kernel[grid](
        dy_c,
        gate_c,
        silu_grad,
        d_up,
        n,
        BLOCK=_BLOCK,
    )
    _swiglu_dgate_kernel[grid](
        dy_c,
        up_c,
        silu_grad,
        d_gate,
        n,
        BLOCK=_BLOCK,
    )
    return d_gate, d_up


def _launch_swiglu_packed_fwd(
    projected: Tensor, output_dtype: torch.dtype, packed_order: SwiGLUPackedOrder
) -> Tensor:
    """Launch one program per row/tile, storing only the final output cast.

    Flatten leading dimensions into rows. Rows use grid axis 0 so large batches
    do not encounter CUDA's smaller grid-axis-1 limit.
    """
    width = projected.shape[-1] // 2
    y = torch.empty((*projected.shape[:-1], width), device=projected.device, dtype=output_dtype)
    if y.numel() == 0:
        return y
    n_rows = y.numel() // width
    device_module = torch.get_device_module(projected.device)
    with device_module.device(projected.device):
        _swiglu_packed_fwd_kernel[(n_rows, triton.cdiv(width, _BLOCK))](
            projected,
            y,
            WIDTH=width,
            GATE_FIRST=packed_order is SwiGLUPackedOrder.GATE_UP,
            BLOCK=_BLOCK,
        )
    return y


def _launch_swiglu_packed_bwd(
    grad_y: Tensor, projected: Tensor, packed_order: SwiGLUPackedOrder
) -> Tensor:
    """Write one packed gradient; both stages share the input device's current stream."""
    grad_projected = torch.empty_like(projected)
    if grad_y.numel() == 0:
        return grad_projected
    grad_y_c = grad_y.contiguous()
    silu_grad = torch.empty_like(grad_y_c, dtype=torch.float32)
    width = projected.shape[-1] // 2
    n_rows = grad_y_c.numel() // width
    grid = (n_rows, triton.cdiv(width, _BLOCK))
    device_module = torch.get_device_module(projected.device)
    with device_module.device(projected.device):
        _swiglu_packed_bwd_up_kernel[grid](
            grad_y_c,
            projected,
            silu_grad,
            grad_projected,
            WIDTH=width,
            GATE_FIRST=packed_order is SwiGLUPackedOrder.GATE_UP,
            BLOCK=_BLOCK,
        )
        _swiglu_packed_bwd_gate_kernel[grid](
            grad_y_c,
            projected,
            silu_grad,
            grad_projected,
            WIDTH=width,
            GATE_FIRST=packed_order is SwiGLUPackedOrder.GATE_UP,
            BLOCK=_BLOCK,
        )
    return grad_projected


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


class _PackedSwiGLUTritonFunction(torch.autograd.Function):
    """Save the packed projection, avoiding independent gate/up copies."""

    @staticmethod
    def forward(
        ctx, projected: Tensor, output_dtype: torch.dtype, packed_order: SwiGLUPackedOrder
    ) -> Tensor:
        projected_c = projected.contiguous()
        y = _launch_swiglu_packed_fwd(projected_c, output_dtype, packed_order)
        ctx.save_for_backward(projected_c)
        ctx.packed_order = packed_order
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        (projected,) = ctx.saved_tensors
        return _launch_swiglu_packed_bwd(grad_y, projected, ctx.packed_order), None, None


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
    """FP32-math SwiGLU for separate operands or one packed projection.

    ``op(gate, up)`` keeps the existing two-tensor interface. For one tensor,
    call ``op(projected, packed_order=SwiGLUPackedOrder.GATE_UP)`` or use
    ``UP_GATE``. The last dimension contains two equal halves; the output is
    half as wide. No broadcasting is performed.

    A contiguous packed input is read directly, without gate/up copies.
    Strided packed inputs are made contiguous once. Backward writes both
    gradient halves into one tensor, preserving the FP32 SiLU-derivative
    workspace and expression order of the separate-input implementation.

    The overloads describe the two accepted input forms for IDEs and type
    checkers. Each method has one runtime implementation below them.
    """

    op_class = "elementwise"

    @overload
    def __call__(self, gate: Tensor, up: Tensor) -> Tensor:
        """Call with separate gate/up tensors."""
        ...

    @overload
    def __call__(self, gate: Tensor, *, packed_order: SwiGLUPackedOrder) -> Tensor:
        """Call with one packed tensor in ``gate`` and an explicit half order."""
        ...

    def __call__(self, *args, **kwargs) -> Tensor:
        """Forward either call form to ``forward``, including its input validation."""
        return self.forward(*args, **kwargs)

    @overload
    def forward(self, gate: Tensor, up: Tensor) -> Tensor:
        """Separate gate/up tensors; return output in the input dtype."""
        ...

    @overload
    def forward(self, gate: Tensor, *, packed_order: SwiGLUPackedOrder) -> Tensor:
        """One packed tensor in ``gate``; its half order must be explicit."""
        ...

    def forward(
        self,
        gate: Tensor,
        up: Tensor | None = None,
        *,
        packed_order: SwiGLUPackedOrder | None = None,
    ) -> Tensor:
        """Compute in FP32, then cast the output to the input dtype.

        ``gate`` is the packed tensor when ``up`` is omitted. Its order must
        be specified; a packed order is rejected when both inputs are given.
        """
        if up is None:
            _validate_swiglu_packed_input(gate, packed_order)
            return _PackedSwiGLUTritonFunction.apply(gate, gate.dtype, packed_order)

        if packed_order is not None:
            raise ValueError("packed_order is only valid with a single packed input")
        _validate_swiglu_separate_inputs(gate, up)
        return _SwiGLUTritonFunction.apply(gate, up)

    @overload
    def forward_fp32(self, gate: Tensor, up: Tensor) -> Tensor:
        """Separate gate/up tensors; return FP32 output."""
        ...

    @overload
    def forward_fp32(self, gate: Tensor, *, packed_order: SwiGLUPackedOrder) -> Tensor:
        """One packed tensor in ``gate``; return FP32 output."""
        ...

    def forward_fp32(
        self,
        gate: Tensor,
        up: Tensor | None = None,
        *,
        packed_order: SwiGLUPackedOrder | None = None,
    ) -> Tensor:
        """Return FP32 output; input gradients retain the original input dtype.

        The packed path widens values in the kernel, avoiding FP32 copies of
        both halves. The separate-input path retains its existing behavior.
        """
        if up is None:
            _validate_swiglu_packed_input(gate, packed_order)
            return _PackedSwiGLUTritonFunction.apply(gate, torch.float32, packed_order)

        if packed_order is not None:
            raise ValueError("packed_order is only valid with a single packed input")
        _validate_swiglu_separate_inputs(gate, up)
        return _SwiGLUTritonFunction.apply(gate.float(), up.float())
