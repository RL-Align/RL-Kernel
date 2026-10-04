# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton MiniMax-H3 rectified-flow Euler step (portable GPU path).

Same contract as ``NativeH3OdeStepOp``; math is fp32 inside the kernels and
rounded once on store. Element-wise, so batch invariance holds bitwise.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_GPU_TYPES = ("cuda", "hip", "xpu", "musa")
_BLOCK = 1024


def _row_count(tensor: Tensor) -> int:
    """Flattened packed-row count: numel divided by the last dim."""
    width = tensor.shape[-1] if tensor.dim() >= 1 else 1
    return tensor.numel() // max(width, 1)


def _as_sigma(value: Any, xt: Tensor, rows: int, name: str) -> tuple[Tensor, bool]:
    """Return ``(sigma, per_row)``; ``per_row`` selects the kernel branch."""
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=torch.float32, device=xt.device)
    if value.device != xt.device:
        raise ValueError(f"{name} must be on xt's device, got {value.device} and {xt.device}.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}.")
    value = value.to(dtype=torch.float32).contiguous()
    if value.numel() == 1:
        return value.reshape(()), False
    if value.numel() == rows:
        return value.reshape(rows), True
    raise ValueError(
        f"{name} must be a scalar or one value per packed row ({rows}), got "
        f"numel={value.numel()}. Refusing to broadcast implicitly."
    )


def _check_sigma_values(sigma: Tensor, sigma_next: Tensor) -> None:
    if not torch.isfinite(sigma).all() or not torch.isfinite(sigma_next).all():
        raise ValueError("sigma and sigma_next must be finite.")
    if bool((sigma <= 0).any()):
        raise ValueError("sigma must be strictly positive; 0 terminates the grid.")
    if bool((sigma_next < 0).any()):
        raise ValueError("sigma_next must be non-negative.")
    if bool((sigma_next > sigma).any()):
        raise ValueError(
            "sigma_next must not exceed sigma; the H3 grid is monotonically decreasing."
        )


@triton.jit
def _ode_step_fwd_kernel(
    xt_ptr,
    v_ptr,
    s_ptr,
    sn_ptr,
    xn_ptr,
    x0_ptr,
    n_elements,
    row_width,
    ROW_SIGMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    xt = tl.load(xt_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if ROW_SIGMA:
        row = offs // row_width
        s = tl.load(s_ptr + row, mask=mask, other=1.0).to(tl.float32)
        sn = tl.load(sn_ptr + row, mask=mask, other=0.0).to(tl.float32)
    else:
        s = tl.load(s_ptr).to(tl.float32)
        sn = tl.load(sn_ptr).to(tl.float32)
    r = tl.div_rn(sn, s)  # correctly-rounded divide is part of the contract
    x0 = xt + s * v
    xn = r * xt + (1.0 - r) * x0  # declared order; do not reassociate
    tl.store(x0_ptr + offs, x0.to(x0_ptr.dtype.element_ty), mask=mask)
    tl.store(xn_ptr + offs, xn.to(xn_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ode_step_bwd_kernel(
    gn_ptr,
    g0_ptr,
    s_ptr,
    sn_ptr,
    gxt_ptr,
    gv_ptr,
    n_elements,
    row_width,
    ROW_SIGMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    gn = tl.load(gn_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g0 = tl.load(g0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if ROW_SIGMA:
        row = offs // row_width
        s = tl.load(s_ptr + row, mask=mask, other=1.0).to(tl.float32)
        sn = tl.load(sn_ptr + row, mask=mask, other=0.0).to(tl.float32)
    else:
        s = tl.load(s_ptr).to(tl.float32)
        sn = tl.load(sn_ptr).to(tl.float32)
    r = tl.div_rn(sn, s)
    # Same expression program as the forward: (1 - r) * gn, not (s - sn) * gn.
    g_x0 = g0 + (1.0 - r) * gn
    g_xt = r * gn + g_x0
    g_v = s * g_x0
    tl.store(gxt_ptr + offs, g_xt.to(gxt_ptr.dtype.element_ty), mask=mask)
    tl.store(gv_ptr + offs, g_v.to(gv_ptr.dtype.element_ty), mask=mask)


def _launch(
    kernel,
    out_a: Tensor,
    out_b: Tensor,
    a: Tensor,
    b: Tensor,
    sigma: Tensor,
    sigma_next: Tensor,
) -> None:
    n = a.numel()
    if n == 0:
        return
    grid = (triton.cdiv(n, _BLOCK),)
    kernel[grid](
        a,
        b,
        sigma,
        sigma_next,
        out_a,
        out_b,
        n,
        a.shape[-1] if a.dim() >= 1 else 1,
        ROW_SIGMA=sigma.dim() > 0,
        BLOCK=_BLOCK,
    )


class _H3OdeStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xt, v, sigma, sigma_next):  # type: ignore[override]
        x_next = torch.empty_like(xt)
        x0 = torch.empty_like(xt)
        _launch(_ode_step_fwd_kernel, x_next, x0, xt, v, sigma, sigma_next)
        ctx.save_for_backward(xt, sigma, sigma_next)
        return x_next, x0

    @staticmethod
    def backward(ctx, grad_next, grad_x0):  # type: ignore[override]
        xt, sigma, sigma_next = ctx.saved_tensors
        needs_xt, needs_v = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (needs_xt or needs_v):
            return None, None, None, None
        if grad_next is None:
            grad_next = torch.zeros_like(xt)
        if grad_x0 is None:
            grad_x0 = torch.zeros_like(xt)
        g_xt = torch.empty_like(xt)
        g_v = torch.empty_like(xt)
        _launch(
            _ode_step_bwd_kernel,
            g_xt,
            g_v,
            grad_next.contiguous(),
            grad_x0.contiguous(),
            sigma,
            sigma_next,
        )
        return (g_xt if needs_xt else None, g_v if needs_v else None, None, None)


class TritonH3OdeStepOp:
    """Triton implementation of the H3 data-ward Euler step."""

    op_class = "elementwise"

    def __call__(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        return self.forward(xt, v, sigma, sigma_next)

    def forward(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _H3OdeStepFunction.apply(xt.contiguous(), v.contiguous(), s, sn)

    def forward_fp32(
        self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any
    ) -> tuple[Tensor, Tensor]:
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _H3OdeStepFunction.apply(xt.float().contiguous(), v.float().contiguous(), s, sn)

    @staticmethod
    def _prepare(xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        for name, tensor in (("xt", xt), ("v", v)):
            if tensor.device.type not in _GPU_TYPES:
                raise RuntimeError(f"{name} must be a GPU tensor, got {tensor.device}.")
            if tensor.dtype not in _SUPPORTED_DTYPES:
                raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {tensor.dtype}.")
        if xt.device != v.device:
            raise ValueError(f"xt and v must share a device, got {xt.device} and {v.device}.")
        if xt.shape != v.shape:
            raise ValueError(
                f"xt and v must share a shape, got {tuple(xt.shape)} and {tuple(v.shape)}."
            )
        rows = _row_count(xt)
        s, _ = _as_sigma(sigma, xt, rows, "sigma")
        sn, _ = _as_sigma(sigma_next, xt, rows, "sigma_next")
        _check_sigma_values(s, sn)
        return s, sn
