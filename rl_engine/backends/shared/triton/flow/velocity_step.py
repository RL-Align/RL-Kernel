# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton MiniMax-H3-Image velocity-form Euler step (portable GPU path, CUDA and ROCm).

Same contract as :class:`~rl_engine.reference.flow.velocity_step.NativeH3ImageVelocityStepOp`:
``x_next = xt + (sigma - sigma_next) * v`` in FP32, rounded once on store. The kernel is
launched with FP fusion disabled, so the multiply and the add round separately, as the
reference (and the pinned Diffusers scheduler) evaluate them. Element-wise, so batch
invariance holds bitwise. Backward: ``d xt = g``, ``d v = (sigma - sigma_next) * g``.
The launch and sigma checks are the H3 ``h3_ode_step`` Triton ones.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor

from rl_engine.backends.shared.triton.flow.ode_step import (
    _BLOCK,
    _GPU_TYPES,
    _SUPPORTED_DTYPES,
    _as_sigma,
    _check_sigma_values,
    _row_count,
)


@triton.jit
def _velocity_step_fwd_kernel(
    xt_ptr,
    v_ptr,
    s_ptr,
    sn_ptr,
    out_ptr,
    n,
    width,
    ROW_SIGMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xt = tl.load(xt_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if ROW_SIGMA:
        row = offs // width
        s = tl.load(s_ptr + row, mask=mask, other=1.0).to(tl.float32)
        sn = tl.load(sn_ptr + row, mask=mask, other=0.0).to(tl.float32)
    else:
        s = tl.load(s_ptr).to(tl.float32)
        sn = tl.load(sn_ptr).to(tl.float32)
    d = s - sn
    out = xt + d * v
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _velocity_step_bwd_kernel(
    g_ptr,
    s_ptr,
    sn_ptr,
    gv_ptr,
    n,
    width,
    ROW_SIGMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if ROW_SIGMA:
        row = offs // width
        s = tl.load(s_ptr + row, mask=mask, other=1.0).to(tl.float32)
        sn = tl.load(sn_ptr + row, mask=mask, other=0.0).to(tl.float32)
    else:
        s = tl.load(s_ptr).to(tl.float32)
        sn = tl.load(sn_ptr).to(tl.float32)
    tl.store(gv_ptr + offs, ((s - sn) * g).to(gv_ptr.dtype.element_ty), mask=mask)


def _launch(kernel, *tensors: Tensor, n: int, width: int, row_sigma: bool) -> None:
    """Launch one kernel over the flattened tensor with FP fusion disabled."""
    if n == 0:
        return
    grid = (triton.cdiv(n, _BLOCK),)
    kernel[grid](*tensors, n, width, ROW_SIGMA=row_sigma, BLOCK=_BLOCK, enable_fp_fusion=False)


class _VelocityStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xt, v, sigma, sigma_next):  # type: ignore[override]
        out = torch.empty_like(xt)
        width = xt.shape[-1] if xt.dim() >= 1 else 1
        row_sigma = sigma.dim() > 0
        _launch(
            _velocity_step_fwd_kernel,
            xt,
            v,
            sigma,
            sigma_next,
            out,
            n=xt.numel(),
            width=width,
            row_sigma=row_sigma,
        )
        ctx.save_for_backward(sigma, sigma_next)
        ctx.width, ctx.row_sigma = width, row_sigma
        return out

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        sigma, sigma_next = ctx.saved_tensors
        needs_xt, needs_v = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        g_v = None
        if needs_v:
            grad = grad.contiguous()
            g_v = torch.empty_like(grad)
            _launch(
                _velocity_step_bwd_kernel,
                grad,
                sigma,
                sigma_next,
                g_v,
                n=grad.numel(),
                width=ctx.width,
                row_sigma=ctx.row_sigma,
            )
        return (grad if needs_xt else None, g_v, None, None)


class H3ImageVelocityStepTritonOp:
    """Triton implementation of the H3-Image velocity-form Euler step."""

    op_class = "elementwise"

    def __call__(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> Tensor:
        return self.forward(xt, v, sigma, sigma_next)

    def forward(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> Tensor:
        """``x_next`` in ``xt``'s dtype; ``v`` is read in its own dtype and widened to FP32."""
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _VelocityStepFunction.apply(xt.contiguous(), v.contiguous(), s, sn)

    def forward_fp32(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> Tensor:
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _VelocityStepFunction.apply(xt.float().contiguous(), v.float().contiguous(), s, sn)

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
        s, s_per_row = _as_sigma(sigma, xt, rows, "sigma")
        sn, sn_per_row = _as_sigma(sigma_next, xt, rows, "sigma_next")
        if s_per_row != sn_per_row:
            raise ValueError(
                "sigma and sigma_next must use the same layout: both scalar or both "
                "one value per packed row."
            )
        _check_sigma_values(s, sn)
        return s, sn
