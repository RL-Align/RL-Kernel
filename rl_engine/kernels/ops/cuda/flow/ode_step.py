# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA MiniMax-H3 rectified-flow Euler step.

Same contract as ``NativeH3OdeStepOp``. The kernel rounds once per operation
(see ``csrc/cuda/flow/ode_step.cu``), so the FP32 result matches the PyTorch
reference exactly; dtype-cast results match within the shared tolerance.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.utils.logger import logger

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _row_count(tensor: Tensor) -> int:
    """Flattened packed-row count: numel divided by the last dim."""
    width = tensor.shape[-1] if tensor.dim() >= 1 else 1
    return tensor.numel() // max(width, 1)


def _as_sigma(value: Any, xt: Tensor, rows: int, name: str) -> Tensor:
    """Return contiguous fp32 sigma as a 0-dim scalar or one value per row."""
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=torch.float32, device=xt.device)
    if value.device != xt.device:
        raise ValueError(f"{name} must be on xt's device, got {value.device} and {xt.device}.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}.")
    value = value.to(dtype=torch.float32).contiguous()
    if value.numel() == 1:
        return value.reshape(())
    if value.numel() == rows:
        return value.reshape(rows)
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


def _require_kernel() -> None:
    if not _EXT_AVAILABLE or _C is None:
        raise RuntimeError("CUDA Euler-step kernels require the compiled rl_engine._C extension.")
    if not hasattr(_C, "ode_step_forward") or not hasattr(_C, "ode_step_backward"):
        raise RuntimeError(
            "ode_step_forward / ode_step_backward are missing from _C. "
            "Rebuild with csrc/cuda/flow/ode_step.cu."
        )


class _H3OdeStepCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xt, v, sigma, sigma_next):  # type: ignore[override]
        xt = xt.contiguous()
        v = v.contiguous()
        out = _C.ode_step_forward(xt, v, sigma, sigma_next)
        ctx.save_for_backward(xt, sigma, sigma_next)
        return out[0], out[1]

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
        grad_xt, grad_v = _C.ode_step_backward(
            grad_next.contiguous(), grad_x0.contiguous(), xt, sigma, sigma_next
        )
        return (grad_xt if needs_xt else None, grad_v if needs_v else None, None, None)


class CudaH3OdeStepOp:
    """CUDA implementation of the H3 data-ward Euler step."""

    op_class = "elementwise"

    def __init__(self) -> None:
        _require_kernel()
        logger.info("Successfully linked to precompiled _C.ode_step_forward kernel.")

    def __call__(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        return self.forward(xt, v, sigma, sigma_next)

    def forward(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _H3OdeStepCudaFunction.apply(xt.contiguous(), v.contiguous(), s, sn)

    def forward_fp32(
        self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any
    ) -> tuple[Tensor, Tensor]:
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _H3OdeStepCudaFunction.apply(xt.float().contiguous(), v.float().contiguous(), s, sn)

    @staticmethod
    def _prepare(xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        for name, tensor in (("xt", xt), ("v", v)):
            if tensor.device.type != "cuda":
                raise RuntimeError(f"{name} must be a CUDA tensor, got {tensor.device}.")
            if tensor.dtype not in _SUPPORTED_DTYPES:
                raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {tensor.dtype}.")
        if xt.device != v.device:
            raise ValueError(f"xt and v must share a device, got {xt.device} and {v.device}.")
        if xt.shape != v.shape:
            raise ValueError(
                f"xt and v must share a shape, got {tuple(xt.shape)} and {tuple(v.shape)}."
            )
        rows = _row_count(xt)
        s = _as_sigma(sigma, xt, rows, "sigma")
        sn = _as_sigma(sigma_next, xt, rows, "sigma_next")
        _check_sigma_values(s, sn)
        return s, sn
