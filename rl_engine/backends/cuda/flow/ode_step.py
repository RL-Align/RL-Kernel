# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA MiniMax-H3 rectified-flow Euler step (strict H3 DSO).

Same contract as ``NativeH3OdeStepOp``. The kernel rounds once per operation
(see ``csrc/cuda/flow/ode_step.cu``), so the FP32 result matches the PyTorch
reference exactly; dtype-cast results match within the shared tolerance.

The symbols live in ``rl_engine._H3_C``, which is only built when
``RL_KERNEL_H3_STRICT=1``. There is no silent fallback: callers that need a
portable path use the shared Triton implementation instead.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import torch
from torch import Tensor

from rl_engine.utils.logger import logger

_C: ModuleType | None
try:
    _C = importlib.import_module("rl_engine._H3_C")
except ImportError:
    _C = None

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_REQUIRED = ("ode_step_forward", "ode_step_backward")


def _row_count(tensor: Tensor) -> int:
    """Flattened packed-row count: numel divided by the last dim."""
    width = tensor.shape[-1] if tensor.dim() >= 1 else 1
    return tensor.numel() // max(width, 1)


def _extension() -> ModuleType:
    """Return the strict H3 DSO or raise: there is no silent fallback."""
    if _C is None or any(not hasattr(_C, name) for name in _REQUIRED):
        raise RuntimeError(
            "H3 CUDA extension unavailable; rebuild RL-Kernel with RL_KERNEL_H3_STRICT=1 "
            "(no fallback)"
        )
    return _C


def _as_sigma(value: Any, xt: Tensor, rows: int, name: str) -> Tensor:
    """Return contiguous fp32 sigma as a 0-dim scalar or one value per row.

    Schedule tensors are detached: they are not differentiable inputs, and the
    kernel exposes gradients only for ``xt`` and ``v``.
    """
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=torch.float32, device=xt.device)
    if value.device != xt.device:
        raise ValueError(f"{name} must be on xt's device, got {value.device} and {xt.device}.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}.")
    value = value.detach().to(dtype=torch.float32).contiguous()
    if value.numel() == 1:
        return value.reshape(())
    if value.numel() == rows:
        return value.reshape(rows)
    raise ValueError(
        f"{name} must be a scalar or one value per packed row ({rows}), got "
        f"numel={value.numel()}. Refusing to broadcast implicitly."
    )


def _check_sigma_values(sigma: Tensor, sigma_next: Tensor) -> None:
    """Reject sigma geometry the pinned H3 shifted grid can never produce.

    The predicates are folded on device and read once, so a per-step call costs a
    single host sync instead of one per condition (the operator runs inside the
    per-step flow loop).
    """
    ok = (
        torch.isfinite(sigma)
        & torch.isfinite(sigma_next)
        & (sigma > 0)
        & (sigma_next >= 0)
        & (sigma_next <= sigma)
    ).all()
    if not bool(ok):
        raise ValueError(
            "sigma and sigma_next must be finite, with sigma > 0 and "
            "0 <= sigma_next <= sigma; the H3 grid is monotonically decreasing."
        )


class _H3OdeStepCudaFunction(torch.autograd.Function):
    """Autograd bridge around the two ``rl_engine._H3_C`` symbols."""

    @staticmethod
    def forward(ctx, xt, v, sigma, sigma_next):  # type: ignore[override]
        """Launch the native forward and save ``xt`` and the sigma pair."""
        xt = xt.contiguous()
        v = v.contiguous()
        out = _extension().ode_step_forward(xt, v, sigma, sigma_next)
        ctx.save_for_backward(xt, sigma, sigma_next)
        return out[0], out[1]

    @staticmethod
    def backward(ctx, grad_next, grad_x0):  # type: ignore[override]
        """Run the native backward; only ``xt`` and ``v`` receive gradients."""
        xt, sigma, sigma_next = ctx.saved_tensors
        needs_xt, needs_v = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (needs_xt or needs_v):
            return None, None, None, None
        if grad_next is None:
            grad_next = torch.zeros_like(xt)
        if grad_x0 is None:
            grad_x0 = torch.zeros_like(xt)
        grad_xt, grad_v = _extension().ode_step_backward(
            grad_next.contiguous(), grad_x0.contiguous(), xt, sigma, sigma_next
        )
        return (grad_xt if needs_xt else None, grad_v if needs_v else None, None, None)


class H3OdeStepCUDAOp:
    """CUDA implementation of the H3 data-ward Euler step."""

    op_class = "elementwise"

    def __init__(self) -> None:
        """Fail at construction when the strict H3 extension is missing."""
        _extension()
        logger.info("Successfully linked to precompiled _H3_C.ode_step_forward kernel.")

    def __call__(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        """Alias for :meth:`forward` so the op can be invoked like a function."""
        return self.forward(xt, v, sigma, sigma_next)

    def forward(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        """Return ``(x_next, x0)`` in the input dtype via the native kernel."""
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _H3OdeStepCudaFunction.apply(xt.contiguous(), v.contiguous(), s, sn)

    def forward_fp32(
        self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any
    ) -> tuple[Tensor, Tensor]:
        """Gold-path entry point: dispatch the native kernel on FP32 copies."""
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _H3OdeStepCudaFunction.apply(xt.float().contiguous(), v.float().contiguous(), s, sn)

    @staticmethod
    def _prepare(xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        """Validate the tensors and normalise both sigma arguments."""
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
        # The kernel takes one broadcast flag derived from sigma and applies it to
        # both tensors, so a mixed pair would read a scalar sigma_next by row.
        if s.dim() != sn.dim():
            raise ValueError(
                "sigma and sigma_next must use the same layout: both scalar or both "
                "one value per packed row."
            )
        _check_sigma_values(s, sn)
        return s, sn
