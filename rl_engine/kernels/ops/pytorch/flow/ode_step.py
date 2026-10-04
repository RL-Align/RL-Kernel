# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""MiniMax-H3 deterministic rectified-flow Euler step (WS1 reference).

    x0     = xt + sigma * v
    r      = sigma_next / sigma
    x_next = r * xt + (1 - r) * x0

FP32 throughout, one cast at the end. Keep this exact expression order: the
equivalent ``xt + (sigma - sigma_next) * v`` reassociates the sum and is a
different FP32 program (RFC #420 ablation probe H13).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _row_count(reference: Tensor) -> int:
    """Flattened packed-row count: numel divided by the last dim."""
    if reference.dim() < 1:
        return 1
    return reference.numel() // max(reference.shape[-1], 1)


def _as_sigma(value: Any, reference: Tensor, name: str) -> Tensor:
    """Return contiguous fp32 sigma as a 0-dim scalar or ``[..., 1]`` per row.

    A bare ``[R]`` must not survive: it broadcasts against the channel axis of
    ``[R, C]``, which is silently wrong whenever ``R == C``.
    """
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=torch.float32, device=reference.device)
    if value.device != reference.device:
        raise ValueError(
            f"{name} must be on xt's device, got {value.device} and {reference.device}."
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}.")
    value = value.to(dtype=torch.float32).contiguous()
    if value.numel() == 1:
        return value.reshape(())
    if value.numel() == _row_count(reference):
        return value.reshape(*reference.shape[:-1], 1)
    return value


def _check_xt_v(xt: Tensor, v: Tensor) -> None:
    for name, tensor in (("xt", xt), ("v", v)):
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {tensor.dtype}.")
    if xt.device != v.device:
        raise ValueError(f"xt and v must share a device, got {xt.device} and {v.device}.")
    if xt.shape != v.shape:
        raise ValueError(
            f"xt and v must share a shape, got {tuple(xt.shape)} and {tuple(v.shape)}."
        )


def _check_sigma_shape(sigma: Tensor, sigma_next: Tensor, xt: Tensor) -> None:
    """Reject anything that is not a scalar or one value per packed row."""
    rows = _row_count(xt)
    for name, tensor in (("sigma", sigma), ("sigma_next", sigma_next)):
        if tensor.numel() not in (1, rows):
            raise ValueError(
                f"{name} must be a scalar or one value per packed row ({rows}), got "
                f"numel={tensor.numel()}. Refusing to broadcast implicitly."
            )


def _check_sigma_values(sigma: Tensor, sigma_next: Tensor) -> None:
    """Reject geometry the pinned H3 shifted grid can never produce."""
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


def _ode_step_fp32(
    xt: Tensor, v: Tensor, sigma: Tensor, sigma_next: Tensor
) -> tuple[Tensor, Tensor]:
    r = sigma_next / sigma
    x0 = xt + sigma * v
    return r * xt + (1.0 - r) * x0, x0


class NativeH3OdeStepOp:
    """PyTorch reference: data-ward ``x0`` plus the FP32 Euler blend."""

    op_class = "elementwise"

    def __call__(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        return self.forward(xt, v, sigma, sigma_next)

    def forward(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        x_next, x0 = _ode_step_fp32(xt.float(), v.float(), s, sn)
        return x_next.to(xt.dtype), x0.to(xt.dtype)

    def forward_fp32(
        self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any
    ) -> tuple[Tensor, Tensor]:
        """gtest gold path: FP32 in, FP32 out."""
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _ode_step_fp32(xt.float(), v.float(), s, sn)

    @staticmethod
    def _prepare(xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        _check_xt_v(xt, v)
        s = _as_sigma(sigma, xt, "sigma")
        sn = _as_sigma(sigma_next, xt, "sigma_next")
        _check_sigma_shape(s, sn, xt)
        _check_sigma_values(s, sn)
        return s, sn
