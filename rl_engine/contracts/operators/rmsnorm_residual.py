# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Task 5 contract: RMSNorm with a byte-preserving fork of the original row.

The operator normalizes a BF16 row and hands the *unnormalized* row to the
residual branch unchanged. It does not add a residual before normalizing.

Frozen arithmetic (RL-Kernel RFC #509, KLR #18):

    s      = sum_d FP32(x[t, d])^2        serial ascending left fold, FP32
    m      = s / D
    r      = rsqrt(m + eps)               rsqrt, never 1 / sqrt(...)
    y      = BF16((FP32(x) * r) * gamma)  one downcast, at the output
    res    = x                            same bytes, separate storage

    u      = FP32(dy) * gamma
    q      = sum_d u[d] * x[d]            serial ascending left fold, FP32
    dx     = (r * u) - (((x * r^3) * q) / D) + FP32(d_residual)
    dgamma = sum_t (dy[t] * x[t]) * r[t]  serial ascending token fold, FP32

The controller RMS of ``fp32_gemm_rms`` (task 1) is ``1 / (sqrt(mean) + eps)``;
the two forms are never interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PROFILE = "dsv4-rmsnorm-residual-fp32-v1"
EPS = 1.0e-6
MAX_HIDDEN = 16384
LONG_REDUCTION_ORDER = "serial-ascending-left-fold"
INPUT_DTYPE = torch.bfloat16
GAIN_DTYPE = torch.float32
GRAD_DTYPES = (torch.bfloat16, torch.float32)


class RMSNormResidualContractError(ValueError):
    """Raised when a task 5 request leaves the frozen contract."""


@dataclass(frozen=True)
class RMSNormResidualSaved:
    """Forward state the backward needs; ``r`` is the FP32 inverse RMS per row."""

    r: Tensor
    hidden: int


def describe() -> dict[str, object]:
    """Provenance fields every backend reports alongside its results."""
    return {
        "profile": PROFILE,
        "eps": EPS,
        "long_reduction_order": LONG_REDUCTION_ORDER,
        "rsqrt_form": "rsqrt(mean + eps)",
        "downcast_points": ["y"],
        "residual": "byte copy of x, separate storage",
        "gradient_dtype": "float32",
    }


def _strided(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise RMSNormResidualContractError(f"{name} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise RMSNormResidualContractError(f"{name} must be a strided tensor")


def validate_forward(x: Tensor, gamma: Tensor, eps: float) -> tuple[int, int]:
    """Check ``x`` BF16 [T, D], ``gamma`` FP32 [D], ``eps == 1e-6``. Returns ``(T, D)``."""
    _strided(x, "x")
    _strided(gamma, "gamma")
    if x.ndim != 2:
        raise RMSNormResidualContractError("x must have shape [T, D]")
    tokens, hidden = x.shape
    if hidden <= 0 or hidden > MAX_HIDDEN:
        raise RMSNormResidualContractError(f"D must be in [1, {MAX_HIDDEN}], got {hidden}")
    if x.dtype is not INPUT_DTYPE:
        raise RMSNormResidualContractError(f"x must be {INPUT_DTYPE}, got {x.dtype}")
    if gamma.shape != (hidden,):
        raise RMSNormResidualContractError(f"gamma must have shape ({hidden},)")
    if gamma.dtype is not GAIN_DTYPE:
        raise RMSNormResidualContractError(
            f"gamma must stay {GAIN_DTYPE}; it is not rounded to BF16 (got {gamma.dtype})"
        )
    if gamma.device != x.device:
        raise RMSNormResidualContractError("gamma must be on the x device")
    if float(eps) != EPS:
        raise RMSNormResidualContractError(f"eps is frozen at {EPS}, got {eps}")
    return tokens, hidden


def validate_backward(
    dy: Tensor, d_residual: Tensor, x: Tensor, gamma: Tensor, saved: RMSNormResidualSaved
) -> tuple[int, int]:
    """Check the upstream gradients and the saved state against the forward inputs."""
    tokens, hidden = validate_forward(x, gamma, EPS)
    for name, value in (("dy", dy), ("d_residual", d_residual)):
        _strided(value, name)
        if value.shape != x.shape:
            raise RMSNormResidualContractError(f"{name} must have the x shape {tuple(x.shape)}")
        if value.dtype not in GRAD_DTYPES:
            raise RMSNormResidualContractError(f"{name} must be BF16 or FP32, got {value.dtype}")
        if value.device != x.device:
            raise RMSNormResidualContractError(f"{name} must be on the x device")
    if not isinstance(saved, RMSNormResidualSaved):
        raise RMSNormResidualContractError("saved must be an RMSNormResidualSaved")
    if saved.hidden != hidden:
        raise RMSNormResidualContractError("saved state belongs to a different hidden size")
    _strided(saved.r, "saved.r")
    if saved.r.shape != (tokens,) or saved.r.dtype is not torch.float32:
        raise RMSNormResidualContractError("saved.r must be FP32 with shape [T]")
    if saved.r.device != x.device:
        raise RMSNormResidualContractError("saved.r must be on the x device")
    return tokens, hidden


__all__ = [
    "EPS",
    "GRAD_DTYPES",
    "LONG_REDUCTION_ORDER",
    "MAX_HIDDEN",
    "PROFILE",
    "RMSNormResidualContractError",
    "RMSNormResidualSaved",
    "describe",
    "validate_backward",
    "validate_forward",
]
