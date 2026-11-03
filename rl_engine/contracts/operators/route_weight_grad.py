# SPDX-License-Identifier: Apache-2.0
"""Task 49's per-dispatched-row FP32 route-weight VJP."""

import torch
from torch import Tensor

PROFILE = "dsv4-route-weight-grad-rocm-fp32-pair256-v1"
BLOCK = 256


def validate_inputs(dh: Tensor, silu: Tensor, up: Tensor) -> None:
    if dh.ndim != 2 or dh.shape[1] == 0:
        raise ValueError("dh must have shape [M, F] with F > 0")
    for name, tensor in (("dh", dh), ("silu", silu), ("up", up)):
        if tensor.layout != torch.strided or tensor.shape != dh.shape or tensor.device != dh.device:
            raise ValueError(f"{name} must be strided with the dh shape and device")
        allowed = (torch.float32,) if name == "silu" else (torch.bfloat16, torch.float32)
        if tensor.dtype not in allowed:
            raise TypeError(f"{name} has unsupported dtype {tensor.dtype}")


def validate_training_inputs(dh, gate, up, p_s, state) -> None:
    validate_inputs(dh, state.silu, up)
    if (
        gate.layout != torch.strided
        or gate.shape != up.shape
        or gate.device != up.device
        or gate.dtype != up.dtype
    ):
        raise ValueError("gate must match up shape, device and dtype")
    if (
        p_s is None
        or p_s.layout != torch.strided
        or p_s.shape != (dh.shape[0],)
        or p_s.device != dh.device
        or p_s.dtype != torch.float32
    ):
        raise ValueError("routed backward requires FP32 p_s[M] on the input device")
