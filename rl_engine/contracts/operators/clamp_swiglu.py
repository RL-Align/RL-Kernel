# SPDX-License-Identifier: Apache-2.0
"""Task 46 activation contract, before the task 43 quantization boundary."""

from dataclasses import dataclass

import torch
from torch import Tensor

PROFILE = "dsv4-clamp-swiglu-rocm-fp32-v1"
GATE_MAX = 10.0
UP_MIN = -10.0
UP_MAX = 10.0


@dataclass(frozen=True)
class ClampSwiGLUState:
    """FP32 intermediates; task 49 must also retain the original gate/up inputs.

    Keep silu separately: ordered dp_s accumulates (dh * silu) * clamped_up,
    which need not have the same bits as dh * unweighted. Never divide by p_s.
    """

    sigmoid: Tensor
    silu: Tensor
    unweighted: Tensor


def validate_inputs(gate: Tensor, up: Tensor, p_s: Tensor | None) -> None:
    if gate.layout != torch.strided or up.layout != torch.strided:
        raise ValueError("gate and up must be strided tensors")
    if gate.ndim != 2 or gate.shape[1] == 0 or gate.shape != up.shape:
        raise ValueError("gate and up must have the same [M, F] shape with F > 0")
    if gate.dtype not in (torch.bfloat16, torch.float32) or up.dtype != gate.dtype:
        raise TypeError("gate and up must have the same BF16 or FP32 dtype")
    if up.device != gate.device:
        raise ValueError("gate and up must be on the same device")
    if p_s is not None:
        if p_s.layout != torch.strided or p_s.shape != (gate.shape[0],):
            raise ValueError("p_s must be a strided [M] tensor")
        if p_s.dtype != torch.float32:
            raise TypeError("p_s must be FP32")
        if p_s.device != gate.device:
            raise ValueError("p_s must be on the gate device")


def validate_backward(
    dh: Tensor, gate: Tensor, up: Tensor, p_s: Tensor | None, state: ClampSwiGLUState
) -> None:
    validate_inputs(gate, up, p_s)
    if dh.layout != torch.strided or dh.shape != gate.shape or dh.device != gate.device:
        raise ValueError("dh must have the gate shape and device and strided layout")
    if dh.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("dh must be BF16 or FP32")
    for name in ("sigmoid", "silu", "unweighted"):
        value = getattr(state, name)
        if (
            value.layout != torch.strided
            or value.shape != gate.shape
            or value.device != gate.device
            or value.dtype != torch.float32
            or not value.is_contiguous()
        ):
            raise ValueError(f"state.{name} must be contiguous FP32 with the gate shape/device")


def split_packed(gate_up: Tensor) -> tuple[Tensor, Tensor]:
    if gate_up.ndim != 2 or gate_up.shape[1] == 0 or gate_up.shape[1] % 2:
        raise ValueError("packed input must have shape [M, 2 * F] with F > 0")
    gate, up = gate_up.chunk(2, dim=1)
    return gate, up
