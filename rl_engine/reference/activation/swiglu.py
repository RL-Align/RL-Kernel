# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import overload

import torch
import torch.nn as nn

from rl_engine.contracts.operators.activation import SwiGLUPackedOrder

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_dtype(x: torch.Tensor, name: str) -> None:
    """Reject unsupported tensor dtypes, naming the offending operand in the error."""
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype in {_SUPPORTED_DTYPES}, got {x.dtype}.")


def _validate_swiglu_separate_inputs(gate: torch.Tensor, up: torch.Tensor) -> None:
    """Validate two operands with matching shape, dtype and device."""
    if gate.shape != up.shape:
        raise ValueError(
            f"gate and up must share shape, got tuple(gate.shape)="
            f"{tuple(gate.shape)} vs tuple(up.shape)={tuple(up.shape)}"
        )
    if gate.device != up.device:
        raise RuntimeError(
            f"gate and up must be on the same device, got '{gate.device}' and '{up.device}'."
        )
    _validate_dtype(gate, "gate")
    _validate_dtype(up, "up")
    if gate.dtype != up.dtype:
        raise TypeError(f"gate and up must share dtype, got {gate.dtype} and {up.dtype}.")


def _validate_swiglu_packed_input(projected: torch.Tensor) -> None:
    """Validate a packed tensor's shape and supported floating dtype.

    The last dimension must be positive and even; leading dimensions may be
    empty. The reference accepts supported floating dtypes on any device.
    """
    if projected.ndim == 0:
        raise ValueError("packed input must have at least one dimension")
    packed_width = projected.shape[-1]
    if packed_width == 0 or packed_width % 2:
        raise ValueError(
            f"packed input must have a positive, even last dimension, got {packed_width}"
        )
    _validate_dtype(projected, "packed input")


class NativeSiLUOp(nn.Module):
    """
    Pure PyTorch native SiLU reference.
    out = x * sigmoid(x)   (a.k.a. Swish)

    Element-wise activation used by the Qwen3 SwiGLU MLP
    (hidden_act="silu"). No hyper-parameters and shape-agnostic, so the
    Qwen3-8B intermediate dim (12288) is just one valid last-dim size.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Canonical entry: compute in fp32, cast the result back to x.dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.
        """
        return self._silu(x, output_dtype=x.dtype)

    def forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        """Ground-truth: compute in fp32 and force fp32 output."""
        return self._silu(x, output_dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _silu(x: torch.Tensor, *, output_dtype: torch.dtype) -> torch.Tensor:
        _validate_dtype(x, "x")
        x_f = x.float()
        out = x_f * torch.sigmoid(x_f)
        return out.to(output_dtype)


class NativeSwiGLUOp(nn.Module):
    """
    Pure PyTorch native SwiGLU reference.
    out = silu(gate) * up = (gate * sigmoid(gate)) * up

    Accepts separate ``gate, up`` tensors of the same shape, or a single
    ``[..., 2 * I]`` tensor with an explicit ``SwiGLUPackedOrder``. Both packed
    orders produce ``[..., I]``. The gate/up and down linear projections
    remain outside this operator. All intermediate arithmetic stays FP32.

    The first parameter is named ``input``: it holds the gate when ``up`` is
    supplied, or the complete packed projection when ``up`` is omitted.

    The overloads describe the two accepted input forms for IDEs and type
    checkers. Each method has one runtime implementation below them.
    Packed callers must pass a ``SwiGLUPackedOrder`` member; string parsing
    belongs to the calling configuration or CLI layer.
    """

    def __init__(self) -> None:
        super().__init__()

    @overload
    def forward(self, input: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Separate gate/up tensors; return output in the input dtype."""
        ...

    @overload
    def forward(self, input: torch.Tensor, *, packed_order: SwiGLUPackedOrder) -> torch.Tensor:
        """One packed tensor in ``input``; its half order must be explicit."""
        ...

    def forward(
        self,
        input: torch.Tensor,
        up: torch.Tensor | None = None,
        *,
        packed_order: SwiGLUPackedOrder | None = None,
    ) -> torch.Tensor:
        """
        Canonical entry: compute in fp32, cast the result back to input.dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.

        ``input`` is the gate when ``up`` is supplied, or the packed projection
        otherwise. A single packed input requires an explicit ``packed_order``.
        """
        return self._swiglu(input, up, output_dtype=input.dtype, packed_order=packed_order)

    @overload
    def forward_fp32(self, input: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Separate gate/up tensors; return FP32 output."""
        ...

    @overload
    def forward_fp32(self, input: torch.Tensor, *, packed_order: SwiGLUPackedOrder) -> torch.Tensor:
        """One packed tensor in ``input``; return FP32 output."""
        ...

    def forward_fp32(
        self,
        input: torch.Tensor,
        up: torch.Tensor | None = None,
        *,
        packed_order: SwiGLUPackedOrder | None = None,
    ) -> torch.Tensor:
        """Compute and return FP32 for either input layout, preserving autograd."""
        return self._swiglu(input, up, output_dtype=torch.float32, packed_order=packed_order)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _swiglu(
        input: torch.Tensor,
        up: torch.Tensor | None,
        *,
        output_dtype: torch.dtype,
        packed_order: SwiGLUPackedOrder | None,
    ) -> torch.Tensor:
        """Validate the chosen input form, then apply the same FP32 expression."""
        if up is None:
            if packed_order is None:
                raise ValueError("packed_order is required with a single packed input")
            projected = input
            _validate_swiglu_packed_input(projected)
            first, second = projected.chunk(2, dim=-1)
            if packed_order is SwiGLUPackedOrder.GATE_UP:
                gate, up = first, second
            else:
                up, gate = first, second
        else:
            if packed_order is not None:
                raise ValueError("packed_order is only valid with a single packed input")
            gate = input
            _validate_swiglu_separate_inputs(gate, up)

        gate_f = gate.float()
        out = gate_f * torch.sigmoid(gate_f) * up.float()
        return out.to(output_dtype)
