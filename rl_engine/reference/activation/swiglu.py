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


def _validate_swiglu_packed_input(packed_input: torch.Tensor) -> None:
    """Validate a packed tensor's shape and supported floating dtype.

    The last dimension must be positive and even; leading dimensions may be
    empty. The reference accepts supported floating dtypes on any device.
    """
    if packed_input.ndim == 0:
        raise ValueError("packed input must have at least one dimension")
    packed_width = packed_input.shape[-1]
    if packed_width == 0 or packed_width % 2:
        raise ValueError(
            f"packed input must have a positive, even last dimension, got {packed_width}"
        )
    _validate_dtype(packed_input, "packed input")


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

    Both call forms require named arguments: ``gate=..., up=...`` for separate
    operands, or ``packed_input=..., packed_order=...`` for one packed projection.
    Positional tensor arguments and mixtures of the two forms are rejected.

    The overloads describe the two accepted input forms for IDEs and type
    checkers. Each method has one runtime implementation below them.
    Packed callers must pass a ``SwiGLUPackedOrder`` member; string parsing
    belongs to the calling configuration or CLI layer.
    """

    def __init__(self) -> None:
        super().__init__()

    @overload
    def forward(self, *, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...

    @overload
    def forward(
        self, *, packed_input: torch.Tensor, packed_order: SwiGLUPackedOrder
    ) -> torch.Tensor: ...

    def forward(
        self,
        *,
        gate: torch.Tensor | None = None,
        up: torch.Tensor | None = None,
        packed_input: torch.Tensor | None = None,
        packed_order: SwiGLUPackedOrder | None = None,
    ) -> torch.Tensor:
        """
        Canonical entry: compute in fp32, cast the result back to the input dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.

        Accept named ``gate`` and ``up``, or named ``packed_input`` and ``packed_order``.
        Requiring keywords prevents a positional swap of the separate operands.
        """
        return self._swiglu(
            gate=gate,
            up=up,
            packed_input=packed_input,
            packed_order=packed_order,
            fp32_output=False,
        )

    @overload
    def forward_fp32(self, *, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor: ...

    @overload
    def forward_fp32(
        self, *, packed_input: torch.Tensor, packed_order: SwiGLUPackedOrder
    ) -> torch.Tensor: ...

    def forward_fp32(
        self,
        *,
        gate: torch.Tensor | None = None,
        up: torch.Tensor | None = None,
        packed_input: torch.Tensor | None = None,
        packed_order: SwiGLUPackedOrder | None = None,
    ) -> torch.Tensor:
        """Compute and return FP32 for either input layout, preserving autograd."""
        return self._swiglu(
            gate=gate, up=up, packed_input=packed_input, packed_order=packed_order, fp32_output=True
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _swiglu(
        *,
        gate: torch.Tensor | None,
        up: torch.Tensor | None,
        packed_input: torch.Tensor | None,
        packed_order: SwiGLUPackedOrder | None,
        fp32_output: bool,
    ) -> torch.Tensor:
        """Validate the chosen input form, then apply the same FP32 expression."""
        if packed_input is not None:
            if gate is not None or up is not None:
                raise TypeError("packed_input cannot be combined with gate or up")
            if packed_order is None:
                raise ValueError("packed_order is required with packed_input")
            _validate_swiglu_packed_input(packed_input)
            first, second = packed_input.chunk(2, dim=-1)
            if packed_order is SwiGLUPackedOrder.GATE_UP:
                gate, up = first, second
            else:
                up, gate = first, second
        else:
            if gate is None or up is None:
                raise TypeError("provide both gate and up, or packed_input with packed_order")
            if packed_order is not None:
                raise ValueError("packed_order is only valid with packed_input")
            _validate_swiglu_separate_inputs(gate, up)

        gate_f = gate.float()
        out = gate_f * torch.sigmoid(gate_f) * up.float()
        output_dtype = torch.float32 if fp32_output else gate.dtype
        return out.to(output_dtype)
