# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import overload

import torch
import torch.nn as nn

from rl_engine.contracts.operators.activation import SwiGLUPackedOrder

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_dtype(x: torch.Tensor, name: str) -> None:
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


def _validate_swiglu_packed_input(
    projected: torch.Tensor, packed_order: SwiGLUPackedOrder | None
) -> None:
    """Validate one packed projection and the explicit order of its equal halves.

    The last dimension must be positive and even; leading dimensions may be
    empty. The reference accepts supported floating dtypes on any device.
    """
    if not isinstance(packed_order, SwiGLUPackedOrder):
        raise ValueError(f"packed_order must be one of {tuple(SwiGLUPackedOrder)}")
    if projected.ndim == 0:
        raise ValueError("packed input must have at least one dimension")
    width = projected.shape[-1]
    if width == 0 or width % 2:
        raise ValueError(f"packed input must have a positive, even last dimension, got {width}")
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

    The overloads describe the two accepted input forms for IDEs and type
    checkers. Each method has one runtime implementation below them.
    """

    def __init__(self) -> None:
        super().__init__()

    @overload
    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Separate gate/up tensors; return output in the input dtype."""
        ...

    @overload
    def forward(self, gate: torch.Tensor, *, packed_order: SwiGLUPackedOrder) -> torch.Tensor:
        """One packed tensor in ``gate``; its half order must be explicit."""
        ...

    def forward(
        self,
        gate: torch.Tensor,
        up: torch.Tensor | None = None,
        *,
        packed_order: SwiGLUPackedOrder | None = None,
    ) -> torch.Tensor:
        """
        Canonical entry: compute in fp32, cast the result back to gate.dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.

        When ``up`` is omitted, ``gate`` is the packed projection and
        ``packed_order`` must identify the order of its two halves.
        """
        return self._swiglu(gate, up, output_dtype=gate.dtype, packed_order=packed_order)

    @overload
    def forward_fp32(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Separate gate/up tensors; return FP32 output."""
        ...

    @overload
    def forward_fp32(self, gate: torch.Tensor, *, packed_order: SwiGLUPackedOrder) -> torch.Tensor:
        """One packed tensor in ``gate``; return FP32 output."""
        ...

    def forward_fp32(
        self,
        gate: torch.Tensor,
        up: torch.Tensor | None = None,
        *,
        packed_order: SwiGLUPackedOrder | None = None,
    ) -> torch.Tensor:
        """Compute and return FP32 for either input layout, preserving autograd."""
        return self._swiglu(gate, up, output_dtype=torch.float32, packed_order=packed_order)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _swiglu(
        gate: torch.Tensor,
        up: torch.Tensor | None,
        *,
        output_dtype: torch.dtype,
        packed_order: SwiGLUPackedOrder | None,
    ) -> torch.Tensor:
        """Validate the chosen input form, then apply the same FP32 expression."""
        if up is None:
            _validate_swiglu_packed_input(gate, packed_order)
            first, second = gate.chunk(2, dim=-1)
            if packed_order is SwiGLUPackedOrder.GATE_UP:
                gate, up = first, second
            else:
                up, gate = first, second
        else:
            if packed_order is not None:
                raise ValueError("packed_order is only valid with a single packed input")
            _validate_swiglu_separate_inputs(gate, up)

        gate_f = gate.float()
        out = gate_f * torch.sigmoid(gate_f) * up.float()
        return out.to(output_dtype)
