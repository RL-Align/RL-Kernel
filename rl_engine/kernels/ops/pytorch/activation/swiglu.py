# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch
import torch.nn as nn


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
        x_f = x.float()
        out = x_f * torch.sigmoid(x_f)
        return out.to(output_dtype)


class NativeSwiGLUOp(nn.Module):
    """
    Pure PyTorch native SwiGLU reference.
    out = silu(gate) * up = (gate * sigmoid(gate)) * up

    Middle stage of the Qwen3/Llama MLP: ``gate`` and ``up`` are the
    gate_proj / up_proj outputs (already at intermediate dim 12288). The
    following down_proj is a plain Matmul and lives in a separate op.
    Element-wise and shape-agnostic; ``gate`` and ``up`` must share shape.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """
        Canonical entry: compute in fp32, cast the result back to gate.dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.
        """
        return self._swiglu(gate, up, output_dtype=gate.dtype)

    def forward_fp32(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Ground-truth: compute in fp32 and force fp32 output."""
        return self._swiglu(gate, up, output_dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _swiglu(
        gate: torch.Tensor,
        up: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if gate.shape != up.shape:
            raise ValueError(
                f"gate and up must share shape, got tuple(gate.shape)="
                f"{tuple(gate.shape)} vs tuple(up.shape)={tuple(up.shape)}"
            )
        gate_f = gate.float()
        out = gate_f * torch.sigmoid(gate_f) * up.float()
        return out.to(output_dtype)
