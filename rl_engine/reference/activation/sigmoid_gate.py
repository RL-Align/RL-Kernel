# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import torch
import torch.nn as nn

from rl_engine.reference.activation.sigmoid import length_invariant_sigmoid

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_dtype(x: torch.Tensor, name: str) -> None:
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {x.dtype}.")


class NativeSigmoidGateOp(nn.Module):
    """
    Pure PyTorch native sigmoid-gate reference.
    out = x * sigmoid(gate)

    Element-wise multiplicative gate used by the Qwen3-Next hybrid stack:
    the full-attention layers gate their attention output
    (``attn_out * sigmoid(output_gate)``) and the Gated DeltaNet layers
    gate their recurrent update through sigmoid(beta); the decay gate
    consumes log decay separately. ``x`` and ``gate`` must share shape, as in
    the SwiGLU reference; no hyper-parameters, shape-agnostic.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Canonical entry: compute in fp32, cast the result back to x.dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.
        """
        return self._sigmoid_gate(x, gate, output_dtype=x.dtype)

    def forward_fp32(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Ground-truth: compute in fp32 and force fp32 output."""
        return self._sigmoid_gate(x, gate, output_dtype=torch.float32)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sigmoid_gate(
        x: torch.Tensor,
        gate: torch.Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if x.shape != gate.shape:
            raise ValueError(
                f"x and gate must share shape, got tuple(x.shape)={tuple(x.shape)} "
                f"vs tuple(gate.shape)={tuple(gate.shape)}"
            )
        if x.device != gate.device:
            raise RuntimeError(
                f"x and gate must be on the same device, got '{x.device}' and '{gate.device}'."
            )
        _validate_dtype(x, "x")
        _validate_dtype(gate, "gate")
        if x.dtype != gate.dtype:
            raise TypeError(f"x and gate must share dtype, got {x.dtype} and {gate.dtype}.")

        out = x.float() * length_invariant_sigmoid(gate.float())
        return out.to(output_dtype)
