# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Causal depthwise conv1d single-token state update (WS1 ground truth, RFC #428 C6).

The trainer-side reference for ``causal_conv1d_update``, which vLLM calls once
per decode token before the Gated DeltaNet recurrence.

The incoming token is rounded to the cache dtype. Products are rounded in
operand dtype before sequential FP32 accumulation, starting from bias (or zero).
This matters for BF16 caches: promoting operands before multiplication loses the
provider's product rounding. Activation may still differ at transcendental ULP
scale; this reference does not establish model-level bitwise equality.

"""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.pytorch.linear_attn.gated_delta_rule import (
    NULL_BLOCK_ID,
    _validate_state_indices,
)

__all__ = ["CausalConv1dUpdateOp"]


class CausalConv1dUpdateOp:
    """One decode token through the paged causal-conv1d cache.

    ============== ========================== ==========================
    tensor         shape                      notes
    ============== ========================== ==========================
    ``x``          ``[B, dim]``
    ``conv_state`` ``[num_blocks, dim, W-1]``  paged; ``dim_first`` layout
    ``weight``     ``[dim, W]``
    ``bias``       ``[dim]`` or ``None``
    ``indices``    ``[B]``                     ``<= 0`` skips
    ============== ========================== ==========================

    Returns ``(out, conv_state)``; the state is updated out of place.
    """

    def __call__(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
        dim_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            x,
            conv_state,
            weight,
            conv_state_indices,
            bias=bias,
            activation=activation,
            dim_first=dim_first,
        )

    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
        dim_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._update(
            x,
            conv_state,
            weight,
            conv_state_indices,
            bias=bias,
            activation=activation,
            dim_first=dim_first,
            output_dtype=x.dtype,
        )

    def forward_fp32(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        conv_state_indices: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        activation: str | None = "silu",
        dim_first: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ground truth: fp32 output, so only the cache dtype rounds."""
        return self._update(
            x,
            conv_state,
            weight,
            conv_state_indices,
            bias=bias,
            activation=activation,
            dim_first=dim_first,
            output_dtype=torch.float32,
        )

    @staticmethod
    def _update(
        x,
        conv_state,
        weight,
        conv_state_indices,
        *,
        bias,
        activation,
        dim_first,
        output_dtype,
    ):
        if activation not in (None, "silu", "swish"):
            raise ValueError(f"activation must be None, 'silu' or 'swish', got {activation!r}")
        if x.dim() != 2:
            raise ValueError(f"x must be 2-D [B, dim], got {tuple(x.shape)}")
        if conv_state.dim() != 3:
            raise ValueError(
                f"conv_state must be 3-D [num_blocks, dim, W-1], got {tuple(conv_state.shape)}"
            )
        if conv_state_indices.dim() != 1 or conv_state_indices.shape[0] != x.shape[0]:
            raise ValueError("conv_state_indices must be 1-D with one entry per row of x")

        # The "SD" layout stores (state_len, dim); the kernels want (dim, state_len).
        state = conv_state if dim_first else conv_state.transpose(-1, -2)
        dim, tail = state.shape[-2], state.shape[-1]
        if weight.ndim != 2 or dim <= 0 or weight.shape[-1] <= 0:
            raise ValueError("weight must be 2-D [dim, W] with positive dimensions")
        if x.shape[1] != dim:
            raise ValueError("x and conv_state must have the same dim")
        _validate_state_indices(conv_state_indices, x.shape[0], state.shape[0], x.device)
        for name, tensor in (
            ("x", x),
            ("conv_state", conv_state),
            ("weight", weight),
            ("bias", bias),
        ):
            if tensor is not None and (tensor.device != x.device or not tensor.is_floating_point()):
                raise ValueError(f"{name} must be floating point on the input device")
        if bias is not None and bias.shape != (dim,):
            raise ValueError("bias must have shape [dim]")
        width = weight.shape[-1]
        if weight.shape[0] != dim:
            raise ValueError(f"weight must be [dim, W] with dim={dim}, got {tuple(weight.shape)}")
        if tail != width - 1:
            raise ValueError(f"conv_state tail must be W-1={width - 1}, got {tail}")

        active = conv_state_indices > NULL_BLOCK_ID
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        new_state = conv_state.clone()  # the cache dtype is the contract here
        if not bool(active.any()):
            return out.to(output_dtype), new_state

        rows = torch.nonzero(active, as_tuple=False).flatten()
        blocks = conv_state_indices[rows].long()

        # The incoming token is rounded to the cache dtype before it is used.
        token = x[rows].to(conv_state.dtype)
        window = torch.cat([state[blocks], token.unsqueeze(-1)], dim=-1)  # [R, dim, W]

        # Match the provider's product rounding and bias-before-taps order.
        acc = torch.zeros(len(rows), dim, dtype=torch.float32, device=x.device)
        if bias is not None:
            acc = acc + bias.float()
        for tap in range(width):
            product = window[..., tap] * weight[:, tap].unsqueeze(0)
            acc = acc + product.float()
        if activation in ("silu", "swish"):
            acc = acc / (1.0 + torch.exp(-acc))

        out[rows] = acc
        rolled = window[..., 1:]
        if dim_first:
            new_state[blocks] = rolled.to(conv_state.dtype)
        else:
            new_state[blocks] = rolled.transpose(-1, -2).to(conv_state.dtype)
        return out.to(output_dtype), new_state
