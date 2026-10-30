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

from rl_engine.reference.linear_attn.gated_delta_rule import NULL_BLOCK_ID, _validate_state_indices

__all__ = ["CausalConv1dUpdateOp"]


class _SiluFp32(torch.autograd.Function):
    """``acc / (1 + exp(-acc))`` in the forward, a backward that cannot overflow.

    ``exp(-acc)`` overflows FP32 for ``acc < -88.7``. The forward then still returns a
    finite ``-0.``, but autograd's backward of the same expression multiplies ``inf`` by
    ``0`` and returns NaN. The forward is kept as is, so the values stay the ones the
    provider comparison was made against; the backward is ``s * (1 + acc * (1 - s))``
    with ``s = sigmoid(acc)``, which is finite for every finite ``acc``.
    """

    @staticmethod
    def forward(ctx, acc: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(acc)
        return acc / (1.0 + torch.exp(-acc))

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        (acc,) = ctx.saved_tensors
        s = torch.sigmoid(acc)
        return grad_out * (s * (1.0 + acc * (1.0 - s)))


class CausalConv1dUpdateOp:
    """One decode token through the paged causal-conv1d cache.

    ============== ========================== ==========================
    tensor         shape                      notes
    ============== ========================== ==========================
    ``x``          ``[B, dim]``
    ``conv_state`` ``[num_blocks, dim, W-1]``  paged; ``dim_first`` layout
    ``weight``     ``[dim, W]``
    ``bias``       ``[dim]`` or ``None``
    ``indices``    ``[B]``                     ``== 0`` skips (as vLLM); ``< 0`` is rejected
    ============== ========================== ==========================

    Returns ``(out, conv_state)``; the state is updated out of place.

    A skipped row (index ``0``) leaves its cache block unchanged and gets an all-zero
    output row. vLLM's kernel leaves ``out`` untouched for such a row instead, so a
    comparison against the provider must not include skipped rows.
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
        # vLLM's causal_conv1d_update skips only the null block id; a negative entry is
        # not a null block there. Reject it instead of treating it as inactive, before
        # any other check, and validate the rest with the conv rule (only 0 is skipped).
        if bool((conv_state_indices < 0).any()):
            raise ValueError("conv state indices must be >= 0; only the null block id is skipped")
        _validate_state_indices(
            conv_state_indices, x.shape[0], state.shape[0], x.device, skip_negative=False
        )
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

        active = conv_state_indices != NULL_BLOCK_ID
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        # The cache dtype is the contract here. `state` above is a view of the caller's
        # `conv_state` (transposed for the SD layout) and is only read; every write goes
        # to this clone, so the caller's cache is never modified.
        new_state = conv_state.clone()
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
            acc = _SiluFp32.apply(acc)

        out[rows] = acc
        rolled = window[..., 1:]
        if dim_first:
            new_state[blocks] = rolled.to(conv_state.dtype)
        else:
            new_state[blocks] = rolled.transpose(-1, -2).to(conv_state.dtype)
        return out.to(output_dtype), new_state
