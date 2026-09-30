# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Causal depthwise conv1d single-token state update (WS1 ground truth, RFC #428 C6).

The trainer-side reference for ``causal_conv1d_update``, which vLLM calls once
per decode token before the Gated DeltaNet recurrence.

Semantics were established by differential testing against the provider rather
than transcribed, and the accumulation order matters:

* ``x`` is cast to ``conv_state.dtype`` **before** anything else, so a bf16
  conv cache rounds the incoming token (``causal_conv1d.py:1160``).
* The window is ``[state[..., -(width-1):], x]``; the taps are accumulated
  **sequentially in tap order**, ``acc = acc + win[t] * w[t]`` starting from
  zero -- not a tree reduction and not an FMA. With an fp32 conv state this
  reproduces the provider bitwise; a tree sum does not.
* Bias and the activation are applied in fp32, with one cast on the way out to
  the input's original dtype.
* The new state is the window minus its oldest column.

Measured against the provider on B200 (``conv_dim=8192``, ``W=4``, B up to 64):

* **The rolled state is bitwise exact in every configuration**, fp32 and bf16
  cache alike -- ``max|diff| = 0``. That is the part the next token consumes,
  so the recurrence carries no error from here.
* With an **fp32** conv state the output is bitwise on all but a handful of
  elements (15 of 524288 at B=64), the residual being fp32 ULP.
* With a **bf16** conv state the output agrees on ~63% of elements, every
  disagreement exactly one bf16 ULP.

The bf16 output gap is a rounding-path difference, not a semantic one: none of
accumulate-in-bf16, bias-in-bf16 or activation-in-bf16 reproduces the provider,
so where it rounds is left unresolved rather than guessed at. An fp32 conv state
is what an exactness claim should use anyway -- the same conclusion the
recurrent state reaches in :mod:`.gated_delta_rule`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.pytorch.linear_attn.gated_delta_rule import NULL_BLOCK_ID

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

        # Sequential tap accumulation from zero, in fp32. A tree sum over the
        # same four terms gives a different last bit and does NOT match.
        acc = torch.zeros(len(rows), dim, dtype=torch.float32, device=x.device)
        w32 = weight.float()
        for tap in range(width):
            acc = acc + window[..., tap].float() * w32[:, tap].unsqueeze(0)
        if bias is not None:
            acc = acc + bias.float()
        if activation in ("silu", "swish"):
            acc = F.silu(acc)

        out[rows] = acc
        rolled = window[..., 1:]
        if dim_first:
            new_state[blocks] = rolled.to(conv_state.dtype)
        else:
            new_state[blocks] = rolled.transpose(-1, -2).to(conv_state.dtype)
        return out.to(output_dtype), new_state
