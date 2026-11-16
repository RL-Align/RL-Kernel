# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Causal depthwise short-conv reference (Qwen3-Next linear-attention stack).

Qwen3-Next feeds qkv through a causal
depthwise conv1d with kernel size 4 before the per-head projections enter
the Gated DeltaNet. This module is the pure-PyTorch golden for that op:

    y[b, t, c] = bias[c] + sum_{j=0}^{K-1} weight[c, j] * x_pad[b, t+j, c]

(Torch ``F.conv1d`` cross-correlation convention: tap ``j=0`` pairs with
the oldest position ``x[t-K+1]``, ``j=K-1`` with the newest ``x[t]`` — the
same pairing HF ``CausalConv1d`` produces with left padding.)

with the leading ``K-1`` positions taken from ``initial_cache`` (left
context carried across prefill/decode boundaries; zeros when absent). The
reduction order over the fixed ``K`` taps is the same for every position,
so the op is batch-invariant by construction — the property the AscendC
kernel (``csrc/ascend/causal_conv1d_ascend.asc``, planned) must reproduce.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_dtype(x: torch.Tensor, name: str) -> None:
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {x.dtype}.")


class NativeCausalConv1dOp(nn.Module):
    """
    Pure PyTorch causal depthwise conv1d reference.

    Shapes (channel-last, matching the HF ``Qwen3Next`` dataflow):
      - x:            [B, T, C]
      - weight:       [C, K]   (one depthwise filter per channel)
      - bias:         [C] or None
      - initial_cache:[B, K-1, C] or None  (left context; None -> zeros)
      - returns y:    [B, T, C]
      - with return_cache=True also new_cache: [B, K-1, C]
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        initial_cache: Optional[torch.Tensor] = None,
        *,
        return_cache: bool = False,
    ):
        """
        Canonical entry: accumulate in fp32, cast the result back to x.dtype.
        """
        return self._conv(
            x, weight, bias, initial_cache, output_dtype=x.dtype, return_cache=return_cache
        )

    def forward_fp32(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        initial_cache: Optional[torch.Tensor] = None,
        *,
        return_cache: bool = False,
    ):
        """Ground-truth: accumulate in fp32 and force fp32 output."""
        return self._conv(
            x, weight, bias, initial_cache, output_dtype=torch.float32, return_cache=return_cache
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _conv(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        initial_cache: Optional[torch.Tensor],
        *,
        output_dtype: torch.dtype,
        return_cache: bool,
    ):
        if x.dim() != 3:
            raise ValueError(f"x must be [B, T, C], got tuple(x.shape)={tuple(x.shape)}")
        _validate_dtype(x, "x")
        _validate_dtype(weight, "weight")
        if bias is not None:
            _validate_dtype(bias, "bias")
        if initial_cache is not None:
            _validate_dtype(initial_cache, "initial_cache")

        batch, seqlen, channels = x.shape
        if weight.dim() != 2 or weight.shape[0] != channels:
            raise ValueError(
                f"weight must be [C, K] with C={channels}, "
                f"got tuple(weight.shape)={tuple(weight.shape)}"
            )
        kernel = weight.shape[1]
        if kernel <= 0 or channels <= 0:
            raise ValueError("kernel width and channel count must be positive")
        for name, tensor in (("weight", weight), ("bias", bias), ("initial_cache", initial_cache)):
            if tensor is not None and tensor.device != x.device:
                raise ValueError(f"{name} must be on the same device as x")
        if bias is not None and bias.shape != (channels,):
            raise ValueError(
                f"bias must be [C] with C={channels}, got tuple(bias.shape)={tuple(bias.shape)}"
            )
        if initial_cache is not None and (
            initial_cache.dim() != 3 or initial_cache.shape != (batch, kernel - 1, channels)
        ):
            raise ValueError(
                f"initial_cache must be [B, {kernel - 1}, C] = "
                f"[{batch}, {kernel - 1}, {channels}], got "
                f"tuple(initial_cache.shape)={tuple(initial_cache.shape)}"
            )

        x_f = x.float()
        pad = kernel - 1
        if initial_cache is None:
            prefix = x_f.new_zeros(batch, pad, channels)
        else:
            prefix = initial_cache.float()
        padded = torch.cat([prefix, x_f], dim=1)  # [B, T + K - 1, C]

        # windows[b, t, c, j] = padded[b, t + j, c]; reduction over the K taps
        # is an elementwise product plus a fixed-length sum, never a BLAS
        # call, so the reduction order cannot vary with batch size — the
        # determinism contract.
        if seqlen:
            windows = padded.unfold(1, kernel, 1)  # [B, T, C, K]
            out = (windows * weight.float()).sum(dim=-1)
        else:
            out = x_f * weight.float().sum(dim=-1)
        if bias is not None:
            out = out + bias.float()

        result = out.to(output_dtype)
        if return_cache:
            # A view would retain the whole prefill allocation for K-1 tokens.
            tail = padded[:, -pad:, :] if pad else padded[:, :0, :]
            new_cache = tail.to(dtype=output_dtype, copy=True)
            return result, new_cache
        return result
