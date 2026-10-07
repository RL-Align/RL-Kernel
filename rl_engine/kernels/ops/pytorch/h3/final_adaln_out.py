# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 final AdaLN output (RFC #420 row ``final_adaln_out``).

``MiniMaxH3AdaLayerNormOut`` runs after the 50 blocks::

    shift, scale = linear(silu(temb).to(bf16)).chunk(2)        # (T, H) each, shift first
    out = norm(x) * (1.0 + scale[timestep_indices]) + shift[timestep_indices]

``linear`` is ``norm_out.linear`` (``2688 -> 2 * 5376``, BF16), ``norm`` is
``norm_out.norm`` (``nn.RMSNorm(5376, eps=1e-5)``), and the table is indexed
by ``timestep_indices`` (one row per distinct timestep), not by
``adaln_indices``. Diffusers then casts the result to the FP32 output heads'
dtype, an exact upcast left to the caller.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.pytorch.h3.rmsnorm import (
    H3_NORM_EPS,
    NativeH3RMSNormOp,
    validate_h3_rmsnorm,
)


def validate_h3_final_adaln_out(
    x, norm_weight, temb, weight, bias, timestep_indices, eps, *, same_dtype: bool = True
) -> int:
    hidden = x.shape[-1]
    if same_dtype:
        validate_h3_rmsnorm(x, norm_weight, eps)
    if temb.dtype != torch.float32:
        raise TypeError(f"temb must be float32 (the SiLU runs before the cast), got {temb.dtype}")
    if temb.dim() != 2 or temb.shape[0] == 0:
        raise ValueError(f"temb must be a non-empty (T, D) matrix, got {tuple(temb.shape)}")
    if weight.shape != (2 * hidden, temb.shape[1]) or bias.shape != (2 * hidden,):
        raise ValueError(
            f"norm_out.linear must be ({2 * hidden}, {temb.shape[1]}) with a ({2 * hidden},) "
            f"bias, got {tuple(weight.shape)} / {tuple(bias.shape)}"
        )
    if bias.dtype != weight.dtype:
        raise TypeError("norm_out.linear weight and bias must share a dtype")
    for name, tensor in (("temb", temb), ("weight", weight), ("bias", bias)):
        if tensor.device != x.device:
            raise ValueError(f"{name} is on {tensor.device}, x is on {x.device}")
    if timestep_indices.dtype != torch.int64 or timestep_indices.dim() != 1:
        raise TypeError("timestep_indices must be a 1-D int64 tensor")
    if x.dim() < 2 or x.shape[-2] != timestep_indices.shape[0]:
        raise ValueError("x must be (..., S, H) with S = len(timestep_indices)")
    if bool(((timestep_indices < 0) | (timestep_indices >= temb.shape[0])).any()):
        raise IndexError(f"timestep_indices must lie in [0, {temb.shape[0]})")
    return hidden


class NativeH3FinalAdaLNOutOp:
    """PyTorch reference: ``forward`` replays diffusers' ``norm_out``;
    ``forward_fp32`` is the FP64 golden, returned in FP32. It rounds only at the
    module boundaries the model declares (the SiLU cast and the BF16 table),
    straight-through for the gradient."""

    op_class = "reduction"

    def __call__(self, x, norm_weight, temb, weight, bias, timestep_indices, eps=H3_NORM_EPS):
        return self.forward(x, norm_weight, temb, weight, bias, timestep_indices, eps)

    def forward(self, x, norm_weight, temb, weight, bias, timestep_indices, eps=H3_NORM_EPS):
        validate_h3_final_adaln_out(x, norm_weight, temb, weight, bias, timestep_indices, eps)
        shift, scale = F.linear(F.silu(temb).to(weight.dtype), weight, bias).chunk(2, dim=-1)
        return NativeH3RMSNormOp().forward_modulated(
            x, norm_weight, shift, scale, timestep_indices, eps
        )

    def forward_fp32(self, x, norm_weight, temb, weight, bias, timestep_indices, eps=H3_NORM_EPS):
        validate_h3_final_adaln_out(
            x, norm_weight, temb, weight, bias, timestep_indices, eps, same_dtype=False
        )
        t64 = temb.double()
        act = t64 * torch.sigmoid(t64)
        act = act + (act.to(weight.dtype).double() - act).detach()  # declared cast, identity VJP
        table = F.linear(act, weight.double(), bias.double())
        # norm_out.linear is a BF16 module: its (T, 2H) output is rounded to the
        # weight dtype, and every position of a timestep shares that row, so the
        # rounding is model semantics (straight-through for the gradient).
        table = table + (table.to(weight.dtype).double() - table).detach()
        shift, scale = table.chunk(2, dim=-1)
        x64 = x.double()
        n = x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + eps) * norm_weight.double()
        out = n * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )
        return out.float()
