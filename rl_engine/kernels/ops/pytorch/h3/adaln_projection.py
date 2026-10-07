# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 three-modality AdaLN projection (RFC #420 row ``adaln_projection_3mod``).

One ``MiniMaxH3AdaLayerNormModulation`` per transformer block::

    act   = silu(temb).to(weight.dtype)          SiLU in FP32, one cast to BF16
    table = act @ W.T + b                        (T, 6 * H * 3), BF16
    rows  = table.view(3 * T, 6 * H)             row t * 3 + m, m in {video, text, audio}
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = rows.chunk(6, -1)

Output channel ``o = m * 6H + c * H + h`` of the projection is chunk ``c``, hidden
index ``h`` of modality ``m``. Each of the six outputs has shape ``(3T, H)``,
and row ``t * 3 + m`` is what ``timestep_indices * 3 + token_tags`` addresses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.pytorch.h3 import H3_ADALN_CHUNKS, H3_MODALITY_NUM

_WEIGHT_DTYPES = (torch.bfloat16, torch.float32)


def adaln_hidden_size(weight: torch.Tensor) -> int:
    rows_per_hidden = H3_ADALN_CHUNKS * H3_MODALITY_NUM
    if weight.dim() != 2 or weight.shape[0] % rows_per_hidden != 0:
        raise ValueError(
            f"AdaLN weight must be (6 * H * 3, D); got {tuple(weight.shape)}, whose first "
            f"dim is not a multiple of {rows_per_hidden}"
        )
    return weight.shape[0] // rows_per_hidden


def validate_h3_adaln_projection(temb: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    for name, tensor in (("temb", temb), ("weight", weight), ("bias", bias)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != temb.device:
            raise ValueError(f"{name} is on {tensor.device}, temb is on {temb.device}")
    if temb.dtype != torch.float32:
        # RFC probe H7: the activation must run at the timestep-embedding
        # precision; a BF16 temb means the cast happened before the SiLU.
        raise TypeError(f"temb must be float32 (the SiLU runs before the cast), got {temb.dtype}")
    if weight.dtype not in _WEIGHT_DTYPES or bias.dtype != weight.dtype:
        raise TypeError(
            f"weight/bias must share a dtype in {_WEIGHT_DTYPES}, got {weight.dtype}/{bias.dtype}"
        )
    hidden = adaln_hidden_size(weight)
    if temb.dim() != 2 or temb.shape[0] == 0 or temb.shape[1] != weight.shape[1]:
        raise ValueError(
            f"temb must be a non-empty (T, {weight.shape[1]}) matrix, got {tuple(temb.shape)}"
        )
    if bias.shape != (weight.shape[0],):
        raise ValueError(f"bias must be ({weight.shape[0]},), got {tuple(bias.shape)}")
    return hidden


def split_adaln_table(table: torch.Tensor, hidden: int) -> tuple[torch.Tensor, ...]:
    """(T, 6H*3) projection output -> six (3T, H) views, diffusers order."""

    return table.view(-1, H3_ADALN_CHUNKS * hidden).chunk(H3_ADALN_CHUNKS, dim=-1)


class NativeH3AdaLNProjectionOp:
    """PyTorch reference for the H3 AdaLN projection.

    ``forward`` is the provider path (``F.silu`` in FP32, cast, ``F.linear``).
    ``forward_fp32`` is the golden: the SiLU in FP64, rounded once to the
    weight dtype at the declared boundary (the cast is model semantics, not
    an implementation detail), then the projection in FP64. The six outputs are
    returned in FP32 without the final BF16 rounding. Its gradient is FP64
    end to end: the cast is applied straight-through (identity VJP).
    """

    op_class = "reduction"

    def __call__(self, temb, weight, bias):
        return self.forward(temb, weight, bias)

    def forward(self, temb, weight, bias) -> tuple[torch.Tensor, ...]:
        hidden = validate_h3_adaln_projection(temb, weight, bias)
        table = F.linear(F.silu(temb).to(weight.dtype), weight, bias)
        return split_adaln_table(table, hidden)

    def forward_fp32(self, temb, weight, bias) -> tuple[torch.Tensor, ...]:
        hidden = validate_h3_adaln_projection(temb, weight, bias)
        t64 = temb.double()
        act = t64 * torch.sigmoid(t64)
        # The declared cast rounds the value; its VJP is the identity. Written
        # straight-through so autograd does not round the FP64 gradient to the
        # weight dtype on its way back (``.to(bf16)`` would).
        act = act + (act.to(weight.dtype).double() - act).detach()
        table = F.linear(act, weight.double(), bias.double())
        return tuple(chunk.float() for chunk in split_adaln_table(table, hidden))
