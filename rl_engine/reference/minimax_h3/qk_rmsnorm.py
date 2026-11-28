# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""H3 per-head Q/K RMSNorm (RFC #420 row ``h3_qk_rmsnorm_d128``).

``attn.norm_q`` and ``attn.norm_k`` are ``nn.RMSNorm(128, eps=1e-5)`` over the last dimension of
the per-head queries and keys, ``(B, S, H, D)`` with ``H = 56`` heads and ``D = 128``, each with
one affine weight of 128 entries shared by every head::

    y = x * rsqrt(mean(x ** 2, dim=-1) + eps) * weight

The heads are sharded by the tensor-parallel grid, so ``H`` may be a divisor of 56
(``56 / tp`` for ``tp`` in 1, 2, 4, 8): the arithmetic is row-local, so a head shard is the same
slice of the full result.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_engine.reference.minimax_h3.rmsnorm import validate_h3_rmsnorm

H3_QK_HEADS = 56
H3_QK_HEAD_DIM = 128
H3_QK_NORM_EPS = 1e-5


def validate_h3_qk_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    """Shapes, dtypes and the pinned ``eps`` of the per-head norm.

    ``x`` is ``(..., H, 128)`` with ``H`` a divisor of 56 (the full heads, or one TP rank's),
    ``weight`` is ``(128,)`` in ``x``'s dtype on ``x``'s device.
    """

    validate_h3_rmsnorm(x, weight, eps)
    if x.dim() < 3:
        raise ValueError(f"x must be (..., heads, {H3_QK_HEAD_DIM}), got {tuple(x.shape)}")
    if x.shape[-1] != H3_QK_HEAD_DIM:
        raise ValueError(f"the head dimension must be {H3_QK_HEAD_DIM}, got {x.shape[-1]}")
    heads = x.shape[-2]
    if heads < 1 or H3_QK_HEADS % heads:
        raise ValueError(f"heads must divide {H3_QK_HEADS}, got {heads}")
    if eps != H3_QK_NORM_EPS:
        raise ValueError(f"eps is pinned to {H3_QK_NORM_EPS}, got {eps}")


class NativeH3QKRMSNormOp:
    """PyTorch reference.

    ``forward`` is the provider path (``F.rms_norm``, which ``nn.RMSNorm`` calls);
    ``forward_fp32`` is the golden: the same formula in FP64, returned in FP32.
    """

    op_class = "reduction"

    def __call__(self, x, weight, eps: float = H3_QK_NORM_EPS):
        return self.forward(x, weight, eps)

    def forward(self, x, weight, eps: float = H3_QK_NORM_EPS) -> torch.Tensor:
        validate_h3_qk_rmsnorm(x, weight, eps)
        return F.rms_norm(x, (x.shape[-1],), weight, eps)

    def forward_fp32(self, x, weight, eps: float = H3_QK_NORM_EPS) -> torch.Tensor:
        validate_h3_qk_rmsnorm(x, weight, eps)
        x64 = x.double()
        rstd = torch.rsqrt(x64.square().mean(dim=-1, keepdim=True) + eps)
        return (x64 * rstd * weight.double()).float()
