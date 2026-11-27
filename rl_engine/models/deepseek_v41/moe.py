# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4.1 MoE: DSv4's (``Dsv4MoE``: router, MXFP4 routed experts, EP) with 384 experts, no
hash layers, and a BF16 shared expert (D12)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.matmul.bf16_linear import Bf16Linear

SWIGLU_LIMIT = 10.0


class Bf16SharedExpert:
    """``fc2(clamp_swiglu(fc1(x)))`` on BF16 GEMMs (the checkpoint's MXFP8 weights
    dequantized, D12). ``fc1`` = ``[w1 (gate); w3 (up)]`` ``[2F, H]``, ``fc2`` ``[H, F]``.
    The SwiGLU is the DSv4 kernel's BF16 output (its FP8 copy is unused)."""

    backward = "bf16"

    def __init__(self, fc1: Bf16Linear, fc2: Bf16Linear, swiglu_limit: float = SWIGLU_LIMIT):
        from rl_engine.kernels.ops.cuda.activation import silu_and_mul_clamp as swiglu

        self.fc1, self.fc2 = fc1, fc2
        self.swiglu_limit = float(swiglu_limit)
        self._swiglu = swiglu
        swiglu._extension()  # fail closed

    def forward(self, x: Tensor, _weights: Any = None) -> tuple[Tensor, dict]:
        z = self.fc1(x.contiguous())
        if not z.shape[0]:
            return torch.empty(0, self.fc2.out_features, device=x.device, dtype=torch.bfloat16), {
                "z": z
            }
        _, _, h = self._swiglu.silu_and_mul_clamp_quant_cuda(z, self.swiglu_limit, return_h=True)
        return self.fc2(h), {"z": z}

    def backward_dx(self, dy: Tensor, saved: dict) -> Tensor:
        dy = dy.to(torch.bfloat16).contiguous()
        if not dy.shape[0]:
            return torch.empty(0, self.fc1.in_features, device=dy.device, dtype=torch.float32)
        dh = self.fc2.backward_dx(dy)
        dz = self._swiglu.silu_and_mul_clamp_backward_cuda(dh, saved["z"], self.swiglu_limit)
        return self.fc1.backward_dx(dz).float()
