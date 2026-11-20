# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Serial FP32 reference for P5 LoRA boundaries; not a bitwise GPU oracle.

The reference deliberately accumulates one product at a time without FMA.
It does not share the Triton tile reduction. Compare numerical accuracy to
this reference separately from same-device, same-profile byte invariance.
"""

from __future__ import annotations

import torch

from rl_engine.contracts.operators.lora import validate_lora_backward, validate_lora_inputs


def _serial_mm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((lhs.shape[0], rhs.shape[1]), device=lhs.device, dtype=torch.float32)
    for k in range(lhs.shape[1]):
        out = out + lhs[:, k].float()[:, None] * rhs[k, :].float()[None, :]
    return out


class ReferenceLoRADelta:
    numeric_profile = "lora-reference-fp32-serial-v1"

    @torch.no_grad()
    def _project(self, x, a):
        return _serial_mm(x, a.t()).to(torch.bfloat16)

    @torch.no_grad()
    def forward(self, x, a, b, alpha):
        alpha = validate_lora_inputs(x, a, b, alpha)
        u = self._project(x, a)
        return _serial_mm(u, b.t()) * alpha, u

    @torch.no_grad()
    def backward(self, dy, x, a, b, alpha, u):
        alpha = validate_lora_inputs(x, a, b, alpha)
        validate_lora_backward(dy, x, a, b, u)
        dv = (dy.float() * alpha).to(torch.bfloat16)
        du = _serial_mm(dv, b).to(torch.bfloat16)
        db = _serial_mm(dv.t(), u)
        da = _serial_mm(du.t(), x)
        dx = _serial_mm(du, a)
        return dx, da, db
