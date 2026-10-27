# SPDX-License-Identifier: Apache-2.0
"""PyTorch diagnostic math; not a bitwise oracle for the ROCm exp implementation."""

import torch
from torch import Tensor

from rl_engine.contracts.operators.clamp_swiglu import (
    GATE_MAX,
    UP_MAX,
    UP_MIN,
    ClampSwiGLUState,
    validate_backward,
    validate_inputs,
)


class ClampSwiGLUReference:
    @staticmethod
    @torch.no_grad()
    def forward(gate: Tensor, up: Tensor, p_s: Tensor | None) -> tuple[Tensor, ClampSwiGLUState]:
        validate_inputs(gate, up, p_s)
        g, u = gate.float(), up.float()
        if p_s is not None:
            g = g.clamp(max=GATE_MAX)
            u = u.clamp(min=UP_MIN, max=UP_MAX)
        sig = torch.sigmoid(g)
        silu = g * sig
        unweighted = silu * u
        weighted = unweighted if p_s is None else unweighted * p_s[:, None]
        return weighted.bfloat16(), ClampSwiGLUState(
            sig.contiguous(), silu.contiguous(), unweighted.contiguous()
        )

    @staticmethod
    @torch.no_grad()
    def backward(
        dh: Tensor, gate: Tensor, up: Tensor, p_s: Tensor | None, state: ClampSwiGLUState
    ) -> tuple[Tensor, Tensor]:
        validate_backward(dh, gate, up, p_s, state)
        g, u = gate.float(), up.float()
        dy = dh.float()
        if p_s is not None:
            g = g.clamp(max=GATE_MAX)
            u = u.clamp(min=UP_MIN, max=UP_MAX)
            dy = dy * p_s[:, None]
        dsilu = state.sigmoid * (1.0 + g * (1.0 - state.sigmoid))
        dg = (dy * u) * dsilu
        du = dy * state.silu
        if p_s is not None:
            # The specified subgradient is zero AT each clamp boundary.
            dg = dg * (gate.float() < GATE_MAX)
            du = du * ((up.float() > UP_MIN) & (up.float() < UP_MAX))
        return dg, du
