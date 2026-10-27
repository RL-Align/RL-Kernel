# SPDX-License-Identifier: Apache-2.0
"""Task 46 public activation and gate/up autograd, before FC2 quantization."""

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from rl_engine.backends.rocm.moe.clamp_swiglu import RocmClampSwiGLUWeighted
from rl_engine.contracts.operators.clamp_swiglu import ClampSwiGLUState, split_packed


class _ClampSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, p_s, recompute):
        h, state = RocmClampSwiGLUWeighted.forward(gate, up, p_s)
        ctx.recompute = recompute
        if recompute:
            ctx.save_for_backward(gate, up, p_s)
        else:
            ctx.save_for_backward(gate, up, p_s, state.sigmoid, state.silu, state.unweighted)
        return h

    @staticmethod
    @once_differentiable
    def backward(ctx, dh):
        gate, up, p_s, *saved = ctx.saved_tensors
        if ctx.recompute:
            _, state = RocmClampSwiGLUWeighted.forward(gate, up, p_s)
        else:
            state = ClampSwiGLUState(*saved)
        dg, du = RocmClampSwiGLUWeighted.backward(dh, gate, up, p_s, state)
        return (
            dg.to(gate.dtype) if ctx.needs_input_grad[0] else None,
            du.to(up.dtype) if ctx.needs_input_grad[1] else None,
            None,
            None,
        )


def clamp_swiglu_weighted(
    gate: Tensor, up: Tensor, p_s: Tensor | None, *, recompute: bool = False
) -> Tensor:
    """Clamp -> SiLU * up -> route weight -> BF16, with FP32 intermediates.

    gate/up are BF16 or FP32 [M, F]; p_s is FP32 [M]. p_s=None selects the
    unweighted shared-expert variant WITHOUT clamping. Gradients at the clamp
    boundaries are zero. BF16 casts use the usual straight-through derivative.

    Trainable p_s requires the separate task 49 dp_s implementation and is
    rejected here while autograd is enabled. The low-level backend exposes
    saved FP32 intermediates for that integration. This function does not
    quantize, run FC2, or provide higher-order gradients.
    """
    if p_s is not None and p_s.requires_grad and torch.is_grad_enabled():
        raise NotImplementedError("trainable p_s requires task 49 route-weight backward")
    return _ClampSwiGLU.apply(gate, up, p_s, recompute)


def clamp_swiglu_weighted_packed(
    gate_up: Tensor, p_s: Tensor | None, *, recompute: bool = False
) -> Tensor:
    """Packed [M, 2*F] input: the gate half comes before the up half."""
    gate, up = split_packed(gate_up)
    return clamp_swiglu_weighted(gate, up, p_s, recompute=recompute)
