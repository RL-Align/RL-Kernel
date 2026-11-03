# SPDX-License-Identifier: Apache-2.0
"""Task 53-55 output boundary; input-gradient branch reduction remains task 56."""

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.rocm.moe.output_combine import RocmOutputCombine
from rl_engine.contracts.operators.fixed_output_combine import OutputCombineTrace


class _OutputCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prepared, expert_output, shared_output, debug):
        ctx.save_for_backward(prepared._rows, prepared._tokens)
        ctx.dtype = expert_output.dtype
        out, steps, before = RocmOutputCombine.forward(
            prepared, expert_output, shared_output, debug
        )
        if debug:
            ctx.mark_non_differentiable(steps, before)
        return out, steps, before

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, _steps, _before):
        _, tokens = ctx.saved_tensors
        dx = RocmOutputCombine.backward(dy, tokens, ctx.dtype) if ctx.needs_input_grad[1] else None
        shared_grad = dy.to(ctx.dtype) if ctx.needs_input_grad[2] else None
        return None, dx, shared_grad, None


def output_combine(prepared, expert_output, shared_output=None, *, mapping_run_id, debug=False):
    """Add slots 0..5 in FP32, add shared once, then cast once to the output dtype.

    Prepare outside capture. This execution entry checks metadata and tensor
    layout without host-reading values. For Graph replay, update value buffers
    only; a different mapping/run requires new preparation and capture.
    """
    prepared.validate_execution(expert_output, shared_output, mapping_run_id)
    out, steps, before = _OutputCombine.apply(prepared, expert_output, shared_output, debug)
    return OutputCombineTrace(out, steps, before) if debug else out
