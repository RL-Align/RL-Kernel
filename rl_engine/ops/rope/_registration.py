# SPDX-License-Identifier: Apache-2.0
"""Dispatcher, fake and recursive input-VJP registration for the RoPE profile."""

import torch

from rl_engine.ops.autograd.backward_runtime import record_backward
from rl_engine.ops.rope.multi_axis_rope import validate_inputs


def register_rotation(name, implementation, *, kernel_id, family):
    @torch.library.custom_op(f"rl_kernel::{name}", mutates_args=(), device_types="cuda")
    def rotation(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, transpose: bool
    ) -> torch.Tensor:
        validate_inputs(x, cos, sin)
        result = implementation(x, cos, sin, transpose)
        if transpose:
            # Keep telemetry inside the opaque custom op; fake/AOT tracing
            # must not execute this Python lock. CUDA Graph replay runs only
            # captured kernels, so execution_count is not a replay counter.
            record_backward(
                "multi_axis_rope",
                kernel_id=kernel_id,
                impl="interleaved_fp32_transpose",
                family=family,
            )
        return result

    @rotation.register_fake
    def fake(x, cos, sin, transpose):
        validate_inputs(x, cos, sin)
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)

    def setup_context(ctx, inputs, output):
        _, cos, sin, transpose = inputs
        ctx.save_for_backward(cos, sin)
        ctx.transpose = transpose

    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        return rotation(grad, cos, sin, not ctx.transpose), None, None, None

    rotation.register_autograd(backward, setup_context=setup_context)
    return rotation
