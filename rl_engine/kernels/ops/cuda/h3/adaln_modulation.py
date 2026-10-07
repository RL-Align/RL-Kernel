# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Fused H3 AdaLN modulation: ``adaln_projection_3mod`` followed by ``adaln_row_gather``.

The forward is exactly the two RFC #420 ops back to back (same kernels, same
bits). The difference is the backward. With two separate ops, autograd hands
the gather's table gradient to the projection in the table's dtype (BF16),
which rounds a long FP32 segment sum once more before it reaches the
projection weights and ``temb``. Here the segment sum stays FP32 into the
projection backward, so the parameter gradients carry no extra BF16 rounding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C
from rl_engine.kernels.ops.cuda.h3.adaln_row_gather import (
    _segment_tiles,
    adaln_row_gather_available,
)
from rl_engine.kernels.ops.cuda.h3.det_linear import (
    det_linear_available,
    det_linear_backward_input,
    det_linear_backward_weight,
    det_linear_forward,
    silu_backward_fp32,
)
from rl_engine.kernels.ops.pytorch.h3 import H3_ADALN_CHUNKS, H3_MODALITY_NUM
from rl_engine.kernels.ops.pytorch.h3.adaln_projection import validate_h3_adaln_projection
from rl_engine.kernels.ops.pytorch.h3.adaln_row_gather import (
    h3_adaln_indices,
    validate_h3_adaln_row_gather,
)

KERNEL_ID = (
    "torch.silu[fp32]+cast[weight_dtype]+rl_engine._C.h3_det_linear_forward"
    "+rl_engine._C.h3_adaln_row_gather_forward"
)
BACKWARD_IMPL = "fp32_segment_sum_into_h3_det_linear_backward"
BACKWARD_KERNEL_ID = (
    "rl_engine._C.h3_adaln_row_gather_backward[fp32]"
    "+rl_engine._C.h3_det_linear_backward_weight"
    "+rl_engine._C.h3_det_linear_backward_input"
    "+rl_engine.kernels.ops.cuda.h3.det_linear.silu_backward_fp32"
)


class _H3AdaLNModulationCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, temb, weight, bias, timestep_indices, token_tags, hidden):
        temb = temb.contiguous()
        act = F.silu(temb).to(weight.dtype)
        (table,) = det_linear_forward(act, weight, bias)
        rows = table.view(-1, H3_ADALN_CHUNKS * hidden)
        packed = _C.h3_adaln_row_gather_forward(
            rows, timestep_indices, token_tags, H3_ADALN_CHUNKS, H3_MODALITY_NUM
        )
        ctx.save_for_backward(temb, act, weight, timestep_indices, token_tags)
        ctx.table_shape = table.shape
        return packed

    @staticmethod
    def backward(ctx, grad_packed):
        temb, act, weight, timestep_indices, token_tags = ctx.saved_tensors
        need_temb, need_weight, need_bias = ctx.needs_input_grad[:3]
        num_timesteps, width = ctx.table_shape
        index = h3_adaln_indices(timestep_indices, token_tags)
        tiles = _segment_tiles(index, num_timesteps * H3_MODALITY_NUM)
        d_rows = _C.h3_adaln_row_gather_backward(grad_packed.contiguous(), *tiles, torch.float32)
        d_table = d_rows.view(num_timesteps, width)
        d_temb = dw = db = None
        if need_weight or need_bias:
            dw, db = det_linear_backward_weight(d_table, act, weight.dtype)
        if need_temb:
            d_temb = silu_backward_fp32(
                det_linear_backward_input(d_table, weight, torch.float32), temb
            )
        record_backward(
            "adaln_modulation_3mod",
            kernel_id=BACKWARD_KERNEL_ID,
            impl=BACKWARD_IMPL,
            family="cuda",
        )
        return (
            d_temb,
            dw if need_weight else None,
            db if need_bias else None,
            None,
            None,
            None,
        )


class H3AdaLNModulationCudaOp:
    """``adaln_projection_3mod`` + ``adaln_row_gather`` with an FP32 table gradient.

    Returns the six ``(S, H)`` modulation tensors for every packed position,
    bitwise equal to running the two ops separately.
    """

    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL

    def __init__(self) -> None:
        if not (det_linear_available() and adaln_row_gather_available()):
            raise RuntimeError("rl_engine._C lacks the h3_det_linear_* / h3_adaln_row_gather_* ops")

    def __call__(self, temb, weight, bias, timestep_indices, token_tags, *, check_range=True):
        hidden = validate_h3_adaln_projection(temb, weight, bias)
        if not temb.is_cuda:
            raise ValueError("H3AdaLNModulationCudaOp needs CUDA tensors")
        _validate_indices(timestep_indices, token_tags, temb.shape[0], check_range)
        packed = _H3AdaLNModulationCuda.apply(
            temb,
            weight,
            bias,
            timestep_indices.contiguous(),
            token_tags.contiguous(),
            hidden,
        )
        return tuple(packed.unbind(0))


def _validate_indices(timestep_indices, token_tags, num_timesteps, check_range):
    # Reuse the gather's checks against a stand-in of the right row count.
    stand_in = torch.empty(
        (num_timesteps * H3_MODALITY_NUM, H3_ADALN_CHUNKS), device=timestep_indices.device
    )
    validate_h3_adaln_row_gather(stand_in, timestep_indices, token_tags, check_range=check_range)
