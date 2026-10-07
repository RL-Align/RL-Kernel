# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CUDA H3 AdaLN row gather (RFC #420 ``adaln_row_gather``)."""

from __future__ import annotations

import torch

from rl_engine.kernels.ops.backward_runtime import record_backward
from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.pytorch.h3 import H3_ADALN_CHUNKS, H3_MODALITY_NUM
from rl_engine.kernels.ops.pytorch.h3.adaln_row_gather import (
    h3_adaln_indices,
    validate_h3_adaln_row_gather,
)

KERNEL_ID = "rl_engine._C.h3_adaln_row_gather_forward"
BACKWARD_IMPL = "stable_sorted_segment_tiles_fp32_ordered_fold"
BACKWARD_KERNEL_ID = "torch.argsort[stable]+rl_engine._C.h3_adaln_row_gather_backward"
# Positions per backward tile; part of the fixed summation order.
BACKWARD_TILE = 256


def adaln_row_gather_available() -> bool:
    return bool(
        _EXT_AVAILABLE
        and hasattr(_C, "h3_adaln_row_gather_forward")
        and hasattr(_C, "h3_adaln_row_gather_backward")
    )


def _segment_tiles(index: torch.Tensor, num_rows: int):
    """Sorted positions and fixed-size tiles per row segment (all integer ops)."""

    sorted_pos = torch.argsort(index, stable=True)
    counts = torch.bincount(index, minlength=num_rows)
    seg_start = torch.cumsum(counts, 0) - counts
    tiles_per_seg = (counts + BACKWARD_TILE - 1) // BACKWARD_TILE
    seg_first_tile = torch.zeros(num_rows + 1, dtype=torch.int64, device=index.device)
    seg_first_tile[1:] = torch.cumsum(tiles_per_seg, 0)
    tile_seg = torch.repeat_interleave(torch.arange(num_rows, device=index.device), tiles_per_seg)
    tile_local = torch.arange(tile_seg.numel(), device=index.device) - seg_first_tile[tile_seg]
    tile_begin = seg_start[tile_seg] + tile_local * BACKWARD_TILE
    tile_end = torch.minimum(tile_begin + BACKWARD_TILE, seg_start[tile_seg] + counts[tile_seg])
    return sorted_pos, tile_begin.contiguous(), tile_end.contiguous(), seg_first_tile


class _H3AdaLNRowGatherCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, timestep_indices, token_tags):
        out = _C.h3_adaln_row_gather_forward(
            rows, timestep_indices, token_tags, H3_ADALN_CHUNKS, H3_MODALITY_NUM
        )
        ctx.save_for_backward(timestep_indices, token_tags)
        ctx.rows_meta = (rows.shape[0], rows.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        timestep_indices, token_tags = ctx.saved_tensors
        num_rows, rows_dtype = ctx.rows_meta
        index = h3_adaln_indices(timestep_indices, token_tags)
        tiles = _segment_tiles(index, num_rows)
        grad_rows = _C.h3_adaln_row_gather_backward(grad_out.contiguous(), *tiles, rows_dtype)
        record_backward(
            "adaln_row_gather", kernel_id=BACKWARD_KERNEL_ID, impl=BACKWARD_IMPL, family="cuda"
        )
        return grad_rows, None, None


class H3AdaLNRowGatherCudaOp:
    """CUDA candidate: one launch gathers all six modulation tensors.

    The forward is a byte copy (bitwise equal to six ``index_select`` calls).
    The backward replaces ``index_select``'s atomic BF16 scatter-add with a
    deterministic FP32 segmented sum in stable sorted order, cast once.
    """

    op_class = "reduction"
    kernel_id = KERNEL_ID
    backward_impl = BACKWARD_IMPL

    def __init__(self) -> None:
        if not adaln_row_gather_available():
            raise RuntimeError(
                "rl_engine._C lacks h3_adaln_row_gather_*; rebuild with "
                "csrc/cuda/h3/adaln_row_gather.cu"
            )

    def __call__(self, rows, timestep_indices, token_tags):
        return self.forward(rows, timestep_indices, token_tags)

    def forward(self, rows, timestep_indices, token_tags, *, check_range: bool = True):
        validate_h3_adaln_row_gather(rows, timestep_indices, token_tags, check_range=check_range)
        if not rows.is_cuda:
            raise ValueError("H3AdaLNRowGatherCudaOp needs CUDA tensors")
        if rows.stride(1) != 1:
            rows = rows.contiguous()
        packed = _H3AdaLNRowGatherCuda.apply(
            rows, timestep_indices.contiguous(), token_tags.contiguous()
        )
        return tuple(packed.unbind(0))

    def gather_chunks(self, chunks, timestep_indices, token_tags, **kwargs):
        """Drop-in for diffusers' six ``(3T, H)`` tensors: concatenate, then gather."""

        if len(chunks) != H3_ADALN_CHUNKS:
            raise ValueError(f"expected {H3_ADALN_CHUNKS} modulation tensors, got {len(chunks)}")
        return self.forward(torch.cat(list(chunks), dim=1), timestep_indices, token_tags, **kwargs)

    def forward_fp32(self, rows, timestep_indices, token_tags):
        return tuple(out.float() for out in self.forward(rows, timestep_indices, token_tags))
