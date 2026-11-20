# SPDX-License-Identifier: Apache-2.0
"""Exact FP8/E8M0 Engram row dequantization for a prepared resident row set."""

import torch
import triton
import triton.language as tl

from rl_engine.kernels.ops.triton.moe.fused_mlp import _decode_e4m3, _e8m0_scale


@triton.jit
def _lookup(POS, W, S, Y, D: tl.constexpr):
    t = tl.program_id(0)
    d = tl.program_id(1) * 256 + tl.arange(0, 256)
    row = tl.load(POS + t)
    w = _decode_e4m3(tl.load(W + row * D + d, d < D, other=0))
    scale = _e8m0_scale(tl.load(S + row * (D // 32) + d // 32, d < D, other=127))
    tl.store(Y + t * D + d, w * scale, d < D)


class ResidentEngramRows(torch.nn.Module):
    """Frozen exact checkpoint rows; a missing row is an error, never row zero.

    A caller may prepare only rows touched by its real input n-grams. This is an
    explicit checkpoint row-loading boundary, not a replacement smaller table.
    """

    def __init__(self, original_row_ids, weight, scale):
        super().__init__()
        if weight.ndim != 2 or not weight.shape[1] or weight.shape[1] % 32:
            raise ValueError("expected FP8 rows with block32-aligned width")
        if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu:
            raise TypeError("expected original E4M3/E8M0 embedding tensors")
        if scale.shape != (weight.shape[0], weight.shape[1] // 32):
            raise ValueError("embedding scale shape mismatch")
        if original_row_ids.dtype != torch.int64 or original_row_ids.shape != (weight.shape[0],):
            raise ValueError("expected one original int64 ID per resident row")
        if not torch.version.hip or weight.device.type != "cuda":
            raise RuntimeError("resident Engram row lookup requires AMD ROCm")
        if scale.device != weight.device or original_row_ids.device != weight.device:
            raise ValueError("all embedding tensors must share the GPU")
        if original_row_ids.numel() == 0 or bool(
            (original_row_ids[1:] <= original_row_ids[:-1]).any()
        ):
            raise ValueError("resident IDs must be nonempty, sorted and unique")
        self.register_buffer("row_ids", original_row_ids.contiguous())
        self.register_buffer("codes", weight.view(torch.uint8).contiguous())
        self.register_buffer("scales", scale.view(torch.uint8).contiguous())

    def forward(self, indices):
        if indices.dtype != torch.int64 or indices.device != self.codes.device:
            raise ValueError("expected int64 IDs on the embedding GPU")
        flat = indices.contiguous().reshape(-1)
        pos = torch.searchsorted(self.row_ids, flat)
        valid = pos < self.row_ids.numel()
        safe = pos.clamp_max(self.row_ids.numel() - 1)
        if not bool((valid & (self.row_ids[safe] == flat)).all()):
            raise KeyError("Engram input references a checkpoint row that is not resident")
        d = self.codes.shape[1]
        y = torch.empty((*indices.shape, d), device=indices.device, dtype=torch.bfloat16)
        with torch.cuda.device(indices.device):
            if flat.numel():
                _lookup[(flat.numel(), triton.cdiv(d, 256))](
                    pos, self.codes, self.scales, y, d, enable_fp_fusion=False
                )
        return y
