# SPDX-License-Identifier: Apache-2.0
"""Production fused GPT-J inverse rotation with explicit transpose VJP."""

import torch
import torch.nn.functional as F


class _Inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tail, positions, cache):
        from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj

        t, b, h, d = tail.shape
        if d != 64 or cache.dtype != torch.float32 or tail.dtype != torch.bfloat16:
            raise ValueError("inverse RoPE requires original BF16 tail and FP32 table")
        padded = F.pad(tail, (448, 0)).reshape(t * b, h, 512).contiguous()
        rotated = _fused_inverse_rope_gptj(padded, positions.reshape(-1), cache, 64)
        ctx.save_for_backward(positions, cache)
        return rotated[..., 448:].reshape(t, b, h, 64).contiguous()

    @staticmethod
    def backward(ctx, dy):
        positions, cache = ctx.saved_tensors
        cs = cache.index_select(0, positions.reshape(-1)).reshape(*positions.shape, 1, 64)
        cos, sin = cs.chunk(2, -1)
        even, odd = dy.float()[..., 0::2], dy.float()[..., 1::2]
        dx = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)
        return dx.to(dy.dtype), None, None


def inverse_rope_tail(tail, positions, cache):
    return _Inverse.apply(tail, positions, cache)
