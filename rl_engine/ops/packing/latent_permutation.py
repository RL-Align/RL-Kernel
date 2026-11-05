# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Inverse-permutation autograd shared by the Qwen-Image latent pack/unpack backends."""

import torch

from rl_engine.reference.model_specific.qwenimage.latent_pack_unpack import (
    dimensions,
    unpack_dimensions,
)


class _LatentPermutation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dims, unpack, kernel):
        ctx.dims, ctx.unpack, ctx.kernel, ctx.shape = dims, unpack, kernel, x.shape
        return kernel(x, *dims, unpack)

    @staticmethod
    def backward(ctx, grad):
        # Re-enter Function.apply so higher-order gradients are also permutations.
        result = _LatentPermutation.apply(grad.contiguous(), ctx.dims, not ctx.unpack, ctx.kernel)
        return result.reshape(ctx.shape), None, None, None


class LatentKernelOp:
    """Device op whose ``kernel(x, B, C, H, W, unpack)`` performs the permutation."""

    unpack = False
    op_class = "permutation"

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        if x.device.type != "cuda":
            raise ValueError("GPU latent backend requires a CUDA tensor")
        dims = (
            unpack_dimensions(x, *args, **kwargs) if self.unpack else dimensions(x, *args, **kwargs)
        )
        return _LatentPermutation.apply(x, dims, self.unpack, self.kernel)
