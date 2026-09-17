# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared Qwen-Image shape contract and inverse-permutation autograd."""

import operator

import torch


def dimensions(x, batch_size, num_channels_latents, height, width, *, unpack=False):
    b, c, h, w = map(operator.index, (batch_size, num_channels_latents, height, width))
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("latent input must be fp32, fp16, or bf16")
    if not x.is_contiguous():
        raise ValueError("latent input must be contiguous")
    if not (0 <= b <= 65535 and c > 0 and h > 0 and w > 0 and h % 2 == w % 2 == 0):
        raise ValueError("expected B in [0,65535], positive C and positive even H,W")
    if c > (2**31 - 1) // 4 or max(h, w, h // 2 * (w // 2)) > 2**31 - 1:
        raise ValueError("latent dimensions exceed indexing limits")
    if h // 2 > 65535:
        raise ValueError("latent height exceeds launch limits")
    shapes = (
        ((b, h // 2 * (w // 2), 4 * c),)
        if unpack
        else ((b, c, h, w), (b, c, 1, h, w), (b, 1, c, h, w))
    )
    if tuple(x.shape) not in shapes:
        raise ValueError(f"latent shape {tuple(x.shape)} does not match {shapes}")
    return b, c, h, w


def unpack_dimensions(x, height, width, vae_scale_factor):
    scale = operator.index(vae_scale_factor)
    if scale <= 0 or x.ndim != 3 or x.shape[-1] % 4:
        raise ValueError("expected positive VAE scale and packed [B,P,4*C] input")
    h, w = (2 * (int(size) // (2 * scale)) for size in (height, width))
    return dimensions(x, x.shape[0], x.shape[-1] // 4, h, w, unpack=True)


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
