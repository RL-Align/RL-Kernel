# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen-Image latent pack/unpack shape contract and PyTorch reference.

The reference is reshape/permute only, with no dtype conversion, so every
backend must reproduce it bit for bit.
"""

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


class NativeLatentPackOp(torch.nn.Module):
    op_class = "permutation"

    def forward(self, x, batch_size, num_channels_latents, height, width):
        b, c, h, w = dimensions(x, batch_size, num_channels_latents, height, width)
        return (
            x.view(b, c, h // 2, 2, w // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b, h // 2 * (w // 2), c * 4)
        )


class NativeLatentUnpackOp(torch.nn.Module):
    op_class = "permutation"

    def forward(self, x, height, width, vae_scale_factor):
        b, c, h, w = unpack_dimensions(x, height, width, vae_scale_factor)
        # The singleton dimension is the frame axis expected by the Qwen-Image VAE.
        return x.view(b, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, 1, h, w)
