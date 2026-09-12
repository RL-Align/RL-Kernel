# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen-Image reference: reshape/permute only, with no dtype conversion."""

import torch

from rl_engine.kernels.ops.latent_layout import dimensions, unpack_dimensions


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
        return x.view(b, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, 1, h, w)  # 1 is frame, just to adapt to the interface
