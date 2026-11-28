# SPDX-License-Identifier: Apache-2.0
"""Independent eager PyTorch reference; use CPU for numerical validation."""

import torch

from rl_engine.ops.norm.latent_normalize import validate_latents


class NativeLatentNormalizeOp(torch.nn.Module):
    op_class = "elementwise"
    inverse = False

    def forward_fp32(self, x, mean, std_inv):
        validate_latents(x, mean, std_inv)
        shape = (1, 16) + (1,) * (x.ndim - 2)
        mean, std_inv = mean.reshape(shape), std_inv.reshape(shape)
        # Separate eager operations preserve each FP32 rounding boundary.
        if self.inverse:
            return x.float() / std_inv + mean
        return (x.float() - mean) * std_inv

    def forward(self, x, mean, std_inv):
        return self.forward_fp32(x, mean, std_inv).to(x.dtype)


class NativeLatentDenormalizeOp(NativeLatentNormalizeOp):
    inverse = True
