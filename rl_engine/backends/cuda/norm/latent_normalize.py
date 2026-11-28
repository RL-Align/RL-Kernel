# SPDX-License-Identifier: Apache-2.0
"""Strict CUDA latent normalization and inverse, with input gradients."""

import torch
from torch.autograd.function import once_differentiable

from rl_engine.backends.extension import _C
from rl_engine.ops.norm.latent_normalize import CONTRACT_ID, validate_latents


class _LatentAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mean, std_inv, inverse, block_size):
        out = _C.latent_normalize_forward(x, mean, std_inv, inverse, block_size)
        ctx.save_for_backward(std_inv)
        ctx.inverse, ctx.block_size = inverse, block_size
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        (std_inv,) = ctx.saved_tensors
        # Expanded upstream gradients (e.g. output.sum()) are valid autograd inputs.
        dx = _C.latent_normalize_backward(
            grad_out.contiguous(), std_inv, ctx.inverse, ctx.block_size
        )
        return dx, None, None, None, None


class LatentNormalizeCudaOp:
    op_class = "elementwise"
    inverse = False

    def __init__(self, block_size=256):
        if torch.version.hip is not None:
            raise RuntimeError("latent_normalize CUDA backend does not support ROCm")
        if _C is None or not all(
            hasattr(_C, name) for name in ("latent_normalize_forward", "latent_normalize_backward")
        ):
            raise RuntimeError("Rebuild rl_engine._C with the CUDA latent_normalize kernels")
        if block_size not in (128, 256, 512):
            raise ValueError("block_size must be 128, 256, or 512")
        self.block_size = block_size
        self.last_trace = None

    def __call__(self, x, mean, std_inv):
        return self.forward(x, mean, std_inv)

    def forward(self, x, mean, std_inv):
        self.last_trace = None
        if x.device.type != "cuda":
            raise ValueError("CUDA latent_normalize requires CUDA tensors")
        validate_latents(x, mean, std_inv)
        if torch.is_grad_enabled() and x.requires_grad:
            out = _LatentAffine.apply(x, mean, std_inv, self.inverse, self.block_size)
        else:
            out = _C.latent_normalize_forward(x, mean, std_inv, self.inverse, self.block_size)
        self.last_trace = {
            "contract": CONTRACT_ID,
            "requested_backend": "cuda",
            "actual_backend": "cuda",
            "kernel": "rl_engine._C.latent_normalize_forward",
            "backward_kernel": "rl_engine._C.latent_normalize_backward",
            "inverse": self.inverse,
            "arithmetic": "separate_round_to_nearest_fp32; no_fma; no_reduction",
            "execution_dtype": str(x.dtype),
            "accumulation_dtype": "torch.float32",
            "output_dtype": str(out.dtype),
            "block_size": self.block_size,
            "fallback": False,
        }
        return out


class LatentDenormalizeCudaOp(LatentNormalizeCudaOp):
    inverse = True
