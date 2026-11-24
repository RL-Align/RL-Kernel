# SPDX-License-Identifier: Apache-2.0
"""INK-07: Inkling 2 Mini norm and clipping cast boundaries."""

import math

import torch

from rl_engine.ops.norm.fixed_tree import fixed_sum_last

QK_THRESHOLD = 22.627416997969522
REL_THRESHOLD = 8.0


def inkling_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP32 input/affine arithmetic followed by one declared output cast.

    Accelerator calls reuse the existing Triton RMSNorm forward/backward;
    CPU is a mathematical reference. No GPU fallback is permitted.
    """
    if x.ndim < 2 or x.shape[-1] == 0 or x.numel() == 0:
        raise ValueError("expected nonempty rows")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if x.dtype not in (torch.float32, torch.bfloat16) or out_dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        raise TypeError("expected BF16 or FP32 input and output")
    if weight is not None:
        if weight.shape != (x.shape[-1],) or weight.device != x.device:
            raise ValueError("weight must be [hidden] on the input device")
        if weight.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("weight must be BF16 or FP32")
    rows = x.float().reshape(-1, x.shape[-1]).contiguous()
    if x.is_cuda:
        from rl_engine.backends.shared.triton.norm.rmsnorm import rmsnorm_triton

        w = rows.new_ones(rows.shape[-1]) if weight is None else weight.float().contiguous()
        y = rmsnorm_triton(rows, w, eps)
    elif x.device.type == "cpu":
        y = rows * torch.rsqrt(fixed_sum_last(rows.square()) / rows.shape[-1] + eps)
        if weight is not None:
            y = y * weight.float()
    else:
        raise RuntimeError("only CPU reference and CUDA/ROCm Triton paths are supported")
    return y.reshape_as(x).to(out_dtype)


def inkling_l2_clip(
    x: torch.Tensor,
    *,
    threshold: float,
    scale: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale in FP32, clip per head, cast, then optionally scale and cast again.

    The portable tree runs on CPU and CUDA/ROCm. It has a finite zero-norm
    backward; distributed scale gradients remain outside this contract.
    """
    if x.ndim != 3 or min(x.shape) == 0:
        raise ValueError("expected nonempty [tokens, heads, head_dim] input")
    if x.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("expected BF16 or FP32 input")
    if x.device.type not in ("cpu", "cuda"):
        raise RuntimeError("expected CPU or CUDA/ROCm input")
    if not math.isfinite(threshold) or threshold <= 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("threshold and eps must be finite and positive")
    value = x.float()
    if scale is not None:
        if scale.shape != x.shape[1:] or scale.device != x.device:
            raise ValueError("scale must be [heads, head_dim] on the input device")
        if scale.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("scale must be BF16 or FP32")
        value = value * scale.float()
    square_sum = fixed_sum_last(value.square())
    # Avoid sqrt(0)'s singular backward, including when clamp selects 1.
    nonzero = square_sum > 0
    safe = torch.where(nonzero, square_sum, torch.ones_like(square_sum))
    norm = torch.where(nonzero, safe.sqrt(), square_sum)
    clipped = (value * (threshold / (norm + eps)).clamp(max=1.0)).to(x.dtype)
    if output_scale is not None:
        if output_scale.shape != (x.shape[0],) or output_scale.device != x.device:
            raise ValueError("output scale must be [tokens] on the input device")
        if output_scale.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("output scale must be BF16 or FP32")
        clipped = (clipped.float() * output_scale.float().view(-1, 1, 1)).to(x.dtype)
    return clipped
