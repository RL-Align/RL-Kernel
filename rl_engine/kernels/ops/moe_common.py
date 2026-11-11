# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Input gates and output packaging shared by the P5-1 kernel backends.

These encode what the CUDA/Triton kernels accept (CUDA tensors, three float
dtypes, a 32-divisible last dim) — backend policy, kept apart from
``rl_engine.moe.mx_format``, which only defines the MX format and its bytes.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.moe.mx_format import MX_BLOCK, MXTensor

ACT_QUANT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Same text as mx_format._check_finite: the fail-closed contract, one wording.
NON_FINITE_MESSAGE = "non-finite values in mx_quantize input; P5 quantization is fail-closed"


def validate_act_quant_input(x: Tensor, backend: str) -> Tensor:
    """Shape/dtype/device gate for the forward; returns ``x`` contiguous."""
    if x.dtype not in ACT_QUANT_DTYPES:
        raise TypeError(f"x must have dtype fp16, bf16, or fp32, got {x.dtype}.")
    if not x.is_cuda:
        raise ValueError(f"{backend} mxfp8_act_quant requires a CUDA tensor.")
    if x.ndim < 1 or x.shape[-1] % MX_BLOCK != 0:
        raise ValueError(f"last dim {tuple(x.shape)} not divisible by MX block {MX_BLOCK}.")
    return x.contiguous()


def validate_ste_grad(dy: Tensor, backend: str) -> Tensor:
    """STE backward gate: any floating dtype (P5-1 spec), CUDA; returns ``dy`` contiguous."""
    if not dy.is_floating_point():
        raise TypeError(f"dy must be a floating-point tensor, got {dy.dtype}.")
    if not dy.is_cuda:
        raise ValueError(f"{backend} mxfp8_act_quant backward requires a CUDA tensor.")
    return dy.contiguous()


def finalize_act_quant(
    codes: Tensor, scales: Tensor, nonfinite: Tensor, check_finite: bool, shape: tuple[int, ...]
) -> MXTensor:
    """Apply the fail-closed read-back (one device sync) and package the MX tensor."""
    if check_finite and bool(nonfinite.item()):
        raise ValueError(NON_FINITE_MESSAGE)
    return MXTensor(codes=codes, scales=scales, elem_format="e4m3", shape=shape)
