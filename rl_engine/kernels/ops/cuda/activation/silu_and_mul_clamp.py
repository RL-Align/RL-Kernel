# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA clamped SwiGLU for the DSv4 shared expert.

Thin wrapper around ``csrc/cuda/activation/silu_and_mul_clamp.cu``: the forward
is fused with ``fp8_group_quant`` and byte-equal to
``fp8_group_quant(silu_and_mul_clamp(z))`` (the vLLM rollout's two kernels);
the backward is byte-equal to ``silu_and_mul_clamp_backward``. Fails closed when
the compiled extension lacks the symbols.
"""

from __future__ import annotations

import importlib
from typing import Literal, overload

import torch
from torch import Tensor
from torch.version import hip as torch_hip

from rl_engine.kernels.ops.pytorch.activation.silu_and_mul_clamp import SWIGLU_LIMIT

_SYMBOLS = ("silu_and_mul_clamp_quant_forward", "silu_and_mul_clamp_backward")


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("silu_and_mul_clamp requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    missing = [name for name in _SYMBOLS if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(f"rl_engine._C lacks {missing}; rebuild the CUDA extension")
    return ext


def _bf16_2d(t: Tensor, name: str) -> Tensor:
    if type(t) is not Tensor or t.ndim != 2 or t.dtype != torch.bfloat16 or not t.is_cuda:
        raise TypeError(f"{name} must be a 2-D BF16 CUDA tensor.")
    t = t.contiguous()
    return t.clone() if t.data_ptr() % 16 else t


@overload
def silu_and_mul_clamp_quant_cuda(
    z: Tensor,
    limit: float = ...,
    *,
    scale_ue8m0: bool = ...,
    return_h: Literal[False] = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def silu_and_mul_clamp_quant_cuda(
    z: Tensor,
    limit: float = ...,
    *,
    scale_ue8m0: bool = ...,
    return_h: Literal[True],
) -> tuple[Tensor, Tensor, Tensor]: ...


def silu_and_mul_clamp_quant_cuda(
    z: Tensor,
    limit: float = SWIGLU_LIMIT,
    *,
    scale_ue8m0: bool = True,
    return_h: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """``z [T, 2F]`` -> FP8 ``h`` ``[T, F]`` and its FP32 scales (DeepGEMM layout).

    ``return_h`` also returns the BF16 ``h`` (for checks; the shared expert
    does not need it)."""
    z = _bf16_2d(z, "z")
    if z.shape[1] % 256:
        raise ValueError("z's last dimension must be a multiple of 256 (F % 128 == 0).")
    q, scale, h = _extension().silu_and_mul_clamp_quant_forward(
        z, float(limit), bool(scale_ue8m0), bool(return_h)
    )
    return (q, scale, h) if return_h else (q, scale)


def silu_and_mul_clamp_backward_cuda(dh: Tensor, z: Tensor, limit: float = SWIGLU_LIMIT) -> Tensor:
    """``dh [T, F]``, ``z [T, 2F]`` -> ``dz [T, 2F]`` BF16."""
    dh = _bf16_2d(dh, "dh")
    z = _bf16_2d(z, "z")
    if z.shape[1] != 2 * dh.shape[1] or z.shape[0] != dh.shape[0] or dh.shape[1] % 8:
        raise ValueError("dh must be [T, F] with F % 8 == 0 for z [T, 2F].")
    return _extension().silu_and_mul_clamp_backward(dh, z, float(limit))
