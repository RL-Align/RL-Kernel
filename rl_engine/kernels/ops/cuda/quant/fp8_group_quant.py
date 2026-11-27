# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA per-token-group FP8 activation quantization for DeepGEMM (DSv4).

Thin wrapper around ``csrc/cuda/quant/fp8_group_quant.cu``, which is
byte-exact with the PyTorch golden
``rl_engine.kernels.ops.pytorch.quant.fp8_group_quant``. Fails closed when the
compiled extension lacks the symbols (rule 7): there is no fallback to the
golden on CUDA tensors.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor
from torch.version import hip as torch_hip

from rl_engine.kernels.ops.pytorch.quant.fp8_group_quant import GROUP_SIZE, INPUT_DTYPES

_SYMBOLS = ("fp8_group_quant_forward", "fp8_group_quant_tokens_forward")


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("fp8_group_quant requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    missing = [name for name in _SYMBOLS if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(f"rl_engine._C lacks {missing}; rebuild the CUDA extension")
    return ext


def _check(x: Tensor) -> Tensor:
    if type(x) is not Tensor:
        raise TypeError("x must be a torch.Tensor.")
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape {tuple(x.shape)}.")
    if x.dtype not in INPUT_DTYPES:
        raise TypeError(f"x must be BF16 or FP16, got {x.dtype}.")
    if not x.is_cuda:
        raise RuntimeError("x must be a CUDA tensor.")
    return x.contiguous()


def fp8_group_quant_cuda(x: Tensor, *, scale_ue8m0: bool = True) -> tuple[Tensor, Tensor]:
    """``x [M, K]`` -> FP8 ``[M, K]`` and FP32 scales ``[M, K/128]`` (DeepGEMM layout)."""
    x = _check(x)
    if x.shape[1] == 0 or x.shape[1] % GROUP_SIZE:
        raise ValueError(f"x's last dimension must be a positive multiple of {GROUP_SIZE}.")
    if x.data_ptr() % 16:
        x = x.clone()
    q, scale = _extension().fp8_group_quant_forward(x, bool(scale_ue8m0))
    return q, scale


def fp8_group_quant_tokens_cuda(x: Tensor, *, scale_ue8m0: bool = True) -> tuple[Tensor, Tensor]:
    """``x [T, C]`` -> FP8 ``[C, T_pad]`` and FP32 scales ``[C, T_pad/128]`` for wgrad."""
    x = _check(x)
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("x must have at least one token and one channel.")
    q, scale = _extension().fp8_group_quant_tokens_forward(x, bool(scale_ue8m0))
    return q, scale


class Fp8GroupQuantCudaOp:
    """CUDA backend for ``fp8_group_quant``; construction fails closed."""

    def __init__(self) -> None:
        _extension()

    def __call__(self, x: Tensor, *, scale_ue8m0: bool = True) -> tuple[Tensor, Tensor]:
        return self.forward(x, scale_ue8m0=scale_ue8m0)

    def forward(self, x: Tensor, *, scale_ue8m0: bool = True) -> tuple[Tensor, Tensor]:
        return fp8_group_quant_cuda(x, scale_ue8m0=scale_ue8m0)

    def forward_tokens(self, x: Tensor, *, scale_ue8m0: bool = True) -> tuple[Tensor, Tensor]:
        return fp8_group_quant_tokens_cuda(x, scale_ue8m0=scale_ue8m0)
