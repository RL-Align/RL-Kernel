# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4 indexer query: RoPE + per-(token, head) FP8 quantization.

Thin wrapper around ``csrc/cuda/indexer/indexer_q_rope_fp8.cu``, byte-equal to
the golden ``rl_engine.kernels.ops.pytorch.indexer.indexer_q_rope_fp8`` (vLLM's
CuTe-DSL ``IndexerQFp8Kernel`` numerics). Fails closed when the compiled
extension lacks the symbol.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor
from torch.version import hip as torch_hip

_SYMBOL = "indexer_q_rope_fp8_forward"


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("indexer_q_rope_fp8 requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    if not hasattr(ext, _SYMBOL):
        raise RuntimeError(f"rl_engine._C lacks {_SYMBOL}; rebuild the CUDA extension")
    return ext


def indexer_q_rope_fp8_cuda(
    q: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    weights: Tensor,
    softmax_scale: float,
    head_scale: float,
) -> tuple[Tensor, Tensor]:
    """``q`` BF16 ``[T, H, 128]`` -> (FP8 E4M3 ``[T, H, 128]``, FP32 weights ``[T, H]``)."""
    q_fp8, weights_out = _extension().indexer_q_rope_fp8_forward(
        q, positions, cos_sin_cache, weights, float(softmax_scale), float(head_scale)
    )
    return q_fp8, weights_out


class IndexerQRopeFp8CudaOp:
    """CUDA backend for ``indexer_q_rope_fp8``; construction fails closed."""

    def __init__(self) -> None:
        _extension()

    def __call__(self, q, positions, cos_sin_cache, weights, softmax_scale, head_scale):
        return indexer_q_rope_fp8_cuda(
            q, positions, cos_sin_cache, weights, softmax_scale, head_scale
        )
