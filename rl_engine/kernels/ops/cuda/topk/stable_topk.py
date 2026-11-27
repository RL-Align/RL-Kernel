# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA stable per-row top-k for the DSv4 sparse-attention indexer.

Thin wrapper around ``csrc/cuda/topk/stable_topk.cu``, byte-equal to the golden
``rl_engine.kernels.ops.pytorch.topk.stable_topk_per_row`` (vLLM's ds4_bi
selector order). Fails closed when the compiled extension lacks the symbol.
Row ranges are not checked on the host (that would synchronize); the kernel
clamps them to ``[0, width]``, while the golden rejects them.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor
from torch.version import hip as torch_hip

from rl_engine.kernels.ops.pytorch.topk.stable_topk import MAX_TOP_K

_SYMBOL = "stable_topk_per_row_out"


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("stable_topk requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    if not hasattr(ext, _SYMBOL):
        raise RuntimeError(f"rl_engine._C lacks {_SYMBOL}; rebuild the CUDA extension")
    return ext


def stable_topk_per_row_cuda(
    logits: Tensor,
    row_starts: Tensor,
    row_ends: Tensor,
    top_k: int,
    *,
    out: Tensor | None = None,
    threads: int = 0,
) -> Tensor:
    """``logits`` FP32 ``[R, W]`` (any strides) -> int32 ``[R, top_k]`` local indices.

    ``out`` may be a row-strided view (e.g. a slice of a persistent index
    buffer) with unit column stride; it is filled in place and returned.
    ``threads`` (256, 512 or 1024; 0 picks by row count) only changes speed:
    the selection is exact, so every block shape writes the same bytes.
    """
    if not isinstance(top_k, int) or not 0 < top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be in (0, {MAX_TOP_K}], got {top_k!r}")
    if out is None:
        out = torch.empty((logits.shape[0], top_k), dtype=torch.int32, device=logits.device)
    _extension().stable_topk_per_row_out(
        logits, row_starts.contiguous(), row_ends.contiguous(), out, top_k, threads
    )
    return out


class StableTopkCudaOp:
    """CUDA backend for ``stable_topk_per_row``; construction fails closed."""

    def __init__(self) -> None:
        _extension()

    def __call__(
        self,
        logits: Tensor,
        row_starts: Tensor,
        row_ends: Tensor,
        top_k: int,
        *,
        out: Tensor | None = None,
    ) -> Tensor:
        return stable_topk_per_row_cuda(logits, row_starts, row_ends, top_k, out=out)
