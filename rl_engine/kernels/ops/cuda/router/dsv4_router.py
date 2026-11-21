# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4 MoE router: top-6 / hash routing and its dX backward.

Thin wrapper around ``csrc/cuda/router/dsv4_router.cu``, byte-equal to the
golden ``rl_engine.kernels.ops.pytorch.router`` (``route_topk`` and
``route_backward_dx``). Fails closed when the compiled extension lacks the symbols.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor
from torch.version import hip as torch_hip

_SYMBOLS = ("dsv4_router_topk", "dsv4_router_backward_dx")
TOP_K = 6


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("the DSv4 router requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    missing = [name for name in _SYMBOLS if not hasattr(ext, name)]
    if missing:
        raise RuntimeError(f"rl_engine._C lacks {missing}; rebuild the CUDA extension")
    return ext


def route_topk_cuda(
    logits: Tensor,
    bias: Tensor | None,
    routed_scaling_factor: float,
    *,
    input_ids: Tensor | None = None,
    tid2eid: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """``(weights FP32 [T, 6], ids int32 [T, 6])``; hash routing when ``tid2eid`` is given."""
    rows = logits.shape[0]
    weights = torch.empty(rows, TOP_K, dtype=torch.float32, device=logits.device)
    ids = torch.empty(rows, TOP_K, dtype=torch.int32, device=logits.device)
    if tid2eid is not None and input_ids is not None and input_ids.dtype != tid2eid.dtype:
        input_ids = input_ids.to(tid2eid.dtype)
    _extension().dsv4_router_topk(
        logits, bias, input_ids, tid2eid, float(routed_scaling_factor), weights, ids
    )
    return weights, ids


def route_backward_dx_cuda(
    logits: Tensor,
    ids: Tensor,
    d_weights: Tensor,
    gate_weight: Tensor,
    routed_scaling_factor: float,
    out_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``dL/dX`` ``[T, H]`` through the frozen BF16 gate ``[E, H]``."""
    dx = torch.empty(logits.shape[0], gate_weight.shape[1], dtype=out_dtype, device=logits.device)
    _extension().dsv4_router_backward_dx(
        logits,
        ids,
        d_weights.to(torch.float32).contiguous(),
        gate_weight,
        float(routed_scaling_factor),
        dx,
    )
    return dx
