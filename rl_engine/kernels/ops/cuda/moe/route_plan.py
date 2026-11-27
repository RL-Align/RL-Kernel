# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4 MoE combine and backward fan-in on a :class:`RoutePlan`.

The plan is a counting sort (``csrc/cuda/moe/moe_route_plan.cu``): a stable sort's
result is unique, so it equals the golden ``build_plan`` exactly. The dispatch gather
and the gradient dispatch are exact copies and reuse the golden
(``rl_engine.kernels.ops.pytorch.moe.route_plan``); the slot-ordered sums run
``csrc/cuda/moe/moe_slot_sum.cu``, byte-equal to the golden ``slot_sum``. Fails closed
when the compiled extension lacks a symbol.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor
from torch.version import hip as torch_hip

from rl_engine.kernels.ops.pytorch.moe.route_plan import TOP_K, RoutePlan


def _extension(symbol: str = "moe_slot_sum"):
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError(f"{symbol} requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    if not hasattr(ext, symbol):
        raise RuntimeError(f"rl_engine._C lacks {symbol}; rebuild the CUDA extension")
    return ext


def build_plan_cuda(ids: Tensor, num_experts: int) -> RoutePlan:
    """:func:`build_plan` on the GPU (no host sync); also caches each row's token id."""
    if ids.dim() != 2 or ids.shape[1] != TOP_K:
        raise ValueError("ids must be [T, 6]")
    perm, token, inv, offsets = _extension("moe_route_plan").moe_route_plan(
        ids.contiguous(), int(num_experts)
    )
    return RoutePlan(perm, inv, offsets, ids.shape[0], int(num_experts), token)


def slot_sum_cuda(
    rows: Tensor,
    plan: RoutePlan,
    shared: Tensor | None = None,
    residual: Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    out = torch.empty(plan.num_tokens, rows.shape[1], dtype=out_dtype, device=rows.device)
    _extension().moe_slot_sum(rows, plan.inv.contiguous(), shared, residual, out)
    return out


def combine_cuda(
    y_rows: Tensor, plan: RoutePlan, shared: Tensor | None, residual: Tensor | None
) -> Tensor:
    """MoE block output BF16 ``[T, H]`` (slot order, + shared once, + residual once)."""
    return slot_sum_cuda(y_rows, plan, shared, residual, torch.bfloat16)


def dispatch_backward_cuda(
    dx_rows: Tensor, plan: RoutePlan, dx_shared: Tensor | None = None
) -> Tensor:
    """``dL/dx`` FP32 ``[T, H]``: slot-ordered fan-in plus the shared expert's ``dx``."""
    return slot_sum_cuda(dx_rows, plan, dx_shared, None, torch.float32)
