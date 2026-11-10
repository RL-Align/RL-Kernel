# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA DSv4 MoE combine and backward fan-in on a :class:`RoutePlan`.

The plan, the dispatch gather and the gradient dispatch are exact integer /
copy work and reuse the golden (``rl_engine.kernels.ops.pytorch.moe.route_plan``);
the slot-ordered sums run ``csrc/cuda/moe/moe_slot_sum.cu``, byte-equal to the
golden ``slot_sum``. Fails closed when the compiled extension lacks the symbol.
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor
from torch.version import hip as torch_hip

from rl_engine.kernels.ops.pytorch.moe.route_plan import RoutePlan


def _extension():
    if torch_hip is not None or not torch.cuda.is_available():
        raise RuntimeError("moe_slot_sum requires NVIDIA CUDA")
    try:
        ext = importlib.import_module("rl_engine._C")
    except ImportError as exc:
        raise RuntimeError("Build the CUDA extension first") from exc
    if not hasattr(ext, "moe_slot_sum"):
        raise RuntimeError("rl_engine._C lacks moe_slot_sum; rebuild the CUDA extension")
    return ext


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
