# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 MoE route plan (golden): dispatch order, combine and the backward fan-in.

From the router's ``ids`` / ``weights`` ``[T, 6]`` (slot order), the plan sorts
the ``T * 6`` (token, slot) pairs by ``(expert, token, slot)``; ``token`` is the
global token id, so with expert parallelism the rows an expert receives keep the
same order whichever rank they come from. Row ``r`` of the dispatched batch is
flat pair ``perm[r] = token * 6 + slot``; ``inv[token, slot]`` is its row. All
plan arithmetic is integer, so it is exact.

The routed-expert kernel (P5) multiplies each row by its route weight ``p_s``
once (``already-weighted-per-slot-rows``, P6 policy), and a row's bytes depend
only on that row, so where a row lands inside its expert does not change it.

Combine (P6 ``fixed_order_combine_fwd`` + ``shared_residual_merge_fwd``), per token,
in FP32: the slot-0 row is copied, slots 1..5 are added in order, then the shared
expert output is added once and the residual once, and the result is rounded to
BF16 (RNE)::

    acc = y[inv[t, 0]];  acc = acc + y[inv[t, s]]  for s = 1..5
    out = bf16((acc + shared[t]) + residual[t])

Backward: the gradient of the combine is ``dy[t]`` for every row of token ``t``
(``gradient dispatch``, exact); the expert-input gradient fans back in with the
same slot order, plus the shared expert's ``dx`` once (P6 ``fused_dx_fanin_bwd``);
the route-weight gradient is ``dp_s[inv]`` (exact).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

TOP_K = 6


@dataclass
class RoutePlan:
    perm: Tensor  # int64 [P], flat (token * 6 + slot) of each dispatched row
    inv: Tensor  # int64 [T, 6], dispatched row of each (token, slot)
    expert_offsets: Tensor  # int32 [E + 1]
    num_tokens: int
    num_experts: int
    token_ids: Tensor | None = None  # int64 [P], perm // 6 when a builder provides it

    @property
    def token(self) -> Tensor:
        """Source token of each dispatched row, int64 [P]."""
        return self.token_ids if self.token_ids is not None else self.perm // TOP_K

    @property
    def rows(self) -> int:
        return self.num_tokens * TOP_K


def build_plan(ids: Tensor, num_experts: int) -> RoutePlan:
    """Plan for ``ids`` int ``[T, 6]`` in ``[0, num_experts)`` (no host sync)."""
    if ids.dim() != 2 or ids.shape[1] != TOP_K:
        raise ValueError("ids must be [T, 6]")
    t = ids.shape[0]
    flat = ids.reshape(-1).to(torch.int64)
    # Stable sort by expert keeps (token, slot) ascending within an expert.
    sorted_ids, perm = torch.sort(flat, stable=True)
    bounds = torch.arange(num_experts + 1, device=ids.device, dtype=torch.int64)
    offsets = torch.searchsorted(sorted_ids, bounds).to(torch.int32)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=ids.device)
    return RoutePlan(perm, inv.view(t, TOP_K), offsets, t, num_experts)


def dispatch(x: Tensor, weights: Tensor, plan: RoutePlan) -> tuple[Tensor, Tensor, Tensor]:
    """``(rows [P, H], p_s FP32 [P], output_slot int32 [P])`` in plan order."""
    rows = x.index_select(0, plan.token)
    p_s = weights.reshape(-1).to(torch.float32).index_select(0, plan.perm)
    return rows, p_s, plan.perm.to(torch.int32)


def slot_sum(
    rows: Tensor,
    plan: RoutePlan,
    shared: Tensor | None = None,
    residual: Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """``((sum_s rows[inv[t, s]]) + shared[t]) + residual[t]`` in FP32, slot order."""
    acc = rows.index_select(0, plan.inv[:, 0]).float()
    for s in range(1, TOP_K):
        acc = acc + rows.index_select(0, plan.inv[:, s]).float()
    if shared is not None:
        acc = acc + shared.float()
    if residual is not None:
        acc = acc + residual.float()
    return acc.to(out_dtype)


def combine(
    y_rows: Tensor, plan: RoutePlan, shared: Tensor | None, residual: Tensor | None
) -> Tensor:
    """MoE block output BF16 ``[T, H]`` from the weighted expert rows."""
    return slot_sum(y_rows, plan, shared, residual, torch.bfloat16)


def combine_backward(dy: Tensor, plan: RoutePlan) -> Tensor:
    """``dL/dy_rows`` FP32 ``[P, H]``: each row gets its token's ``dy``."""
    return dy.float().index_select(0, plan.token)


def dispatch_backward(dx_rows: Tensor, plan: RoutePlan, dx_shared: Tensor | None = None) -> Tensor:
    """``dL/dx`` FP32 ``[T, H]``: slot-ordered fan-in plus the shared expert's ``dx``."""
    return slot_sum(dx_rows, plan, dx_shared, None, torch.float32)


def weights_backward(dp_s: Tensor, plan: RoutePlan) -> Tensor:
    """``dL/dweights`` FP32 ``[T, 6]`` from ``dL/dp_s`` ``[P]``."""
    return dp_s.to(torch.float32)[plan.inv]
