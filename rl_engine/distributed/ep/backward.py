# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Expert gradients along a saved forward route.

Follows the row orders of the forward exchange: ``dy`` travels from ``(token, slot)``
through send and arrival order to expert order, and ``dX`` comes back the same way.
Per-token gradients are accumulated in FP32 from +0.0 in ascending slot order and cast
once, matching the single-device router backward bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.distributed.ep.exchange import EpExchange, EpRoute
from rl_engine.distributed.ep.plan import EpPlanError

_OP_DY, _OP_DX = 23, 24
_WIRE_DTYPES = {torch.bfloat16: 1}


def accumulate_slots(rows: torch.Tensor, top_k: int, dtype: torch.dtype) -> torch.Tensor:
    """``[T * top_k, H]`` rows in ``(token, slot)`` order to ``[T, H]`` per-token values."""
    lanes = rows.view(rows.shape[0] // top_k, top_k, *rows.shape[1:])
    total = torch.zeros_like(lanes[:, 0], dtype=torch.float32)
    for slot in range(top_k):
        total = total + lanes[:, slot].to(torch.float32)
    return total.to(dtype)


@dataclass(eq=False)
class SavedEpRoute:
    """A returned strict route; carries one ``dy`` then one ``dX``, and never again."""

    route: EpRoute
    phase: str = "returned"


class EpGradExchange(EpExchange):
    """Forward exchange plus the reverse trip. Every step is agreed before payload moves."""

    def agree(self, op: int, *, run_tag: int = 0, width: int = 0, error=None) -> None:
        """All-rank agreement on an operation; a local ``error`` makes every rank raise."""
        self._agree(op, run_tag=run_tag, width=width, error=error)

    def save_route(self, route: EpRoute) -> SavedEpRoute:
        if not (self.strict and route.returned and route.owner is self):
            raise EpPlanError("backward needs a returned route from this strict exchange")
        send = route.send
        if send.num_rows != send.num_tokens * send.top_k:
            raise EpPlanError("backward needs every Top-K slot of every token")
        return SavedEpRoute(route)

    def _grad_signature(
        self, saved: SavedEpRoute, phase: str, rows: torch.Tensor, num_rows: int
    ) -> dict[str, int]:
        if saved.route.owner is not self:
            raise EpPlanError("route belongs to another exchange")
        if saved.phase != phase:
            raise EpPlanError(f"backward step out of order: route is at {saved.phase}")
        if rows.dtype not in _WIRE_DTYPES:
            raise EpPlanError(f"unsupported gradient dtype {rows.dtype}")
        if rows.device != self.transport.device:
            raise EpPlanError("gradient is not on the exchange device")
        if rows.requires_grad:
            raise EpPlanError("higher-order gradients through the exchange are unsupported")
        if rows.ndim != 2 or rows.shape[0] != num_rows:
            raise EpPlanError(f"gradient must have {num_rows} rows, got {tuple(rows.shape)}")
        return {
            "run_tag": saved.route.send.run_tag,
            "dtype": _WIRE_DTYPES[rows.dtype],
            "width": rows.shape[1],
        }

    def dispatch_dy(self, saved: SavedEpRoute, dy: torch.Tensor) -> torch.Tensor:
        """``[S, H]`` in ``(token, slot)`` order to expert order on the owning ranks."""
        route = saved.route
        rows: Any = None
        output: Any = None
        signature: dict[str, int] = {}
        error = None
        try:
            signature = self._grad_signature(saved, "returned", dy, route.send.num_rows)
            rows = torch.empty_like(dy)
            rows[route.send.to_canonical] = dy
            output = self._allocate((route.recv.num_rows, dy.shape[1]), dy.dtype)
        except (EpPlanError, RuntimeError) as exc:
            error = exc
        self._agree(_OP_DY, **signature, error=error)
        self._move(output, rows, route.recv_counts, route.send_counts)
        saved.phase = "dispatched"
        with self.guard():
            return route.recv.to_expert_order(output)

    def return_dx(
        self, saved: SavedEpRoute, dx_rows: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        """Expert-order ``[R, H]`` back to the sources, accumulated to ``[T, H]``."""
        route = saved.route
        rows: Any = None
        home: Any = None
        signature: dict[str, int] = {}
        error = None
        try:
            signature = self._grad_signature(saved, "dispatched", dx_rows, route.recv.num_rows)
            rows = route.recv.to_arrival_order(dx_rows)
            home = self._allocate((route.send.num_rows, dx_rows.shape[1]), dx_rows.dtype)
        except (EpPlanError, RuntimeError) as exc:
            error = exc
        self._agree(_OP_DX, **signature, error=error)
        self._move(home, rows, route.send_counts, route.recv_counts)
        saved.phase = "used"
        with self.guard():
            canonical = home.index_select(0, route.send.to_canonical)
            return accumulate_slots(canonical, route.send.top_k, dtype)


__all__ = ["EpGradExchange", "SavedEpRoute", "accumulate_slots"]
