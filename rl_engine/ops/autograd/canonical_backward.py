# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Autograd-lifetime canonical parameter-gradient reductions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch

WeightReducer = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class _Use:
    keys: torch.Tensor
    rows: torch.Tensor | None = None
    grads: torch.Tensor | None = None


@dataclass
class CanonicalBackwardSession:
    """Collect a graph's row-local parameter VJPs in logical-key order."""

    uses: dict[str, list[_Use]] = field(default_factory=dict)
    received: dict[str, int] = field(default_factory=dict)
    backward_started: bool = False

    def register(self, parameter_id: str, keys: torch.Tensor) -> int:
        if self.backward_started:
            raise RuntimeError("cannot register canonical rows after backward started")
        if keys.dim() != 2 or keys.shape[1] not in (2, 3):
            raise ValueError("logical keys must have shape [rows, 2] or [rows, 3]")
        slot = len(self.uses.setdefault(parameter_id, []))
        self.uses[parameter_id].append(_Use(keys=keys.detach()))
        return slot

    def submit_linear(
        self,
        parameter_id: str,
        slot: int,
        rows: torch.Tensor,
        grads: torch.Tensor,
        reducer: WeightReducer,
    ) -> torch.Tensor | None:
        self.backward_started = True
        entries = self.uses.get(parameter_id)
        if entries is None or slot >= len(entries):
            raise RuntimeError(f"unregistered canonical use: {parameter_id}:{slot}")
        entry = entries[slot]
        entry.rows = rows.reshape(-1, rows.shape[-1]).detach()
        entry.grads = grads.reshape(-1, grads.shape[-1]).detach()
        if entry.rows.shape[0] != entry.keys.shape[0]:
            raise ValueError("logical key count does not match linear rows")
        count = self.received.get(parameter_id, 0) + 1
        self.received[parameter_id] = count
        if count != len(entries):
            return None
        all_keys = torch.cat([item.keys for item in entries], dim=0)
        all_rows = torch.cat([item.rows for item in entries if item.rows is not None], dim=0)
        all_grads = torch.cat([item.grads for item in entries if item.grads is not None], dim=0)
        valid = all_keys[:, 0] >= 0
        all_keys = all_keys[valid]
        all_rows = all_rows[valid]
        all_grads = all_grads[valid]
        if all_keys.shape[0] == 0:
            raise RuntimeError(f"no active logical rows for {parameter_id}")
        order = torch.arange(all_keys.shape[0], device=all_keys.device)
        for column in range(all_keys.shape[1] - 1, -1, -1):
            values = all_keys.index_select(0, order)[:, column]
            order = order.index_select(0, torch.argsort(values, stable=True))
        return reducer(all_rows.index_select(0, order), all_grads.index_select(0, order))

    def submit_rows(
        self,
        parameter_id: str,
        slot: int,
        rows: torch.Tensor,
        reducer: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor | None:
        self.backward_started = True
        entries = self.uses.get(parameter_id)
        if entries is None or slot >= len(entries):
            raise RuntimeError(f"unregistered canonical use: {parameter_id}:{slot}")
        entry = entries[slot]
        entry.rows = rows.reshape(rows.shape[0], -1).detach()
        if entry.rows.shape[0] != entry.keys.shape[0]:
            raise ValueError("logical key count does not match contribution rows")
        count = self.received.get(parameter_id, 0) + 1
        self.received[parameter_id] = count
        if count != len(entries):
            return None
        all_keys = torch.cat([item.keys for item in entries], dim=0)
        all_rows = torch.cat([item.rows for item in entries if item.rows is not None], dim=0)
        valid = all_keys[:, 0] >= 0
        all_keys, all_rows = all_keys[valid], all_rows[valid]
        order = torch.arange(all_keys.shape[0], device=all_keys.device)
        for column in range(all_keys.shape[1] - 1, -1, -1):
            values = all_keys.index_select(0, order)[:, column]
            order = order.index_select(0, torch.argsort(values, stable=True))
        return reducer(all_rows.index_select(0, order))

    def validate_complete(self) -> None:
        missing = {
            name: len(entries) - self.received.get(name, 0)
            for name, entries in self.uses.items()
            if self.received.get(name, 0) != len(entries)
        }
        if missing:
            raise RuntimeError(f"incomplete canonical submissions: {missing}")


_ACTIVE: ContextVar[CanonicalBackwardSession | None] = ContextVar(
    "rl_kernel_canonical_backward", default=None
)


def active_session() -> CanonicalBackwardSession | None:
    return _ACTIVE.get()


@contextmanager
def canonical_backward_session() -> Iterator[CanonicalBackwardSession]:
    if _ACTIVE.get() is not None:
        raise RuntimeError("canonical backward sessions cannot be nested")
    session = CanonicalBackwardSession()
    token = _ACTIVE.set(session)
    try:
        yield session
    finally:
        _ACTIVE.reset(token)


__all__ = ["CanonicalBackwardSession", "active_session", "canonical_backward_session"]
