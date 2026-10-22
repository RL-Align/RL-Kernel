# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Autograd-lifetime canonical parameter-gradient reductions."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch

WeightReducer = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class _Use:
    keys: torch.Tensor
    contribution: object | None = None
    rows: torch.Tensor | None = None
    grads: torch.Tensor | None = None


@dataclass
class CanonicalBackwardSession:
    """Collect a graph's row-local parameter VJPs in logical-key order."""

    uses: dict[str, list[_Use]] = field(default_factory=dict)
    received: dict[str, int] = field(default_factory=dict)
    backward_started: bool = False
    _normalized: dict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, repr=False
    )
    _shareable: set[tuple[object, ...]] | None = field(default=None, repr=False)

    def register(
        self,
        parameter_id: str,
        keys: torch.Tensor,
        *,
        contribution: object | None = None,
    ) -> int:
        if self.backward_started:
            raise RuntimeError("cannot register canonical rows after backward started")
        if keys.dim() != 2 or keys.shape[1] not in (2, 3):
            raise ValueError("logical keys must have shape [rows, 2] or [rows, 3]")
        slot = len(self.uses.setdefault(parameter_id, []))
        self.uses[parameter_id].append(
            _Use(
                keys=keys.detach(),
                contribution=contribution,
            )
        )
        return slot

    @staticmethod
    def _concatenate(values: list[torch.Tensor]) -> torch.Tensor:
        return values[0] if len(values) == 1 else torch.cat(values, dim=0)

    def _normalize(
        self, parameter_id: str, values: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        entries = self.uses[parameter_id]
        # A token names one forward's shared weight/bias contribution. Matching
        # complete token sequences prove that both parameters receive the same
        # keys and upstream gradients, without a device equality check.
        signature = tuple(item.contribution for item in entries)
        if self._shareable is None:
            # Forward registration is complete before backward. Build this once
            # so a model's parameters do not each scan the complete session.
            signatures = Counter(
                tuple(item.contribution for item in other)
                for other in self.uses.values()
                if all(item.contribution is not None for item in other)
            )
            self._shareable = {key for key, count in signatures.items() if count == 2}
        shared = signature in self._shareable
        if shared and signature in self._normalized:
            return self._normalized.pop(signature)
        keys = self._concatenate([item.keys for item in entries])
        valid = torch.nonzero(keys[:, 0] >= 0, as_tuple=True)[0]
        active_keys = keys.index_select(0, valid)
        order = torch.arange(valid.numel(), device=keys.device)
        for column in range(keys.shape[1] - 1, -1, -1):
            column_values = active_keys.index_select(0, order)[:, column]
            order = order.index_select(0, torch.argsort(column_values, stable=True))
        order = valid.index_select(0, order)
        normalized = order, self._concatenate(values).index_select(0, order)
        if shared:
            self._normalized[signature] = normalized
        return normalized

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
        order, grads = self._normalize(
            parameter_id, [item.grads for item in entries if item.grads is not None]
        )
        if order.numel() == 0:
            raise RuntimeError(f"no active logical rows for {parameter_id}")
        rows = self._concatenate([item.rows for item in entries if item.rows is not None])
        return reducer(rows.index_select(0, order), grads)

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
        _, rows = self._normalize(
            parameter_id, [item.rows for item in entries if item.rows is not None]
        )
        return reducer(rows)

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
