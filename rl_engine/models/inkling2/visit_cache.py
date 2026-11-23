# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini per-visit KV and convolution cache (INK-22)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, order=True)
class VisitKey:
    physical_layer_id: int
    pass_index: int
    logical_visit_id: int


@dataclass
class VisitState:
    kv: torch.Tensor | None
    conv: torch.Tensor | None


class VisitStateCache:
    """Store cloned KV/conv states under physical-layer and logical-visit IDs."""

    def __init__(self) -> None:
        self._states: dict[VisitKey, VisitState] = {}

    def put(
        self,
        key: VisitKey,
        *,
        kv: torch.Tensor | None = None,
        conv: torch.Tensor | None = None,
    ) -> None:
        self._states[key] = VisitState(
            None if kv is None else kv.clone(),
            None if conv is None else conv.clone(),
        )

    def get(self, key: VisitKey) -> VisitState | None:
        state = self._states.get(key)
        if state is None:
            return None
        return VisitState(
            None if state.kv is None else state.kv.clone(),
            None if state.conv is None else state.conv.clone(),
        )

    def require(self, key: VisitKey) -> VisitState:
        state = self.get(key)
        if state is None:
            raise KeyError(f"no cached state for {key}")
        return state

    def clear_visit(self, key: VisitKey) -> None:
        self._states.pop(key, None)

    def clear_pass(self, pass_index: int) -> None:
        for key in tuple(self._states):
            if key.pass_index == pass_index:
                del self._states[key]

    def clear(self) -> None:
        self._states.clear()

    def keys(self) -> tuple[VisitKey, ...]:
        return tuple(sorted(self._states))
