# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini per-visit KV and convolution cache (INK-22)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def clone_visit_state_cuda(source: torch.Tensor) -> torch.Tensor:
    """Clone a CUDA KV or convolution state through the native copy kernel."""
    from rl_engine.backends.extension import _C, _EXT_AVAILABLE

    if not _EXT_AVAILABLE or not hasattr(_C, "inkling2_visit_state_clone"):
        raise RuntimeError("Inkling 2 visit-state CUDA extension is unavailable")
    if not source.is_cuda or source.ndim == 0:
        raise ValueError("visit state must be a non-scalar CUDA tensor")
    if source.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("native visit-state clone supports float16, bfloat16, and float32")
    return _C.inkling2_visit_state_clone(source.contiguous())


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

    @staticmethod
    def _clone_state(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.is_cuda:
            from rl_engine.backends.extension import _C, _EXT_AVAILABLE

            if _EXT_AVAILABLE and hasattr(_C, "inkling2_visit_state_clone"):
                return clone_visit_state_cuda(tensor)
        return tensor.clone()

    def put(
        self,
        key: VisitKey,
        *,
        kv: torch.Tensor | None = None,
        conv: torch.Tensor | None = None,
    ) -> None:
        self._states[key] = VisitState(
            self._clone_state(kv),
            self._clone_state(conv),
        )

    def get(self, key: VisitKey) -> VisitState | None:
        state = self._states.get(key)
        if state is None:
            return None
        return VisitState(
            self._clone_state(state.kv),
            self._clone_state(state.conv),
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
