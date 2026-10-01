# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Framework-light automatic capture for models without a live adapter."""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .evidence import Capture


def _safe_id(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return value or "root"


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _rows(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        raise ValueError("automatic capture only supports tensor values with a row dimension")
    if value.ndim == 1:
        return value.reshape(1, -1)
    return value.detach().reshape(-1, value.shape[-1])


def _leaf_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name and not any(module.children())
    ]


def _selected(
    leaves: list[tuple[str, torch.nn.Module]],
    include: Iterable[str] | None,
    exclude: Iterable[str] | None,
    max_modules: int | None,
) -> list[tuple[str, torch.nn.Module]]:
    include_set = set(include or ())
    exclude_set = set(exclude or ())
    leaves = [
        (name, module)
        for name, module in leaves
        if (
            not include_set
            or any(name == item or name.startswith(item + ".") for item in include_set)
        )
        and not any(name == item or name.startswith(item + ".") for item in exclude_set)
    ]
    if max_modules is not None:
        if max_modules < 1:
            raise ValueError("max_modules must be positive or None")
        leaves = leaves[:max_modules]
    if not leaves:
        raise ValueError("automatic capture found no eligible leaf modules")
    return leaves


def _module_ids(
    leaves: list[tuple[str, torch.nn.Module]],
) -> list[tuple[str, str, torch.nn.Module]]:
    used: set[str] = set()
    result = []
    for name, module in leaves:
        stem = _safe_id(name)
        module_id = stem
        suffix = 1
        while module_id in used:
            suffix += 1
            module_id = f"{stem}_{suffix}"
        used.add(module_id)
        result.append((name, module_id, module))
    return result


def prepare_auto(
    directory: Path,
    model: torch.nn.Module,
    *,
    positions: Iterable[int],
    scope: str,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    max_modules: int | None = 256,
) -> list[str]:
    """Create a dynamic-width manifest from a model's leaf modules."""
    selected_positions = list(positions)
    if not selected_positions or any(type(p) is not int or p < 0 for p in selected_positions):
        raise ValueError("positions must be a nonempty list of nonnegative integers")
    leaves = _selected(_leaf_modules(model), include, exclude, max_modules)
    boundaries = []
    for name, module_id, module in _module_ids(leaves):
        base = f"auto.{module_id}"
        for stage, inputs in (("input", []), ("output", [f"{base}.input"])):
            boundaries.append(
                {
                    "id": f"{base}.{stage}",
                    "module": "auto",
                    "stage": stage,
                    "layer": None,
                    "positions": selected_positions,
                    "width": None,
                    "dynamic_width": True,
                    "inputs": inputs,
                    "contracts": [],
                    "module_path": name,
                    "route": type(module).__qualname__,
                }
            )
    Capture.prepare(directory, boundaries, scope=scope)
    return [name for name, _, _ in _module_ids(leaves)]


class AutoCapture:
    """Observe selected leaf-module inputs and outputs without an adapter."""

    def __init__(
        self,
        directory: Path,
        side: str,
        identity: Mapping[str, Any],
        model: torch.nn.Module,
        *,
        positions: Iterable[int],
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
        max_modules: int | None = 256,
    ):
        self.capture = Capture(directory, side, identity)
        self.positions = list(positions)
        self.leaves = _module_ids(_selected(_leaf_modules(model), include, exclude, max_modules))
        self.handles: list[Any] = []
        self.seen: set[str] = set()

    def _record(self, boundary: str, value: Any, route: str) -> None:
        # A reused module can execute more than once. The generic manifest has
        # one semantic event per module path; retain its first event rather
        # than misclassifying a later call as a replica disagreement.
        if boundary in self.seen:
            return
        tensor = _first_tensor(value)
        if tensor is None:
            return
        rows = _rows(tensor)
        if rows.shape[0] != len(self.positions):
            return
        self.capture.record(boundary, rows, positions=self.positions, route=route)
        self.seen.add(boundary)

    def __enter__(self):
        for _name, module_id, module in self.leaves:
            base = f"auto.{module_id}"
            route = type(module).__qualname__

            def before(mod, args, kwargs, base=base, route=route):
                self._record(f"{base}.input", args, route)

            def after(mod, args, kwargs, output, base=base, route=route):
                self._record(f"{base}.output", output, route)

            self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(module.register_forward_hook(after, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


@contextmanager
def auto_capture(*args, **kwargs):
    with AutoCapture(*args, **kwargs) as capture:
        yield capture
