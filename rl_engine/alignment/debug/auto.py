# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Native eager capture: all calls, explicit cross-framework boundary mappings."""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .core import fingerprint
from .events import EventCapture, encode_tree


def _inventory_identity(value):
    """Local immutable check; global equality is verified by sharded events."""
    if isinstance(value, torch.Tensor):
        return fingerprint(value)
    if isinstance(value, dict):
        return {key: _inventory_identity(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_inventory_identity(child) for child in value]
    return value


def _weight_specs():
    return [
        {"id": f"weights.{stage}", "module": "weights", "stage": stage}
        for stage in ("before", "after")
    ]


@dataclass(frozen=True)
class ModuleBoundary:
    """Bind a logical role to a native module, without replacing its forward.

    inputs(module, args, kwargs) and outputs(module, args, kwargs, output) may
    return tensor trees or LogicalTensor views. Selectors must be observational.
    Omitting selectors observes ALL positional/keyword inputs and output leaves.
    """

    path: str
    inputs: Callable[..., Any] | None = None
    outputs: Callable[..., Any] | None = None
    layer: int | None = None
    observe_inputs: bool = True


def _bindings(
    model: torch.nn.Module,
    include: Iterable[str] | None,
    exclude: Iterable[str] | None,
    max_modules: int | None,
    bindings: Mapping[str, ModuleBoundary] | None,
) -> dict[str, ModuleBoundary]:
    if bindings is not None:
        if include is not None or exclude is not None or max_modules is not None:
            raise ValueError("explicit bindings cannot be combined with automatic module filters")
        selected = dict(bindings)
    else:
        included, excluded = set(include or ()), set(exclude or ())
        leaves = [
            name
            for name, module in model.named_modules()
            if not any(module.children())
            and (not included or any(name == p or name.startswith(p + ".") for p in included))
            and not any(name == p or name.startswith(p + ".") for p in excluded)
        ]
        if max_modules is not None:
            if max_modules < 1:
                raise ValueError("max_modules must be positive or None")
            leaves = leaves[:max_modules]
        selected = {}
        for name in leaves:
            stem = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "root"
            role, suffix = f"auto.{stem}", 1
            while role in selected:
                suffix += 1
                role = f"auto.{stem}_{suffix}"
            selected[role] = ModuleBoundary(name)
    if not selected or any(not isinstance(role, str) or not role for role in selected):
        raise ValueError("capture requires nonempty logical module roles")
    modules = []
    for binding in selected.values():
        modules.append(model.get_submodule(binding.path))
    if len({id(module) for module in modules}) != len(modules):
        raise ValueError("bind each module once; repeated calls are captured as invocations")
    return selected


def _specs(bindings: Mapping[str, ModuleBoundary], capture_root: bool) -> list[dict]:
    specs = [
        {
            "id": f"{role}.{stage}",
            "module": role,
            "layer": binding.layer,
            "stage": stage,
        }
        for role, binding in bindings.items()
        for stage in (("input", "output") if binding.observe_inputs else ("output",))
    ]
    if capture_root:
        specs += [
            {"id": f"model.{stage}", "module": "model", "layer": None, "stage": stage}
            for stage in ("input", "output")
        ]
    return specs


def prepare_auto(
    directory: Path,
    model: torch.nn.Module,
    *,
    positions: Iterable[int],
    scope: str,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    max_modules: int | None = None,
    bindings: Mapping[str, ModuleBoundary] | None = None,
    capture_root: bool = True,
    ranks: Mapping[str, Sequence[int]] | None = None,
    sharded_weights: bool = False,
) -> list[str]:
    positions = list(positions)
    if not positions or any(type(p) is not int or p < 0 for p in positions):
        raise ValueError("positions must be a nonempty list of nonnegative integers")
    selected = _bindings(model, include, exclude, max_modules, bindings)
    EventCapture.prepare(
        directory,
        _specs(selected, capture_root) + (_weight_specs() if sharded_weights else []),
        scope=scope,
        ranks=ranks,
        coverage={
            "roles": list(selected),
            "model_endpoints": capture_root,
            "selection": "explicit semantic mapping"
            if bindings is not None
            else "automatic leaves",
            "max_modules": max_modules,
            "positions": positions,
            "sharded_weights": sharded_weights,
            "claim": "declared observed boundaries only, not every operation in the model",
        },
    )
    return [binding.path for binding in selected.values()]


class AutoCapture:
    """Observe every invocation and both model endpoints during native eager replay.

    Mapping weights explicitly enables independently verified inventories across
    renamed, fused or sharded runtimes. Without a mapping, actual state_dict
    names/values must agree. Capture never silently trusts a checkpoint path.
    """

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
        max_modules: int | None = None,
        bindings: Mapping[str, ModuleBoundary] | None = None,
        capture_root: bool = True,
        root_inputs: Callable[..., Any] | None = None,
        root_outputs: Callable[..., Any] | None = None,
        weights: Callable[[], Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]]]
        | None = None,
        rank: int = 0,
        sharded_weights: Callable[[], Mapping[str, Any]] | None = None,
    ):
        self.capture = EventCapture(directory, side, identity, rank=rank)
        self.model, self.positions = model, list(positions)
        if self.positions != self.capture.manifest["coverage"]["positions"]:
            raise ValueError("capture positions differ from prepared scope")
        self.bindings = _bindings(model, include, exclude, max_modules, bindings)
        expected = {spec["id"] for spec in _specs(self.bindings, capture_root)}
        if sharded_weights is not None:
            expected.update(spec["id"] for spec in _weight_specs())
        if (sharded_weights is not None) != self.capture.manifest["coverage"].get(
            "sharded_weights", False
        ):
            raise ValueError("sharded weight capture differs from prepared scope")
        if sharded_weights is not None and weights is not None:
            raise ValueError("choose a replicated or sharded weight reader, not both")
        if expected != set(self.capture.specs):
            raise ValueError("semantic boundary coverage differs from manifest")
        self.capture_root, self.root_inputs, self.root_outputs = (
            capture_root,
            root_inputs,
            root_outputs,
        )
        self.weights = weights or model.state_dict
        self.sharded_weights = sharded_weights
        self.handles: list[Any] = []
        self.frame: str | None = None
        self.frame_count = 0
        self.calls: dict[str, int] = {}
        self.stacks: dict[str, list[tuple]] = {role: [] for role in self.bindings}
        self.started = False

    def _weights(self) -> dict[str, str]:
        values = self.weights()
        items = values.items() if isinstance(values, Mapping) else values
        result = {}
        for name, value in items:
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                raise ValueError("weight inventory must map canonical names to actual tensors")
            if name in result:
                raise ValueError(f"duplicate canonical weight: {name}")
            result[name] = fingerprint(value)
        return result

    def _root_before(self, module, args, kwargs):
        if self.frame is not None:
            raise ValueError("recursive/concurrent root execution needs an explicit event adapter")
        self.frame = f"forward:{self.frame_count}"
        self.frame_count += 1
        self.calls.clear()
        if self.capture_root:
            value = (
                self.root_inputs(module, args, kwargs)
                if self.root_inputs
                else {"args": args, "kwargs": kwargs}
            )
            self.capture.record("model.input", value, frame=self.frame)

    def _root_after(self, module, args, kwargs, output):
        if self.capture_root:
            value = self.root_outputs(module, args, kwargs, output) if self.root_outputs else output
            self.capture.record(
                "model.output",
                value,
                frame=self.frame,
                inputs=[(self.frame, "model.input", 0)],
            )
        self.frame = None

    def __enter__(self):
        if self.started:
            raise RuntimeError("AutoCapture sessions are single-use")
        self.started = True
        if self.sharded_weights is None:
            self.initial_weights = self._weights()
            self.capture.identity["observed_weights"] = self.initial_weights
        else:
            values = self.sharded_weights()
            if not isinstance(values, Mapping) or not values:
                raise ValueError("sharded weight reader must return a nonempty mapping")
            self.initial_weights = _inventory_identity(encode_tree(values))
        self.capture.__enter__()
        try:
            if self.sharded_weights is not None:
                self.capture.record("weights.before", values, frame="weights")
            self.handles.append(
                self.model.register_forward_pre_hook(self._root_before, with_kwargs=True)
            )
            for role, binding in self.bindings.items():
                module = self.model.get_submodule(binding.path)
                route = type(module).__qualname__

                def before(mod, args, kwargs, role=role, binding=binding, route=route):
                    if self.frame is None:
                        raise ValueError("observed module called outside the model forward")
                    invocation = self.calls.get(role, 0)
                    self.calls[role] = invocation + 1
                    key = (self.frame, f"{role}.input", invocation)
                    if binding.observe_inputs:
                        value = (
                            binding.inputs(mod, args, kwargs)
                            if binding.inputs
                            else {"args": args, "kwargs": kwargs}
                        )
                        self.capture.record(
                            key[1],
                            value,
                            frame=key[0],
                            invocation=key[2],
                            route=route,
                        )
                    self.stacks[role].append(key)

                def after(mod, args, kwargs, output, role=role, binding=binding, route=route):
                    key = self.stacks[role].pop()
                    value = (
                        binding.outputs(mod, args, kwargs, output) if binding.outputs else output
                    )
                    self.capture.record(
                        f"{role}.output",
                        value,
                        frame=key[0],
                        invocation=key[2],
                        inputs=[key] if binding.observe_inputs else [],
                        route=route,
                    )

                self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                self.handles.append(module.register_forward_hook(after, with_kwargs=True))
            self.handles.append(
                self.model.register_forward_hook(self._root_after, with_kwargs=True)
            )
            return self
        except Exception as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        try:
            if self.frame is not None or any(self.stacks.values()):
                self.capture.errors.append("unfinished forward invocation")
            if self.sharded_weights is None:
                final_weights = self._weights()
            else:
                values = self.sharded_weights()
                final_weights = _inventory_identity(encode_tree(values))
                self.capture.record("weights.after", values, frame="weights")
            if final_weights != self.initial_weights:
                self.capture.errors.append("actual weights changed during observation")
        except Exception as error:
            self.capture.errors.append(f"weight verification failed: {error}")
        return self.capture.__exit__(exc_type, exc, traceback)


@contextmanager
def auto_capture(*args, **kwargs):
    with AutoCapture(*args, **kwargs) as capture:
        yield capture
