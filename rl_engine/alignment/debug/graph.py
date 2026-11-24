# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Preallocated graph observers: device copies in capture, host I/O after replay."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from contextlib import contextmanager
from typing import Any

import torch

from .events import EventCapture, encode_tree


def _allocate(value):
    if isinstance(value, torch.Tensor):
        return torch.empty_like(value, memory_format=torch.contiguous_format, requires_grad=False)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{f.name: _allocate(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, dict):
        return {k: _allocate(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_allocate(v) for v in value)
    if isinstance(value, range):
        return value
    return value


def _copy(target, value, devices):
    if isinstance(target, torch.Tensor):
        if (
            not isinstance(value, torch.Tensor)
            or target.shape != value.shape
            or target.dtype != value.dtype
            or target.device != value.device
        ):
            raise ValueError("graph probe shape/dtype/device changed; recapture required")
        devices.add(target.device)
        target.copy_(value.detach())
    elif is_dataclass(target) and not isinstance(target, type):
        if type(target) is not type(value):
            raise ValueError("graph probe view type changed")
        for field in fields(target):
            _copy(getattr(target, field.name), getattr(value, field.name), devices)
    elif isinstance(target, dict):
        if not isinstance(value, dict) or target.keys() != value.keys():
            raise ValueError("graph probe fields changed")
        for key in target:
            _copy(target[key], value[key], devices)
    elif isinstance(target, (tuple, list)):
        if type(target) is not type(value) or len(target) != len(value):
            raise ValueError("graph probe structure changed")
        for a, b in zip(target, value, strict=True):
            _copy(a, b, devices)
    elif type(target) is not type(value) or target != value:
        raise ValueError("graph probe static metadata changed; use a tensor for dynamic state")


class GraphCapture:
    """Instrument a CUDA/HIP graph without CPU copies, hashing or I/O inside it.

    Declare slots before warmup/capture. Call ``stage`` at real native boundaries
    INSIDE graph capture. Each slot must execute exactly once on every replay;
    repeated invocations need distinct slots/boundaries. Use ``replay`` for each
    diagnostic replay so buffers are frozen before the next execution mutates
    them. Never wrap capture itself with AutoCapture's eager hooks.
    """

    def __init__(self, capture: EventCapture):
        self.capture = capture
        self.slots: dict[str, dict[str, Any]] = {}
        self.replays = 0
        self.sealed = False
        self.sequence = None

    def declare(
        self,
        boundary: str,
        template,
        *,
        inputs: tuple[str, ...] = (),
        route: str = "native_graph",
        contracts: dict | None = None,
    ):
        if self.sealed or boundary in self.slots or boundary not in self.capture.specs:
            raise ValueError("declare unique prepared graph slots before replay")
        if any(parent not in self.slots for parent in inputs):
            raise ValueError("graph inputs must be previously declared slots")
        encode_tree(template)  # Validate supported evidence outside graph capture.
        buffer = _allocate(template)
        devices = set()
        with torch.no_grad():
            _copy(buffer, template, devices)
        if len(devices) != 1:
            raise ValueError("each graph slot needs tensors on exactly one device")
        device = devices.pop()
        if self.slots and device != next(iter(self.slots.values()))["counter"].device:
            raise ValueError("one graph observer must use one device")
        if self.sequence is None:
            self.sequence = torch.zeros((), dtype=torch.int64, device=device)
        self.slots[boundary] = {
            "buffer": buffer,
            "counter": torch.zeros((), dtype=torch.int64, device=device),
            "order": torch.zeros((), dtype=torch.int64, device=device),
            "inputs": inputs,
            "route": route,
            "contracts": dict(contracts or {}),
        }

    def stage(self, boundary: str, value) -> None:
        """Only device operations; called by the native instrumented graph."""
        try:
            slot = self.slots[boundary]
            with torch.no_grad():
                _copy(slot["buffer"], value, set())
                slot["order"].copy_(self.sequence)
                self.sequence.add_(1)
                slot["counter"].add_(1)
        except Exception as exc:
            self.capture.errors.append(f"graph staging failed: {exc}")
            raise

    @contextmanager
    def observe(self, model, bindings):
        """Install ModuleBoundary hooks for native warmup/graph construction.

        Slots must already be declared. Selectors must be graph-safe and only
        observe native values. Hooks are removed after capture; the device copy
        operations they recorded remain part of the graph and run on replay.
        """
        handles = []
        try:
            for role, binding in bindings.items():
                module = model.get_submodule(binding.path)
                stages = ("input", "output") if binding.observe_inputs else ("output",)
                if any(f"{role}.{stage}" not in self.slots for stage in stages):
                    raise ValueError("declare graph slots for every mapped module boundary")

                def before(mod, args, kwargs, role=role, binding=binding):
                    value = (
                        binding.inputs(mod, args, kwargs)
                        if binding.inputs
                        else {"args": args, "kwargs": kwargs}
                    )
                    self.stage(f"{role}.input", value)

                def after(mod, args, kwargs, output, role=role, binding=binding):
                    value = (
                        binding.outputs(mod, args, kwargs, output) if binding.outputs else output
                    )
                    self.stage(f"{role}.output", value)

                if binding.observe_inputs:
                    handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after, with_kwargs=True))
            yield self
        except Exception as exc:
            self.capture.errors.append(f"graph boundary observation failed: {exc}")
            raise
        finally:
            for handle in handles:
                handle.remove()

    def replay(self, graph, *, frame: str | None = None) -> None:
        self.sealed = True
        frame = f"graph:{self.replays}" if frame is None else frame
        self.replays += 1
        try:
            if not self.slots:
                raise ValueError("no graph probe slots declared")
            if any(slot["counter"].is_cuda for slot in self.slots.values()):
                if torch.cuda.is_current_stream_capturing():
                    raise ValueError("graph evidence must be flushed outside stream capture")
            for slot in self.slots.values():
                slot["counter"].zero_()
            self.sequence.zero_()
            graph.replay()
            # Read counters after native replay. A hook that ran only during
            # warmup/capture cannot accidentally pass with stale tensor buffers.
            if any(slot["counter"].item() != 1 for slot in self.slots.values()):
                raise ValueError("missing/repeated graph probe execution; stale evidence rejected")
            if any(slot["order"].item() != order for order, slot in enumerate(self.slots.values())):
                raise ValueError("native graph probe order differs from declared dependencies")
            for boundary, slot in self.slots.items():
                self.capture.record(
                    boundary,
                    slot["buffer"],
                    frame=frame,
                    inputs=[(frame, parent, 0) for parent in slot["inputs"]],
                    route=slot["route"],
                    contracts=slot["contracts"],
                )
        except Exception as exc:
            self.capture.errors.append(f"graph replay failed: {exc}")
            raise
