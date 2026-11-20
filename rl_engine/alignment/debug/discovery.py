# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Automatic, fail-closed discovery of corresponding native decoder blocks.

No model names, layer-path conventions, numerical-error minimization or reference
forwards are used. Evidence comes from executed state flow and actual parameters.
Opaque residual tuples remain opaque; verified affine primitives can still be
compared inside such blocks without reconstructing an unobserved residual sum.
"""

from __future__ import annotations

from collections import defaultdict
import inspect
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .core import atomic_save, fingerprint, snapshot
from .discovery_weights import (
    parameter_inventory,
)

DISCOVERY_SCHEMA = "rlkernel.block_discovery.v1"
SIDES = ("training", "rollout")


class MappingRequired(ValueError):
    """Evidence cannot certify an automatic correspondence."""


def _stacks(model) -> list[dict]:
    candidates = {}
    for path, parent in model.named_modules():
        children = [
            (f"{path}.{name}" if path else name, module)
            for name, module in parent.named_children()
            # Pure vector-normalization siblings cannot supply the affine
            # evidence required by this discovery mode. Avoid quadratic
            # candidate searches over every pair of norms in a deep model.
            if any(parameter.ndim == 2 for parameter in module.parameters())
        ]
        groups = []
        if isinstance(parent, (torch.nn.ModuleList, torch.nn.Sequential, torch.nn.ModuleDict)):
            groups.append(children)
        else:
            shapes = defaultdict(list)
            for name, module in children:
                signature = tuple(sorted(tuple(p.shape) for p in module.parameters()))
                shapes[signature].append((name, module))
            groups.extend(shapes.values())
        for group in groups:
            if len(group) >= 2:
                names = tuple(name for name, _ in group)
                candidates[names] = {"container": path, "blocks": list(names)}
    return list(candidates.values())


def _buffers(model):
    return {name: fingerprint(value) for name, value in model.named_buffers()}


def _bound_arguments(module, args, kwargs):
    try:
        return {
            **inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments,
            **kwargs,
        }
    except (ValueError, TypeError) as exc:
        raise MappingRequired("cannot inspect the native forward input interface") from exc


def _find_argument(arguments, names):
    matches = [arguments[name] for name in names if isinstance(arguments.get(name), torch.Tensor)]
    if len(matches) != 1:
        raise MappingRequired(f"provide exactly one actual native input from {names}")
    return matches[0]


def _root_rows(model, args, kwargs, case):
    arguments = _bound_arguments(model, args, kwargs)
    tokens = snapshot(_find_argument(arguments, ("input_ids", "tokens", "token_ids")))
    positions = snapshot(_find_argument(arguments, ("position_ids", "positions")))
    if any(value.dtype not in (torch.int32, torch.int64) for value in (tokens, positions)):
        raise MappingRequired("actual token and position IDs must be integer tensors, not casts")
    tokens, positions = tokens.long(), positions.long()
    if arguments.get("use_cache") is True:
        raise MappingRequired(
            "automatic discovery requires a no-reuse prefill, not cached decoding"
        )
    for name in ("past_key_values", "past_key_value", "layer_past"):
        if arguments.get(name) is not None:
            raise MappingRequired(f"{name} is opaque/nonempty; explicit cache evidence is required")
    mask = arguments.get("attention_mask")
    if mask is not None and (
        not isinstance(mask, torch.Tensor)
        or mask.shape != case["mask"].shape
        or not torch.equal(snapshot(mask), case["mask"].to(mask.dtype))
    ):
        raise MappingRequired("actual attention mask differs or is not the declared 2-D token mask")
    expected = case["tokens"]
    batch, length = expected.shape
    if tokens.ndim == 2 and tokens.shape == expected.shape:
        if positions.shape != tokens.shape:
            raise MappingRequired("actual position tensor shape differs from input tokens")
        sequences = list(zip(tokens.tolist(), positions.tolist(), strict=True))
        layout = "matrix"
    elif (
        tokens.ndim == 1 and tokens.numel() == expected.numel() and positions.shape == tokens.shape
    ):
        sequences = list(
            zip(
                tokens.reshape(batch, length).tolist(),
                positions.reshape(batch, length).tolist(),
                strict=True,
            )
        )
        layout = "packed"
    else:
        raise MappingRequired("chunked, padded or sharded native rows need an explicit logical map")
    signatures = [
        (tuple(t), tuple(p))
        for t, p in zip(expected.tolist(), case["positions"].tolist(), strict=True)
    ]
    if len(set(signatures)) != batch:
        raise MappingRequired(
            "identical samples have ambiguous ownership; supply explicit sample IDs"
        )
    rows, seen = [], set()
    for token_row, position_row in sequences:
        signature = (tuple(token_row), tuple(position_row))
        if signature not in signatures:
            raise MappingRequired("actual native tokens/positions differ from the frozen case")
        sample = signatures.index(signature)
        if sample in seen:
            raise MappingRequired("duplicate or missing native sample")
        seen.add(sample)
        rows.extend((sample, position) for position in position_row)
    return {
        "rows": rows,
        "layout": layout,
        "mask_evidence": "observed" if mask is not None else "declared",
    }


class DiscoverCapture:
    """Capture an unfamiliar model without ModuleBoundary/weight/shape mappings.

    The caller still runs its real native forward and provides the frozen input
    case. Initial automatic scope: eval-mode eager, TP=CP=1, complete unpadded
    prefill. Unsupported evidence produces a mapping request, never a fake pass.
    """

    def __init__(
        self,
        directory: Path,
        side: str,
        model: torch.nn.Module,
        *,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        context: Mapping[str, Any] | None = None,
        stack: str | None = None,
        max_capture_bytes: int = 512 * 1024 * 1024,
    ):
        if side not in SIDES:
            raise ValueError("side must be training or rollout")
        tokens = snapshot(tokens)
        if (
            tokens.ndim != 2
            or not tokens.numel()
            or tokens.dtype not in (torch.int32, torch.int64)
            or bool((tokens < 0).any())
        ):
            raise ValueError("provide nonempty integer [batch, sequence] frozen tokens")
        positions = (
            torch.arange(tokens.shape[1]).expand_as(tokens).clone()
            if positions is None
            else snapshot(positions)
        )
        mask = torch.ones_like(tokens, dtype=torch.bool) if mask is None else snapshot(mask)
        if (
            positions.shape != tokens.shape
            or mask.shape != tokens.shape
            or not bool((mask == 1).all())
            or positions.dtype not in (torch.int32, torch.int64)
            or not torch.equal(positions, torch.arange(tokens.shape[1]).expand_as(positions))
        ):
            raise ValueError(
                "automatic discovery currently requires complete unpadded prefill rows"
            )
        if max_capture_bytes <= 0:
            raise ValueError("max_capture_bytes must be positive")
        self.directory, self.side, self.model = Path(directory), side, model
        self.case = {
            "tokens": tokens.long(),
            "positions": positions.long(),
            "mask": mask.bool(),
            "context": json.loads(
                json.dumps(dict(context or {"phase": "prefill"}), allow_nan=False)
            ),
        }
        self.candidates = _stacks(model)
        if stack is not None:
            self.candidates = [
                candidate for candidate in self.candidates if candidate["container"] == stack
            ]
        self.limit, self.used = max_capture_bytes, 0
        self.handles, self.frames, self.errors = [], [], []
        self.frame = None
        self.clock = 0
        self.pending = defaultdict(list)
        self.started = False

    def _tensors(self, value) -> dict:
        tensors, opaque = {}, []

        def visit(item, path):
            if isinstance(item, torch.Tensor):
                self.used += item.numel() * item.element_size()
                if self.used > self.limit:
                    raise MappingRequired(
                        "capture byte limit exceeded; shorten the frozen replay or raise the limit"
                    )
                key = json.dumps(path, separators=(",", ":"))
                tensors[key] = {
                    "value": snapshot(item),
                    "storage": [
                        str(item.device),
                        item.untyped_storage().data_ptr(),
                        item.storage_offset(),
                        item.numel(),
                        str(item.dtype),
                    ],
                }
            elif isinstance(item, Mapping):
                for name, child in item.items():
                    if not isinstance(name, str):
                        opaque.append(f"{path}: non-string mapping key")
                    else:
                        visit(child, [*path, name])
            elif isinstance(item, (tuple, list)):
                for index, child in enumerate(item):
                    visit(child, [*path, index])
            elif item is not None and type(item) not in (bool, int, float, str):
                opaque.append(f"{path}: {type(item).__qualname__}")

        visit(value, [])
        return {"tensors": tensors, "opaque": opaque}

    def _root_before(self, module, args, kwargs):
        if self.frame is not None:
            raise MappingRequired(
                "concurrent or recursive root forwards require explicit event mapping"
            )
        self.frame = {
            "index": len(self.frames),
            "calls": [],
            "closed": False,
            **_root_rows(module, args, kwargs, self.case),
        }
        self.frames.append(self.frame)

    def _root_after(self, module, args, kwargs, output):
        self.frame["closed"] = True
        self.frame = None

    def _before(self, module, args, kwargs, *, path, kind):
        if self.frame is None:
            raise MappingRequired(f"{path}: observed call outside the model forward")
        self.clock += 1
        call = {
            "path": path,
            "kind": kind,
            "start": self.clock,
            "end": None,
            "input": self._tensors({"args": args, "kwargs": kwargs}),
        }
        self.frame["calls"].append(call)
        self.pending[(kind, path)].append(call)

    def _after(self, module, args, kwargs, output, *, path, kind):
        self.clock += 1
        call = self.pending[(kind, path)].pop()
        call["end"] = self.clock
        call["output"] = self._tensors(output)

    def _write(self, state):
        atomic_save(
            self.folder / "probe.pt",
            {
                "schema_version": DISCOVERY_SCHEMA,
                "state": state,
                "errors": self.errors,
                "case": self.case,
                "candidates": self.candidates,
                "inventory": getattr(self, "inventory", {}),
                "frames": self.frames,
                "side": self.side,
            },
        )

    def __enter__(self):
        from functools import partial

        if self.started:
            raise RuntimeError("discovery sessions are single-use")
        self.started = True
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "capture-manifest.json").exists():
            raise ValueError("use a new directory for automatic discovery, not a mapped bundle")
        try:
            with (self.directory / "discovery-manifest.json").open("x", encoding="utf-8") as handle:
                json.dump({"schema_version": DISCOVERY_SCHEMA}, handle)
        except FileExistsError:
            pass
        self.folder = self.directory / "discovery" / self.side
        self.folder.mkdir(parents=True, exist_ok=False)
        self._write("open")
        try:
            if any(module.training for module in self.model.modules()):
                raise MappingRequired(
                    "use explicit eval-mode frozen replay; discovery never changes model mode"
                )
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
                raise MappingRequired(
                    "automatic shard ownership is not proven; use the mapped TP/CP capture API"
                )
            if hasattr(self.model, "_orig_mod"):
                raise MappingRequired("compiled model: use an explicitly scoped eager replay")
            self.inventory = parameter_inventory(self.model)
            self.initial_buffers = _buffers(self.model)
            self.handles.append(
                self.model.register_forward_pre_hook(self._root_before, with_kwargs=True)
            )
            modules = dict(self.model.named_modules())
            blocks = sorted({path for candidate in self.candidates for path in candidate["blocks"]})
            for path in blocks:
                self.handles.append(
                    modules[path].register_forward_pre_hook(
                        partial(self._before, path=path, kind="block"),
                        with_kwargs=True,
                    )
                )
            for path in self.inventory["owners"]:
                self.handles.append(
                    modules[path].register_forward_pre_hook(
                        partial(self._before, path=path, kind="primitive"),
                        with_kwargs=True,
                    )
                )
                self.handles.append(
                    modules[path].register_forward_hook(
                        partial(self._after, path=path, kind="primitive"),
                        with_kwargs=True,
                    )
                )
            for path in blocks:
                self.handles.append(
                    modules[path].register_forward_hook(
                        partial(self._after, path=path, kind="block"),
                        with_kwargs=True,
                    )
                )
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
        if exc is not None:
            self.errors.append(f"{exc_type.__name__}: {exc}")
        if self.frame is not None or any(self.pending.values()):
            self.errors.append("unfinished forward invocation")
        if hasattr(self, "inventory"):
            try:
                if parameter_inventory(self.model) != self.inventory:
                    self.errors.append("actual parameters or primitive storage contracts changed")
                if _buffers(self.model) != self.initial_buffers:
                    self.errors.append(
                        "runtime buffers changed; warm up or explicitly map mutable state"
                    )
            except Exception as error:
                self.errors.append(f"state verification failed: {error}")
        self._write("failed" if self.errors else "complete")
        return False
