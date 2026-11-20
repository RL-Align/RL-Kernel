# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Execution-event evidence, independent of models, kernels and launchers.

Unlike the legacy, explicitly ordered Capture API, this format observes every
invocation. A successful close is part of the evidence, not inferred from the
presence of some tensor files. Physical rank order is never semantic order.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .core import atomic_save, bitdiff, snapshot
from .evidence import IDENTITIES, diagnosis
from .shards import ShardSet, TensorShard, encode_shards, merge_shards
from .routing import RoutedTensor, encode_routes, merge_routes

EVENT_SCHEMA = "rlkernel.events.v1"
SIDES = ("training", "rollout")


def _load_snapshot(path: Path | str) -> dict[str, Any]:
    # Torch's safe unpickler can raise IndexError/AssertionError as well as
    # UnpicklingError for truncated/corrupt payloads. Never turn these into a
    # pass, or retry with unsafe weights_only=False.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("snapshot payload must be a mapping")
        return payload
    except Exception as exc:
        raise ValueError(
            f"unreadable snapshot {Path(path).name}: {type(exc).__name__}: {exc}"
        ) from exc


@dataclass(frozen=True)
class LogicalTensor:
    """Explicit [rows, features] view; rows are (sample_id, token_position).

    Transposes, QKV splits and residual reconstruction belong in the caller's
    semantic binding. Never cast values merely to make a bitwise test pass.
    """

    value: torch.Tensor
    rows: Sequence[tuple[int, int]]
    feature_offset: int = 0
    feature_width: int | None = None


def encode_tree(value: Any) -> dict[str, Any]:
    if isinstance(value, RoutedTensor):
        return encode_routes(value)
    if isinstance(value, TensorShard):
        return encode_shards([value])
    if isinstance(value, ShardSet):
        return encode_shards(value.shards)
    if isinstance(value, LogicalTensor):
        if value.value.ndim != 2:
            raise ValueError("logical tensors require explicit [rows, features] views")
        rows = [list(row) for row in value.rows]
        width = value.feature_width if value.feature_width is not None else value.value.shape[-1]
        if (
            value.value.ndim != 2
            or value.value.shape[0] != len(rows)
            or not rows
            or any(len(row) != 2 or any(type(x) is not int or x < 0 for x in row) for row in rows)
            or len({tuple(row) for row in rows}) != len(rows)
            or value.feature_offset < 0
            or value.feature_offset + value.value.shape[1] > width
        ):
            raise ValueError("invalid logical sample/token/feature mapping")
        return {
            "kind": "logical",
            "value": snapshot(value.value),
            "rows": rows,
            "offset": value.feature_offset,
            "width": width,
        }
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "value": snapshot(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("capture mappings require string keys")
        return {
            "kind": "mapping",
            "items": {key: encode_tree(value[key]) for key in sorted(value)},
        }
    if isinstance(value, (tuple, list)):
        return {"kind": type(value).__name__, "items": [encode_tree(v) for v in value]}
    if value is None or type(value) in (bool, int, float, str):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("nonfinite scalar state")
        return {"kind": type(value).__name__, "value": value}
    raise TypeError(
        f"unobservable state {type(value).__qualname__}; supply an explicit semantic selector"
    )


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != EVENT_SCHEMA:
        raise ValueError("unsupported execution-event schema")
    specs = manifest["boundaries"]
    names = [spec["id"] for spec in specs]
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("event boundaries must have unique nonempty IDs")
    if any(not spec.get("stage") for spec in specs):
        raise ValueError("event boundaries require a stage")
    for side in SIDES:
        ranks = manifest["ranks"][side]
        if (
            not ranks
            or len(set(ranks)) != len(ranks)
            or any(type(r) is not int or r < 0 for r in ranks)
        ):
            raise ValueError("declare the expected worker ranks on each side")


class EventCapture:
    """Append-only native execution observer with explicit completion evidence."""

    @staticmethod
    def prepare(
        directory: Path,
        boundaries: list[dict[str, Any]],
        *,
        scope: str,
        ranks: Mapping[str, Sequence[int]] | None = None,
        coverage: Mapping[str, Any] | None = None,
    ) -> None:
        manifest = {
            "schema_version": EVENT_SCHEMA,
            "scope": scope,
            "boundaries": boundaries,
            "ranks": dict(ranks or {side: [0] for side in SIDES}),
            "coverage": dict(coverage or {}),
        }
        _validate_manifest(manifest)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "capture-manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)

    def __init__(self, directory: Path, side: str, identity: Mapping[str, Any], *, rank: int = 0):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "capture-manifest.json").read_text("utf-8"))
        _validate_manifest(self.manifest)
        if side not in SIDES or rank not in self.manifest["ranks"][side]:
            raise ValueError("worker not declared in capture manifest")
        if any(identity.get(key) is None for key in IDENTITIES):
            raise ValueError("actual weights/tokens/positions/mask/sampling identities required")
        # JSON roundtrip freezes nested metadata and rejects unserializable state.
        self.identity = json.loads(json.dumps(dict(identity), allow_nan=False))
        self.side, self.rank = side, rank
        self.specs = {spec["id"]: spec for spec in self.manifest["boundaries"]}
        self.folder = self.directory / "events" / f"{side}-rank{rank}"
        self.count = 0
        self.keys: set[tuple[str, str, int]] = set()
        self.opened = False
        self.errors: list[str] = []

    def __enter__(self):
        if self.opened:
            raise RuntimeError("capture sessions are single-use")
        self.folder.mkdir(parents=True, exist_ok=False)
        self.opened = True
        self._session("open")
        return self

    def _session(self, state: str) -> None:
        atomic_save(
            self.folder / "session.pt",
            {
                "schema_version": EVENT_SCHEMA,
                "side": self.side,
                "rank": self.rank,
                "identity": self.identity,
                "state": state,
                "count": self.count,
                "errors": self.errors,
            },
        )

    def record(
        self,
        boundary: str,
        value: Any,
        *,
        frame: str = "forward:0",
        invocation: int = 0,
        inputs: Sequence[tuple[str, str, int]] = (),
        route: str = "unknown",
        contracts: Mapping[str, Any] | None = None,
    ) -> tuple[str, str, int]:
        if not self.opened or boundary not in self.specs:
            raise ValueError("open the session and declare the boundary before recording")
        key = (frame, boundary, invocation)
        try:
            if (
                not isinstance(frame, str)
                or not frame
                or type(invocation) is not int
                or invocation < 0
            ):
                raise ValueError("invalid event identity")
            if key in self.keys:
                raise ValueError(f"duplicate execution event: {key}")
            if any(tuple(parent) not in self.keys for parent in inputs):
                raise ValueError("event dependencies must already have been observed")
            tree = encode_tree(value)
            observed_contracts = json.loads(json.dumps(dict(contracts or {}), allow_nan=False))
            atomic_save(
                self.folder / f"{self.count:08d}.pt",
                {
                    "schema_version": EVENT_SCHEMA,
                    "key": list(key),
                    "order": self.count,
                    "inputs": [list(parent) for parent in inputs],
                    "route": route,
                    "contracts": observed_contracts,
                    "tree": tree,
                },
            )
            self.keys.add(key)
            self.count += 1
            return key
        except Exception as exc:
            self.errors.append(f"{boundary}: {type(exc).__name__}: {exc}")
            raise

    def __exit__(self, exc_type, exc, traceback):
        if exc is not None:
            self.errors.append(f"{exc_type.__name__}: {exc}")
        self._session("failed" if self.errors else "complete")
        self.opened = False
        return False


def _merge_trees(trees: list[dict[str, Any]]) -> dict[str, Any]:
    """Join explicitly mapped shards; ordinary tensors must be exact replicas."""
    first = trees[0]
    kind = first["kind"]
    if any(tree["kind"] != kind for tree in trees):
        raise ValueError("replica value structures differ")
    if kind == "sharded":
        return merge_shards(trees)
    if kind == "routed":
        return merge_routes(trees)
    if kind == "logical":
        widths = {tree["width"] for tree in trees}
        dtypes = {tree["value"].dtype for tree in trees}
        if len(widths) != 1 or len(dtypes) != 1:
            raise ValueError("logical shard widths/dtypes differ")
        rows: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
        for tree in trees:
            if tree["value"].ndim != 2 or len(tree["rows"]) != tree["value"].shape[0]:
                raise ValueError("invalid logical shard rows")
            for index, position in enumerate(tree["rows"]):
                pieces = rows.setdefault(tuple(position), {})
                offset, value = tree["offset"], tree["value"][index]
                if offset in pieces and not bitdiff(pieces[offset], value).get("equal"):
                    raise ValueError("logical replicas disagree")
                pieces[offset] = value
        joined = []
        for position in sorted(rows):
            cursor, pieces = 0, []
            for offset, value in sorted(rows[position].items()):
                if offset != cursor:
                    raise ValueError("logical feature gap/overlap")
                cursor += value.numel()
                pieces.append(value)
            if cursor != first["width"]:
                raise ValueError("incomplete logical feature coverage")
            joined.append(torch.cat(pieces))
        if not joined:
            raise ValueError("empty logical tensor")
        return {"kind": "logical", "rows": sorted(rows), "value": torch.stack(joined)}
    if kind in ("mapping", "tuple", "list"):
        items = first["items"]
        keys = list(items) if kind == "mapping" else list(range(len(items)))
        for tree in trees:
            other = tree["items"]
            if (list(other) if kind == "mapping" else list(range(len(other)))) != keys:
                raise ValueError("replica container structures differ")
        merged = {key: _merge_trees([tree["items"][key] for tree in trees]) for key in keys}
        return {"kind": kind, "items": merged if kind == "mapping" else list(merged.values())}
    for other in trees[1:]:
        if kind == "tensor":
            if not bitdiff(first["value"], other["value"]).get("equal"):
                raise ValueError("unmapped rank tensors are not identical replicas")
        elif first != other:
            raise ValueError("replica scalar state differs")
    return first


def _compare_trees(left: dict, right: dict, path: str = "$") -> tuple[list[dict], list[str]]:
    deltas, errors = [], []
    kind = left["kind"]
    if right["kind"] != kind:
        return [], [f"{path}: value structures differ"]
    if kind in ("tensor", "logical"):
        if kind == "logical" and left["rows"] != right["rows"]:
            return [], [f"{path}: logical sample/token coverage differs"]
        delta = bitdiff(left["value"], right["value"])
        if not delta.get("comparable") or not delta.get("finite"):
            return [], [f"{path}: invalid tensor comparison ({delta.get('reason', 'nonfinite')})"]
        if not delta["equal"]:
            delta["tensor_path"] = path
            if kind == "logical":
                index = delta["first_index"]
                delta.update(
                    sample_id=left["rows"][index[0]][0],
                    position=left["rows"][index[0]][1],
                    feature=index[1],
                )
            deltas.append(delta)
    elif kind in ("mapping", "tuple", "list"):
        a, b = left["items"], right["items"]
        keys = list(a) if kind == "mapping" else list(range(len(a)))
        other_keys = list(b) if kind == "mapping" else list(range(len(b)))
        if keys != other_keys:
            return [], [f"{path}: container fields differ"]
        for key in keys:
            differences, issues = _compare_trees(a[key], b[key], f"{path}[{key!r}]")
            deltas.extend(differences)
            errors.extend(issues)
    elif left["value"] != right["value"]:
        deltas.append(
            {
                "tensor_path": path,
                "equal": False,
                "training": left["value"],
                "rollout": right["value"],
            }
        )
    return deltas, errors


def analyze_events(directory: Path) -> dict[str, Any]:
    """Report first observed event only when both observed sequences agree."""
    directory = Path(directory)
    manifest = json.loads((directory / "capture-manifest.json").read_text("utf-8"))
    _validate_manifest(manifest)
    specs = {spec["id"]: spec for spec in manifest["boundaries"]}
    errors, identities, workers = [], [], {}
    expected_folders = {f"{side}-rank{rank}" for side in SIDES for rank in manifest["ranks"][side]}
    actual_folders = {path.name for path in (directory / "events").glob("*")}
    if actual_folders != expected_folders:
        errors.append("missing or undeclared workers")
    for side in SIDES:
        workers[side] = []
        for rank in manifest["ranks"][side]:
            folder = directory / "events" / f"{side}-rank{rank}"
            try:
                session = _load_snapshot(folder / "session.pt")
                if (
                    session["schema_version"] != EVENT_SCHEMA
                    or session["side"] != side
                    or session["rank"] != rank
                    or session["state"] != "complete"
                    or session["errors"]
                ):
                    errors.append(f"{side}/rank{rank}: incomplete or failed execution")
                identity = session["identity"]
                if any(identity.get(key) is None for key in IDENTITIES):
                    errors.append(f"{side}/rank{rank}: incomplete runtime identity")
                identities.append(identity)
                count = session["count"]
                if type(count) is not int or count < 1:
                    raise ValueError("no completed execution events")
                paths = sorted(folder.glob("[0-9]*.pt"))
                if [path.name for path in paths] != [f"{i:08d}.pt" for i in range(count)]:
                    raise ValueError("missing or extra event records")
                events, seen = [], set()
                for order, path in enumerate(paths):
                    event = _load_snapshot(path)
                    key = tuple(event["key"])
                    if (
                        event["schema_version"] != EVENT_SCHEMA
                        or event["order"] != order
                        or len(key) != 3
                        or key[1] not in specs
                        or key in seen
                        or any(tuple(parent) not in seen for parent in event["inputs"])
                    ):
                        raise ValueError("invalid event identity/order/dependencies")
                    # Keep only the index in memory; load one paired boundary
                    # at a time below instead of retaining the whole model trace.
                    event.pop("tree")
                    event["_path"] = str(path)
                    events.append(event)
                    seen.add(key)
                observed = {event["key"][1] for event in events}
                missing = set(specs) - observed
                if missing:
                    errors.append(f"{side}/rank{rank}: unobserved boundaries: {sorted(missing)}")
                workers[side].append(events)
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                EOFError,
            ) as exc:
                errors.append(f"{side}/rank{rank}: {exc}")
    if not identities or any(identity != identities[0] for identity in identities[1:]):
        errors.append("runtime weights/tokens/positions/mask/sampling identities differ or missing")
    sequences = [
        [tuple(e["key"]) for e in events] for sides in workers.values() for events in sides
    ]
    order_valid = bool(sequences) and all(sequence == sequences[0] for sequence in sequences)
    if not order_valid:
        errors.append("execution event order/count differs; semantic correspondence is not proven")
    comparisons, first, previous, evidence, prior = [], None, None, None, {}
    # Do not fabricate correspondence from alphabetical names or registration order.
    if order_valid and all(workers[side] for side in SIDES):
        for index, key in enumerate(sequences[0]):
            spec = specs[key[1]]
            records = {side: [worker[index] for worker in workers[side]] for side in SIDES}
            issues, contracts, trees = [], {}, {}
            dependencies = records["training"][0]["inputs"]
            if any(record["inputs"] != dependencies for side in SIDES for record in records[side]):
                issues.append("observed input dependencies differ")
            for side in SIDES:
                try:
                    trees[side] = _merge_trees(
                        [_load_snapshot(record["_path"])["tree"] for record in records[side]]
                    )
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    EOFError,
                ) as exc:
                    issues.append(f"{side}: {exc}")
                contracts[side] = records[side][0]["contracts"]
                if any(record["contracts"] != contracts[side] for record in records[side]):
                    issues.append(f"{side}: replica contracts differ")
                for required in spec.get("contracts", []):
                    if contracts[side].get(required) is None:
                        issues.append(f"{side}: missing contract {required}")
            contract_differences = {
                name: {side: contracts[side].get(name) for side in SIDES}
                for name in contracts["training"].keys() | contracts["rollout"].keys()
                if contracts["training"].get(name) != contracts["rollout"].get(name)
            }
            if contract_differences:
                issues.append("observed calculation contracts differ")
            deltas = []
            if len(trees) == 2:
                try:
                    deltas, tensor_errors = _compare_trees(trees["training"], trees["rollout"])
                    issues.extend(tensor_errors)
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    IndexError,
                    AttributeError,
                    RuntimeError,
                ) as exc:
                    issues.append(f"invalid tensor-tree evidence: {exc}")
            parent_status = [prior.get(tuple(parent), "not_comparable") for parent in dependencies]
            row = {
                "boundary": key[1],
                "frame": key[0],
                "invocation": key[2],
                "module": spec.get("module", "mapped"),
                "layer": spec.get("layer"),
                "stage": spec["stage"],
                "errors": issues,
                "contract_differences": contract_differences,
                "routes": {side: sorted({r["route"] for r in records[side]}) for side in SIDES},
                "status": "not_comparable" if issues else "diverged" if deltas else "equal",
                "input_evidence": (
                    "unknown"
                    if not parent_status or "not_comparable" in parent_status
                    else "different"
                    if "diverged" in parent_status
                    else "equal"
                ),
            }
            errors.extend(f"{key}: {issue}" for issue in issues)
            comparisons.append(row)
            prior[key] = row["status"]
            if deltas and first is None:
                first = {**row, **deltas[0], "last_equal_boundary": previous}
                evidence = trees
            if row["status"] == "equal":
                previous = {"boundary": key[1], "frame": key[0], "invocation": key[2]}
    report = {
        "schema_version": "rlkernel.diagnostic_report.v3",
        "status": "not_comparable" if errors else "diverged" if first else "equal",
        "scope": manifest["scope"],
        "coverage": {
            **manifest["coverage"],
            "ordering": "paired observed event order; not a causal DAG",
            "hidden_state": "cache/mask/fused internals require explicit bindings",
        },
        "first_divergence": first,
        "errors": errors,
        "boundaries": comparisons,
        "runtime_identity": identities[0] if identities else None,
        "replacement": {
            "status": "unsupported",
            "reason": "Native execution evidence needs no RL Kernel replacement or live adapter.",
        },
    }
    report["diagnosis"] = diagnosis(report)
    # Replace stale evidence, including after a previously divergent bundle becomes invalid.
    atomic_save(directory / "first-divergence.pt", {"first_divergence": first, "values": evidence})
    return report
