# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Portable semantic evidence. No Qwen, Megatron, vLLM or RL Kernel dependency.

Applications declare corresponding boundaries and their logical layouts. Hooks
observe the original forward; they never synthesize reference intermediates.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from .core import _stitch, atomic_save, bitdiff

SCHEMA = "rlkernel.capture.v1"
IDENTITIES = ("weights", "tokens", "positions", "mask", "sampling")


def _specifications(boundaries: list[dict[str, Any]]) -> None:
    seen = set()
    if not boundaries:
        raise ValueError("declare at least one semantic boundary")
    for spec in boundaries:
        name = spec["id"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("boundary IDs must be unique nonempty strings")
        if not spec.get("module") or not spec.get("stage"):
            raise ValueError(f"{name}: module and stage are required")
        if any(parent not in seen for parent in spec.get("inputs", [])):
            raise ValueError(f"{name}: inputs must precede output in semantic execution order")
        positions = spec["positions"]
        dynamic_width = bool(spec.get("dynamic_width", False))
        if (
            not positions
            or any(type(p) is not int or p < 0 for p in positions)
            or len(set(positions)) != len(positions)
            or (not dynamic_width and spec["width"] < 1)
            or (dynamic_width and spec.get("width") not in (None, 0))
        ):
            raise ValueError(f"{name}: declare nonempty logical positions and feature width")
        if not isinstance(spec.get("contracts", []), list):
            raise ValueError(f"{name}: contracts must list required observed contract keys")
        seen.add(name)


class Capture:
    """Explicit adapter API for standard tensors from any production model.

    Prepare the manifest once before distributed workers start. Each worker
    constructs Capture(directory, side, identity, rank=N) and records normalized
    [logical_rows, local_features] tensors. Identity values must describe actual
    runtime inputs/weights, not merely the intended configuration.
    """

    @staticmethod
    def prepare(directory: Path, boundaries: list[dict[str, Any]], *, scope: str) -> None:
        _specifications(boundaries)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "capture-manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {"schema_version": SCHEMA, "scope": scope, "boundaries": boundaries},
                handle,
                indent=2,
            )

    def __init__(self, directory: Path, side: str, identity: Mapping[str, Any], *, rank: int = 0):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "capture-manifest.json").read_text())
        if self.manifest.get("schema_version") != SCHEMA:
            raise ValueError("unsupported capture manifest version")
        _specifications(self.manifest["boundaries"])
        if side not in {"training", "rollout"} or rank < 0:
            raise ValueError("capture side must be training/rollout and rank nonnegative")
        if any(identity.get(key) is None for key in IDENTITIES):
            raise ValueError("actual weights/tokens/positions/mask/sampling identities required")
        self.side, self.rank, self.identity = side, rank, dict(identity)
        self.specs = {s["id"]: s for s in self.manifest["boundaries"]}
        self.calls = 0

    def record(
        self,
        boundary: str,
        value: torch.Tensor,
        *,
        positions: list[int],
        offset: int = 0,
        contracts: Mapping[str, Any] | None = None,
        route: str = "unknown",
    ) -> None:
        spec = self.specs[boundary]
        if (
            value.ndim != 2
            or value.shape[0] != len(positions)
            or offset < 0
            or (
                not spec.get("dynamic_width", False)
                and offset + value.shape[1] > spec["width"]
            )
            or not set(positions).issubset(spec["positions"])
        ):
            raise ValueError(f"{boundary}: invalid logical row/feature mapping")
        # Stable exclusive reservation prevents a second process from silently
        # overwriting rank evidence. Atomic tensor saves keep interrupted files out.
        folder = self.directory / "captures"
        folder.mkdir(exist_ok=True)
        stem = f"{self.side}-rank{self.rank}-call{self.calls}"
        with (folder / f"{stem}.lock").open("x"):
            pass
        atomic_save(
            folder / f"{stem}.pt",
            {
                "schema_version": SCHEMA,
                "boundary": boundary,
                "side": self.side,
                "identity": self.identity,
                "contracts": dict(contracts or {}),
                "route": route,
                "tensors": {"value": value},
                "metadata": {
                "stage_positions": {"value": torch.tensor(positions)},
                "feature_offsets": {"value": offset},
                "feature_sizes": {
                    "value": value.shape[1] if spec.get("dynamic_width") else spec["width"]
                },
                },
            },
        )
        self.calls += 1

    @contextmanager
    def observe(self, bindings: Mapping[str, tuple[torch.nn.Module, Callable[..., dict]]]):
        """Map boundary IDs to (module, extractor(module, args, kwargs, output)).

        Extractors return record() keyword arguments (value/positions/contracts/
        route/offset). They must declare the exact semantic boundary, including
        residual/norm placement. For recurrent layers use explicit event IDs and
        record(); arbitrary module names are not a semantic adapter.
        """
        handles = []
        try:
            for boundary, (module, extract) in bindings.items():
                if boundary not in self.specs:
                    raise ValueError(f"undeclared boundary: {boundary}")

                def hook(mod, args, kwargs, output, boundary=boundary, extract=extract):
                    self.record(boundary, **extract(mod, args, kwargs, output))

                handles.append(module.register_forward_hook(hook, with_kwargs=True))
            yield self
        finally:
            for handle in handles:
                handle.remove()


def analyze_bundle(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "capture-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported portable capture version")
    specs = manifest["boundaries"]
    _specifications(specs)
    pairs = {s["id"]: {side: [] for side in ("training", "rollout")} for s in specs}
    errors, identities = [], []
    paths = sorted((directory / "captures").glob("*.pt"))
    for lock in (directory / "captures").glob("*.lock"):
        if not lock.with_suffix(".pt").is_file():
            errors.append(f"incomplete capture: {lock.stem}")
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record.get("schema_version") != SCHEMA or record.get("boundary") not in pairs:
            raise ValueError(f"invalid capture record: {path}")
        side = record["side"]
        if side not in {"training", "rollout"}:
            raise ValueError(f"invalid capture side: {side}")
        identity = record.get("identity", {})
        if any(identity.get(key) is None for key in IDENTITIES):
            errors.append(f"{path.name}: incomplete runtime identity")
        identities.append(identity)
        pairs[record["boundary"]][side].append(record)
    if not identities or any(v != identities[0] for v in identities):
        errors.append("runtime weights/tokens/positions/mask/sampling identities differ or missing")
    comparisons, first, values = [], None, {}
    for spec in specs:
        name, selected = spec["id"], set(spec["positions"])
        pair, contract_diff, boundary_errors, routes = pairs[name], {}, [], {}
        values[name] = {}
        observed_contracts = {}
        for side, records in pair.items():
            rows, issues = _stitch(records, "value", selected)
            boundary_errors.extend(issues)
            if set(rows) != selected:
                boundary_errors.append(f"{side}: missing logical row coverage")
            for record in records:
                observed_width = record["metadata"]["feature_sizes"]["value"]
                if spec.get("dynamic_width"):
                    if observed_width < 1:
                        boundary_errors.append(f"{side}: dynamic feature width is empty")
                elif observed_width != spec["width"]:
                    boundary_errors.append(f"{side}: feature width differs from manifest")
            values[name][side] = rows
            routes[side] = sorted({r.get("route", "unknown") for r in records})
            contracts = [r.get("contracts", {}) for r in records]
            if not contracts or any(c != contracts[0] for c in contracts):
                boundary_errors.append(f"{side}: missing or inconsistent replica contracts")
            observed_contracts[side] = contracts[0] if contracts else {}
            for key in spec.get("contracts", []):
                if observed_contracts[side].get(key) is None:
                    boundary_errors.append(f"{side}: missing required contract {key}")
        left_contract, right_contract = observed_contracts.values()
        for key in left_contract.keys() | right_contract.keys():
            if left_contract.get(key) != right_contract.get(key):
                contract_diff[key] = {
                    "training": left_contract.get(key),
                    "rollout": right_contract.get(key),
                }
        if contract_diff:
            boundary_errors.append("observed calculation contracts differ")
        deltas = []
        for position in sorted(values[name]["training"].keys() & values[name]["rollout"].keys()):
            delta = bitdiff(values[name]["training"][position], values[name]["rollout"][position])
            if not delta.get("comparable") or not delta.get("finite", False):
                boundary_errors.append(f"invalid tensor comparison at position {position}")
            if not delta.get("equal"):
                deltas.append({"position": position, **delta})
        errors.extend(f"{name}: {e}" for e in boundary_errors)
        row = {
            "boundary": name,
            "module": spec["module"],
            "stage": spec["stage"],
            "layer": spec.get("layer"),
            "errors": boundary_errors,
            "contract_differences": contract_diff,
            "routes": routes,
            "status": "not_comparable" if boundary_errors else "diverged" if deltas else "equal",
        }
        prior = {r["boundary"]: r for r in comparisons}
        inputs = spec.get("inputs", [])
        row["input_evidence"] = (
            "unknown"
            if not inputs or any(prior[p]["status"] == "not_comparable" for p in inputs)
            else "different"
            if any(prior[p]["status"] == "diverged" for p in inputs)
            else "equal"
        )
        comparisons.append(row)
        if deltas and first is None:
            first = {**row, **deltas[0]}
            atomic_save(
                directory / "first-divergence.pt",
                {
                    "first_divergence": first,
                    "boundaries": {key: values[key] for key in [*inputs, name]},
                },
            )
    report = {
        "schema_version": "rlkernel.diagnostic_report.v2",
        "status": "not_comparable" if errors else "diverged" if first else "equal",
        "scope": manifest.get("scope", "explicitly declared boundaries only"),
        "first_divergence": first,
        "errors": errors,
        "boundaries": comparisons,
        "runtime_identity": identities[0] if identities else None,
        "replacement": {
            "status": "unsupported",
            "reason": "Portable evidence has no live adapter; diagnosis is available.",
        },
    }
    report["diagnosis"] = diagnosis(report)
    return report


def diagnosis(report: dict[str, Any]) -> dict[str, Any]:
    """Observed facts and investigation directions, never an invented kernel cause."""
    first = report.get("first_divergence")
    differences = {
        row["boundary"]: row["contract_differences"]
        for row in report.get("boundaries", [])
        if row.get("contract_differences")
    }
    if differences:
        return {
            "level": "contract_difference",
            "reason": "Observed contracts differ; a numerical root cause is not established.",
            "contract_differences": differences,
            "next_checks": [
                f"Align observed contract at {name}: {diff}" for name, diff in differences.items()
            ],
        }
    if report.get("errors"):
        return {
            "level": "inconclusive",
            "reason": "Comparison gates failed.",
            "next_checks": report["errors"][:5],
        }
    if not first:
        return {
            "level": "not_reproduced",
            "reason": "No mismatch in captured boundaries.",
            "next_checks": ["Check original batching/graph/cache execution if it failed there."],
        }
    input_evidence = first.get("input_evidence", "unknown")
    reason = {
        "equal": "Declared inputs agree; first observed output differs.",
        "different": "Observed input already differs; follow the upstream boundary.",
        "unknown": "First observed difference; equivalent operator inputs are not proven.",
    }
    return {
        "level": "localized",
        "input_evidence": input_evidence,
        "reason": reason[input_evidence],
        "next_checks": report.get("next_checks")
        or [
            "Check uncaptured state and contracts before kernel arithmetic.",
            "If complete inputs match, inspect accumulation dtype, cast and reduction order.",
        ],
    }
