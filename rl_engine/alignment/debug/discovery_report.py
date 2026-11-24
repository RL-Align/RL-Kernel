# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU-only validation, automatic matching and reporting of discovery probes."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

import torch

from .core import atomic_save, bitdiff, fingerprint
from .discovery import DISCOVERY_SCHEMA, SIDES, MappingRequired
from .discovery_weights import (
    align_primitive_views,
    in_scope,
    scope_inventory,
    scope_parameters,
)
from .events import EventCapture, LogicalTensor, _load_snapshot, analyze_events


def _normalize(leaf, frame, case):
    value = leaf["value"]
    batch, sequence = case["tokens"].shape
    count = batch * sequence
    if not value.is_floating_point() or value.ndim not in (2, 3) or value.numel() == 0:
        raise MappingRequired("not a floating hidden-state tensor")
    if value.ndim == 2:
        if value.shape[0] != count or (frame["layout"] != "packed" and batch > 1 and sequence > 1):
            raise MappingRequired("flattened hidden-state row ownership is ambiguous")
        rows = value
    else:
        leading = tuple(value.shape[:2])
        if batch == sequence and batch > 1:
            raise MappingRequired(
                "batch and sequence axes are indistinguishable; provide a logical layout"
            )
        if leading == (batch, sequence):
            rows = value.reshape(count, -1)
        elif leading == (sequence, batch):
            rows = value.transpose(0, 1).reshape(count, -1)
        else:
            raise MappingRequired("hidden-state dimensions do not cover the complete frozen batch")
    return LogicalTensor(rows, frame["rows"])


def _eligible(tree, frame, case, width=None):
    result = {}
    for path, leaf in tree["tensors"].items():
        try:
            logical = _normalize(leaf, frame, case)
        except MappingRequired:
            continue
        if width is None or logical.value.shape[-1] == width:
            result[path] = (leaf, logical)
    return result


def _same_handoff(left, right):
    # Only within ONE execution: permit an exact value handoff. Storage pointers
    # alone are not identities; allocators may reuse released addresses.
    return bitdiff(left[1].value, right[1].value).get("equal", False)


def _unique(tree, frame, case, width):
    values = _eligible(tree, frame, case, width)
    unique = {}
    for path, item in values.items():
        unique.setdefault(tuple(item[0]["storage"]), (path, item[1]))
    if len(unique) != 1:
        raise MappingRequired(f"expected one observable width-{width} tensor, found {len(unique)}")
    return next(iter(unique.values()))


def _calls(probe, candidate, frame):
    selected = [
        call
        for call in frame["calls"]
        if call["kind"] == "block" and call["path"] in candidate["blocks"]
    ]
    if Counter(call["path"] for call in selected) != Counter(candidate["blocks"]):
        raise MappingRequired("stack members were skipped, reused or never observed")
    selected.sort(key=lambda call: call["start"])
    if any(
        left["end"] >= right["start"] for left, right in zip(selected, selected[1:], strict=False)
    ):
        raise MappingRequired(
            "stack execution is nested/concurrent rather than a sequential block chain"
        )
    return selected


def _atoms(probe, block, frame):
    owners = probe["inventory"]["owners"]
    primitives = [
        call
        for call in frame["calls"]
        if call["kind"] == "primitive"
        and in_scope(call["path"], block["path"])
        and block["start"] <= call["start"] < call["end"] <= block["end"]
    ]
    seen = set()
    atoms = []
    for call in primitives:
        for atom in owners[call["path"]]["atoms"]:
            seen.update(atom["parameters"])
            if atom["kind"] == "affine":
                atoms.append({**atom, "call": call, "signature": tuple(atom["keys"])})
    expected = scope_parameters(probe["inventory"], block["path"])
    if seen != expected:
        raise MappingRequired("some parameter-owning modules bypassed observable calls")
    return atoms


def _projection_groups(left, right):
    """Union physical fusions; never assert a temporal order inside a fused op."""
    a, b = [atom["signature"] for atom in left], [atom["signature"] for atom in right]
    if len(set(a)) != len(a) or len(set(b)) != len(b):
        raise MappingRequired(
            "repeated identical affine parameters have ambiguous call correspondence"
        )
    if set(a) != set(b):
        raise MappingRequired("affine parameter/use contracts do not correspond")
    positions = {signature: index for index, signature in enumerate(a)}
    parent = list(range(len(a)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for atoms in (left, right):
        by_call = defaultdict(list)
        for atom in atoms:
            by_call[atom["call"]["start"]].append(positions[atom["signature"]])
        for group in by_call.values():
            for index in group[1:]:
                parent[find(index)] = find(group[0])
    groups = []
    for index in range(len(a)):
        if not any(index in group for group in groups):
            groups.append([i for i in range(len(a)) if find(i) == find(index)])
    for atoms in (left, right):
        order = []
        for atom in atoms:
            group = next(
                i for i, items in enumerate(groups) if positions[atom["signature"]] in items
            )
            if not order or order[-1] != group:
                order.append(group)
        if order != list(range(len(groups))):
            raise MappingRequired(
                "projection-group execution orders differ; no common first-event order"
            )
    right_by_key = {atom["signature"]: atom for atom in right}
    return [[(left[index], right_by_key[a[index]]) for index in group] for group in groups]


def _anchor(atom, frame, case):
    if not atom["input_visible"] or not atom["output_visible"]:
        raise MappingRequired("projection bias/normalization boundary is not directly observable")
    call = atom["call"]
    if call["input"]["opaque"] or call["output"]["opaque"]:
        raise MappingRequired("opaque state at a projection boundary needs an explicit selector")
    input_path, inputs = _unique(call["input"], frame, case, atom["in_features"])
    # A native fused projection returns its complete physical output.
    eligible = _eligible(call["output"], frame, case)
    output_candidates = [
        (path, logical)
        for path, (_, logical) in eligible.items()
        if logical.value.shape[1] >= atom["output_slice"][1]
    ]
    if len(output_candidates) != 1:
        raise MappingRequired("projection output tensor is ambiguous")
    output_path, outputs = output_candidates[0]
    start, stop = atom["output_slice"]
    outputs = LogicalTensor(outputs.value[:, start:stop], outputs.rows)
    return (
        inputs,
        outputs,
        {
            "module": call["path"],
            "input": input_path,
            "output": output_path,
            "output_columns": [start, stop],
        },
    )


def _group_values(pairs, side_index, frame, case):
    values = [_anchor(pair[side_index], frame, case) for pair in pairs]
    first_input = values[0][0]
    if any(
        item[0].rows != first_input.rows
        or not bitdiff(item[0].value, first_input.value).get("equal")
        for item in values[1:]
    ):
        raise MappingRequired("a fused projection group does not share the same actual input")
    return (
        first_input,
        {str(index): value[1] for index, value in enumerate(values)},
        [value[2] for value in values],
    )


def _candidate_pair(left, right, a, b):
    if len(a["blocks"]) != len(b["blocks"]):
        raise MappingRequired("block counts differ")
    plans, normalized = [], {side: [] for side in SIDES}
    execution_order = None
    for left_frame, right_frame in zip(left["frames"], right["frames"], strict=True):
        lc, rc = _calls(left, a, left_frame), _calls(right, b, right_frame)
        order = [[call["path"] for call in calls] for calls in (lc, rc)]
        if execution_order is not None and order != execution_order:
            raise MappingRequired("block execution order changes between forwards")
        execution_order = order
        frame_events = {side: [] for side in SIDES}
        frame_plans = []
        widths = {side: [] for side in SIDES}
        for layer, (lb, rb) in enumerate(zip(lc, rc, strict=True)):
            if any(block[stage]["opaque"] for block in (lb, rb) for stage in ("input", "output")):
                raise MappingRequired(
                    f"block {layer}: opaque state requires an explicit boundary contract"
                )
            if scope_inventory(left["inventory"], lb["path"]) != scope_inventory(
                right["inventory"], rb["path"]
            ):
                raise MappingRequired(
                    f"block {layer}: actual parameter contents/layouts do not correspond"
                )
            la, ra = _atoms(left, lb, left_frame), _atoms(right, rb, right_frame)
            groups = _projection_groups(la, ra)
            if not groups:
                raise MappingRequired("no verified affine anchors in the discovered block")
            hidden_width = groups[0][0][0]["in_features"]
            details = {"layer": layer, "training": lb["path"], "rollout": rb["path"], "anchors": []}
            endpoints = []
            for block, frame, probe in ((lb, left_frame, left), (rb, right_frame, right)):
                try:
                    endpoints.append(_unique(block["input"], frame, probe["case"], hidden_width))
                except MappingRequired:
                    endpoints.append(None)
            if all(endpoints):
                for side, (_path, value) in zip(SIDES, endpoints, strict=True):
                    frame_events[side].append((f"block.{layer}.input", layer, "input", value, []))
                details["block_input"] = {
                    side: item[0] for side, item in zip(SIDES, endpoints, strict=True)
                }
            for index, group in enumerate(groups):
                anchors = [
                    _group_values(group, which, frame, probe["case"])
                    for which, (frame, probe) in enumerate(
                        ((left_frame, left), (right_frame, right))
                    )
                ]
                base = f"block.{layer}.projection.{index}"
                for side, (inputs, outputs, _) in zip(SIDES, anchors, strict=True):
                    frame_events[side].append((f"{base}.input", layer, "input", inputs, []))
                    frame_events[side].append(
                        (f"{base}.output", layer, "output", outputs, [f"{base}.input"])
                    )
                details["anchors"].append(
                    {
                        "id": base,
                        **{side: value[2] for side, value in zip(SIDES, anchors, strict=True)},
                        "fused_components": len(group),
                    }
                )
            endpoints = []
            for block, frame, probe in ((lb, left_frame, left), (rb, right_frame, right)):
                try:
                    endpoints.append(_unique(block["output"], frame, probe["case"], hidden_width))
                except MappingRequired:
                    endpoints.append(None)
            if all(endpoints):
                for side, (_path, value) in zip(SIDES, endpoints, strict=True):
                    deps = [f"block.{layer}.input"] if "block_input" in details else []
                    frame_events[side].append(
                        (f"block.{layer}.output", layer, "output", value, deps)
                    )
                details["block_output"] = {
                    side: item[0] for side, item in zip(SIDES, endpoints, strict=True)
                }
            for side in SIDES:
                widths[side].append(hidden_width)
            frame_plans.append(details)

        # Validate the actual sequential state handoff independently on each side.
        # Multi-state residual representations are allowed, never added together.
        for side, calls, frame, probe in (
            ("training", lc, left_frame, left),
            ("rollout", rc, right_frame, right),
        ):
            for index, (previous, following) in enumerate(zip(calls, calls[1:], strict=False)):
                outputs = list(
                    _eligible(
                        previous["output"], frame, probe["case"], widths[side][index]
                    ).values()
                )
                inputs = list(
                    _eligible(
                        following["input"], frame, probe["case"], widths[side][index + 1]
                    ).values()
                )
                matches = [
                    [
                        right_index
                        for right_index, value in enumerate(inputs)
                        if _same_handoff(output, value)
                    ]
                    for output in outputs
                ]
                if (
                    not outputs
                    or not inputs
                    or any(len(indices) != 1 for indices in matches)
                    or len({indices[0] for indices in matches}) != len(matches)
                ):
                    raise MappingRequired(
                        f"{side}: adjacent blocks do not form a verified state stream"
                    )
        if plans and frame_plans != plans:
            raise MappingRequired("selected boundary fields change between forwards")
        plans = frame_plans
        for side in SIDES:
            normalized[side].append(frame_events[side])
    return {
        "blocks": plans,
        "containers": {"training": a["container"], "rollout": b["container"]},
    }, normalized


def _contains(outer, inner):
    return all(
        any(in_scope(block[side], parent[side]) for parent in outer["blocks"])
        for side in SIDES
        for block in inner["blocks"]
    )


def _failure(directory, reasons, candidates=None):
    plan = {
        "schema_version": DISCOVERY_SCHEMA,
        "status": "needs_mapping",
        "automatic": True,
        "reasons": reasons,
        "candidates": candidates or [],
        "required_actions": [
            "Verify actual weights/loading or supply a lossless primitive-layout mapping.",
            "For ambiguous stacks, select the intended container with stack= on each side.",
            "For opaque state or non-linear execution, use explicit "
            "ModuleBoundary/EventCapture evidence.",
        ],
    }
    (directory / "mapping-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    atomic_save(directory / "first-divergence.pt", {"first_divergence": None})
    return {
        "schema_version": "rlkernel.diagnostic_report.v3",
        "status": "not_comparable",
        "first_divergence": None,
        "boundaries": [],
        "errors": reasons,
        "scope": "automatic block discovery; correspondence not certified",
        "automatic_mapping": plan,
        "diagnosis": {
            "level": "needs_mapping",
            "reason": reasons[0],
            "next_checks": plan["required_actions"],
        },
        "replacement": {
            "status": "unsupported",
            "reason": "No RL Kernel replacement is needed for diagnosis.",
        },
    }


def analyze_discovery(directory: Path):
    directory = Path(directory).resolve()
    try:
        marker = json.loads((directory / "discovery-manifest.json").read_text("utf-8"))
        if marker.get("schema_version") != DISCOVERY_SCHEMA:
            raise ValueError("unsupported automatic-discovery schema")
    except (ValueError, OSError, AttributeError) as exc:
        return _failure(directory, [str(exc)])
    probes, reasons = {}, []
    for side in SIDES:
        try:
            probe = _load_snapshot(directory / "discovery" / side / "probe.pt")
            if (
                probe["schema_version"] != DISCOVERY_SCHEMA
                or probe["side"] != side
                or probe["state"] != "complete"
                or probe["errors"]
                or not probe["frames"]
                or not all(frame["closed"] for frame in probe["frames"])
            ):
                raise MappingRequired(
                    f"{side}: incomplete/failed observation: {probe.get('errors', [])}"
                )
            probes[side] = probe
        except (ValueError, KeyError, TypeError) as exc:
            reasons.append(str(exc))
    if reasons:
        return _failure(directory, reasons)
    left, right = probes["training"], probes["rollout"]
    align_primitive_views(left["inventory"], right["inventory"])
    for key in ("tokens", "positions", "mask"):
        if not torch.equal(left["case"][key], right["case"][key]):
            reasons.append(f"frozen {key} differ")
    if left["case"]["context"] != right["case"]["context"]:
        reasons.append("declared execution contexts differ")
    if len(left["frames"]) != len(right["frames"]):
        reasons.append("forward counts differ")
    if scope_inventory(left["inventory"]) != scope_inventory(right["inventory"]):
        reasons.append(
            "actual full-model parameter contents/layouts differ; no automatic weight equivalence"
        )
    if reasons:
        return _failure(directory, reasons)
    valid, attempts = [], []
    for a in left["candidates"]:
        for b in right["candidates"]:
            if len(a["blocks"]) != len(b["blocks"]):
                continue
            try:
                plan, normalized = _candidate_pair(left, right, a, b)
                valid.append((plan, normalized))
            except (MappingRequired, ValueError, KeyError, TypeError, IndexError) as exc:
                attempts.append(
                    {"training": a["container"], "rollout": b["container"], "reason": str(exc)}
                )
    maximal = [
        pair
        for pair in valid
        if not any(other is not pair and _contains(other[0], pair[0]) for other in valid)
    ]
    if len(maximal) != 1:
        why = (
            "multiple independent block stacks are valid; choose a scope, do not guess"
            if len(maximal) > 1
            else "no uniquely validated executed block correspondence"
        )
        return _failure(directory, [why], attempts)
    plan, normalized = maximal[0]
    plan.update(
        schema_version=DISCOVERY_SCHEMA,
        status="validated",
        automatic=True,
        validation=[
            "actual full-model and per-block parameters, with explicit primitive storage contracts",
            "observed block execution order and within-run state handoffs",
            "unique affine parameter correspondence and common projection-group order",
            "actual token/position ownership; no cross-side activation-error matching",
        ],
        limits=[
            "Eager eval prefill, TP=CP=1; masks/cache not exposed by a backend "
            "remain declared state.",
            "Fused projection components have no asserted internal temporal order.",
            "Unmatched block endpoints and output-head/scoring internals remain outside coverage.",
        ],
    )
    (directory / "mapping-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    case = left["case"]
    identity = {
        "weights": sorted(scope_inventory(left["inventory"]).items()),
        "tokens": fingerprint(case["tokens"]),
        "positions": case["positions"].tolist(),
        "mask": case["mask"].tolist(),
        "sampling": {},
        "context": case["context"],
    }
    specs = [
        {"id": name, "module": "automatic_block", "layer": layer, "stage": stage}
        for name, layer, stage, _, _ in normalized["training"][0]
    ]
    # Re-normalize from immutable probes on every report-only invocation. Never
    # trust an edited cached mapping plan, or silently reuse stale derived data.
    with tempfile.TemporaryDirectory(prefix=".discovery-", dir=directory) as temporary:
        root = Path(temporary).resolve()
        if not root.is_relative_to(directory):
            raise ValueError("temporary normalization directory escaped the evidence directory")
        EventCapture.prepare(
            root,
            specs,
            scope="automatically validated decoder blocks and native affine boundaries",
            coverage={
                "discovery": "validated",
                "layers": len(plan["blocks"]),
                "limits": plan["limits"],
            },
        )
        for side in SIDES:
            with EventCapture(root, side, identity) as capture:
                for index, frame in enumerate(normalized[side]):
                    for name, _, _, value, dependencies in frame:
                        capture.record(
                            name,
                            value,
                            frame=f"forward:{index}",
                            inputs=[(f"forward:{index}", parent, 0) for parent in dependencies],
                        )
        report = analyze_events(root)
        atomic_save(directory / "first-divergence.pt", _load_snapshot(root / "first-divergence.pt"))
    report["automatic_mapping"] = plan
    return report
