# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Validate strict-provider identity, configuration and execution provenance.

The runner fixes the expectation *before* execution. Coverage is never inferred
from traces. Call records must identify the declared inputs/state boundary and
the allowed execution plan. Numerical output acceptance belongs to the existing
operator/model harness; this module does not accept or compare output values.
Evidence checks rely on trusted adapters; they do not establish GPU qualification.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from rl_engine.contracts.strict_provider import (
    CaseSpec,
    CallContext,
    KernelPlan,
    ProviderLock,
    ScopeSpec,
    SnapshotActivationReceipt,
    StrictProviderError,
    WorkerObservation,
    canonical_json,
    fingerprint,
)

EXPECTATION_SCHEMA = "rl_kernel.strict_provider.expectation.v1"
SUMMARY_SCHEMA = "rl_kernel.strict_provider.summary.v1"
CASES_SCHEMA = "rl_kernel.strict_provider.cases.v1"
ARTIFACTS_SCHEMA = "rl_kernel.strict_provider.artifacts.v1"
RESULT_SCHEMA = "rl_kernel.strict_provider.result.v1"


def _reject(code: str, message: str, field: str | None = None, **details: Any) -> None:
    raise StrictProviderError(code, message, field=field, **details)


def _dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if not isinstance(value, Mapping):
        _reject("invalid_evidence", "Expected a JSON object")
    return dict(value)


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _reject("invalid_evidence", "Duplicate JSON key", key)
        result[key] = value
    return result


def read_json(path: str | Path) -> dict[str, Any]:
    """Read strict JSON, rejecting duplicate keys and nonstandard constants."""
    return _parse_json(Path(path).read_bytes())


def _parse_json(content: bytes) -> dict[str, Any]:
    def invalid_constant(value: str) -> None:
        _reject("nonfinite_evidence", "Nonfinite JSON constant", observed=value)

    value = json.loads(
        content.decode("utf-8"),
        object_pairs_hook=_unique_json_pairs,
        parse_constant=invalid_constant,
    )
    canonical_json(value)  # Also rejects overflowed floats such as 1e999.
    return _dict(value)


def artifact_reference(path: str | Path, evidence_root: str | Path) -> dict[str, Any]:
    """Create a reference at collection time; validation never refreshes it."""
    root = Path(evidence_root).resolve(strict=True)
    actual = Path(path).resolve(strict=True)
    try:
        relative = actual.relative_to(root)
    except ValueError:
        _reject("evidence_outside_attempt", "Artifact is outside evidence root")
    content = actual.read_bytes()
    return {
        "path": relative.as_posix(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        _reject("invalid_expectation", "Expected a single safe path component", field)
    if value in {".", ".."}:
        _reject("invalid_expectation", "Invalid path component", field)
    return value


def _call_key(call: Mapping[str, Any]) -> tuple[Any, ...]:
    required = ("case_id", "role", "rank", "layer_id", "node", "phase", "call_id")
    if any(key not in call for key in required):
        _reject("invalid_coverage", "Call lacks a required coverage coordinate")
    rank = call["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        _reject("invalid_coverage", "Invalid rank")
    layer = call["layer_id"]
    if isinstance(layer, bool) or not isinstance(layer, (str, int)):
        _reject("invalid_coverage", "Invalid layer identifier")
    if any(
        not isinstance(call[k], str) or not call[k]
        for k in required
        if k not in {"rank", "layer_id"}
    ):
        _reject("invalid_coverage", "Empty coverage coordinate")
    # Layer IDs come from JSON object keys in the model inventory.
    return tuple(str(call[k]) if k == "layer_id" else call[k] for k in required)


def build_expectation(
    lock: ProviderLock | Mapping[str, Any],
    scope: ScopeSpec | Mapping[str, Any],
    cases: Sequence[CaseSpec | Mapping[str, Any]],
    worker_roster: Sequence[Mapping[str, Any]],
    model_inventory: Mapping[str, Sequence[str]],
    *,
    run_id: str,
    attempt: str,
    weight_snapshot_id: str,
    identity_artifacts: Sequence[Mapping[str, Any]],
    snapshot_generation: int = 0,
) -> dict[str, Any]:
    """Freeze runner-owned inventory, case calls and roster before any worker runs."""
    lock = lock if isinstance(lock, ProviderLock) else ProviderLock.from_dict(dict(lock))
    scope = scope if isinstance(scope, ScopeSpec) else ScopeSpec.from_dict(dict(scope))
    data = {
        "schema_version": EXPECTATION_SCHEMA,
        "lock_id": lock.lock_id,
        "run_id": _identifier(run_id, "run_id"),
        "attempt": _identifier(attempt, "attempt"),
        "weight_snapshot_id": weight_snapshot_id,
        "scope": scope.to_dict(),
        "cases": [
            c.to_dict() if isinstance(c, CaseSpec) else CaseSpec.from_dict(dict(c)).to_dict()
            for c in cases
        ],
        "worker_roster": [dict(w) for w in worker_roster],
        "model_inventory": {str(k): list(v) for k, v in model_inventory.items()},
        "identity_artifacts": [dict(ref) for ref in identity_artifacts],
        "snapshot_generation": snapshot_generation,
    }
    _expected_calls(lock, data)
    data["expectation_id"] = fingerprint(data)
    return data


def _expected_calls(lock: ProviderLock, expectation: Mapping[str, Any]) -> Counter:
    if expectation.get("schema_version") != EXPECTATION_SCHEMA:
        _reject("invalid_expectation", "Unknown expectation schema")
    if expectation.get("lock_id") != lock.lock_id:
        _reject("stale_evidence", "Expectation belongs to another lock")
    if lock.execution["graph_mode"] != "eager":
        _reject("unsupported_graph", "v1 validator supports eager execution only")
    _identifier(expectation.get("run_id"), "run_id")
    _identifier(expectation.get("attempt"), "attempt")
    if (
        not isinstance(expectation.get("weight_snapshot_id"), str)
        or not expectation["weight_snapshot_id"]
    ):
        _reject("invalid_expectation", "Missing active weight snapshot")
    if (
        not isinstance(expectation.get("identity_artifacts"), list)
        or not expectation["identity_artifacts"]
    ):
        _reject("missing_artifact_evidence", "Fixed original artifact references are required")
    if (
        type(expectation.get("snapshot_generation")) is not int
        or expectation["snapshot_generation"] < 0
    ):
        _reject("invalid_expectation", "Invalid snapshot generation")
    scope = ScopeSpec.from_dict(_dict(expectation.get("scope"))).to_dict()
    if scope["kind"] != "component" or not scope["scope_id"].startswith("component."):
        _reject(
            "unsupported_scope",
            "v1 supports component scopes; full-model registry is not connected",
        )
    if lock.to_dict()["scope"] != scope["scope_id"]:
        _reject("unsupported_scope", "Scope does not match lock")
    inventory = _dict(expectation.get("model_inventory"))
    if not inventory or fingerprint(inventory) != scope["inventory_fingerprint"]:
        _reject("invalid_inventory", "Model inventory is missing or has changed")
    for layer, nodes in inventory.items():
        if (
            not layer
            or not isinstance(nodes, list)
            or not nodes
            or any(not isinstance(n, str) or not n for n in nodes)
            or len(nodes) != len(set(nodes))
        ):
            _reject("invalid_inventory", "Invalid model inventory entry", layer)
    inventory_nodes = {n for nodes in inventory.values() for n in nodes}
    required_nodes = set(scope["required_nodes"])
    if not required_nodes <= inventory_nodes:
        _reject("invalid_inventory", "Scope contains nodes absent from model inventory")
    if not required_nodes <= set(lock.to_dict()["nodes"]):
        _reject("unsupported_scope", "Lock lacks required scope nodes")
    roster = expectation.get("worker_roster")
    if not isinstance(roster, list) or not roster:
        _reject("missing_worker", "Worker roster must be nonempty")
    worker_ids: set[str] = set()
    coordinates: set[tuple[str, int]] = set()
    for worker in roster:
        if not isinstance(worker, dict) or set(worker) != {"worker_id", "role", "rank"}:
            _reject("invalid_expectation", "Worker roster has invalid fields")
        if not isinstance(worker["worker_id"], str) or not worker["worker_id"]:
            _reject("invalid_expectation", "Invalid worker identifier")
        if (
            worker["role"] not in {"rollout", "replay"}
            or isinstance(worker["rank"], bool)
            or (not isinstance(worker["rank"], int) or worker["rank"] < 0)
        ):
            _reject("invalid_expectation", "Invalid worker role/rank")
        coord = (worker["role"], worker["rank"])
        if worker["worker_id"] in worker_ids or coord in coordinates:
            _reject("duplicate_worker", "Duplicate worker identity or coordinate")
        worker_ids.add(worker["worker_id"])
        coordinates.add(coord)
    if {r for r, _ in coordinates} != {"rollout", "replay"}:
        _reject("missing_worker", "Both rollout and replay workers are required")
    if {rank for role, rank in coordinates if role == "rollout"} != {
        rank for role, rank in coordinates if role == "replay"
    }:
        _reject("missing_worker", "Unpaired worker rank coordinates")
    cases = expectation.get("cases")
    if not isinstance(cases, list) or not cases:
        _reject("empty_cases", "No required validation cases")
    case_ids: set[str] = set()
    expected: Counter = Counter()
    for raw in cases:
        case = CaseSpec.from_dict(_dict(raw)).to_dict()
        case_id = case["case_id"]
        if case_id in case_ids:
            _reject("duplicate_case", "Duplicate mandatory case", case_id)
        case_ids.add(case_id)
        per_case: set[tuple[str, int, str, str]] = set()
        for call in case["required_calls"]:
            if set(call) != {"role", "rank", "layer_id", "node", "phase", "call_id", "count"}:
                _reject("invalid_coverage", "Invalid required-call fields")
            key = _call_key({"case_id": case_id, **call})
            _, role, rank, layer, node, _, _ = key
            if (
                (role, rank) not in coordinates
                or node not in required_nodes
                or (layer not in inventory or node not in inventory[layer])
            ):
                _reject("invalid_coverage", "Required call is outside fixed inventory/roster")
            count = call["count"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                _reject("invalid_coverage", "Call multiplicity must be a positive integer")
            if count != 1:
                _reject("invalid_coverage", "v1 requires a distinct call_id for every execution")
            if key in expected:
                _reject("invalid_coverage", "Duplicate required-call coordinate")
            expected[key] = count
            per_case.add((role, rank, layer, node))
        # Every selected inventory node must be represented on every worker.
        needed = {
            (role, rank, layer, node)
            for role, rank in coordinates
            for layer, nodes in inventory.items()
            for node in nodes
            if node in required_nodes
        }
        if not needed <= per_case:
            _reject("incomplete_expectation", "Case omits a selected model layer/node")
    return expected


def _read_artifact_bytes(
    reference: Mapping[str, Any], root: Path, expectation: Mapping[str, Any], seen: set[Path]
) -> bytes:
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256", "size_bytes"}:
        _reject("invalid_evidence", "Evidence must have path, SHA256 and size")
    if not isinstance(reference["path"], str) or not reference["path"]:
        _reject("invalid_evidence", "Evidence path must be a nonempty string")
    relative = Path(reference["path"])
    if relative.is_absolute() or ".." in relative.parts:
        _reject("evidence_outside_attempt", "Evidence path must be relative to current attempt")
    attempt_root = (root / expectation["run_id"] / expectation["attempt"]).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    try:
        path.relative_to(attempt_root)
        attempt_root.relative_to(root)
    except ValueError:
        _reject("evidence_outside_attempt", "Evidence escapes the current attempt")
    if path in seen:
        _reject("duplicate_evidence", "An evidence file was supplied more than once")
    seen.add(path)
    if not path.is_file():
        _reject("invalid_evidence", "Evidence is not a regular file")
    content = path.read_bytes()
    if type(reference["size_bytes"]) is not int or reference["size_bytes"] != len(content):
        _reject("evidence_hash_mismatch", "Evidence size differs from fixed reference")
    if reference["sha256"] != hashlib.sha256(content).hexdigest():
        _reject("evidence_hash_mismatch", "Evidence content differs from fixed reference")
    return content


def _load_artifact(
    reference: Mapping[str, Any], root: Path, expectation: Mapping[str, Any], seen: set[Path]
) -> dict[str, Any]:
    return _parse_json(_read_artifact_bytes(reference, root, expectation, seen))


def _check_context(data: Mapping[str, Any], expectation: Mapping[str, Any]) -> None:
    for key in ("lock_id", "run_id", "attempt", "weight_snapshot_id"):
        if data.get(key) != expectation[key]:
            _reject(
                "stale_evidence",
                "Evidence belongs to another execution context",
                key,
                expected=expectation[key],
                observed=data.get(key),
            )


def _verify_identity_artifacts(
    lock: ProviderLock, expectation: Mapping[str, Any], root: Path, seen: set[Path]
) -> set[str]:
    """Bind original file bytes to the provider/source/binary inventory identities."""
    inventories: set[str] = set()
    for reference in expectation["identity_artifacts"]:
        manifest = _load_artifact(reference, root, expectation, seen)
        _check_context(manifest, expectation)
        if manifest.get("schema_version") != ARTIFACTS_SCHEMA:
            _reject("invalid_artifact_inventory", "Unknown artifact inventory schema")
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            _reject("invalid_artifact_inventory", "Artifact inventory must contain original files")
        files = []
        logical_paths: set[str] = set()
        file_paths: set[Path] = set()
        for record in records:
            if not isinstance(record, dict) or set(record) != {"logical_path", "reference"}:
                _reject("invalid_artifact_inventory", "Invalid artifact inventory record")
            logical = record["logical_path"]
            if (
                not isinstance(logical, str)
                or not logical
                or "\\" in logical
                or (
                    ":" in logical
                    or logical.startswith("/")
                    or ".." in logical.split("/")
                    or any(part in {"", "."} for part in logical.split("/"))
                )
                or logical in logical_paths
            ):
                _reject(
                    "invalid_artifact_inventory", "Artifact logical paths must be unique/relative"
                )
            logical_paths.add(logical)
            ref = record["reference"]
            _read_artifact_bytes(ref, root, expectation, file_paths)
            files.append({"path": logical, "size": ref["size_bytes"], "sha256": ref["sha256"]})
        inventory_id = fingerprint(sorted(files, key=lambda f: f["path"]))
        if manifest.get("inventory_sha256") != inventory_id:
            _reject(
                "evidence_hash_mismatch", "Artifact inventory identity does not match file bytes"
            )
        if inventory_id in inventories:
            _reject("duplicate_evidence", "Artifact inventory identity was supplied twice")
        inventories.add(inventory_id)
    required = {
        lock.to_dict()["nodes"][node]["bindings"][role]["artifacts_ref"]
        for node in expectation["scope"]["required_nodes"]
        for role in ("rollout", "replay")
    }
    if not required <= inventories:
        _reject(
            "missing_artifact_evidence", "Original artifacts do not cover locked provider bindings"
        )
    return inventories


def _verify_summary(
    lock: ProviderLock,
    summary: Mapping[str, Any],
    expectation: Mapping[str, Any],
    verified_inventories: set[str],
) -> tuple[Counter, dict[tuple[Any, ...], list[str]]]:
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        _reject("invalid_evidence", "Unknown worker summary schema")
    _check_context(summary, expectation)
    if type(summary.get("rank")) is not int:
        _reject("worker_mismatch", "Worker rank must be an integer")
    if any(summary.get(name) for name in ("errors", "failures", "skipped")):
        _reject("incomplete_calls", "Worker reports failed or skipped execution")
    if summary.get("summary_id") != fingerprint(
        {k: v for k, v in summary.items() if k != "summary_id"}
    ):
        _reject("evidence_hash_mismatch", "Worker summary content identity mismatch")
    if summary.get("state") != "finished":
        _reject("incomplete_calls", "Worker has not finished successfully")
    calls = summary.get("calls")
    footer = summary.get("footer", {})
    if (
        not isinstance(calls, list)
        or not calls
        or not isinstance(footer, dict)
        or (
            footer.get("complete") is not True
            or type(footer.get("call_count")) is not int
            or footer["call_count"] != len(calls)
        )
    ):
        _reject("incomplete_calls", "Missing completion footer or nonempty call records")
    ld = lock.to_dict()
    required_nodes = expectation["scope"]["required_nodes"]
    observations = _dict(summary.get("observations"))
    bindings = _dict(summary.get("bindings"))
    for node in required_nodes:
        observation = WorkerObservation.from_dict(_dict(observations.get(node))).to_dict()
        for field in ("model", "platform", "execution"):
            if observation[field] != ld[field]:
                _reject("binding_mismatch", "Worker identity does not match lock", field)
        for field in ("worker_id", "role", "rank"):
            if observation.get(field, 0 if field == "rank" else None) != summary.get(field):
                _reject("binding_mismatch", "Observation does not match summary", field)
        expected_binding = ld["nodes"][node]["bindings"].get(summary["role"])
        if observation["bindings"].get(node) != expected_binding:
            _reject("binding_mismatch", "Provider binding differs from lock", node)
        if bindings.get(node) != expected_binding:
            _reject("binding_mismatch", "Summary bindings differ from observation", node)
    generation = expectation["snapshot_generation"]
    if type(summary.get("snapshot_generation")) is not int or (
        summary["snapshot_generation"] != generation
    ):
        _reject("snapshot_inactive", "Summary has a stale snapshot generation")
    history = summary.get("snapshot_history")
    if not isinstance(history, list) or not history:
        _reject("snapshot_inactive", "Missing worker-load activation receipt")
    receipt = SnapshotActivationReceipt.from_dict(_dict(history[-1]))
    receipt.validate()
    if (
        receipt.evidence_kind != "worker_load"
        or receipt.worker_id != summary["worker_id"]
        or (
            receipt.snapshot_id != summary["weight_snapshot_id"] or receipt.generation != generation
        )
    ):
        _reject("snapshot_inactive", "Snapshot receipt does not match active worker snapshot")
    observed: Counter = Counter()
    alignments: dict[tuple[Any, ...], list[str]] = {}
    tickets: set[str] = set()
    cases = {case["case_id"]: case for case in expectation["cases"]}
    for call in calls:
        context = CallContext.from_dict(_dict(call.get("context"))).to_dict()
        if call.get("context_id") != fingerprint(context):
            _reject("stale_evidence", "Call context identity does not match content")
        if type(call.get("snapshot_generation")) is not int or (
            call["snapshot_generation"] != generation
        ):
            _reject("snapshot_inactive", "Call used a stale snapshot generation")
        for field in ("worker_id", "role", "run_id", "attempt", "weight_snapshot_id"):
            if context.get(field) != summary[field]:
                _reject("stale_evidence", "Call context differs from worker summary", field)
        case = cases.get(context["case_id"])
        if case is None:
            _reject("invalid_coverage", "Call belongs to an undeclared case", context["case_id"])
        actual_context = context.get("metadata", {}).get("comparison_context")
        if not isinstance(actual_context, Mapping) or canonical_json(
            actual_context
        ) != canonical_json(case["comparison_context"]):
            _reject(
                "execution_context_mismatch",
                "Call inputs/state boundary differ from the predeclared case",
                "context.metadata.comparison_context",
            )
        context["rank"] = summary["rank"]
        key = _call_key(context)
        if key in observed:
            _reject("duplicate_call", "A logical call coordinate was completed more than once")
        if (
            call.get("status") != "completed"
            or call.get("states") != ["prepared", "authorized", "launched", "completed"]
            or call.get("completion", {}).get("status") != "completed"
        ):
            _reject("incomplete_calls", "Call lacks authorization or successful completion")
        kind = call["completion"].get("kind")
        if (
            not isinstance(kind, str)
            or not kind
            or (ld["platform"]["kind"] in {"rocm", "cuda"} and kind == "synchronous")
        ):
            _reject("incomplete_calls", "GPU completion must be explicitly checked")
        if any(call.get(name) for name in ("errors", "failures", "skipped")):
            _reject("incomplete_calls", "Call reports failed or skipped execution")
        runtime_id = call.get("runtime_id")
        if not isinstance(runtime_id, str) or not runtime_id:
            _reject("invalid_evidence", "Resolved-call facts fingerprint is missing")
        ticket_id = call.get("call_ticket_id")
        if not isinstance(ticket_id, str) or not ticket_id or ticket_id in tickets:
            _reject("duplicate_call", "Authorization ticket is missing or was reused")
        tickets.add(ticket_id)
        plan = KernelPlan.from_dict(_dict(call.get("plan")))
        pd = plan.to_dict()
        if any(
            pd["code"][field] not in verified_inventories
            for field in ("source_inventory_sha256", "binary_inventory_sha256")
        ):
            _reject(
                "missing_artifact_evidence",
                "Plan code is not covered by verified original artifacts",
            )
        if pd["node"] != context["node"] or pd["phase"] != context["phase"]:
            _reject("plan_mismatch", "Plan belongs to another node/phase")
        if call.get("plan_id") != plan.plan_id or call.get("alignment_id") != plan.alignment_id:
            _reject("plan_mismatch", "Plan identity does not match its contents")
        node = ld["nodes"].get(context["node"])
        if not node or plan.plan_id not in node["allowed_plans"].get(summary["role"], []):
            _reject("plan_mismatch", "Actual plan was not allowed by lock")
        if pd["implementation"] != bindings[context["node"]]["implementation"]:
            _reject("plan_mismatch", "Plan implementation differs from bound provider")
        if pd["launch"]["graph_state"] != ld["execution"]["graph_mode"]:
            _reject("plan_mismatch", "Plan graph mode differs from locked execution")
        observed[key] += 1
        # Pairing excludes the role, retaining case/rank/layer/node/phase/call.
        pair_key = (key[0], *key[2:])
        alignments.setdefault(pair_key, []).append(plan.alignment_id)
    return observed, alignments


def validate_run(
    lock: ProviderLock | Mapping[str, Any],
    expectation: Mapping[str, Any],
    worker_summaries: Sequence[Mapping[str, Any]],
    evidence_root: str | Path,
    *,
    case_results: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Check manifest evidence; numerical and hardware acceptance remain unevaluated."""
    lock = lock if isinstance(lock, ProviderLock) else ProviderLock.from_dict(dict(lock))
    expectation = _dict(expectation)
    identity = expectation.get("expectation_id")
    if identity != fingerprint({k: v for k, v in expectation.items() if k != "expectation_id"}):
        _reject("invalid_expectation", "Runner expectation content identity mismatch")
    expected = _expected_calls(lock, expectation)
    root = Path(evidence_root).resolve(strict=True)
    seen: set[Path] = set()
    verified_inventories = _verify_identity_artifacts(lock, expectation, root, seen)
    roster = {w["worker_id"]: w for w in expectation["worker_roster"]}
    received: set[str] = set()
    observed: Counter = Counter()
    alignment_by_role: dict[str, dict[tuple[Any, ...], list[str]]] = {"rollout": {}, "replay": {}}
    for reference in worker_summaries:
        summary = _load_artifact(reference, root, expectation, seen)
        worker_id = summary.get("worker_id")
        if worker_id not in roster or worker_id in received:
            _reject("duplicate_or_unknown_worker", "Worker summary is duplicate or unexpected")
        received.add(worker_id)
        if any(summary.get(k) != roster[worker_id][k] for k in ("role", "rank")):
            _reject("worker_mismatch", "Worker coordinate does not match fixed roster")
        counts, alignments = _verify_summary(lock, summary, expectation, verified_inventories)
        observed.update(counts)
        alignment_by_role[summary["role"]].update(alignments)
    if received != set(roster):
        _reject("missing_worker", "A required worker summary is missing")
    if observed != expected:
        _reject(
            "coverage_mismatch",
            "Actual calls differ from fixed expected multiplicities",
            expected={str(k): v for k, v in expected.items()},
            observed={str(k): v for k, v in observed.items()},
        )
    if alignment_by_role["rollout"] != alignment_by_role["replay"]:
        _reject("alignment_mismatch", "Rollout/replay numerical plans are not aligned")
    required_cases = {c["case_id"] for c in expectation["cases"]}
    executed_cases: set[str] = set()
    if not case_results:
        _reject("missing_case_results", "Mandatory-case execution evidence is required")
    for reference in case_results:
        report = _load_artifact(reference, root, expectation, seen)
        _check_context(report, expectation)
        if report.get("schema_version") != CASES_SCHEMA:
            _reject("invalid_case_results", "Unknown case report schema")
        for record in report.get("cases", []):
            case_id = record.get("case_id")
            if case_id not in required_cases or case_id in executed_cases:
                _reject("duplicate_or_unknown_case", "Mandatory case is duplicate or unexpected")
            executed_cases.add(case_id)
            if (
                record.get("outcome") != "passed"
                or type(record.get("tests")) is not int
                or (record["tests"] <= 0)
                or any(
                    type(record.get(k)) is not int or record[k] != 0
                    for k in ("failures", "errors", "skipped")
                )
            ):
                _reject("mandatory_case_failed", "Mandatory case failed, skipped or ran zero tests")
    if executed_cases != required_cases:
        _reject("missing_case_results", "Mandatory cases did not all execute")
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed",
        "lock_id": lock.lock_id,
        "expectation_id": identity,
        "run_id": expectation["run_id"],
        "attempt": expectation["attempt"],
        "weight_snapshot_id": expectation["weight_snapshot_id"],
        "scope": expectation["scope"]["scope_id"],
        "validation_scope": "identity_configuration_provenance",
        "manifest_enforcement": "passed",
        "coverage": "passed",
        "numerical_parity": "not_evaluated",
        "hardware_validation": "not_evaluated",
        "case_count": len(required_cases),
        "worker_count": len(received),
        "call_count": sum(observed.values()),
        "evidence": {
            "worker_summaries": list(worker_summaries),
            "case_results": list(case_results),
        },
    }
    result["result_id"] = fingerprint(result)
    return result
