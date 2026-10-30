# SPDX-License-Identifier: Apache-2.0
"""Executable negative fixtures, with inputs and expected versioned failure codes."""

import copy
from dataclasses import asdict

from .contract import (
    ContractError,
    Identity,
    Status,
    digest,
    runtime_policy,
    validate_runtime_policy,
)
from .fixtures import state_sequence
from .planner import (
    candidate_plan,
    topology,
    validate_candidates,
    validate_snapshot,
    validate_topology,
)


def negative_cases() -> list[dict]:
    cases = []

    def add(name, kind, candidate, status):
        record = {
            "case_id": f"P2-N-{name}.v1",
            "kind": kind,
            "candidate": candidate,
            "expected_status": status.value,
        }
        cases.append({**record, "checksum": digest(record)})

    state = state_sequence("C4", 4, "training")[-1]
    for name, status in (
        ("state_corruption", Status.STATE_BYTES_MISMATCH),
        ("wrong_page", Status.INVALID_PAGE_OR_GENERATION),
        ("wrong_generation", Status.INVALID_PAGE_OR_GENERATION),
        ("early_commit", Status.EARLY_OR_DUPLICATE_COMMIT),
        ("duplicate_commit", Status.EARLY_OR_DUPLICATE_COMMIT),
    ):
        s = copy.deepcopy(state)
        if name == "state_corruption":
            s["index"]["rows"][0]["data"] = "AQE="
        elif name == "wrong_page":
            s["main"]["rows"][0]["page"] = 999
        elif name == "wrong_generation":
            s["generation"] = 0
        elif name == "early_commit":
            s["main"]["partial"].append("AAAAAA==")
        else:
            s["main"]["rows"].append(copy.deepcopy(s["main"]["rows"][0]))
        add(name, "snapshot", s, status)
    identity = asdict(Identity("C4"))
    identity["index"] = identity["main"]
    add("main_index_alias", "identity", identity, Status.MAIN_INDEX_IDENTITY_ALIAS)
    identity = asdict(Identity("C4"))
    identity["layer"] = "C8"
    add("invalid_layer", "identity", identity, Status.INVALID_LAYER_TYPE)
    for field, value, status in (
        ("fallback", True, Status.SILENT_FALLBACK),
        ("num_splits", 2, Status.FORBIDDEN_SPLIT_REDUCTION),
        ("dynamic_partition", True, Status.FORBIDDEN_SPLIT_REDUCTION),
        ("atomic_partial_accumulation", True, Status.FORBIDDEN_ATOMIC_REDUCTION),
        ("round_points", {}, Status.ROUND_POINT_MISMATCH),
        ("readback_kind", "configured", Status.MISSING_PROVENANCE),
        ("backend", "ascend", Status.UNSUPPORTED_CAPABILITY),
        ("backend", "rocm", Status.UNSUPPORTED_CAPABILITY),
    ):
        policy = runtime_policy()
        policy[field] = value
        add(f"policy_{field}_{value}", "runtime", policy, status)
    for field, value, status in (
        ("softmax_denominators", 2, Status.MULTIPLE_SOFTMAX_DENOMINATORS),
        ("sink_has_value", True, Status.INVALID_SINK_SEMANTICS),
        ("order", "recent_first", Status.INVALID_CANDIDATE_ORDER),
        ("compressed", [99], Status.INVALID_TOPK_ORDER),
    ):
        plan = candidate_plan("C4", 3, [0])
        plan[field] = value
        add(field, "candidates", plan, status)
    plan = topology(257, 4, 8)
    plan["global_index_visibility"] = []
    add("missing_global_index", "topology", plan, Status.MISSING_GLOBAL_VISIBILITY)
    plan = topology(257, 4, 8)
    plan["c128_owners"][0] = -1
    add("duplicate_owner", "topology", plan, Status.DUPLICATE_LOGICAL_OWNER)
    return cases


def evaluate_negative(record: dict) -> str:
    dispatch = {
        "snapshot": validate_snapshot,
        "identity": lambda x: Identity(**x).validate(),
        "runtime": validate_runtime_policy,
        "candidates": validate_candidates,
        "topology": lambda x: validate_topology(x, 257),
    }
    try:
        dispatch[record["kind"]](record["candidate"])
    except ContractError as exc:
        return exc.status.value
    return Status.PASS.value
