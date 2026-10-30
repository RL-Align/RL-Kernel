# SPDX-License-Identifier: Apache-2.0
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from rl_engine.p2.contract import (
    BOUNDARIES,
    MODES,
    OPERATORS,
    ContractError,
    Identity,
    Status,
    check_compatibility,
    digest,
    manifest,
)
from rl_engine.p2.fixtures import catalog, state_sequence
from rl_engine.p2.planner import (
    PlannerMock,
    candidate_plan,
    compare_snapshots,
    ordered_collective,
    topology,
    validate_candidates,
    validate_snapshot,
    validate_topology,
)


def fails(status, fn, *args, **kwargs):
    with pytest.raises(ContractError) as err:
        fn(*args, **kwargs)
    assert err.value.status == status


def test_frozen_manifest_and_catalog():
    data = Path(__file__).parent / "data"
    assert manifest() == json.loads((data / "contract.v1.json").read_text())
    assert catalog() == json.loads((data / "catalog.v1.json").read_text())
    check_compatibility(json.loads(json.dumps(manifest())))
    assert len(OPERATORS) == 15
    assert len({op.name for op in OPERATORS}) == 15
    assert len(BOUNDARIES) == 13
    assert len({c["case_id"] for c in catalog()}) == len(catalog())
    assert len({c["group"] for c in catalog()}) == 17


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "unknown"),
        ("foundation_abi", "v2"),
        ("modes", ["prefill"]),
        ("operators", []),
        ("model", {"hidden_size": 1}),
    ],
)
def test_manifest_drift(field, value):
    candidate = manifest()
    candidate[field] = value
    fails(Status.SCHEMA_MISMATCH, check_compatibility, candidate)


@pytest.mark.parametrize(
    "change,status",
    [
        ({"layer": "C8"}, Status.INVALID_LAYER_TYPE),
        ({"absolute_layer": True}, Status.SCHEMA_MISMATCH),
        ({"schema": "unknown"}, Status.SCHEMA_MISMATCH),
        ({"profile": "cuda_fallback"}, Status.UNSUPPORTED_CAPABILITY),
        ({"index": "synthetic-main.v1"}, Status.MAIN_INDEX_IDENTITY_ALIAS),
        ({"checkpoint": ""}, Status.IDENTITY_DRIFT),
    ],
)
def test_identity_negative(change, status):
    fails(status, replace(Identity("C4"), **change).validate)


@pytest.mark.parametrize("component", ["weight", "ape", "norm", "state", "page", "quant"])
def test_main_index_identity_isolation(component):
    identity = Identity("C4")
    assert identity.stream_id("main", component) != identity.stream_id("index", component)
    altered = replace(identity, index="different-index")
    assert altered.stream_id("main", component) == identity.stream_id("main", component)
    assert altered.stream_id("index", component) != identity.stream_id("index", component)


@pytest.mark.parametrize("layer", ["C0", "C4", "C128"])
@pytest.mark.parametrize("tokens", [1, 3, 4, 7, 8, 127, 128, 129, 257])
@pytest.mark.parametrize("mode", MODES)
def test_modes_chunks_every_token(layer, tokens, mode):
    baseline = state_sequence(layer, tokens, "training")
    chunks = tuple([3] * (tokens // 3) + ([tokens % 3] if tokens % 3 else []))
    candidate = state_sequence(layer, tokens, mode, chunks, resume_at=min(tokens, 5))
    assert candidate == baseline
    for state in candidate:
        validate_snapshot(state)
    last = candidate[-1]
    cr = {"C0": 0, "C4": 4, "C128": 128}[layer]
    assert len(last["main"]["rows"]) == (tokens // cr if cr else 0)
    assert len(last["index"]["rows"]) == (tokens // 4 if layer == "C4" else 0)


@pytest.mark.parametrize("layer", ["C0", "C4", "C128"])
def test_long_pages_chunks_resume(layer):
    first = state_sequence(layer, 1025, "prefill", (1, 127, 128, 257, 512), page_size=4)
    second = state_sequence(layer, 1025, "graph_decode", resume_at=513, page_size=4)
    assert [s["state_hash"] for s in first] == [s["state_hash"] for s in second]
    validate_snapshot(second[-1])
    assert second[-1]["recent"][0]["logical"] == 897


def test_layer_table_interleaved():
    table = ["C0", "C4", "C128", "C0", "C4"]
    assert [
        PlannerMock(Identity(layer, i)).snapshot()["identity"]["absolute_layer"]
        for i, layer in enumerate(table)
    ] == list(range(5))


def test_failed_commit_is_atomic():
    planner = PlannerMock(Identity("C4"))
    before = planner.snapshot()
    fails(
        Status.EARLY_OR_DUPLICATE_COMMIT,
        planner.step,
        0,
        b"r",
        b"1234",
        b"5678",
        b"early",
        b"early",
    )
    assert planner.snapshot() == before
    fails(Status.SCHEMA_MISMATCH, planner.step, 0, b"r", b"x", b"5678")
    assert planner.snapshot() == before
    planner.step(0, b"r", b"1234", b"5678")
    fails(Status.INVALID_GLOBAL_POSITION, planner.step, 0, b"r", b"1234", b"5678")


@pytest.mark.parametrize(
    "kind,status",
    [
        ("page", Status.INVALID_PAGE_OR_GENERATION),
        ("generation", Status.INVALID_PAGE_OR_GENERATION),
        ("early", Status.EARLY_OR_DUPLICATE_COMMIT),
        ("duplicate", Status.EARLY_OR_DUPLICATE_COMMIT),
        ("bytes", Status.STATE_BYTES_MISMATCH),
    ],
)
def test_state_corruption(kind, status):
    s = state_sequence("C4", 8, "training")[-1]
    if kind == "page":
        s["recent"][0]["page"] = 100
    elif kind == "generation":
        s["generation"] += 1
    elif kind == "early":
        s["main"]["partial"].append("AAAAAA==")
    elif kind == "duplicate":
        s["main"]["rows"].append(copy.deepcopy(s["main"]["rows"][0]))
    else:
        s["index"]["rows"][0]["data"] = "AQE="
    fails(status, validate_snapshot, s)


def test_comparison_identity_before_state():
    a = state_sequence("C4", 4, "training")[-1]
    b = copy.deepcopy(a)
    b["identity"]["checkpoint"] = "changed"
    b["state_hash"] = "broken"
    fails(Status.IDENTITY_DRIFT, compare_snapshots, a, b)


def test_resealed_byte_difference_still_fails():
    a = state_sequence("C4", 4, "training")[-1]
    b = copy.deepcopy(a)
    b["index"]["rows"][0]["data"] = "AQE="
    b["state_hash"] = digest({k: v for k, v in b.items() if k != "state_hash"})
    validate_snapshot(b)
    fails(Status.STATE_BYTES_MISMATCH, compare_snapshots, a, b)


@pytest.mark.parametrize("layer", ["C0", "C4", "C128"])
@pytest.mark.parametrize("position", [0, 3, 4, 127, 128, 1024])
def test_candidate_range(layer, position):
    selection = list(range(min((position + 1) // 4, 512))) if layer == "C4" else None
    plan = candidate_plan(layer, position, selection)
    validate_candidates(plan)
    assert plan["recent"] == list(range(max(0, position - 127), position + 1))


@pytest.mark.parametrize(
    "key,value,status",
    [
        ("order", "recent_then_compressed", Status.INVALID_CANDIDATE_ORDER),
        ("softmax_denominators", 2, Status.MULTIPLE_SOFTMAX_DENOMINATORS),
        ("sink_has_value", True, Status.INVALID_SINK_SEMANTICS),
        ("recent", [], Status.INVALID_CANDIDATE_ORDER),
        ("compressed", [1, 1], Status.INVALID_TOPK_ORDER),
        ("compressed", [999], Status.INVALID_TOPK_ORDER),
    ],
)
def test_invalid_candidates(key, value, status):
    plan = candidate_plan("C4", 127, [1, 0])
    plan[key] = value
    fails(status, validate_candidates, plan)


@pytest.mark.parametrize("cp", [1, 2, 4])
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_ws2_global_topology_and_ordered_tree(cp, tp):
    plan = topology(513, cp, tp)
    validate_topology(plan, 513)
    heads = torch.tensor([1e20, 1.0, -1e20, 1.0] * 16).reshape(64, 1)
    shards = dict(reversed(list(enumerate(heads.chunk(tp)))))
    expected = ordered_collective({0: heads}, 1)
    assert torch.equal(ordered_collective(shards, tp), expected)
    for i, owner in enumerate(plan["c4_owners"]):
        start, end = plan["ranges"][owner]
        assert start <= i * 4 + 3 < end
    bad = copy.deepcopy(plan)
    bad["global_index_visibility"] = []
    fails(Status.MISSING_GLOBAL_VISIBILITY, validate_topology, bad, 513)


def test_ws2_negative_owner_and_missing_rank():
    plan = topology(257, 4, 8)
    plan["c128_owners"][0] = -1
    fails(Status.DUPLICATE_LOGICAL_OWNER, validate_topology, plan, 257)
    fails(Status.MISSING_RANK, ordered_collective, {0: torch.zeros(32, 2)}, 2)
    fails(
        Status.SCHEMA_MISMATCH, ordered_collective, {0: torch.zeros(1, 2), 1: torch.zeros(1, 2)}, 2
    )


def test_snapshot_roundtrip_is_detached():
    snapshot = state_sequence("C128", 127, "training")[-1]
    restored = PlannerMock.restore(snapshot)
    assert restored.snapshot() == snapshot
    snapshot["main"]["partial"][0] = "AAAAAA=="
    assert restored.snapshot() != snapshot
    assert restored.identity == Identity(**asdict(Identity("C128")))
