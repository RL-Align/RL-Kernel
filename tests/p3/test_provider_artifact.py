import json
import struct
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pytest

from rl_engine.p3.artifact import create, verify
from rl_engine.p3.checker import check_anchor
from rl_engine.p3.contract import (
    OP_ABI,
    SAVED_ROUTE_SCHEMA,
    UNSET,
    P3Error,
    P3OpCtxHost,
    P3Verdict,
    SavedRouteSealedV1,
)
from rl_engine.p3.fixtures import catalog
from rl_engine.p3.mocks import materialize_tokens, validate_ownership, validate_unsharded_gate
from rl_engine.p3.negative import run_negative_fixtures
from rl_engine.p3.provider import (
    InvocationAllocator,
    RecordedProvider,
    readback_verdict,
    validate_saved,
)
from rl_engine.p3.recordings import record_case
from rl_engine.p3.reference import ReferenceProvider


def test_negative_fixture_catalog_executes():
    result = run_negative_fixtures()
    assert len(result["fixtures"]) >= 24
    assert all(x["result"] == "EXPECTED_REJECTION" for x in result["fixtures"])


@pytest.mark.parametrize(
    "status,expected",
    [
        (UNSET, P3Verdict.PASS),
        (1, P3Verdict.NON_FINITE),
        (2, P3Verdict.HASH_TABLE_INDEX_OUT_OF_RANGE),
    ],
)
def test_readback_success_and_legal_device_errors(status, expected):
    assert readback_verdict(struct.pack("<iiQ", status, 0, 33), 33) == expected
    assert (
        readback_verdict(struct.pack("<iiQ", status, 0, 33), 33, synchronized=False)
        == P3Verdict.INCOMPLETE_ARTIFACT
    )


def test_allocator_durable_restart_concurrency_and_stale_attempt(tmp_path):
    path = tmp_path / "journal.json"
    allocator = InvocationAllocator(path, "run", "engine", 0)
    first = allocator.reserve()
    assert first == (1 << 32) | 1
    assert InvocationAllocator(path, "run", "engine", 0).reserve() == first + 1
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: allocator.reserve(), range(40)))
    assert len(set(ids)) == 40
    assert max(ids) == (1 << 32) | 42
    assert InvocationAllocator(path, "run", "engine", 0, 2).reserve() == (2 << 32) | 1
    with pytest.raises(P3Error) as exc:
        allocator.reserve()
    assert exc.value.verdict == P3Verdict.STALE_RUN_METADATA


def test_allocator_scope_corrupt_and_exhaustion(tmp_path):
    path = tmp_path / "journal.json"
    allocator = InvocationAllocator(path, "run", "engine", 0)
    allocator.reserve()
    with pytest.raises(P3Error) as exc:
        InvocationAllocator(path, "wrong-run", "engine", 0).reserve()
    assert exc.value.verdict == P3Verdict.IDENTITY_DRIFT
    data = json.loads(path.read_text())
    data["state"]["counter"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(P3Error) as exc:
        allocator.reserve()
    assert exc.value.verdict == P3Verdict.CORRUPT_ARTIFACT
    from rl_engine.p3.serialization import fingerprint

    data["state"]["counter"] = 0xFFFFFFFF
    data["checksum"] = fingerprint(data["state"])
    path.write_text(json.dumps(data))
    with pytest.raises(P3Error) as exc:
        allocator.reserve()
    assert exc.value.verdict == P3Verdict.CORRUPT_ARTIFACT


def test_saved_validation_precedes_zero_and_checks_padding():
    case = catalog()[0]
    recording = record_case(case)
    saved = SavedRouteSealedV1(**deepcopy(recording["saved"]["route"]))
    ctx = P3OpCtxHost(
        "run",
        "recorded",
        0,
        case.row_active.copy(),
        recording["bundle"]["RoutePlan"]["route_artifact_fingerprint"],
    )
    validate_saved(saved, SAVED_ROUTE_SCHEMA, ctx)
    saved.payload["p"][3, 0] = np.nan
    # Integrity failure still applies to inactive raw saved bytes.
    assert (
        ReferenceProvider().hash_route_bwd(ctx, case.dweights, saved).verdict
        == P3Verdict.CORRUPT_ARTIFACT
    )
    ctx.row_active[:] = False
    assert (
        ReferenceProvider().hash_route_bwd(ctx, case.dweights, saved.payload).verdict
        == P3Verdict.SCHEMA_MISMATCH
    )


def test_zero_active_no_saved_and_no_allocator(tmp_path):
    case = catalog()[6]
    ctx = P3OpCtxHost(
        "run",
        "recorded",
        0,
        case.row_active,
        allocator=InvocationAllocator(tmp_path / "journal", "run", "recorded", 0),
    )
    result = ReferenceProvider().stable_topk6_fwd(ctx, np.zeros((4, 256), np.float32))
    assert result.verdict == P3Verdict.ZERO_ACTIVE_TOKENS and result.payload is None
    assert ctx.invocation_id == 0
    assert not (tmp_path / "journal").exists()


def test_artifact_offline_replay_input_bound_provider_and_resume(tmp_path):
    path = tmp_path / "sealed"
    artifact = create(path)
    assert len(verify(path)["recordings"]) == len(catalog())
    assert sum(len(r["operators"]) for r in artifact["recordings"]) == 5 * sum(
        c.row_active.any() for c in catalog()
    )
    assert create(path, resume=True)["request"] == artifact["request"]
    case = catalog()[0]
    provider = RecordedProvider.from_artifact(path, case.case_id)
    ctx = P3OpCtxHost("run", "recorded", 0, case.row_active.copy())
    op = artifact["recordings"][0]["operators"]["router_sqrt_softplus_fwd"]
    assert provider.router_sqrt_softplus_fwd(ctx, *op["inputs"]).verdict == P3Verdict.PASS
    changed = op["inputs"][0].copy()
    changed[0, 0] += 1
    assert (
        provider.router_sqrt_softplus_fwd(ctx, changed, case.round_policy).verdict
        == P3Verdict.IDENTITY_DRIFT
    )
    with pytest.raises(P3Error) as exc:
        create(path, backend="cuda", resume=True)
    assert exc.value.verdict == P3Verdict.IDENTITY_DRIFT
    with pytest.raises(P3Error):
        create(path)


@pytest.mark.parametrize("mutation", ["bitflip", "extra", "incomplete", "old"])
def test_artifact_corruption_rejected(tmp_path, mutation):
    path = tmp_path / "sealed"
    create(path)
    if mutation == "bitflip":
        with (path / "artifact.json").open("a") as file:
            file.write(" ")
    elif mutation == "extra":
        (path / "unsealed-extra").write_text("extra")
    elif mutation == "incomplete":
        (path / "seal.json").rename(path / "unfinished-seal")
    else:
        seal = json.loads((path / "seal.json").read_text())
        seal["schema"] = OP_ABI
        (path / "seal.json").write_text(json.dumps(seal))
    with pytest.raises(P3Error):
        verify(path)


def test_distributed_hook_and_fail_closed_gate():
    case = catalog()[0]
    partition = materialize_tokens(case, [203, 305])
    plan = record_case(partition)["bundle"]["RoutePlan"]
    validate_ownership(plan, [203, 305])
    with pytest.raises(P3Error) as exc:
        validate_ownership(plan, [203])
    assert exc.value.verdict == P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING
    with pytest.raises(P3Error) as exc:
        validate_ownership(plan, [101, 203, 305])
    assert exc.value.verdict == P3Verdict.INCOMPLETE_ARTIFACT
    declaration = {"gate": "unsharded", "sp": True}
    validate_unsharded_gate(declaration, declaration, (2, 4096), (2, 256))
    with pytest.raises(P3Error) as exc:
        validate_unsharded_gate(declaration, declaration, (2, 4096), (2, 128))
    assert exc.value.verdict == P3Verdict.FORBIDDEN_LOCAL_SHARD_TOPK
    with pytest.raises(P3Error) as exc:
        check_anchor()
    assert exc.value.verdict == P3Verdict.MISSING_PROVENANCE


def test_schema_is_checked_before_zero_active_and_padding_is_ignored():
    active = np.array([True, False])
    ctx = P3OpCtxHost("run", "recorded", 0, active)
    q = np.ones((2, 256), np.float32)
    q[1] = np.nan
    provider = ReferenceProvider()
    assert provider.stable_topk6_fwd(ctx, q).verdict == P3Verdict.PASS
    ctx.row_active[:] = False
    assert provider.stable_topk6_fwd(ctx, q[:, :128]).verdict == P3Verdict.SCHEMA_MISMATCH
    assert (
        provider.router_sqrt_softplus_fwd(ctx, q.astype(np.float64), "fp32_direct").verdict
        == P3Verdict.SCHEMA_MISMATCH
    )


def test_first_mismatch_byte_precision_and_identity_priority():
    from rl_engine.p3.checker import compare_recordings

    base = record_case(catalog()[0])
    changed = deepcopy(base)
    changed["operators"]["learned_route_fwd"]["payload"]["weights"].view(np.uint32)[1, 4] ^= 1
    diff = compare_recordings(base, changed)
    assert diff["verdict"] == "ROUTE_WEIGHT_BYTES_MISMATCH"
    assert diff["first_mismatch"]["global_token_id"] == 203
    assert diff["first_mismatch"]["column"] == 4
    assert diff["first_mismatch"]["owner"] == "T04"
    changed["identity"]["checkpoint_id"] = "wrong-checkpoint"
    assert compare_recordings(base, changed)["verdict"] == "IDENTITY_DRIFT"


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("physical", P3Verdict.INVALID_PLACEMENT_MAP),
        ("version", P3Verdict.PLACEMENT_MAP_VERSION_MISMATCH),
        ("tie", P3Verdict.TIE_BREAK_POLICY_MISMATCH),
        ("round", P3Verdict.LOGIT_ROUND_POINT_MISMATCH),
    ],
)
def test_handoff_mapping_and_policy_validation(mutation, expected):
    from rl_engine.p3.checker import validate_plan

    bundle = record_case(catalog()[0])["bundle"]
    plan = bundle["RoutePlan"]
    if mutation == "physical":
        plan["envelope"][0]["physical_expert_id"] = 999
    elif mutation == "version":
        plan["envelope"][0]["placement_map_version"] = "stale"
    elif mutation == "tie":
        plan["core"][0]["tie_break_policy"] = "unstable"
    else:
        plan["core"][0]["logit_round_point"] = "wrong"
    with pytest.raises(P3Error) as exc:
        validate_plan(bundle)
    assert exc.value.verdict == expected
