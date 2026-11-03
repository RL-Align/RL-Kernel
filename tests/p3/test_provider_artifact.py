import json
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rl_engine.p3.artifact import create, verify
from rl_engine.p3.assembler import assemble, combine_seed, seal_forward_saved
from rl_engine.p3.boundary import consume
from rl_engine.p3.checker import check_anchor, compare_plans, validate_plan
from rl_engine.p3.contract import (
    CORE_SCHEMA,
    ENVELOPE_SCHEMA_V2,
    OP_ABI,
    SAVED_ROUTE_SCHEMA,
    UNSET,
    P3Error,
    P3OpCtxHost,
    P3OpResult,
    P3Verdict,
    SavedRouteSealedV1,
)
from rl_engine.p3.fixtures import catalog
from rl_engine.p3.mocks import (
    OWNERSHIP,
    materialize_tokens,
    validate_ownership,
    validate_unsharded_gate,
)
from rl_engine.p3.negative import run_negative_fixtures
from rl_engine.p3.provider import (
    InvocationAllocator,
    RecordedProvider,
    readback_verdict,
    validate_saved,
)
from rl_engine.p3.recordings import record_case
from rl_engine.p3.reference import ReferenceProvider
from rl_engine.p3.serialization import canonical, decode, encode, fingerprint, ordered_record


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


# T05 coverage stays in this existing handoff/artifact script.
def forward_results(case):
    recording = record_case(case)
    provenance = recording["provenance"]
    score = recording["operators"]["router_sqrt_softplus_fwd"]["payload"]
    route = recording["operators"][case.router_mode + "_route_fwd"]["payload"]
    return P3OpResult(P3Verdict.PASS, score, provenance), P3OpResult(
        P3Verdict.PASS, route, provenance
    )


def runtime_for(case):
    return {
        "group": "tp1.cp2.dp1.pp1.ep2.v1",
        "topology": "synthetic.ws2.v1",
        "token_ownership": dict(OWNERSHIP),
        "expected_token_set": sorted(case.global_token_id[case.row_active].tolist()),
    }


@pytest.mark.parametrize("case_index", [0, 1, 4, 5])
def test_t05_training_inference_pack_padding_placement_invariance(case_index):
    case = catalog()[case_index]
    score, route = forward_results(case)
    training = assemble(case, score, route, engine_id="training")
    packed_case = case.rows([3, 2, 0, 3, 1])  # Reorder tokens and add another padding row.
    packed_score, packed_route = forward_results(packed_case)
    inference = assemble(
        packed_case,
        packed_score,
        packed_route,
        run_id="inference-run",
        engine_id="inference",
        attempt_id=3,
        rank=2,
        placement=list(reversed(range(256))),
        placement_version="reverse.v1",
        envelope_version=ENVELOPE_SCHEMA_V2,
        runtime=runtime_for(packed_case),
    )
    assert compare_plans(training, inference) == P3Verdict.CASE_PASS
    a, b = training["RoutePlan"], inference["RoutePlan"]
    assert a["per_token_fingerprints"] == b["per_token_fingerprints"]
    assert a["route_semantic_fingerprint"] == b["route_semantic_fingerprint"]
    assert a["route_artifact_fingerprint"] != b["route_artifact_fingerprint"]
    assert canonical(training["CombinePlanSeed"]) == canonical(inference["CombinePlanSeed"])
    assert canonical(training) == canonical(assemble(case, score, route, engine_id="training"))
    # Per-token Core bytes must agree independently of the hash comparison.
    for token in case.global_token_id[case.row_active]:
        left = [r for r in a["core"] if r["global_token_id"] == token]
        right = [r for r in b["core"] if r["global_token_id"] == token]
        assert canonical(left) == canonical(right)


def test_t05_v1_handoff_bytes_and_schema_order_remain_frozen():
    case = catalog()[0]
    score, route = forward_results(case)
    actual = assemble(case, score, route)
    example = Path("examples/dsv4_p3_startkit/handoff.v1.json")
    assert canonical(actual) == canonical(decode(json.loads(example.read_text())))
    row = actual["RoutePlan"]["core"][0]
    assert canonical(ordered_record(CORE_SCHEMA, dict(reversed(list(row.items()))))) == canonical(
        row
    )


@pytest.mark.parametrize(
    "fault",
    [
        "saved-score",
        "selection-buffer",
        "saved-ids",
        "pre-bias",
        "weights",
        "z",
        "slots",
        "dtype",
        "provenance",
        "failed",
    ],
)
def test_t05_rejects_inconsistent_forward_results(fault):
    case = catalog()[0]
    score, route = forward_results(case)
    if fault == "saved-score":
        score.payload["saved_score"]["s"][0, 0] += 1
    elif fault == "selection-buffer":
        route.payload["q"][0, 0] += 1
    elif fault == "saved-ids":
        route.payload["saved_route"]["ids"][0, 0] ^= 1
    elif fault == "pre-bias":
        route.payload["saved_route"]["a"][0, 0] += 1
    elif fault == "weights":
        route.payload["weights"].view(np.uint32)[0, 0] ^= 1
    elif fault == "z":
        score.payload["saved_score"]["z_prime"][0, 0] += 1
    elif fault == "slots":
        route.payload["ids"][0, 0] = route.payload["ids"][0, 1]
        route.payload["saved_route"]["ids"] = route.payload["ids"].copy()
    elif fault == "dtype":
        route.payload["weights"] = route.payload["weights"].astype(np.float64)
    elif fault == "provenance":
        route.provenance["fallback_reason"] = "cpu fallback"
    else:
        route = P3OpResult(P3Verdict.NON_FINITE)
    with pytest.raises(P3Error):
        assemble(case, score, route)


@pytest.mark.parametrize(
    "field,value",
    [
        ("topk_index", True),
        ("valid", 1),
        ("logical_expert_id", 1.0),
        ("route_weight", np.float64(0.1)),
    ],
)
def test_t05_schema_rejects_coerced_field_types(field, value):
    bundle = record_case(catalog()[0])["bundle"]
    bundle["RoutePlan"]["core"][0][field] = value
    with pytest.raises(P3Error) as exc:
        validate_plan(bundle)
    assert exc.value.verdict == P3Verdict.SCHEMA_MISMATCH


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("missing", P3Verdict.INCOMPLETE_ARTIFACT),
        ("unauthorized", P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING),
        ("duplicate", P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING),
        ("ownership", P3Verdict.SCHEMA_MISMATCH),
        ("schema", P3Verdict.SCHEMA_MISMATCH),
        ("placement", P3Verdict.INVALID_PLACEMENT_MAP),
    ],
)
def test_t05_ws2_rejects_invalid_declarations(fault, expected):
    case = catalog()[0]
    score, route = forward_results(case)
    runtime = runtime_for(case)
    kwargs = {"runtime": runtime, "envelope_version": ENVELOPE_SCHEMA_V2}
    if fault == "missing":
        runtime["expected_token_set"].append(999)
    elif fault == "unauthorized":
        runtime["expected_token_set"].pop()
    elif fault == "duplicate":
        runtime["expected_token_set"].append(101)
    elif fault == "ownership":
        runtime["token_ownership"]["ep"] = "partition"
    elif fault == "schema":
        kwargs["envelope_version"] = "RoutePlanEnvelope.v0"
    else:
        kwargs["placement"] = [False] + list(range(1, 256))
    with pytest.raises(P3Error) as exc:
        assemble(case, score, route, **kwargs)
    assert exc.value.verdict == expected


def test_t05_seed_preserves_duplicate_hash_slots_and_detects_mutation():
    case = next(c for c in catalog() if c.case_id == "hash-duplicates.fp32_direct")
    bundle = record_case(case)["bundle"]
    seed = bundle["CombinePlanSeed"]
    assert canonical(seed) == canonical(combine_seed(bundle["RoutePlan"]))
    for token in seed["tokens"]:
        source_row = np.flatnonzero(case.global_token_id == token["global_token_id"])[0]
        assert [s["topk_index"] for s in token["slots"]] == list(range(6))
        assert [s["logical_expert_id"] for s in token["slots"]] == case.table[
            case.input_token_id[source_row]
        ].tolist()
    assert any(len({s["logical_expert_id"] for s in t["slots"]}) < 6 for t in seed["tokens"])
    seed["tokens"][0]["slots"][0]["route_weight"] = np.float32(-1)
    with pytest.raises(P3Error) as exc:
        validate_plan(bundle)
    assert exc.value.verdict == P3Verdict.INVALID_DISCRETE_PLAN


def test_t05_ws2_consumers_read_bundle_in_separate_process(tmp_path):
    case = catalog()[0]
    score, route = forward_results(case)
    runtime = runtime_for(case)
    bundle = assemble(case, score, route, envelope_version=ENVELOPE_SCHEMA_V2, runtime=runtime)
    plan = bundle["RoutePlan"]
    assert plan["expected_token_set"] == runtime["expected_token_set"]
    assert all("expected_token_set" not in env for env in plan["envelope"])
    changed = deepcopy(bundle)
    changed["RoutePlan"]["expected_token_set"] = list(reversed(runtime["expected_token_set"]))
    with pytest.raises(P3Error) as exc:
        validate_plan(changed)
    assert exc.value.verdict == P3Verdict.CORRUPT_ARTIFACT
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(encode(bundle)))
    program = """
import json, sys
from rl_engine.p3.boundary import consume
from rl_engine.p3.serialization import decode, fingerprint
b = decode(json.load(open(sys.argv[1])))
identity = b['RoutePlan']['identity']
results = {p: fingerprint(consume(b, p, expected_identity=identity))
           for p in ('P4', 'P6', 'P7')}
print(json.dumps(results))
"""
    observed = json.loads(
        subprocess.check_output([sys.executable, "-c", program, str(path)], text=True)
    )
    for consumer in ("P4", "P6", "P7"):
        result = consume(
            bundle,
            consumer,
            expected_identity=case.identity(),
            expected_runtime=runtime,
            expected_placement_version="identity.v1",
        )
        assert observed[consumer] == fingerprint(result)
    with pytest.raises(P3Error) as exc:
        consume(
            bundle, "P4", expected_identity=case.identity(), expected_placement_version="stale.v1"
        )
    assert exc.value.verdict == P3Verdict.PLACEMENT_MAP_VERSION_MISMATCH
    with pytest.raises(P3Error) as exc:
        consume(bundle, "P7", expected_identity=case.identity(), expected_runtime={"rank": 99})
    assert exc.value.verdict == P3Verdict.IDENTITY_DRIFT


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "run",
        "attempt",
        "id-reuse",
        "mask",
        "plan",
        "payload",
        "saved-logits",
        "saved-normalization",
    ],
)
def test_t05_sealed_saved_binds_forward_invocations(fault):
    case = catalog()[0]
    score, route = forward_results(case)
    bundle = assemble(case, score, route)
    plan = bundle["RoutePlan"]
    ctx = P3OpCtxHost(
        "p3-synthetic",
        "recorded",
        0,
        case.row_active.copy(),
        plan["route_artifact_fingerprint"],
        invocation_id=(1 << 32) | 1,
    )
    route_ctx = replace(ctx, invocation_id=(1 << 32) | 2)
    if fault == "run":
        route_ctx.run_id = "other"
    elif fault == "attempt":
        route_ctx.invocation_id = (2 << 32) | 2
    elif fault == "id-reuse":
        route_ctx.invocation_id = ctx.invocation_id
    elif fault == "mask":
        route_ctx.row_active = ~case.row_active
    elif fault == "plan":
        route_ctx.route_artifact_fingerprint = "ab" * 32
    elif fault == "payload":
        route.payload["weights"].view(np.uint32)[0, 0] ^= 1
    elif fault == "saved-logits":
        score.payload["saved_score"]["z_prime"][0, 0] += 1
    elif fault == "saved-normalization":
        route.payload["saved_route"]["Z"][0, 0] += 1
    if fault:
        with pytest.raises(P3Error):
            seal_forward_saved(case, bundle, score, route, ctx, route_ctx)
    else:
        sealed = seal_forward_saved(case, bundle, score, route, ctx, route_ctx)
        expected = record_case(case)["saved"]
        assert canonical(
            {k: {"header": v.header, "payload": v.payload} for k, v in sealed.items()}
        ) == canonical(expected)
        sealed["route"].payload["p"].view(np.uint32)[3, 0] ^= 1
        with pytest.raises(P3Error) as exc:
            validate_saved(sealed["route"], SAVED_ROUTE_SCHEMA, route_ctx)
        assert exc.value.verdict == P3Verdict.CORRUPT_ARTIFACT


def test_t05_padding_saved_bytes_are_archived_without_affecting_semantics():
    case = catalog()[0]
    score, route = forward_results(case)
    baseline = assemble(case, score, route)
    score.payload["s"][~case.row_active] = np.nan
    for value in score.payload["saved_score"].values():
        value[~case.row_active] = np.nan
    route.payload["ids"][~case.row_active] = -999
    route.payload["weights"][~case.row_active] = np.nan
    for name, value in route.payload["saved_route"].items():
        value[~case.row_active] = -999 if name == "ids" else np.nan
    bundle = assemble(case, score, route)
    assert canonical(bundle) == canonical(baseline)
    fingerprint_value = bundle["RoutePlan"]["route_artifact_fingerprint"]
    ctx = P3OpCtxHost(
        "p3-synthetic",
        "recorded",
        0,
        case.row_active.copy(),
        fingerprint_value,
        invocation_id=(1 << 32) | 1,
    )
    sealed = seal_forward_saved(
        case, bundle, score, route, ctx, replace(ctx, invocation_id=(1 << 32) | 2)
    )
    validate_saved(sealed["route"], SAVED_ROUTE_SCHEMA, ctx)
    assert np.isnan(sealed["route"].payload["p"][~case.row_active]).all()


def test_t05_runtime_stream_handles_do_not_change_artifact_identity():
    from rl_engine.p3.assembler import artifact_fingerprint

    plan = record_case(catalog()[0])["bundle"]["RoutePlan"]
    envelopes = deepcopy(plan["envelope"])
    for env in envelopes:
        env["stream"] = 123
    a = artifact_fingerprint(plan["identity"]["case_id"], plan["core"], envelopes)
    for env in envelopes:
        env["stream"] = 456
    b = artifact_fingerprint(plan["identity"]["case_id"], plan["core"], envelopes)
    assert a == b


@pytest.mark.parametrize("case_index", [6, 7, 14, 15])
def test_t05_zero_active_recordings_never_have_saved_state(case_index):
    case = catalog()[case_index]
    assert not case.row_active.any()
    recording = record_case(case)
    assert recording["saved"] == {} and recording["bundle"] is None
    zero = P3OpResult(P3Verdict.ZERO_ACTIVE_TOKENS)
    with pytest.raises(P3Error):
        assemble(case, zero, zero)


@pytest.mark.parametrize(
    "field,value",
    [
        ("table_present", True),
        ("bias_present", False),
        ("weight_source", "post_bias_score"),
        ("selection_source", "tid2eid.table_slot"),
    ],
)
def test_t05_padding_metadata_cannot_be_resealed_as_valid(field, value):
    from rl_engine.p3.assembler import artifact_fingerprint

    bundle = record_case(catalog()[0])["bundle"]
    plan = bundle["RoutePlan"]
    plan["core"][-1][field] = value
    digest = artifact_fingerprint(plan["identity"]["case_id"], plan["core"], plan["envelope"])
    plan["route_artifact_fingerprint"] = digest
    for env in plan["envelope"]:
        env["route_artifact_fingerprint"] = digest
    with pytest.raises(P3Error) as exc:
        validate_plan(bundle)
    assert exc.value.verdict == P3Verdict.IDENTITY_DRIFT
