"""The sole assembler_abi.v1 entry point for validated, deterministic handoffs."""

from copy import deepcopy

import numpy as np

from .contract import (
    CAPACITY_POLICY,
    CORE_SCHEMA,
    ENVELOPE_SCHEMA,
    ENVELOPE_SCHEMA_V2,
    EPSILON,
    MODEL_FIELDS,
    SAVED_FIELDS,
    SAVED_ROUTE_SCHEMA,
    SAVED_SCORE_SCHEMA,
    SCALE,
    SEED_SCHEMA,
    SLOT_FIELDS,
    TIE_POLICY,
    E,
    K,
    P3Verdict,
    envelope_fields,
)
from .oracle import require, tensor
from .serialization import canonical, checked_structure, fingerprint, ordered_record


@checked_structure
def validate_case(case):
    T = len(case.z)
    tensor(case.z, "float32", (T, E))
    tensor(case.row_active, "bool", (T,))
    tensor(case.global_token_id, "int64", (T,))
    tensor(case.input_token_id, "int64", (T,))
    tensor(case.dweights, "float32", (T, K))
    tensor(case.bias, "float32", (E,))
    tensor(case.hidden, "uint16", (T, 4096))
    require(
        case.hidden_shape == (T, 4096) and case.hidden_dtype in ("bfloat16", "float16"),
        P3Verdict.SCHEMA_MISMATCH,
        "hidden must retain full 4096 features",
    )
    require(
        case.absolute_layer >= 0
        and case.router_mode == ("hash" if case.absolute_layer < 3 else "learned"),
        P3Verdict.INVALID_DISCRETE_PLAN,
        "layer/hash/learned XOR",
    )
    active_ids = case.global_token_id[case.row_active]
    require(
        (active_ids >= 0).all() and len(np.unique(active_ids)) == len(active_ids),
        P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING,
        "duplicate/invalid global token identity",
    )
    require(
        (case.input_token_id[case.row_active] >= 0).all(),
        P3Verdict.IDENTITY_DRIFT,
        "negative active input token",
    )
    require(
        case.table.dtype == np.int32
        and case.table.ndim == 2
        and case.table.shape[1] == K
        and len(case.table) > 0
        and ((case.table >= 0) & (case.table < E)).all(),
        P3Verdict.HASH_TABLE_MISMATCH,
        "table and sentinel row must be valid",
    )


def _validate_payloads(active, score_result, route_result):
    """Shared forward/saved consistency checks for both assembly and sealing."""
    require(
        score_result.verdict == route_result.verdict == P3Verdict.PASS,
        P3Verdict.INCOMPLETE_ARTIFACT,
        "assembler requires completed forward operators",
    )
    T = len(active)
    score, route = score_result.payload, route_result.payload
    tensor(score["s"], "float32", (T, E))
    tensor(route["ids"], "int32", (T, K))
    tensor(route["weights"], "float32", (T, K))
    require(
        active.any(), P3Verdict.ZERO_ACTIVE_TOKENS, "zero-active handoff has no plan or saved state"
    )
    for version, payload in (
        (SAVED_SCORE_SCHEMA, score["saved_score"]),
        (SAVED_ROUTE_SCHEMA, route["saved_route"]),
    ):
        require(
            set(payload) == set(SAVED_FIELDS[version]), P3Verdict.SCHEMA_MISMATCH, "saved fields"
        )
        for name, value in payload.items():
            shape = (T, E) if version == SAVED_SCORE_SCHEMA else (T, 1 if name == "Z" else K)
            tensor(value, "int32" if name == "ids" else "float32", shape)
            require(
                np.isfinite(value[active]).all(), P3Verdict.NON_FINITE, "active saved nonfinite"
            )
    ids = route["ids"][active]
    require(
        ((ids >= 0) & (ids < E)).all(), P3Verdict.INVALID_DISCRETE_PLAN, "invalid logical expert"
    )
    for actual, expected in (
        (score["saved_score"]["s"][active], score["s"][active]),
        (route["saved_route"]["ids"][active], ids),
        (route["saved_route"]["a"][active], np.take_along_axis(score["s"][active], ids, axis=1)),
    ):
        require(
            canonical(actual) == canonical(expected),
            P3Verdict.IDENTITY_DRIFT,
            "saved/forward drift",
        )
    from . import bitmath

    saved = route["saved_route"]
    require(
        (saved["Z"][active] > 0).all(), P3Verdict.NON_FINITE, "invalid normalization denominator"
    )
    for actual, expected in (
        (saved["Z"][active], bitmath.sum6(saved["a"][active])[:, None] + EPSILON),
        (saved["p"][active], saved["a"][active] / saved["Z"][active]),
        (route["weights"][active], saved["p"][active] * SCALE),
    ):
        require(
            canonical(actual) == canonical(expected),
            P3Verdict.IDENTITY_DRIFT,
            "normalization drift",
        )
    require(
        np.isfinite(score["s"][active]).all() and np.isfinite(route["weights"][active]).all(),
        P3Verdict.NON_FINITE,
        "nonfinite active handoff",
    )
    return score, route, ids


@checked_structure
def validate_forward(case, score_result, route_result):
    """Validate handoff consistency without rerunning score or Top-K kernels."""
    from . import bitmath
    from .checker import check_actual_provenance
    from .contract import ROUND_POLICIES

    validate_case(case)
    active = case.row_active
    score, route, ids = _validate_payloads(active, score_result, route_result)
    require(
        case.round_policy in ROUND_POLICIES, P3Verdict.LOGIT_ROUND_POINT_MISMATCH, "round policy"
    )
    for result in (score_result, route_result):
        check_actual_provenance(result.provenance)
        require(
            result.provenance.get("round_policy", case.round_policy)
            in (case.round_policy, "operator-input-or-sealed-forward"),
            P3Verdict.LOGIT_ROUND_POINT_MISMATCH,
            "forward round policy drift",
        )
    require(
        score_result.provenance["backend_profile"] == route_result.provenance["backend_profile"],
        P3Verdict.INVALID_PROFILE,
        "score and route execution profiles differ",
    )
    expected_z = case.z[active]
    if case.round_policy == "bf16_round_then_widen":
        expected_z = bitmath.apply(expected_z, "bf16")
    require(
        canonical(score["saved_score"]["z_prime"][active]) == canonical(expected_z),
        P3Verdict.IDENTITY_DRIFT,
        "score saved logits differ from case and round policy",
    )
    if case.router_mode == "hash":
        tokens = case.input_token_id[active]
        require(
            (tokens < len(case.table)).all(), P3Verdict.HASH_TABLE_INDEX_OUT_OF_RANGE, "table index"
        )
        require(
            np.array_equal(ids, case.table[tokens]),
            P3Verdict.TOPK_ORDER_MISMATCH,
            "Hash must preserve table slots",
        )
    else:
        require(np.isfinite(case.bias).all(), P3Verdict.NON_FINITE, "nonfinite bias")
        if "q" in route:
            tensor(route["q"], "float32", score["s"].shape)
            require(
                canonical(route["q"][active]) == canonical(score["s"][active] + case.bias),
                P3Verdict.IDENTITY_DRIFT,
                "selection buffer differs from post-bias scores",
            )
        selection = np.take_along_axis(score["s"][active], ids, axis=1) + case.bias[ids]
        require(np.isfinite(selection).all(), P3Verdict.NON_FINITE, "nonfinite selection")
        require(
            all(len(set(row.tolist())) == K for row in ids)
            and np.all(
                (selection[:, :-1] > selection[:, 1:])
                | ((selection[:, :-1] == selection[:, 1:]) & (ids[:, :-1] < ids[:, 1:]))
            ),
            P3Verdict.TOPK_ORDER_MISMATCH,
            "Learned slots must retain deterministic selection order",
        )


def token_fingerprint(rows):
    first = rows[0]
    model = {k: first[k] for k in MODEL_FIELDS}
    for key in ("table_fingerprint", "bias_fingerprint"):
        model[key] = bytes.fromhex(model[key])
    return fingerprint([model, [{k: row[k] for k in SLOT_FIELDS} for row in rows]])


def artifact_fingerprint(case_id, rows, envelopes):
    """Schema-ordered artifact identity; runtime stream handles are archive-only."""
    body = []
    for env in envelopes:
        record = {k: env[k] for k in envelope_fields(env["envelope_schema_version"])}
        if isinstance(record["stream"], (int, np.integer)):
            record["stream"] = "archive-only.runtime-stream"
        body.append(record)
    return fingerprint({"case_id": case_id, "core": rows, "envelope": body})


@checked_structure
def assemble(
    case,
    score_result,
    route_result,
    *,
    run_id="p3-synthetic",
    engine_id="recorded",
    attempt_id=1,
    rank=0,
    placement=None,
    placement_version="identity.v1",
    envelope_version=ENVELOPE_SCHEMA,
    runtime=None,
):
    """Receive case + PASS operator payloads; never compute routing in the assembler."""
    validate_forward(case, score_result, route_result)
    fields = envelope_fields(envelope_version)
    runtime = deepcopy(runtime)
    if envelope_version == ENVELOPE_SCHEMA_V2:
        require(
            isinstance(runtime, dict)
            and set(runtime) == {"group", "topology", "token_ownership", "expected_token_set"},
            P3Verdict.SCHEMA_MISMATCH,
            "Envelope v2 requires explicit ownership and topology",
        )
        from .mocks import OWNERSHIP, validate_ownership_metadata

        validate_ownership_metadata(runtime["token_ownership"], runtime["expected_token_set"])
        runtime["token_ownership"] = {k: runtime["token_ownership"][k] for k in OWNERSHIP}
        runtime["expected_token_set"] = sorted(runtime["expected_token_set"])
    else:
        require(runtime is None, P3Verdict.SCHEMA_MISMATCH, "runtime metadata requires Envelope v2")
    runtime_envelope = (
        None
        if runtime is None
        else {
            **{k: v for k, v in runtime.items() if k != "expected_token_set"},
            "expected_token_set_fingerprint": fingerprint(runtime["expected_token_set"]),
        }
    )
    placement = list(range(E)) if placement is None else list(placement)
    require(
        len(placement) == E
        and all(type(e) is int for e in placement)
        and sorted(placement) == list(range(E)),
        P3Verdict.INVALID_PLACEMENT_MAP,
        "placement must be a versioned bijection",
    )
    require(
        isinstance(placement_version, str) and bool(placement_version),
        P3Verdict.PLACEMENT_MAP_VERSION_MISMATCH,
        "placement version required",
    )
    s = score_result.payload["s"]
    ids, weights = route_result.payload["ids"], route_result.payload["weights"]
    active = case.row_active
    provenance = route_result.provenance
    recorded = provenance["actual_backend"] == "recorded-cpu"
    table_hash = fingerprint(case.table) if case.router_mode == "hash" else "00" * 32
    bias_hash = fingerprint(case.bias) if case.router_mode == "learned" else "00" * 32
    rows, envelopes, per_token = [], [], {}
    for row in range(len(case.z)):
        valid = bool(active[row])
        token_rows = []
        for slot in range(K):
            expert = int(ids[row, slot]) if valid else -1
            score = s[row, expert] if valid else np.float32(0)
            selection = (
                (score + case.bias[expert] if case.router_mode == "learned" else score)
                if valid
                else np.float32(0)
            )
            record = {
                "core_schema_version": CORE_SCHEMA,
                "checkpoint_id": case.checkpoint_id,
                "weight_id": case.weight_id,
                "weight_fingerprint": case.weight_fingerprint,
                "absolute_layer": case.absolute_layer,
                "router_mode": case.router_mode,
                "global_token_id": int(case.global_token_id[row]) if valid else -1,
                "input_token_id": int(case.input_token_id[row]) if valid else -1,
                "capacity_policy": CAPACITY_POLICY,
                "overflow_policy": CAPACITY_POLICY,
                "logit_round_point": case.round_policy,
                "tie_break_policy": TIE_POLICY,
                "table_present": case.router_mode == "hash",
                "table_fingerprint": table_hash,
                "bias_present": case.router_mode == "learned",
                "bias_fingerprint": bias_hash,
                "selection_source": (
                    "tid2eid.table_slot" if case.router_mode == "hash" else "post_bias_score"
                ),
                "weight_source": "pre_bias_score",
                "topk_index": slot,
                "logical_expert_id": expert,
                "valid": valid,
                "invalid_reason": "none" if valid else "padding",
                "route_weight": weights[row, slot] if valid else np.float32(0),
                "weight_score": score,
                "selection_score": selection,
                "capacity": -1,
            }
            token_rows.append(record)
            envelope = {
                "envelope_schema_version": envelope_version,
                "run_id": run_id,
                "engine_id": engine_id,
                "attempt_id": attempt_id,
                "source_row": row,
                "physical_expert_id": placement[expert] if valid else -1,
                "placement_map_version": placement_version,
                "rank": rank,
                "group": "local",
                "topology": "synthetic.single_rank.v1",
                "backend_profile": route_result.provenance.get("backend_profile", ""),
                "kernel": provenance["kernel"],
                "build": (
                    "p3-start-kit.v1"
                    if recorded
                    else fingerprint([provenance["source_fingerprint"], provenance["build_flags"]])
                ),
                "device": provenance["device"],
                "stream": provenance["stream"],
                "event": "handoff" if recorded else provenance["event"],
            }
            if runtime_envelope is not None:
                envelope.update(deepcopy(runtime_envelope))
            envelopes.append({k: envelope[k] for k in fields})
        semantic = token_fingerprint(token_rows) if valid else "00" * 32
        if valid:
            per_token[str(case.global_token_id[row])] = semantic
        rows.extend(
            ordered_record(CORE_SCHEMA, {**r, "route_semantic_fingerprint": semantic})
            for r in token_rows
        )
    per_token = dict(sorted(per_token.items(), key=lambda item: int(item[0])))
    semantic_case = fingerprint([(int(k), v) for k, v in per_token.items()])
    artifact_hash = artifact_fingerprint(case.case_id, rows, envelopes)
    for env in envelopes:
        env["route_artifact_fingerprint"] = artifact_hash
        ordered_record(envelope_version, env)
    plan = {
        "core": rows,
        "envelope": envelopes,
        "per_token_fingerprints": per_token,
        "route_semantic_fingerprint": semantic_case,
        "route_artifact_fingerprint": artifact_hash,
        "padding_sentinel_token_id": 0,
        "padding_sentinel_identity": fingerprint(case.table[0]),
        "identity": case.identity(),
        "placement_map": {"version": placement_version, "logical_to_physical": placement},
    }
    if runtime is not None:
        plan["expected_token_set"] = runtime["expected_token_set"]
    bundle = {"RoutePlan": plan, "CombinePlanSeed": combine_seed(plan)}
    from .checker import validate_plan

    validate_plan(bundle)
    return bundle


@checked_structure
def seal_forward_saved(case, bundle, score_result, route_result, score_ctx, route_ctx):
    """Bind raw forward state to the validated plan and actual forward invocations."""
    from .checker import validate_plan
    from .provider import seal_saved, validate_saved

    validate_plan(bundle)
    validate_forward(case, score_result, route_result)
    plan = bundle["RoutePlan"]
    require(
        canonical(case.identity()) == canonical(plan["identity"]),
        P3Verdict.IDENTITY_DRIFT,
        "saved source case differs from assembled identity",
    )
    active = np.array([r["valid"] for r in plan["core"][::K]], dtype=np.bool_)
    require(active.any(), P3Verdict.ZERO_ACTIVE_TOKENS, "zero-active handoff has no saved state")
    s, ids, weights = (
        score_result.payload["s"],
        route_result.payload["ids"],
        route_result.payload["weights"],
    )
    cores = [r for r in plan["core"] if r["valid"]]
    expected_ids = np.array([r["logical_expert_id"] for r in cores], np.int32).reshape(-1, K)
    require(
        np.array_equal(ids[active], expected_ids)
        and weights[active].tobytes()
        == np.array([r["route_weight"] for r in cores], np.float32).tobytes()
        and np.take_along_axis(s[active], expected_ids, axis=1).tobytes()
        == np.array([r["weight_score"] for r in cores], np.float32).tobytes(),
        P3Verdict.IDENTITY_DRIFT,
        "forward results differ from sealed plan",
    )
    require(
        score_ctx.invocation_id != route_ctx.invocation_id,
        P3Verdict.CORRUPT_ARTIFACT,
        "forward invocations must be distinct",
    )
    require(
        isinstance(score_ctx.manifest_source, str)
        and score_ctx.manifest_source
        and score_ctx.manifest_source == route_ctx.manifest_source,
        P3Verdict.IDENTITY_DRIFT,
        "forward manifest sources differ",
    )
    saved = {}
    for name, version, result, ctx in (
        ("score", SAVED_SCORE_SCHEMA, score_result, score_ctx),
        ("route", SAVED_ROUTE_SCHEMA, route_result, route_ctx),
    ):
        tensor(ctx.row_active, "bool", active.shape)
        require(
            type(ctx.invocation_id) is int
            and 0 < ctx.invocation_id < 2**64
            and ctx.invocation_id & 0xFFFFFFFF != 0,
            P3Verdict.CORRUPT_ARTIFACT,
            "invalid forward invocation counter",
        )
        require(
            np.array_equal(ctx.row_active, active)
            and ctx.backend_tag == result.provenance["actual_backend"]
            and ctx.route_artifact_fingerprint == plan["route_artifact_fingerprint"]
            and all(
                [ctx.run_id, ctx.engine_id, ctx.rank, ctx.invocation_id >> 32]
                == [e["run_id"], e["engine_id"], e["rank"], e["attempt_id"]]
                for e in plan["envelope"]
            ),
            P3Verdict.IDENTITY_DRIFT,
            "forward context differs from assembled identity",
        )
        raw = result.payload["saved_" + name]
        state = seal_saved(version, raw, ctx, result)
        validate_saved(state, version, ctx)
        saved[name] = state
    return saved


@checked_structure
def combine_seed(plan):
    tokens = {}
    for row in plan["core"]:
        if row["global_token_id"] < 0:
            continue
        key = str(row["global_token_id"])
        token = tokens.setdefault(
            key,
            {
                "global_token_id": row["global_token_id"],
                "slots": [],
                "route_semantic_fingerprint": row["route_semantic_fingerprint"],
            },
        )
        token["slots"].append(
            {k: row[k] for k in ("topk_index", "logical_expert_id", "route_weight", "valid")}
        )
    require(
        all(
            [s["topk_index"] for s in token["slots"]] == list(range(K)) for token in tokens.values()
        ),
        P3Verdict.TOPK_ORDER_MISMATCH,
        "seed requires exactly six ordered slots per token",
    )
    return {
        "schema_version": SEED_SCHEMA,
        "tokens": [tokens[k] for k in sorted(tokens, key=int)],
        "route_semantic_fingerprint": plan["route_semantic_fingerprint"],
    }
