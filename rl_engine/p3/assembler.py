"""The sole assembler_abi.v1 stub. T05 replaces this module in place."""

import numpy as np

from .contract import (
    CAPACITY_POLICY,
    CORE_SCHEMA,
    ENVELOPE_FIELDS,
    ENVELOPE_SCHEMA,
    MODEL_FIELDS,
    SEED_SCHEMA,
    SLOT_FIELDS,
    TIE_POLICY,
    E,
    K,
    P3Verdict,
)
from .oracle import require, tensor
from .serialization import fingerprint


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


def token_fingerprint(rows):
    first = rows[0]
    model = {k: first[k] for k in MODEL_FIELDS}
    for key in ("table_fingerprint", "bias_fingerprint"):
        model[key] = bytes.fromhex(model[key])
    return fingerprint([model, [{k: row[k] for k in SLOT_FIELDS} for row in rows]])


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
):
    """Receive case + PASS operator payloads; never compute routing in the assembler."""
    validate_case(case)
    require(
        score_result.verdict == route_result.verdict == P3Verdict.PASS,
        P3Verdict.INCOMPLETE_ARTIFACT,
        "assembler requires completed forward operators",
    )
    placement = list(range(E)) if placement is None else list(placement)
    require(
        len(placement) == E and sorted(placement) == list(range(E)),
        P3Verdict.INVALID_PLACEMENT_MAP,
        "placement must be a versioned bijection",
    )
    require(
        bool(placement_version),
        P3Verdict.PLACEMENT_MAP_VERSION_MISMATCH,
        "placement version required",
    )
    s = score_result.payload["s"]
    ids, weights = route_result.payload["ids"], route_result.payload["weights"]
    tensor(s, "float32", (len(case.z), E))
    tensor(ids, "int32", (len(case.z), K))
    tensor(weights, "float32", (len(case.z), K))
    active = case.row_active
    require(
        ((ids[active] >= 0) & (ids[active] < E)).all(),
        P3Verdict.INVALID_DISCRETE_PLAN,
        "invalid logical expert",
    )
    require(
        np.isfinite(s[active]).all() and np.isfinite(weights[active]).all(),
        P3Verdict.NON_FINITE,
        "nonfinite active handoff",
    )
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
                "envelope_schema_version": ENVELOPE_SCHEMA,
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
                "kernel": "recorded-cpu.oracle.v1",
                "build": "p3-start-kit.v1",
                "device": "cpu",
                "stream": "synchronous",
                "event": "handoff",
            }
            envelopes.append({k: envelope[k] for k in ENVELOPE_FIELDS})
        semantic = token_fingerprint(token_rows) if valid else "00" * 32
        if valid:
            per_token[str(case.global_token_id[row])] = semantic
        rows.extend({**r, "route_semantic_fingerprint": semantic} for r in token_rows)
    per_token = dict(sorted(per_token.items(), key=lambda item: int(item[0])))
    semantic_case = fingerprint([(int(k), v) for k, v in per_token.items()])
    artifact_hash = fingerprint({"case_id": case.case_id, "core": rows, "envelope": envelopes})
    for env in envelopes:
        env["route_artifact_fingerprint"] = artifact_hash
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
    return {"RoutePlan": plan, "CombinePlanSeed": combine_seed(plan)}


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
    return {
        "schema_version": SEED_SCHEMA,
        "tokens": [tokens[k] for k in sorted(tokens, key=int)],
        "route_semantic_fingerprint": plan["route_semantic_fingerprint"],
    }
