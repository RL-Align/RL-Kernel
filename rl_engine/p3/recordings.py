"""Input-bound recordings covering the seven interfaces across Hash/Learned cases."""

import numpy as np

from . import oracle
from .assembler import assemble, validate_case
from .checker import validate_events, validate_plan
from .contract import (
    EVENT_MANIFEST,
    OP_ABI,
    RECORDING_SCHEMA,
    SAVED_ROUTE_SCHEMA,
    SAVED_SCORE_SCHEMA,
    P3OpCtxHost,
    P3OpResult,
    P3Verdict,
)
from .diagnostics import paired_diagnostics
from .provenance import cpu_provenance
from .provider import seal_saved
from .reference import ReferenceProvider
from .serialization import fingerprint
from .torch_reference import forward as torch_forward


def record_case(case, *, engine_id="recorded", run_id="p3-synthetic", attempt_id=1):
    validate_case(case)
    provenance = cpu_provenance((case.z, case.table, case.bias), round_policy=case.round_policy)
    if not case.row_active.any():
        return {
            "schema_version": RECORDING_SCHEMA,
            "case_id": case.case_id,
            "source_checksum": case.checksum(),
            "identity": case.identity(),
            "producer_verdict": "ZERO_ACTIVE_TOKENS",
            "row_active": case.row_active.copy(),
            "operators": {},
            "saved": {},
            "bundle": None,
            "torch_paired": {"status": "ZERO_ACTIVE_TOKENS"},
            "events": [],
            "provenance": provenance,
        }
    # All rows are passed to the oracle. Provider owns inactive sanitization;
    # the assembler alone canonicalizes archived route records.
    active = case.row_active
    z = case.z.copy()
    z[~active] = 0
    token_ids = case.input_token_id.copy()
    token_ids[~active] = 0
    score = oracle.router_sqrt_softplus_fwd(z, case.round_policy)
    s = score["s"]
    route = (
        oracle.hash_route_fwd(token_ids, s, case.table)
        if case.router_mode == "hash"
        else oracle.learned_route_fwd(s, case.bias)
    )
    score_result = P3OpResult(P3Verdict.PASS, score, provenance)
    route_result = P3OpResult(P3Verdict.PASS, route, provenance)
    bundle = assemble(
        case, score_result, route_result, run_id=run_id, engine_id=engine_id, attempt_id=attempt_id
    )
    ctx = P3OpCtxHost(
        run_id,
        engine_id,
        0,
        active.copy(),
        bundle["RoutePlan"]["route_artifact_fingerprint"],
        invocation_id=attempt_id << 32 | 1,
    )
    score_saved = seal_saved(SAVED_SCORE_SCHEMA, score["saved_score"], ctx, score_result)
    ctx.invocation_id += 1
    route_saved = seal_saved(SAVED_ROUTE_SCHEMA, route["saved_route"], ctx, route_result)
    provider = ReferenceProvider()
    rb = provider.learned_route_bwd(ctx, case.dweights, route_saved)
    sb = provider.router_sqrt_softplus_bwd(ctx, rb.payload["ds"], score_saved)
    oracle.require(
        rb.verdict == sb.verdict == P3Verdict.PASS,
        P3Verdict.INCOMPLETE_ARTIFACT,
        "recorded backward incomplete",
    )
    q = route.get("q", s.copy())
    inputs_outputs = {
        "router_sqrt_softplus_fwd": ([z, case.round_policy], score),
        "router_sqrt_softplus_bwd": ([rb.payload["ds"], score_saved.payload], sb.payload),
        "stable_topk6_fwd": ([q], {"ids": oracle.stable_topk6(q)}),
    }
    inputs_outputs[case.router_mode + "_route_fwd"] = (
        [token_ids, s, case.table] if case.router_mode == "hash" else [s, case.bias],
        route,
    )
    inputs_outputs[case.router_mode + "_route_bwd"] = (
        [case.dweights, route_saved.payload],
        rb.payload,
    )
    operators = {
        name: {
            "inputs": inputs,
            "input_checksum": fingerprint(inputs),
            "payload": payload,
            "verdict": "PASS",
            "provenance": cpu_provenance(inputs, round_policy=case.round_policy),
        }
        for name, (inputs, payload) in inputs_outputs.items()
    }
    paired = torch_forward(
        z,
        case.round_policy,
        token_ids=token_ids if case.router_mode == "hash" else None,
        table=case.table if case.router_mode == "hash" else None,
        bias=case.bias if case.router_mode == "learned" else None,
    )
    paired_trace = {k: v.detach().numpy().copy() for k, v in paired.items()}
    diagnostic = {
        **paired_diagnostics(paired_trace, s, route, q, active),
        "status": "RECORDED_DIAGNOSTIC",
        "backend": "torch-cpu",
        "max_abs_score_error": float(np.max(np.abs(paired_trace["s"][active] - s[active]))),
        "ids_match": bool(np.array_equal(paired_trace["ids"][active], route["ids"][active])),
        "trace": paired_trace,
        "checksum": fingerprint(paired_trace),
        "strict_verdict_unchanged": True,
    }
    events = [
        {
            **event,
            "operator_abi": OP_ABI,
            "producer_verdict": "PASS",
            "upstream_contract_id": "p3-synthetic-boundary.v1",
            "producer_schema": RECORDING_SCHEMA,
            "branch": case.router_mode,
            "absolute_layer": case.absolute_layer,
            "fixture_checksum": case.checksum(),
        }
        for event in EVENT_MANIFEST
    ]
    validate_plan(bundle)
    validate_events(events)
    return {
        "schema_version": RECORDING_SCHEMA,
        "case_id": case.case_id,
        "source_checksum": case.checksum(),
        "identity": case.identity(),
        "producer_verdict": "PASS",
        "row_active": case.row_active.copy(),
        "operators": operators,
        "saved": {
            "score": {"header": score_saved.header, "payload": score_saved.payload},
            "route": {"header": route_saved.header, "payload": route_saved.payload},
        },
        "bundle": bundle,
        "torch_paired": diagnostic,
        "events": events,
        "provenance": provenance,
    }
