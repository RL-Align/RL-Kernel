"""T01 compatibility stub. T09 extends this module with L1–L3b attribution."""

from functools import wraps

import numpy as np

from .assembler import combine_seed, token_fingerprint
from .contract import (
    CAPACITY_POLICY,
    CORE_FIELDS,
    CORE_SCHEMA,
    ENVELOPE_FIELDS,
    ENVELOPE_SCHEMA,
    EVENT_MANIFEST,
    MILES_ANCHOR,
    MODEL_FIELDS,
    OP_ABI,
    OPERATORS,
    PROVENANCE_SCHEMA,
    RECORDING_SCHEMA,
    REDUCTION_TREE,
    ROUND_POLICIES,
    SAVED_ROUTE_SCHEMA,
    SAVED_SCORE_SCHEMA,
    TIE_POLICY,
    E,
    K,
    P3Error,
    P3OpCtxHost,
    P3Verdict,
    SavedRouteSealedV1,
    SavedScoreSealedV1,
)
from .oracle import require, tensor
from .serialization import canonical, fingerprint


def checked_structure(fn):
    @wraps(fn)
    def checked(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except P3Error:
            raise
        except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as exc:
            raise P3Error(P3Verdict.SCHEMA_MISMATCH, f"malformed {fn.__name__} input") from exc

    return checked


def check_anchor(anchor=None):
    anchor = MILES_ANCHOR if anchor is None else anchor
    require(
        set(anchor) == set(MILES_ANCHOR)
        and all(isinstance(v, str) and v and "PLACEHOLDER" not in v for v in anchor.values()),
        P3Verdict.MISSING_PROVENANCE,
        "Miles anchor five fields not recorded",
    )


@checked_structure
def check_compatibility(producer):
    require(
        producer.get("producer_schema") == RECORDING_SCHEMA,
        P3Verdict.SCHEMA_MISMATCH,
        "unknown producer schema",
    )
    require(
        producer.get("operator_abi") == OP_ABI, P3Verdict.SCHEMA_MISMATCH, "operator ABI mismatch"
    )
    require(
        producer.get("producer_verdict") is not None,
        P3Verdict.UPSTREAM_VERDICT_MISSING,
        "producer verdict absent",
    )
    require(
        producer["producer_verdict"] == "PASS",
        P3Verdict.UPSTREAM_EVIDENCE_MISSING,
        "producer did not pass",
    )
    require(
        producer.get("upstream_contract_id") == "p3-synthetic-boundary.v1",
        P3Verdict.UPSTREAM_CONTRACT_MISMATCH,
        "T01 only knows its explicit synthetic boundary",
    )
    require(
        is_digest(producer.get("fixture_checksum")),
        P3Verdict.MISSING_PROVENANCE,
        "fixture checksum absent",
    )


@checked_structure
def validate_plan(bundle):
    plan = bundle["RoutePlan"]
    rows, envelopes = plan["core"], plan["envelope"]
    placement = plan["placement_map"]
    mapping = placement["logical_to_physical"]
    require(
        len(mapping) == E and sorted(mapping) == list(range(E)),
        P3Verdict.INVALID_PLACEMENT_MAP,
        "archived placement not bijective",
    )
    identity = plan["identity"]
    expected_tokens = dict(identity["tokens"])
    require(
        len(rows) == len(envelopes) and len(rows) % 6 == 0,
        P3Verdict.INCOMPLETE_ARTIFACT,
        "six slots and matching envelopes required",
    )
    for row in rows:
        require(
            tuple(row) == CORE_FIELDS and row["core_schema_version"] == CORE_SCHEMA,
            P3Verdict.SCHEMA_MISMATCH,
            "Core field order/version mismatch",
        )
    for env in envelopes:
        require(
            tuple(env) == ENVELOPE_FIELDS + ("route_artifact_fingerprint",)
            and env["envelope_schema_version"] == ENVELOPE_SCHEMA,
            P3Verdict.SCHEMA_MISMATCH,
            "Envelope field order/version mismatch",
        )
    tokens = {}
    for offset in range(0, len(rows), 6):
        token_rows = rows[offset : offset + 6]
        token = token_rows[0]["global_token_id"]
        require(
            [r["topk_index"] for r in token_rows] == list(range(6)),
            P3Verdict.TOPK_ORDER_MISMATCH,
            "duplicate/reordered slot",
        )
        require(
            all(r["global_token_id"] == token for r in token_rows),
            P3Verdict.IDENTITY_DRIFT,
            "slot token identity changed",
        )
        if token < 0:
            require(
                all(
                    r["global_token_id"] == r["input_token_id"] == r["logical_expert_id"] == -1
                    and not r["valid"]
                    and r["invalid_reason"] == "padding"
                    and all(
                        canonical(r[k]) == canonical(np.float32(0))
                        for k in ("route_weight", "weight_score", "selection_score")
                    )
                    for r in token_rows
                ),
                P3Verdict.INVALID_DISCRETE_PLAN,
                "noncanonical padding",
            )
            continue
        require(
            token in expected_tokens
            and all(
                r["checkpoint_id"] == identity["checkpoint_id"]
                and r["weight_id"] == identity["weight_id"]
                and r["weight_fingerprint"] == identity["weight_fingerprint"]
                and r["absolute_layer"] == identity["layer"]
                and r["router_mode"] == identity["mode"]
                and r["input_token_id"] == expected_tokens[token]
                for r in token_rows
            ),
            P3Verdict.IDENTITY_DRIFT,
            "Core disagrees with case/model identity",
        )
        require(
            all(r["tie_break_policy"] == TIE_POLICY for r in token_rows),
            P3Verdict.TIE_BREAK_POLICY_MISMATCH,
            "unsupported tie policy",
        )
        require(
            all(
                r["logit_round_point"] == identity["round_policy"]
                and r["logit_round_point"] in ROUND_POLICIES
                for r in token_rows
            ),
            P3Verdict.LOGIT_ROUND_POINT_MISMATCH,
            "round policy drift",
        )
        require(
            all(
                canonical({k: r[k] for k in MODEL_FIELDS})
                == canonical({k: token_rows[0][k] for k in MODEL_FIELDS})
                for r in token_rows
            ),
            P3Verdict.IDENTITY_DRIFT,
            "model/policy identity must agree across all six slots",
        )
        require(
            identity["layer"] >= 0
            and identity["mode"] == ("hash" if identity["layer"] < 3 else "learned")
            and all(
                r["capacity_policy"] == r["overflow_policy"] == CAPACITY_POLICY
                and r["valid"]
                and r["invalid_reason"] == "none"
                and 0 <= r["logical_expert_id"] < E
                and r["weight_source"] == "pre_bias_score"
                and r["selection_source"]
                == ("tid2eid.table_slot" if identity["mode"] == "hash" else "post_bias_score")
                and r["table_present"] is (identity["mode"] == "hash")
                and r["bias_present"] is (identity["mode"] == "learned")
                and r["table_fingerprint"] == (identity["table"] or "00" * 32)
                and r["bias_fingerprint"] == (identity["bias"] or "00" * 32)
                and r["capacity"] == -1
                and np.isfinite([r["route_weight"], r["weight_score"], r["selection_score"]]).all()
                for r in token_rows
            ),
            P3Verdict.INVALID_DISCRETE_PLAN,
            "invalid active dropless Core",
        )
        require(
            str(token) not in tokens, P3Verdict.AMBIGUOUS_GLOBAL_TOKEN_MAPPING, "duplicate token"
        )
        semantic = token_fingerprint(token_rows)
        require(
            all(r["route_semantic_fingerprint"] == semantic for r in token_rows),
            P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH,
            "per-token semantic corrupt",
        )
        tokens[str(token)] = semantic
    tokens = dict(sorted(tokens.items(), key=lambda item: int(item[0])))
    require(
        set(map(int, tokens)) == set(expected_tokens),
        P3Verdict.INCOMPLETE_ARTIFACT,
        "active token set incomplete",
    )
    for row, (core, env) in enumerate(zip(rows, envelopes, strict=True)):
        require(
            env["placement_map_version"] == placement["version"],
            P3Verdict.PLACEMENT_MAP_VERSION_MISMATCH,
            "envelope placement version mismatch",
        )
        expert = core["logical_expert_id"]
        require(
            env["source_row"] == row // 6
            and env["physical_expert_id"] == (mapping[expert] if expert >= 0 else -1),
            P3Verdict.INVALID_PLACEMENT_MAP,
            "logical/physical mapping mismatch",
        )
    require(
        tokens == plan["per_token_fingerprints"]
        and fingerprint([(int(k), v) for k, v in tokens.items()])
        == plan["route_semantic_fingerprint"],
        P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH,
        "case/per-token map mismatch",
    )
    envelope_body = [{k: env[k] for k in ENVELOPE_FIELDS} for env in envelopes]
    expected = fingerprint(
        {"case_id": plan["identity"]["case_id"], "core": rows, "envelope": envelope_body}
    )
    require(
        expected == plan["route_artifact_fingerprint"]
        and all(e["route_artifact_fingerprint"] == expected for e in envelopes),
        P3Verdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH,
        "artifact hash corrupt",
    )
    require(
        canonical(bundle["CombinePlanSeed"]) == canonical(combine_seed(plan)),
        P3Verdict.INVALID_DISCRETE_PLAN,
        "CombinePlanSeed cannot be rebuilt from Core",
    )


def compare_plans(left, right, *, same_config=False):
    validate_plan(left)
    validate_plan(right)
    a, b = left["RoutePlan"], right["RoutePlan"]
    require(
        canonical(a["identity"]) == canonical(b["identity"]),
        P3Verdict.IDENTITY_DRIFT,
        "case/model/token identity first",
    )
    require(
        a["per_token_fingerprints"] == b["per_token_fingerprints"],
        P3Verdict.ROUTE_SEMANTIC_FINGERPRINT_MISMATCH,
        "per-token semantic mismatch",
    )
    if same_config:
        require(
            a["route_artifact_fingerprint"] == b["route_artifact_fingerprint"],
            P3Verdict.ROUTE_ARTIFACT_FINGERPRINT_MISMATCH,
            "repeat artifact mismatch",
        )
    return P3Verdict.CASE_PASS


@checked_structure
def validate_events(events):
    require(
        len(events) == len(EVENT_MANIFEST),
        P3Verdict.MISSING_BOUNDARY_TRACE,
        "forward boundary sequence incomplete",
    )
    for expected, actual in zip(EVENT_MANIFEST, events, strict=False):
        require(
            all(actual.get(k) == v for k, v in expected.items()),
            P3Verdict.INVALID_DISCRETE_PLAN,
            "event order/branch changed",
        )
        check_compatibility(actual)
        layer = actual.get("absolute_layer", -1)
        require(
            layer >= 0 and actual.get("branch") == ("hash" if layer < 3 else "learned"),
            P3Verdict.INVALID_DISCRETE_PLAN,
            "boundary Hash/Learned XOR mismatch",
        )


@checked_structure
def check_actual_provenance(provenance):
    require(
        provenance.get("requested_backend")
        and provenance.get("actual_backend")
        and "fast_math" in provenance
        and "fallback_reason" in provenance,
        P3Verdict.MISSING_PROVENANCE,
        "actual readback missing",
    )
    require(
        provenance["requested_backend"] == provenance["actual_backend"]
        and provenance["fast_math"] is False
        and not provenance["fallback_reason"],
        P3Verdict.SILENT_FALLBACK,
        "backend/fast-math/fallback changed",
    )
    common = (
        "schema_version",
        "backend_profile",
        "kernel",
        "source_fingerprint",
        "build_flags",
        "target_arch",
        "device",
        "stream",
        "event",
    )
    require(
        all(k in provenance and provenance[k] is not None and provenance[k] != "" for k in common),
        P3Verdict.MISSING_PROVENANCE,
        "kernel/build/target/execution metadata missing",
    )
    require(
        provenance["schema_version"] == PROVENANCE_SCHEMA,
        P3Verdict.SCHEMA_MISMATCH,
        "old provenance schema",
    )
    require(
        is_digest(provenance["source_fingerprint"]), P3Verdict.MISSING_PROVENANCE, "source hash"
    )
    if provenance["actual_backend"] == "recorded-cpu":
        from . import bitmath

        require(
            provenance["backend_profile"] == "synthetic-cpu.v1"
            and provenance["kernel"] == "recorded-cpu.oracle.v1"
            and provenance["target_arch"] == "host"
            and provenance["device"] == "cpu"
            and provenance.get("certifies_cuda") is False,
            P3Verdict.INVALID_PROFILE,
            "CPU reference is not hardware evidence",
        )
        require(
            provenance["source_fingerprint"] == bitmath.source_fingerprint()
            and provenance["build_flags"] == list(bitmath.HOST_FLAGS)
            and provenance.get("math_path") == "p3-bitmath.v1"
            and provenance.get("reduction_tree_id") == REDUCTION_TREE
            and provenance.get("tie_policy") == TIE_POLICY,
            P3Verdict.SILENT_FALLBACK,
            "unapproved CPU arithmetic path",
        )
        require(
            isinstance(provenance.get("input_tensors"), list)
            and is_digest(provenance.get("inputs_fingerprint"))
            and provenance.get("round_policy")
            and provenance.get("launch") == {"block": None, "warps": None, "stages": None},
            P3Verdict.MISSING_PROVENANCE,
            "CPU tensor/rounding/launch metadata",
        )
    elif provenance["actual_backend"] == "cuda":
        from .provider import readback_verdict
        from .stable_topk6 import cuda_source_fingerprint

        required = (
            "dtype",
            "shape",
            "stride",
            "block",
            "grid",
            "topk_abi",
            "tie_policy",
            "invocation_id",
            "status_readback",
            "run_id",
            "engine_id",
            "rank",
            "attempt_id",
            "gpu",
            "torch",
            "cuda",
            "actual_binary_arch",
            "capabilities",
        )
        require(
            all(k in provenance for k in required),
            P3Verdict.MISSING_PROVENANCE,
            "CUDA readback fields",
        )
        require(
            provenance["backend_profile"] == "cuda.sm90.h100.v1"
            and provenance["kernel"] == "p3_topk_kernel"
            and provenance["target_arch"] == "sm90"
            and provenance["actual_binary_arch"] == 90
            and "H100" in provenance["gpu"]
            and provenance["source_fingerprint"] == cuda_source_fingerprint()
            and provenance["build_flags"]
            == "sm_90;ftz=false;fmad=false;prec-div=true;prec-sqrt=true;fast-math=false"
            and provenance["topk_abi"] == "stable_topk6_device_abi.v1"
            and provenance["tie_policy"] == TIE_POLICY,
            P3Verdict.SILENT_FALLBACK,
            "unapproved CUDA binary/profile",
        )
        require(
            provenance["dtype"] == "torch.float32"
            and len(provenance["shape"]) == 2
            and provenance["shape"][1] == E
            and provenance["stride"] == [E, 1]
            and provenance["block"] in (1, 32, 64, 128)
            and provenance["grid"]
            == (provenance["shape"][0] + provenance["block"] - 1) // provenance["block"]
            and provenance["capabilities"]
            == {"int32_atomicMin": True, "uint64_atomicExch": True, "device_fence": True}
            and provenance["event"] == "same_stream_D2H_then_synchronize",
            P3Verdict.MISSING_PROVENANCE,
            "CUDA tensor/launch/capability drift",
        )
        require(
            provenance["invocation_id"] >> 32 == provenance["attempt_id"]
            and provenance["run_id"]
            and provenance["engine_id"]
            and readback_verdict(
                bytes.fromhex(provenance["status_readback"]), provenance["invocation_id"]
            )
            == P3Verdict.PASS,
            P3Verdict.CORRUPT_ARTIFACT,
            "CUDA status/echo/attempt evidence",
        )
    else:
        require(False, P3Verdict.UNSUPPORTED_CAPABILITY, "unsupported actual backend")


def is_digest(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


@checked_structure
def validate_operator_inputs(operator, inputs, active, *, raw_saved=False):
    require(operator in OPERATORS, P3Verdict.SCHEMA_MISMATCH, "unknown operator")
    require(
        isinstance(active, np.ndarray) and active.ndim == 1, P3Verdict.SCHEMA_MISMATCH, "row mask"
    )
    tensor(active, "bool", (len(active),))
    T = len(active)
    expected_count = (
        3 if operator == "hash_route_fwd" else (1 if operator == "stable_topk6_fwd" else 2)
    )
    require(len(inputs) == expected_count, P3Verdict.SCHEMA_MISMATCH, "input arity")
    if operator == "hash_route_fwd":
        tensor(inputs[0], "int64", (T,))
        tensor(inputs[1], "float32", (T, E))
        table = inputs[2]
        require(
            isinstance(table, np.ndarray) and table.ndim == 2,
            P3Verdict.SCHEMA_MISMATCH,
            "table rank",
        )
        tensor(table, "int32", (len(table), K))
    else:
        tensor(
            inputs[0],
            "float32",
            (T, K if operator in ("hash_route_bwd", "learned_route_bwd") else E),
        )
        if operator == "learned_route_fwd":
            tensor(inputs[1], "float32", (E,))
        elif operator == "router_sqrt_softplus_fwd":
            require(isinstance(inputs[1], str), P3Verdict.SCHEMA_MISMATCH, "round policy type")
        elif operator.endswith("_bwd") and not raw_saved:
            cls = SavedScoreSealedV1 if operator.startswith("router_sqrt") else SavedRouteSealedV1
            require(type(inputs[-1]) is cls, P3Verdict.SCHEMA_MISMATCH, "sealed saved required")


@checked_structure
def validate_recording(recording, *, check_diagnostics=True):
    """Validate transport evidence before a provider exposes any recorded payload."""
    require(
        recording.get("schema_version") == RECORDING_SCHEMA,
        P3Verdict.SCHEMA_MISMATCH,
        "recording schema",
    )
    active = recording["row_active"]
    tensor(active, "bool", (len(active),))
    check_actual_provenance(recording["provenance"])
    if not active.any():
        require(
            recording["producer_verdict"] == "ZERO_ACTIVE_TOKENS"
            and recording["operators"] == {}
            and recording["saved"] == {}
            and recording["bundle"] is None
            and recording["events"] == [],
            P3Verdict.CORRUPT_ARTIFACT,
            "zero-active recording must not contain payload/saved",
        )
        return
    require(
        recording.get("producer_verdict") == "PASS",
        P3Verdict.UPSTREAM_EVIDENCE_MISSING,
        "producer failed",
    )
    validate_plan(recording["bundle"])
    require(
        canonical(recording["identity"]) == canonical(recording["bundle"]["RoutePlan"]["identity"]),
        P3Verdict.IDENTITY_DRIFT,
        "recording/plan identity",
    )
    validate_events(recording["events"])
    require(
        all(
            e["fixture_checksum"] == recording["source_checksum"]
            and e["absolute_layer"] == recording["identity"]["layer"]
            and e["branch"] == recording["identity"]["mode"]
            for e in recording["events"]
        ),
        P3Verdict.IDENTITY_DRIFT,
        "boundary source/branch identity",
    )
    branch = recording["identity"]["mode"]
    expected_ops = {
        "router_sqrt_softplus_fwd",
        "router_sqrt_softplus_bwd",
        "stable_topk6_fwd",
        branch + "_route_fwd",
        branch + "_route_bwd",
    }
    require(
        set(recording["operators"]) == expected_ops, P3Verdict.SCHEMA_MISMATCH, "operator coverage"
    )
    from .provenance import tensor_metadata

    for name, op in recording["operators"].items():
        require(op.get("verdict") == "PASS", P3Verdict.UPSTREAM_EVIDENCE_MISSING, "operator failed")
        require(
            isinstance(op.get("payload"), dict),
            P3Verdict.SCHEMA_MISMATCH,
            "missing operator payload",
        )
        validate_operator_inputs(name, op["inputs"], active, raw_saved=True)
        require(
            fingerprint(op["inputs"]) == op["input_checksum"],
            P3Verdict.CORRUPT_ARTIFACT,
            "operator input checksum",
        )
        check_actual_provenance(op["provenance"])
        require(
            op["provenance"]["inputs_fingerprint"] == op["input_checksum"]
            and op["provenance"]["input_tensors"] == [tensor_metadata(v) for v in op["inputs"]]
            and op["provenance"]["round_policy"] == recording["identity"]["round_policy"],
            P3Verdict.IDENTITY_DRIFT,
            "operator input/round provenance",
        )
        output_shapes = {
            "router_sqrt_softplus_fwd": {"s": (len(active), E)},
            "router_sqrt_softplus_bwd": {"dz": (len(active), E)},
            "stable_topk6_fwd": {"ids": (len(active), K)},
            "hash_route_fwd": {"ids": (len(active), K), "weights": (len(active), K)},
            "learned_route_fwd": {"ids": (len(active), K), "weights": (len(active), K)},
            "hash_route_bwd": {"ds": (len(active), E)},
            "learned_route_bwd": {"ds": (len(active), E)},
        }
        for field, shape in output_shapes[name].items():
            tensor(op["payload"][field], "int32" if field == "ids" else "float32", shape)
            require(
                np.isfinite(op["payload"][field][active]).all(),
                P3Verdict.NON_FINITE,
                "active output nonfinite",
            )
    from .provider import validate_saved

    ctx = P3OpCtxHost(
        "recorded",
        "recorded",
        0,
        active,
        recording["bundle"]["RoutePlan"]["route_artifact_fingerprint"],
    )
    for key, cls, version in (
        ("score", SavedScoreSealedV1, SAVED_SCORE_SCHEMA),
        ("route", SavedRouteSealedV1, SAVED_ROUTE_SCHEMA),
    ):
        validate_saved(cls(**recording["saved"][key]), version, ctx)
    paired = recording.get("torch_paired", {})
    require(
        paired.get("status") == "RECORDED_DIAGNOSTIC"
        and paired.get("strict_verdict_unchanged") is True
        and "trace" in paired
        and paired.get("checksum") == fingerprint(paired["trace"]),
        P3Verdict.MISSING_PROVENANCE,
        "paired Torch evidence absent or corrupt",
    )
    from .diagnostics import paired_diagnostics

    score = recording["operators"]["router_sqrt_softplus_fwd"]["payload"]["s"]
    route = recording["operators"][branch + "_route_fwd"]["payload"]
    q = recording["operators"]["stable_topk6_fwd"]["inputs"][0]
    diagnostic = paired_diagnostics(paired["trace"], score, route, q, active)
    require(
        all(k in paired for k in diagnostic),
        P3Verdict.MISSING_PROVENANCE,
        "paired diagnostic fields",
    )
    if check_diagnostics:
        require(
            canonical(diagnostic) == canonical({k: paired[k] for k in diagnostic}),
            P3Verdict.CORRUPT_ARTIFACT,
            "diagnostic metadata disagrees with trace",
        )


def compare_recordings(left, right):
    """Small T01 comparator hook. T09 extends its ladder and distributed attribution."""
    from .contract import P3Error

    try:
        require(
            canonical(left["identity"]) == canonical(right["identity"]),
            P3Verdict.IDENTITY_DRIFT,
            "identity before numerical comparison",
        )
        for recording in (left, right):
            require(
                recording["producer_verdict"] == "PASS",
                P3Verdict.UPSTREAM_EVIDENCE_MISSING,
                "non-PASS producer cannot be compared",
            )
            check_actual_provenance(recording["provenance"])
            validate_events(recording["events"])
            validate_recording(recording, check_diagnostics=False)
        branch = left["identity"]["mode"]
        checks = [
            (
                branch + "_route_fwd",
                ("ids",),
                P3Verdict.TOPK_ORDER_MISMATCH,
                "selection",
                "T03" if branch == "hash" else "T01",
                42 if branch == "hash" else 43,
            ),
            (
                "router_sqrt_softplus_fwd",
                ("s",),
                P3Verdict.SCORE_BYTES_MISMATCH,
                "score",
                "T02",
                41,
            ),
            (
                branch + "_route_fwd",
                ("weights",),
                P3Verdict.ROUTE_WEIGHT_BYTES_MISMATCH,
                "weight",
                "T03" if branch == "hash" else "T04",
                42 if branch == "hash" else 44,
            ),
            (
                branch + "_route_bwd",
                ("ds",),
                P3Verdict.GRADIENT_BYTES_MISMATCH,
                "backward",
                "T06",
                42 if branch == "hash" else 44,
            ),
            (
                "router_sqrt_softplus_bwd",
                ("dz",),
                P3Verdict.GRADIENT_BYTES_MISMATCH,
                "backward",
                "T02",
                41,
            ),
        ]
        for operator, keys, verdict, site, owner, issue in checks:
            for key in keys:
                a = left["operators"][operator]["payload"][key]
                b = right["operators"][operator]["payload"][key]
                require(
                    a.dtype == b.dtype and a.shape == b.shape,
                    P3Verdict.SCHEMA_MISMATCH,
                    "candidate payload schema mismatch",
                )
                if a.dtype == np.float32:
                    mismatch = a.view(np.uint32) != b.view(np.uint32)
                else:
                    mismatch = a != b
                mismatch[~left["row_active"]] = False
                positions = np.argwhere(mismatch)
                if positions.size:
                    row, column = map(int, positions[0])
                    token = left["bundle"]["RoutePlan"]["core"][row * 6]["global_token_id"]
                    return {
                        "verdict": verdict.name,
                        "first_mismatch": {
                            "absolute_layer": left["identity"]["layer"],
                            "site": site,
                            "pass": "backward" if site == "backward" else "forward",
                            "event_index": {"score": 0, "selection": 1, "weight": 2, "backward": 4}[
                                site
                            ],
                            "global_token_id": token,
                            "rank": 0,
                            "source_row": row,
                            "column": column,
                            "operator": operator,
                            "field": key,
                            "owner": owner,
                            "issue": issue,
                            "phase": "WS1",
                            "boundary": "p3-synthetic-boundary.v1",
                            "artifact": left["bundle"]["RoutePlan"]["route_artifact_fingerprint"],
                        },
                    }
        compare_plans(left["bundle"], right["bundle"])
        for recording in (left, right):
            validate_recording(recording)
        return {"verdict": "CASE_PASS", "first_mismatch": None}
    except P3Error as exc:
        return {"verdict": exc.verdict.name, "first_mismatch": {"detail": str(exc)}}
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        return {"verdict": "SCHEMA_MISMATCH", "first_mismatch": {"detail": type(exc).__name__}}
