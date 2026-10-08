"""Single machine-readable source of the p3-router-task-contract.v23 ABI."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol

import numpy as np

CONTRACT_VERSION = "p3-router-task-contract.v23"
OP_ABI = "p3-op-abi.v4"
CORE_SCHEMA = "RoutePlanCore.v1"
ENVELOPE_SCHEMA = "RoutePlanEnvelope.v1"
SAVED_SCORE_SCHEMA = "SavedScoreSealedV1"
SAVED_ROUTE_SCHEMA = "SavedRouteSealedV1"
SEED_SCHEMA = "CombinePlanSeed.v1"
ASSEMBLER_ABI = "assembler_abi.v1"
TOPK_ABI = "stable_topk6_device_abi.v1"
BITMATH_VERSION = "p3-bitmath.v1"
ARTIFACT_SCHEMA = "p3-start-kit-artifact.v2"
RECORDING_SCHEMA = "p3-recording.v2"
PROVENANCE_SCHEMA = "p3-provenance.v2"
FIXTURE_VERSION = "p3-fixtures.v2"
GOLDEN_SCHEMA = "p3-golden.v2"
H, E, K = 4096, 256, 6
EPSILON, SCALE = np.float32(1e-20), np.float32(1.5)
UNSET = 0x7FFFFFFF
TIE_POLICY = "score_desc_logical_expert_id_asc.v1"
REDUCTION_TREE = "((a0+a1)+(a2+a3))+(a4+a5).fp32.v1"
ROUND_POLICIES = ("fp32_direct", "bf16_round_then_widen")
CAPACITY_POLICY = "dropless_v1"


class P3Verdict(IntEnum):
    PASS = 0
    NON_FINITE = 1
    HASH_TABLE_INDEX_OUT_OF_RANGE = 2
    IDENTITY_DRIFT = 10
    SCHEMA_MISMATCH = 11
    CORRUPT_ARTIFACT = 12
    INCOMPLETE_ARTIFACT = 13
    LOGIT_ROUND_POINT_MISMATCH = 14
    HASH_TABLE_MISMATCH = 15
    UNSUPPORTED_CAPABILITY = 16
    ZERO_ACTIVE_TOKENS = 17
    UPSTREAM_NON_FINITE = 18
    GATE_SHARDING_MISMATCH = 19
    MISSING_RANK = 20
    PRE_UPDATE_WEIGHT_DRIFT = 21
    STALE_RUN_METADATA = 22
    CASE_PASS = 50
    ROUTE_WEIGHT_BYTES_MISMATCH = 51
    SCORE_BYTES_MISMATCH = 52
    GRADIENT_BYTES_MISMATCH = 53
    BYTE_MISMATCH = 54
    TOPK_ORDER_MISMATCH = 55
    TIE_BREAK_POLICY_MISMATCH = 56
    INVALID_DISCRETE_PLAN = 57
    INVALID_PROFILE = 58
    ROUTE_SEMANTIC_FINGERPRINT_MISMATCH = 59
    ROUTE_ARTIFACT_FINGERPRINT_MISMATCH = 60
    SELECTION_GRADIENT_PRESENT = 61
    FORBIDDEN_LOCAL_SHARD_TOPK = 62
    AMBIGUOUS_GLOBAL_TOKEN_MAPPING = 63
    INVALID_PLACEMENT_MAP = 64
    PLACEMENT_MAP_VERSION_MISMATCH = 65
    SILENT_FALLBACK = 66
    MISSING_PROVENANCE = 67
    MISSING_BOUNDARY_TRACE = 68
    UPSTREAM_CONTRACT_MISMATCH = 69
    UPSTREAM_VERDICT_MISSING = 70
    UPSTREAM_EVIDENCE_MISSING = 71
    NATURAL_ROUTE_MISMATCH = 72


class P3Error(ValueError):
    def __init__(self, verdict: P3Verdict, detail: str):
        super().__init__(detail)
        self.verdict = verdict


@dataclass(frozen=True)
class P3OpResult:
    verdict: P3Verdict
    payload: Any = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.verdict != P3Verdict.PASS and self.payload is not None:
            raise ValueError("non-PASS operator payload must be absent")
        if 50 <= self.verdict:
            raise ValueError("operator API cannot return runner verdicts")


@dataclass
class P3OpCtxHost:
    run_id: str
    engine_id: str
    rank: int
    row_active: np.ndarray
    route_artifact_fingerprint: str = ""
    backend_tag: str = "recorded-cpu"
    stream: Any = None
    allocator: Any = None
    invocation_id: int = 0
    status_record: Any = None
    manifest_source: str = "synthetic.v1"


@dataclass(frozen=True)
class SavedScoreSealedV1:
    header: dict
    payload: dict[str, np.ndarray]


@dataclass(frozen=True)
class SavedRouteSealedV1:
    header: dict
    payload: dict[str, np.ndarray]


class RouterProvider(Protocol):
    def router_sqrt_softplus_fwd(self, ctx, z, logit_round_point) -> P3OpResult: ...
    def router_sqrt_softplus_bwd(self, ctx, saved_ds, saved_score_sealed) -> P3OpResult: ...
    def hash_route_fwd(self, ctx, input_token_id, s, tid2eid) -> P3OpResult: ...
    def hash_route_bwd(self, ctx, dweights, saved_route_sealed) -> P3OpResult: ...
    def stable_topk6_fwd(self, ctx, q) -> P3OpResult: ...
    def learned_route_fwd(self, ctx, s, b) -> P3OpResult: ...
    def learned_route_bwd(self, ctx, dweights, saved_route_sealed) -> P3OpResult: ...


# Serialization and all downstream schemas are generated from these ordered fields.
MODEL_FIELDS = (
    "core_schema_version",
    "checkpoint_id",
    "weight_id",
    "weight_fingerprint",
    "absolute_layer",
    "router_mode",
    "global_token_id",
    "input_token_id",
    "capacity_policy",
    "overflow_policy",
    "logit_round_point",
    "tie_break_policy",
    "table_present",
    "table_fingerprint",
    "bias_present",
    "bias_fingerprint",
    "selection_source",
    "weight_source",
)
SLOT_FIELDS = (
    "topk_index",
    "logical_expert_id",
    "valid",
    "invalid_reason",
    "route_weight",
    "weight_score",
    "selection_score",
    "capacity",
)
CORE_FIELDS = MODEL_FIELDS + SLOT_FIELDS + ("route_semantic_fingerprint",)
ENVELOPE_FIELDS = (
    "envelope_schema_version",
    "run_id",
    "engine_id",
    "attempt_id",
    "source_row",
    "physical_expert_id",
    "placement_map_version",
    "rank",
    "group",
    "topology",
    "backend_profile",
    "kernel",
    "build",
    "device",
    "stream",
    "event",
)
SAVED_FIELDS = {SAVED_SCORE_SCHEMA: ("z_prime", "s"), SAVED_ROUTE_SCHEMA: ("ids", "a", "Z", "p")}
SAVED_HEADER_FIELDS = (
    "version",
    "operator_abi",
    "row_active",
    "route_artifact_fingerprint",
    "forward_invocation_id",
    "manifest_source",
    "saved_payload_checksum",
)
OPERATORS = {
    "router_sqrt_softplus_fwd": {"owner": "T02", "inputs": ["z:f32[T,256]", "round_policy"]},
    "router_sqrt_softplus_bwd": {"owner": "T02", "inputs": ["ds:f32[T,256]", SAVED_SCORE_SCHEMA]},
    "hash_route_fwd": {
        "owner": "T03",
        "inputs": ["input_token_id:i64[T]", "s:f32[T,256]", "tid2eid:i32[V,6]"],
    },
    "hash_route_bwd": {"owner": "T06", "inputs": ["dw:f32[T,6]", SAVED_ROUTE_SCHEMA]},
    "stable_topk6_fwd": {"owner": "T01", "inputs": ["q:f32[T,256]"]},
    "learned_route_fwd": {"owner": "T04", "inputs": ["s:f32[T,256]", "b:f32[256]"]},
    "learned_route_bwd": {"owner": "T06", "inputs": ["dw:f32[T,6]", SAVED_ROUTE_SCHEMA]},
}
EVENT_MANIFEST = tuple(
    {
        "site": site,
        "event_index": i,
        "predecessor": None if i == 0 else i - 1,
        "phase": "forward",
        "comparison_mode": "byte_exact",
        "tolerance_version": "strict.v1",
    }
    for i, site in enumerate(("score", "selection", "weight", "handoff"))
)
MILES_ANCHOR = dict.fromkeys(
    (
        "miles_commit",
        "router_kernel/build",
        "recorded_trace_checksum",
        "tie_break_source",
        "logit_round_point_source",
    ),
    "PLACEHOLDER",
)


def manifest():
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "anchor_pending",
        "artifact_schema": ARTIFACT_SCHEMA,
        "recording_schema": RECORDING_SCHEMA,
        "provenance_schema": PROVENANCE_SCHEMA,
        "fixture_version": FIXTURE_VERSION,
        "operator_abi": OP_ABI,
        "assembler_abi": ASSEMBLER_ABI,
        "topk_abi": TOPK_ABI,
        "bitmath_version": BITMATH_VERSION,
        "model": {"H": H, "E": E, "K": K},
        "round_policies": ROUND_POLICIES,
        "capacity_policy": CAPACITY_POLICY,
        "tie_policy": TIE_POLICY,
        "reduction_tree_id": REDUCTION_TREE,
        "schemas": {
            CORE_SCHEMA: CORE_FIELDS,
            ENVELOPE_SCHEMA: ENVELOPE_FIELDS + ("route_artifact_fingerprint",),
            **SAVED_FIELDS,
            SEED_SCHEMA: ("global_token_id", "slots", "route_semantic_fingerprint"),
        },
        "operators": OPERATORS,
        "saved_header_fields": SAVED_HEADER_FIELDS,
        "saved_payload_types": {
            SAVED_SCORE_SCHEMA: {"z_prime": "float32[T,256]", "s": "float32[T,256]"},
            SAVED_ROUTE_SCHEMA: {
                "ids": "int32[T,6]",
                "a": "float32[T,6]",
                "Z": "float32[T,1]",
                "p": "float32[T,6]",
            },
        },
        "operator_outputs": {
            "router_sqrt_softplus_fwd": ["s:float32[T,256]", "saved_score_raw"],
            "router_sqrt_softplus_bwd": ["dz:float32[T,256]"],
            "hash_route_fwd": ["ids:int32[T,6]", "weights:float32[T,6]", "saved_route_raw"],
            "hash_route_bwd": ["ds:float32[T,256]"],
            "stable_topk6_fwd": ["ids:int32[T,6]"],
            "learned_route_fwd": ["ids:int32[T,6]", "weights:float32[T,6]", "saved_route_raw"],
            "learned_route_bwd": ["ds:float32[T,256]"],
        },
        "serialization": {
            "version": "p3-canonical.v1",
            "endianness": "little",
            "integer": "tag i + int64; archive uint64 above INT64_MAX uses tag u",
            "float": "tag f + raw IEEE binary32",
            "length": "uint32",
            "semantic_model_order": MODEL_FIELDS,
            "semantic_slot_order": SLOT_FIELDS,
            "artifact_envelope_order": ENVELOPE_FIELDS,
            "irrelevant_table_bias": "present=false + 32 zero bytes",
            "padding_in_semantic": False,
            "padding_in_artifact": True,
        },
        "events": EVENT_MANIFEST,
        "verdicts": {x.name: x.value for x in P3Verdict},
        "device_status": {"UNSET": UNSET, "NON_FINITE": 1, "HASH_TABLE_INDEX_OUT_OF_RANGE": 2},
        "miles_router_anchor": MILES_ANCHOR,
        "foundation_compatibility": "UNVERIFIED",
        "scope": {
            "release": "T01_S0_DEVELOPMENT",
            "hardware_profiles": ["cuda.sm90.h100.v1"],
            "cpu_profile": "synthetic-cpu.v1",
            "capacity": "dropless-only; finite-capacity/overflow requires a separate delta",
            "foundation": "local boundary checks only; official version/owner approval pending",
            "miles": "real anchor and recorded L3b required before WS1",
        },
    }
