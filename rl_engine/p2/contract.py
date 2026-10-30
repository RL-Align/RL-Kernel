# SPDX-License-Identifier: Apache-2.0
"""Versioned, fail-closed P2 start-kit contract (no kernel dispatch)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from functools import wraps
from typing import Any

CONTRACT_VERSION = "p2-task-contract.v1"
ABI_VERSION = "foundation-attention-boundary.v1"
SCHEMA_VERSION = "p2-start-kit.v1"
ERROR_VERSION = "p2-error.v1"
REFERENCE_PROFILE = "p2.synthetic-fp32.v1"
MODES = ("training", "prefill", "eager_decode", "graph_decode")
LAYERS = ("C0", "C4", "C128")


class Status(str, Enum):
    PASS = "PASS"
    BYTE_MISMATCH = "BYTE_MISMATCH"
    STATE_BYTES_MISMATCH = "STATE_BYTES_MISMATCH"
    IDENTITY_DRIFT = "IDENTITY_DRIFT"
    INVALID_LAYER_TYPE = "INVALID_LAYER_TYPE"
    INVALID_COMPRESSION_PLAN = "INVALID_COMPRESSION_PLAN"
    INVALID_GLOBAL_POSITION = "INVALID_GLOBAL_POSITION"
    INVALID_PAGE_OR_GENERATION = "INVALID_PAGE_OR_GENERATION"
    EARLY_OR_DUPLICATE_COMMIT = "EARLY_OR_DUPLICATE_COMMIT"
    MAIN_INDEX_IDENTITY_ALIAS = "MAIN_INDEX_IDENTITY_ALIAS"
    INVALID_CANDIDATE_ORDER = "INVALID_CANDIDATE_ORDER"
    MULTIPLE_SOFTMAX_DENOMINATORS = "MULTIPLE_SOFTMAX_DENOMINATORS"
    INVALID_SINK_SEMANTICS = "INVALID_SINK_SEMANTICS"
    INVALID_TOPK_ORDER = "INVALID_TOPK_ORDER"
    AMBIGUOUS_LOGICAL_INDEX = "AMBIGUOUS_LOGICAL_INDEX"
    INVALID_ROPE_VARIANT = "INVALID_ROPE_VARIANT"
    ROUND_POINT_MISMATCH = "ROUND_POINT_MISMATCH"
    FORBIDDEN_SPLIT_REDUCTION = "FORBIDDEN_SPLIT_REDUCTION"
    FORBIDDEN_ATOMIC_REDUCTION = "FORBIDDEN_ATOMIC_REDUCTION"
    MISSING_GLOBAL_VISIBILITY = "MISSING_GLOBAL_VISIBILITY"
    DUPLICATE_LOGICAL_OWNER = "DUPLICATE_LOGICAL_OWNER"
    MISSING_RANK = "MISSING_RANK"
    MISSING_PROVENANCE = "MISSING_PROVENANCE"
    SILENT_FALLBACK = "SILENT_FALLBACK"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    NON_FINITE = "NON_FINITE"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    INCOMPLETE_ARTIFACT = "INCOMPLETE_ARTIFACT"
    CORRUPT_ARTIFACT = "CORRUPT_ARTIFACT"
    NATURAL_ROUTE_MISMATCH = "NATURAL_ROUTE_MISMATCH"


class ContractError(ValueError):
    def __init__(self, status: Status, detail: str):
        self.status = status
        super().__init__(f"{status.value}: {detail}")


def require(condition: bool, status: Status, detail: str) -> None:
    if not condition:
        raise ContractError(status, detail)


def schema_guard(function):
    """Malformed public JSON must fail with a versioned status, not a traceback."""

    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ContractError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as exc:
            raise ContractError(Status.SCHEMA_MISMATCH, function.__name__) from exc

    return checked


def integer(value: Any, name: str, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, Status.SCHEMA_MISMATCH, name)
    return value


def canonical(value: Any) -> bytes:
    """One portable JSON representation; NaN/Inf are never serializable."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


MODEL = {
    "hidden_size": 4096,
    "main_q_heads": 64,
    "main_kv_heads": 1,
    "main_dim": 512,
    "index_heads": 64,
    "index_dim": 128,
    "rope_dim": 64,
    "main_rotary_range": [448, 512],
    "index_rotary_range": [64, 128],
    "recent_window": 128,
    "topk": 512,
    "output_groups": 8,
    "heads_per_group": 8,
    "o_lora_rank": 1024,
}
COMPRESSION = {
    "C0": {"cr": 0, "coff": 0, "overlap": False, "index": False},
    "C4": {"cr": 4, "coff": 2, "overlap": True, "index": True},
    "C128": {"cr": 128, "coff": 1, "overlap": False, "index": False},
}


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    owner: str
    boundary: str
    forward: str
    backward: str
    saved: tuple[str, ...]
    comparison: str = "strict_bytes; reference_fp32_is_not_production_certification"


OPERATORS = (
    OperatorSpec(
        "indexer_scale",
        "T02",
        "M2.indexer.score",
        "u:fp32[T,64] -> w:fp32[T,64]",
        "dw -> du",
        ("global_heads", "head_dim", "scale_bytes", "association_version"),
    ),
    OperatorSpec(
        "rope_gptj_interleaved_partial",
        "T02",
        "M2.output_projection",
        "x,cos,sin,global_position,rope_dim,inverse -> y",
        "dy,saved -> dx",
        ("table_hash", "global_position", "rotary_range", "variant", "inverse"),
    ),
    OperatorSpec(
        "recent128_cache_update",
        "T03",
        "M2.state.recent",
        "KV_t,logical_position,cache -> state_commit",
        "functional_scatter_only",
        ("slot", "page", "generation", "source_token", "scatter_index"),
    ),
    OperatorSpec(
        "indexer_projection",
        "T05",
        "M2.indexer.projection",
        "Q_r,X,W_qI,W_w -> Q_I,u",
        "dQ_I,du,saved -> dQ_r,dW_qI,dX,dW_w",
        ("inputs", "weights", "gemm_fingerprint", "row_head_layout"),
    ),
    OperatorSpec(
        "indexer_relu_sum",
        "T05",
        "M2.indexer.score",
        "A:fp32[T,64,N],w:fp32[T,64] -> I:fp32[T,N]",
        "dI,saved -> dA,dw",
        ("relu_mask_zero_derivative", "fixed_head_tree", "A", "R", "w"),
    ),
    OperatorSpec(
        "indexer_q_rope_hadamard_mxfp4",
        "T05",
        "M2.indexer.q_repr",
        "Q_I -> q4_packed,ue8m0,Q_H_ref",
        "reference_dQ_H,saved -> dQ_I",
        ("Q_H", "hadamard_version", "rope_identity", "packing_manifest"),
    ),
    OperatorSpec(
        "indexer_icv_matmul",
        "T05",
        "M2.indexer.icv",
        "Q:fp32[T,64,128],K:fp32[N,128] -> A:fp32[T,64,N]",
        "dA -> dQ,dK",
        ("Q", "K", "block_scale", "inner_tree", "outer_tree"),
    ),
    OperatorSpec(
        "indexer_topk512",
        "T05",
        "M2.indexer.topk",
        "I,valid,global_logical_index -> ids:int64[T,512],valid_count",
        "non_differentiable",
        ("raw_score_bytes", "valid", "global_ids", "total_order"),
    ),
    OperatorSpec(
        "kv_compressor_c128",
        "T04",
        "M2.compressor.main_c128",
        "K,S,APE,norm_w -> completed_row",
        "drow,saved -> dK,dS,dAPE,dNorm",
        ("A", "m", "alpha", "P", "norm", "rope"),
    ),
    OperatorSpec(
        "kv_compressor_c4",
        "T04",
        "M2.compressor.main_c4",
        "K_first,K_second,S_first,S_second,APE,norm_w -> completed_row",
        "drow,saved -> dK_first,dK_second,dS_first,dS_second,dAPE,dNorm",
        ("A_before_overlap", "previous_current_map", "alpha", "P", "inverse_overlap"),
    ),
    OperatorSpec(
        "index_compressor_c4",
        "T05",
        "M2.compressor.index_c4",
        "X,W_kvI,W_gI,APE_I,norm_I -> K_index",
        "dK_index,saved -> dX,dW_kvI,dW_gI,dAPE_I,dNorm_I",
        ("independent_identities", "c4_map", "norm", "rope", "hadamard"),
    ),
    OperatorSpec(
        "kv_compressor_state_cache_update",
        "T03",
        "M2.state.main",
        "K_proj_t,S_proj_t,global_position,state,cache -> commit_record",
        "functional_tensors_and_saved_map_only",
        ("generation", "partial_fp32", "page", "format"),
    ),
    OperatorSpec(
        "index_compressor_state_cache_update_fp4",
        "T05",
        "M2.state.index",
        "K_I_t,S_I_t,global_position,state,index_cache -> commit_record",
        "reference_prequant_only",
        ("prequant_row", "fp4_bytes", "scale", "independent_page", "generation"),
    ),
    OperatorSpec(
        "o_proj_grouped",
        "T06",
        "M2.output_projection",
        "O:[T,64,512],W_a[8],W_b -> Y:[T,4096]",
        "dY,saved -> dO,dW_a,dW_b",
        ("O_tilde", "Z_g", "Z", "group_order", "gemm_identity", "rope_identity"),
    ),
    OperatorSpec(
        "mqa_joint_attention_sink",
        "T06",
        "M2.attention",
        "Q,KV,sink,candidate_plan -> O",
        "dO,saved -> dQ,dKV,dsink",
        ("candidate_plan", "Q", "K", "V", "m", "Z", "p", "dequant_manifest"),
    ),
)
BOUNDARIES = tuple(sorted({op.boundary for op in OPERATORS}))

ARITHMETIC = {
    "reduction": "fp32-adjacent-pair-zero-pad-to-power-of-two.v1",
    "forbidden": [
        "split_k",
        "stream_k",
        "split_kv",
        "dynamic_num_splits",
        "atomic_partial_accumulation",
        "runtime_dependent_partition",
    ],
    "rope": "gptj_interleaved_partial; caller_supplied_table_bytes; out_of_place_inverse",
    "topk": "finite_score_desc_global_id_asc; invalid_removed; pad_-1_to_512.v1",
    "candidate_order": "compressed_prefix_then_recent; sink_denominator_only",
    "indexer_scale": "(u * fp32(64**-0.5)) * fp32(128**-0.5)",
    "mxfp4_reference": {
        "version": "p2.mxfp4-reference.v1",
        "block": 32,
        "element": "E2M1",
        "scale": "UE8M0",
        "scale_selection": "ceil(log2(max(amax,6*2**-126)/6)); clamp[-127,127]",
        "zero_block_scale_byte": 1,
        "rounding": "nearest_ties_even_saturate_6",
        "packing": "even_element_low_nibble",
        "quant_backward": "unsupported",
    },
    "round_points": {
        "normalized_input": "bf16",
        "pool_and_partial_state": "fp32",
        "reference_completed_main": "fp32 (NOT production FP8)",
        "reference_index": "fp32 prequant -> E2M1/UE8M0",
        "production_fp8_layout_and_megatron_round_points": "requires_owner_profile",
    },
    "primitive_reuse": {
        "gemm": "rl_engine.kernels.ops.cuda.matmul.det_gemm",
        "rmsnorm": "rl_engine.kernels.ops.pytorch.norm.rms_norm",
        "rope_extension": "rl_engine.kernels.ops.cuda.rotary_embedding.rope",
    },
}


@dataclass(frozen=True)
class Identity:
    layer: str
    absolute_layer: int = 0
    checkpoint: str = "synthetic-checkpoint.v1"
    main: str = "synthetic-main.v1"
    index: str = "synthetic-index.v1"
    schema: str = SCHEMA_VERSION
    profile: str = REFERENCE_PROFILE

    def validate(self) -> None:
        require(self.schema == SCHEMA_VERSION, Status.SCHEMA_MISMATCH, "identity schema")
        require(self.layer in LAYERS, Status.INVALID_LAYER_TYPE, str(self.layer))
        integer(self.absolute_layer, "absolute_layer")
        require(self.profile == REFERENCE_PROFILE, Status.UNSUPPORTED_CAPABILITY, self.profile)
        for value in (self.checkpoint, self.main, self.index):
            require(isinstance(value, str) and bool(value), Status.IDENTITY_DRIFT, "empty identity")
        require(self.main != self.index, Status.MAIN_INDEX_IDENTITY_ALIAS, "Main == Index")

    def stream_id(self, stream: str, component: str) -> str:
        require(stream in ("recent", "main", "index"), Status.SCHEMA_MISMATCH, stream)
        root = self.index if stream == "index" else self.main
        return digest(
            {
                "root": root,
                "stream": stream,
                "component": component,
                "layer": self.absolute_layer,
                "checkpoint": self.checkpoint,
            }
        )


def manifest() -> dict:
    result = {
        "contract_version": CONTRACT_VERSION,
        "foundation_abi": ABI_VERSION,
        "schema_version": SCHEMA_VERSION,
        "error_version": ERROR_VERSION,
        "model": MODEL,
        "compression": COMPRESSION,
        "modes": list(MODES),
        "operators": [asdict(op) for op in OPERATORS],
        "boundaries": list(BOUNDARIES),
        "arithmetic": ARITHMETIC,
        "statuses": [s.value for s in Status],
        "state_schema": {
            "version": "p2-state.v1",
            "endian": "little",
            "streams": ["recent", "main", "index"],
            "per_token": ["position", "generation", "identity", "recent", "main", "index"],
            "recent": "slot=position%128; page=slot//page_size; generation=position//128+1",
            "compressed": "logical_row=(position+1)//cr-1; page=row//page_size",
            "commit": "one_recent_per_token; compressed_only_at_(position+1)%cr==0",
        },
        "evidence_levels": ["synthetic_reference", "live_ws1", "live_ws2", "integration"],
        "runtime_policy": runtime_policy(),
    }
    return json.loads(canonical(result))


def check_compatibility(candidate: dict) -> None:
    require(candidate == manifest(), Status.SCHEMA_MISMATCH, "contract/ABI/profile drift")


def runtime_policy() -> dict:
    return {
        "schema": "p2-runtime-policy.v1",
        "profile": REFERENCE_PROFILE,
        "backend": "cpu",
        "fallback": False,
        "num_splits": 1,
        "split_k": False,
        "stream_k": False,
        "split_kv": False,
        "dynamic_partition": False,
        "atomic_partial_accumulation": False,
        "tree": ARITHMETIC["reduction"],
        "round_points": ARITHMETIC["round_points"].copy(),
        "readback_kind": "actual",
        "kernel_fingerprint": None,
    }


@schema_guard
def validate_runtime_policy(policy: dict) -> None:
    require(
        policy["schema"] == "p2-runtime-policy.v1", Status.SCHEMA_MISMATCH, "runtime policy schema"
    )
    require(
        policy["profile"] == REFERENCE_PROFILE and policy["backend"] == "cpu",
        Status.UNSUPPORTED_CAPABILITY,
        "unregistered backend/profile",
    )
    require(policy["fallback"] is False, Status.SILENT_FALLBACK, "fallback")
    require(
        type(policy["num_splits"]) is int
        and policy["num_splits"] == 1
        and all(
            policy[k] is False for k in ("split_k", "stream_k", "split_kv", "dynamic_partition")
        ),
        Status.FORBIDDEN_SPLIT_REDUCTION,
        "partition",
    )
    require(
        policy["atomic_partial_accumulation"] is False, Status.FORBIDDEN_ATOMIC_REDUCTION, "atomics"
    )
    require(
        policy["round_points"] == ARITHMETIC["round_points"],
        Status.ROUND_POINT_MISMATCH,
        "round points",
    )
    require(
        policy["tree"] == ARITHMETIC["reduction"], Status.FORBIDDEN_SPLIT_REDUCTION, "fixed tree"
    )
    require(
        policy["readback_kind"] == "actual" and policy["kernel_fingerprint"] is None,
        Status.MISSING_PROVENANCE,
        "CPU reference cannot claim native kernel readback",
    )
    require(set(policy) == set(runtime_policy()), Status.SCHEMA_MISMATCH, "policy keys")
