# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C2 (#268) canonical workload: logical identity, fixtures, and manifest API.

This module freezes the full Qwen3-8B Dense logical sample workload used by later
gates (C3–C11). It does not run the full model or assert #150 numerical thresholds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_MANIFEST_PATH = Path(__file__).with_name("ws1_manifest.json")

_REQUIRED_TOP_LEVEL = (
    "version",
    "workload_id",
    "seed",
    "model_identity",
    "chain_semantics",
    "stochastic_policy",
    "primary_matrix",
    "fixtures",
    "logical_identity",
    "capabilities",
    "backend_profiles",
    "representative_cases",
    "provenance_boundary",
    "fixture_identity_sha256",
)

_REQUIRED_MATRIX_CELLS = (
    "B1-singleton_aggregate/full",
    "BN/full",
    "B1-singleton_aggregate/chunked",
    "BN/chunked",
)

_REQUIRED_PROFILES = ("cuda_bf16", "triton_cuda_bf16", "ascend_bf16")

_REQUIRED_CHAIN_NODES = (
    "embedding",
    "rms_norm",
    "det_gemm",
    "qk_norm",
    "rope",
    "attention",
    "swiglu",
    "silu",
    "lm_head",
    "logprob",
    "batch_invariant_logp",
)

_FORBIDDEN_COMPARISON_ROLES = frozenset({"baseline", "singleton_aggregate"})

_OFFICIAL_FINGERPRINT = {
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}


class WorkloadError(ValueError):
    """Raised when the WS1 workload manifest or fixture is invalid."""


def _require(mapping: Mapping[str, Any], key: str, *, context: str) -> Any:
    """Return mapping[key] or raise WorkloadError (never bare KeyError)."""
    if key not in mapping:
        raise WorkloadError(f"{context} missing {key!r}")
    return mapping[key]


@dataclass(frozen=True)
class LogicalToken:
    """One active or inactive logical token position."""

    sample_id: str
    token_position: int
    token_id: int
    is_active: bool


@dataclass(frozen=True)
class LogicalSample:
    """One logical sequence with identity recoverable after layout transforms."""

    sample_id: str
    token_ids: tuple[int, ...]
    prompt_len: int
    seq_len: int

    def tokens(self) -> tuple[LogicalToken, ...]:
        out: list[LogicalToken] = []
        for pos, tid in enumerate(self.token_ids):
            out.append(
                LogicalToken(
                    sample_id=self.sample_id,
                    token_position=pos,
                    token_id=int(tid),
                    is_active=pos >= self.prompt_len,
                )
            )
        return tuple(out)

    def active_tokens(self) -> tuple[LogicalToken, ...]:
        return tuple(t for t in self.tokens() if t.is_active)


@dataclass(frozen=True)
class LogicalBatch:
    """Ordered multiset of logical samples for one workload cell."""

    workload_id: str
    seed: int
    samples: tuple[LogicalSample, ...]
    cell_id: str | None = None

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(s.sample_id for s in self.samples)

    def logical_keys(self, *, active_only: bool = False) -> tuple[tuple[str, int], ...]:
        keys: list[tuple[str, int]] = []
        for sample in self.samples:
            for tok in sample.tokens():
                if active_only and not tok.is_active:
                    continue
                keys.append((tok.sample_id, tok.token_position))
        return tuple(keys)

    def active_token_count(self) -> int:
        return sum(1 for s in self.samples for t in s.tokens() if t.is_active)

    def token_multiset(self, *, active_only: bool = True) -> tuple[tuple[str, int, int], ...]:
        """Return (sample_id, token_position, token_id) multiset in fixed sample order."""
        items: list[tuple[str, int, int]] = []
        for sample in self.samples:
            for tok in sample.tokens():
                if active_only and not tok.is_active:
                    continue
                items.append((tok.sample_id, tok.token_position, tok.token_id))
        return tuple(items)


@dataclass(frozen=True)
class PaddedBatch:
    """Right- or left-padded physical layout with restore indices."""

    physical_token_ids: tuple[tuple[int, ...], ...]
    physical_attention_mask: tuple[tuple[int, ...], ...]
    physical_loss_mask: tuple[tuple[int, ...], ...]
    physical_position_ids: tuple[tuple[int, ...], ...]
    pad_side: str
    pad_token_id: int
    padded_len: int
    # For each physical (batch_idx, phys_pos) -> (sample_id, token_position) or None if pad
    restore_map: tuple[tuple[tuple[str, int] | None, ...], ...]
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalLayout:
    """Flattened physical tokens plus an unambiguous logical restore map."""

    layout_kind: str
    physical_token_ids: tuple[int, ...]
    physical_loss_mask: tuple[int, ...]
    restore_map: tuple[tuple[str, int], ...]
    segment_offsets: tuple[int, ...]
    segment_lengths: tuple[int, ...]


@dataclass(frozen=True)
class ChunkPlan:
    """Chunked-prefill plan for one logical sequence length."""

    seq_len: int
    chunk_size: int
    chunk_spans: tuple[tuple[int, int], ...]  # half-open [start, end)

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_spans)


@dataclass(frozen=True)
class SingletonAggregatePlan:
    """B=1 × N schedule that must match one B=N run of the same multiset."""

    sample_ids: tuple[str, ...]
    run_sample_ids: tuple[tuple[str, ...], ...]  # each run is a 1-tuple
    aggregation_order: tuple[str, ...]
    denominator: str
    token_multiset: tuple[tuple[str, int, int], ...]


@dataclass
class WS1Manifest:
    """Validated in-memory view of ws1_manifest.json."""

    raw: dict[str, Any]
    path: Path = field(default=_MANIFEST_PATH)

    @property
    def version(self) -> str:
        return str(self.raw["version"])

    @property
    def workload_id(self) -> str:
        return str(self.raw["workload_id"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def model_identity(self) -> dict[str, Any]:
        return dict(self.raw["model_identity"])

    @property
    def chain_semantics(self) -> dict[str, Any]:
        return dict(self.raw["chain_semantics"])

    @property
    def clip_interval(self) -> tuple[float, float]:
        interval = self.raw["chain_semantics"]["clip_interval"]
        return (float(interval[0]), float(interval[1]))

    @property
    def primary_matrix(self) -> dict[str, Any]:
        return dict(self.raw["primary_matrix"])

    @property
    def fixtures(self) -> dict[str, Any]:
        return dict(self.raw["fixtures"])

    @property
    def backend_profiles(self) -> dict[str, Any]:
        return dict(self.raw["backend_profiles"])

    @property
    def representative_cases(self) -> list[dict[str, Any]]:
        return list(self.raw["representative_cases"])


def default_manifest_path() -> Path:
    return _MANIFEST_PATH


def load_manifest(path: str | Path | None = None) -> WS1Manifest:
    manifest_path = Path(path) if path is not None else _MANIFEST_PATH
    with manifest_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise WorkloadError("manifest root must be a JSON object")
    validate_manifest(raw)
    return WS1Manifest(raw=raw, path=manifest_path)


def validate_manifest(raw: Mapping[str, Any]) -> None:
    """Hard-fail if any required C2 pin is missing or inconsistent."""
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in raw]
    if missing:
        raise WorkloadError(f"manifest missing top-level keys: {missing}")

    _validate_model_identity(raw["model_identity"])
    _validate_chain_semantics(raw["chain_semantics"])
    _validate_stochastic_policy(raw["stochastic_policy"])
    _validate_primary_matrix(raw["primary_matrix"], raw["fixtures"])
    _validate_fixtures(raw["fixtures"], raw["primary_matrix"])
    _validate_logical_identity(raw["logical_identity"])
    _validate_capabilities(raw["capabilities"])
    _validate_backend_profiles(raw["backend_profiles"], raw["capabilities"])
    _validate_representative_cases(raw["representative_cases"])
    _validate_fixture_case_bindings(raw["fixtures"], raw["representative_cases"])
    expected_identity = manifest_identity_hash(raw)
    if raw["fixture_identity_sha256"] != expected_identity:
        raise WorkloadError(
            "fixture_identity_sha256 does not match manifest; change workload_id/version "
            "and regenerate the identity for any numerics-affecting edit"
        )


def _validate_model_identity(identity: Mapping[str, Any]) -> None:
    for key in ("model_id", "revision", "config_fingerprint", "weight_snapshot"):
        if key not in identity:
            raise WorkloadError(f"model_identity missing {key!r}")
    fp = identity["config_fingerprint"]
    if not isinstance(fp, Mapping):
        raise WorkloadError("config_fingerprint must be an object")
    for key, expected in _OFFICIAL_FINGERPRINT.items():
        if key not in fp:
            raise WorkloadError(f"config_fingerprint missing {key!r}")
        if fp[key] != expected:
            raise WorkloadError(
                f"config_fingerprint {key}={fp[key]!r} does not match official "
                f"Qwen3-8B Dense pin {expected!r}; architecture shrink is forbidden"
            )
    if not identity.get("exit_forbids_architecture_shrink", False):
        raise WorkloadError("exit_forbids_architecture_shrink must be true")
    weight = identity["weight_snapshot"]
    for key in (
        "pin_method",
        "total_size_bytes",
        "index_file",
        "index_sha256",
        "content_hash_algorithm",
        "content_hash",
        "shards",
        "weight_files_total_size_bytes",
    ):
        if key not in weight:
            raise WorkloadError(f"weight_snapshot missing {key!r}")
    shards = weight["shards"]
    if not isinstance(shards, list) or not shards:
        raise WorkloadError("weight_snapshot.shards must be a non-empty list")
    filenames = [str(shard.get("filename", "")) for shard in shards]
    if len(set(filenames)) != len(filenames) or any(not name for name in filenames):
        raise WorkloadError("weight_snapshot shard filenames must be unique and non-empty")
    if int(weight["weight_files_total_size_bytes"]) != sum(int(s["size_bytes"]) for s in shards):
        raise WorkloadError("weight_snapshot file total does not match shard sizes")
    for shard in shards:
        digest = str(shard.get("sha256", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise WorkloadError("every weight shard must pin a lowercase SHA-256")
    index_digest = str(weight["index_sha256"])
    if len(index_digest) != 64 or any(c not in "0123456789abcdef" for c in index_digest):
        raise WorkloadError("weight_snapshot.index_sha256 must be a lowercase SHA-256")
    expected_content_hash = weight_snapshot_hash(shards)
    if weight["content_hash_algorithm"] != "sha256-of-sorted-shard-records-v1":
        raise WorkloadError("unsupported weight_snapshot content_hash_algorithm")
    if weight["content_hash"] != expected_content_hash:
        raise WorkloadError("weight_snapshot content_hash does not match shard records")


def _validate_chain_semantics(sem: Mapping[str, Any]) -> None:
    for key in (
        "execution_dtype",
        "reference_dtype",
        "clip_interval",
        "aggregates",
        "forbidden_comparison_roles",
        "tf32_policy_ref",
        "report_naming",
        "backend_actual_semantics",
    ):
        if key not in sem:
            raise WorkloadError(f"chain_semantics missing {key!r}")
    if sem["execution_dtype"] != "bfloat16":
        raise WorkloadError("execution_dtype must be bfloat16 for WS1")
    if sem["reference_dtype"] != "float32":
        raise WorkloadError("reference_dtype must be float32 for WS1")
    interval = sem["clip_interval"]
    if not (isinstance(interval, (list, tuple)) and len(interval) == 2):
        raise WorkloadError("clip_interval must be a length-2 list")
    if float(interval[0]) >= float(interval[1]):
        raise WorkloadError("clip_interval lower bound must be < upper bound")
    aggregates = list(sem["aggregates"])
    for name in ("max_abs_dlogp", "approx_kl0", "clipfrac0"):
        if name not in aggregates:
            raise WorkloadError(f"aggregates must include {name}")
    forbidden = set(sem["forbidden_comparison_roles"])
    if not _FORBIDDEN_COMPARISON_ROLES.issubset(forbidden):
        raise WorkloadError(
            f"forbidden_comparison_roles must include {_FORBIDDEN_COMPARISON_ROLES}"
        )
    if "tolerance_contract.json" not in str(sem["tf32_policy_ref"]):
        raise WorkloadError("tf32_policy_ref must point at the C1 tolerance contract")
    report_naming = sem["report_naming"]
    if not isinstance(report_naming, Mapping):
        raise WorkloadError("report_naming must be an object")
    report_forbidden = set(report_naming.get("forbidden_in_reports", []))
    if not _FORBIDDEN_COMPARISON_ROLES.issubset(report_forbidden):
        raise WorkloadError(
            "report_naming.forbidden_in_reports must include baseline and singleton_aggregate"
        )
    if report_naming.get("singleton_aggregate_is") != "c2_execution_aggregation_mode_only":
        raise WorkloadError(
            "report_naming must declare singleton_aggregate as c2 execution mode only"
        )
    actual_sem = sem["backend_actual_semantics"]
    if not isinstance(actual_sem, Mapping):
        raise WorkloadError("backend_actual_semantics must be an object")
    if actual_sem.get("c2_representative_actual_source") != (
        "scripts/ws1_candidate_evidence.py runtime execution"
    ):
        raise WorkloadError("C2 representative actual provenance must come from runtime execution")
    if "C8" not in actual_sem.get("full_model_runtime_observed_actual_owner", []):
        raise WorkloadError("backend_actual_semantics must assign full-model actuals to C8+")


def _validate_stochastic_policy(policy: Mapping[str, Any]) -> None:
    for key in ("dropout", "sampling_in_logprob_parity", "undeclared_randomness"):
        if key not in policy:
            raise WorkloadError(f"stochastic_policy missing {key!r}")
    if float(policy["dropout"]) != 0.0:
        raise WorkloadError("canonical gate dropout must be 0.0")
    if policy.get("sampling_in_logprob_parity", True):
        raise WorkloadError("sampling_in_logprob_parity must be false")
    if policy["undeclared_randomness"] != "hard_fail":
        raise WorkloadError("undeclared_randomness must be hard_fail")


def _validate_primary_matrix(matrix: Mapping[str, Any], fixtures: Mapping[str, Any]) -> None:
    n = int(_require(matrix, "N", context="primary_matrix"))
    if n <= 1:
        raise WorkloadError("primary_matrix.N must be > 1")
    sample_ids = list(_require(matrix, "sample_ids", context="primary_matrix"))
    if len(sample_ids) != n:
        raise WorkloadError("sample_ids length must equal N")
    if len(set(sample_ids)) != n:
        raise WorkloadError("sample_ids must be unique")
    perm = matrix.get("batch_permutation", {})
    if perm.get("enabled"):
        p = list(_require(perm, "permutation", context="primary_matrix.batch_permutation"))
        if sorted(p) != list(range(n)):
            raise WorkloadError("batch_permutation.permutation must be a permutation of [0..N)")
    chunk = _require(matrix, "chunk", context="primary_matrix")
    if not isinstance(chunk, Mapping):
        raise WorkloadError("primary_matrix.chunk must be an object")
    chunk_size = int(_require(chunk, "chunk_size_tokens", context="primary_matrix.chunk"))
    seq_len = int(_require(fixtures, "primary_seq_len", context="fixtures"))
    if chunk_size <= 0:
        raise WorkloadError("chunk_size_tokens must be positive")
    plan = build_chunk_plan(seq_len, chunk_size)
    if chunk.get("require_ge_2_chunks") and plan.num_chunks < 2:
        raise WorkloadError("chunk plan must create >= 2 chunks")
    if chunk.get("non_divisible_case") and seq_len % chunk_size == 0:
        raise WorkloadError("non_divisible_case requires seq_len % chunk_size != 0")

    cells = _require(matrix, "cells", context="primary_matrix")
    if not isinstance(cells, list):
        raise WorkloadError("primary_matrix.cells must be a list")
    cell_ids = [c["cell_id"] for c in cells]
    if set(cell_ids) != set(_REQUIRED_MATRIX_CELLS):
        raise WorkloadError(
            f"primary_matrix.cells must be exactly {_REQUIRED_MATRIX_CELLS}, got {cell_ids}"
        )
    for cell in cells:
        mode = cell["batch_mode"]
        if mode not in ("singleton_aggregate", "batched"):
            raise WorkloadError(f"unknown batch_mode {mode!r}")
        for role_key in ("comparison_lhs_role", "comparison_rhs_role"):
            role = str(cell.get(role_key, ""))
            if role in _FORBIDDEN_COMPARISON_ROLES:
                raise WorkloadError(
                    f"cell {cell.get('cell_id')!r}: {role_key} must not use "
                    f"forbidden comparison role {role!r}"
                )


def _validate_fixtures(fixtures: Mapping[str, Any], matrix: Mapping[str, Any]) -> None:
    samples = fixtures.get("samples")
    if not isinstance(samples, list) or not samples:
        raise WorkloadError("fixtures.samples must be a non-empty list")
    expected_ids = list(matrix["sample_ids"])
    got_ids = [s["sample_id"] for s in samples]
    if got_ids != expected_ids:
        raise WorkloadError(
            f"fixtures.samples order/ids must match primary_matrix.sample_ids "
            f"{expected_ids}, got {got_ids}"
        )
    primary_seq = int(_require(fixtures, "primary_seq_len", context="fixtures"))
    declared_varlen = [int(x) for x in _require(fixtures, "varlen_seq_lens", context="fixtures")]
    if declared_varlen != [int(s["seq_len"]) for s in samples]:
        raise WorkloadError("varlen_seq_lens must match fixtures.samples seq_len values")
    for sample in samples:
        tids = sample["token_ids"]
        if len(tids) != int(sample["seq_len"]):
            raise WorkloadError(
                f"sample {sample['sample_id']} token_ids length {len(tids)} "
                f"!= sample seq_len {sample['seq_len']}"
            )
        if not 0 < int(sample["prompt_len"]) < int(sample["seq_len"]):
            raise WorkloadError(f"sample {sample['sample_id']} prompt_len is invalid")
    if max(declared_varlen) != primary_seq:
        raise WorkloadError("primary_seq_len must equal the maximum varlen sequence length")
    # Per-sample prompt/completion lengths are authoritative (no stale scalar pin).
    expected_prompt_lens = [int(s["prompt_len"]) for s in samples]
    expected_completion_lens = [int(s["seq_len"]) - int(s["prompt_len"]) for s in samples]
    if list(fixtures.get("prompt_lens", [])) != expected_prompt_lens:
        raise WorkloadError("fixtures.prompt_lens must match per-sample prompt_len values")
    if list(fixtures.get("completion_lens", [])) != expected_completion_lens:
        raise WorkloadError("fixtures.completion_lens must match per-sample (seq_len - prompt_len)")
    if int(fixtures.get("max_completion_len", -1)) != max(expected_completion_lens):
        raise WorkloadError("fixtures.max_completion_len must equal max(completion_lens)")
    if "primary_completion_len" in fixtures:
        raise WorkloadError(
            "fixtures.primary_completion_len is forbidden under varlen primary samples; "
            "use completion_lens / max_completion_len"
        )
    padding = _require(fixtures, "padding", context="fixtures")
    if not isinstance(padding, Mapping):
        raise WorkloadError("fixtures.padding must be an object")
    if "right" not in padding["modes"] or "left" not in padding["modes"]:
        raise WorkloadError("padding.modes must include left and right")
    packing = _require(fixtures, "packing", context="fixtures")
    if not isinstance(packing, Mapping):
        raise WorkloadError("fixtures.packing must be an object")
    if packing["status"] not in {
        "supported",
        "n_a_with_capability_proof",
        "unsupported",
        "supported_op_not_in_exit_matrix",
    }:
        raise WorkloadError(f"unknown packing status {packing['status']!r}")
    if packing["status"] != "supported":
        raise WorkloadError("packing op is present, so C2 must pin a supported packed fixture")
    if not packing.get("packed_fixture"):
        raise WorkloadError("supported packing requires packed_fixture")
    for name in (
        "short_full_model_fixture",
        "long_full_model_fixture",
        "representative_full_model_fixture",
    ):
        fixture = fixtures[name]
        if "token_ids" in fixture and len(fixture["token_ids"]) != int(fixture["seq_len"]):
            raise WorkloadError(f"{name} token_ids length mismatch")
        if not fixture.get("candidate_case_ids"):
            raise WorkloadError(f"{name} must reference representative case IDs")


def _validate_logical_identity(logical: Mapping[str, Any]) -> None:
    key = list(logical.get("key", []))
    if key != ["sample_id", "token_position"]:
        raise WorkloadError("logical_identity.key must be [sample_id, token_position]")
    grad = logical.get("gradient_singleton_aggregate", {})
    if not grad.get("forbid_different_sample_sets", False):
        raise WorkloadError("gradient_singleton_aggregate must forbid different sample sets")


def _validate_capabilities(caps: Mapping[str, Any]) -> None:
    for key in ("packing", "qk_norm", "required_chain_ops", "operator_spec_map"):
        if key not in caps:
            raise WorkloadError(f"capabilities missing {key!r}")
    ops = {entry["op"]: entry["status"] for entry in caps["required_chain_ops"]}
    for op in _REQUIRED_CHAIN_NODES:
        if op not in ops:
            raise WorkloadError(f"required_chain_ops missing {op!r}")
        if op not in caps["operator_spec_map"]:
            raise WorkloadError(f"operator_spec_map missing {op!r}")


def _validate_backend_profiles(
    profiles: Mapping[str, Any], capabilities: Mapping[str, Any]
) -> None:
    for name in _REQUIRED_PROFILES:
        if name not in profiles:
            raise WorkloadError(f"backend_profiles missing required profile {name!r}")
    required_ops = [
        e["op"] for e in capabilities["required_chain_ops"] if e["status"] == "required"
    ]
    for name, profile in profiles.items():
        nodes = profile.get("required_nodes")
        if not isinstance(nodes, list) or not nodes:
            raise WorkloadError(f"profile {name} must declare required_nodes")
        node_names = [n["node"] for n in nodes]
        missing = [op for op in required_ops if op not in node_names]
        if missing:
            raise WorkloadError(
                f"profile {name} missing required chain nodes {missing}; "
                "undeclared missing nodes are forbidden (use status=missing_required)"
            )
        for node in nodes:
            status = node.get("status")
            if status not in {"declared", "missing_required"}:
                raise WorkloadError(
                    f"profile {name} node {node.get('node')}: status must be "
                    f"declared or missing_required, got {status!r}"
                )
            if status == "missing_required":
                if node.get("expected_backend_id") not in (None, ""):
                    raise WorkloadError(
                        f"profile {name} node {node['node']}: missing_required must not "
                        "claim an expected_backend_id"
                    )
            else:
                for field_name in (
                    "expected_backend_id",
                    "expected_kernel_config_id",
                    "algorithm_property",
                ):
                    if not node.get(field_name):
                        raise WorkloadError(
                            f"profile {name} node {node['node']} missing {field_name}"
                        )


def _validate_representative_cases(cases: Sequence[Mapping[str, Any]]) -> None:
    if not cases:
        raise WorkloadError("representative_cases must be non-empty")
    ids = [c["case_id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise WorkloadError("representative_cases case_id values must be unique")
    families = {c["family"] for c in cases}
    for family in ("gemm", "attention", "logprob"):
        if family not in families:
            raise WorkloadError(f"representative_cases must include family {family!r}")
    for case in cases:
        for key in (
            "case_id",
            "family",
            "shape",
            "expected_backend_id",
            "expected_kernel_config_id",
            "actual_backend_id",
            "actual_kernel_config_id",
            "provenance_status",
            "provenance_evidence",
            "algorithm_property",
            "architecture_identity",
            "fixture_id",
            "operator_spec",
        ):
            if key not in case:
                raise WorkloadError(f"case {case.get('case_id')} missing {key!r}")
        if case["architecture_identity"] != "full_qwen3_8b_dense":
            raise WorkloadError(
                f"case {case['case_id']} must pin architecture_identity=full_qwen3_8b_dense"
            )
        if case["provenance_status"] != "runtime_evidence_required":
            raise WorkloadError(f"case {case['case_id']} must require runtime candidate evidence")
        if case["actual_backend_id"] != case["expected_backend_id"]:
            raise WorkloadError(f"case {case['case_id']} actual backend mismatch")
        if case["actual_kernel_config_id"] != case["expected_kernel_config_id"]:
            raise WorkloadError(f"case {case['case_id']} actual kernel mismatch")
        evidence = case["provenance_evidence"]
        if evidence.get("kind") != "runtime_execution_via_operator_specs":
            raise WorkloadError(f"case {case['case_id']} lacks runtime provenance command")
        if evidence.get("resolved_path") != case["actual_kernel_config_id"]:
            raise WorkloadError(f"case {case['case_id']} evidence path mismatch")
        if not evidence.get("algorithm_source"):
            raise WorkloadError(f"case {case['case_id']} lacks algorithm source proof")
        if not evidence.get("runtime_evidence_command"):
            raise WorkloadError(f"case {case['case_id']} lacks runtime evidence command")
    for profile in _REQUIRED_PROFILES:
        profile_cases = [c for c in cases if profile in c.get("profile_ids", [])]
        for family in ("gemm", "attention", "logprob"):
            count = sum(c["family"] == family for c in profile_cases)
            if not 1 <= count <= 3:
                raise WorkloadError(
                    f"profile {profile} must have 1-3 {family} representative cases"
                )
        gemm_m = {int(c["shape"]["M"]) for c in profile_cases if c["family"] == "gemm"}
        if len(gemm_m) < 2:
            raise WorkloadError(f"profile {profile} GEMM cases require multiple M values")
        attn_modes = {c["shape"]["mode"] for c in profile_cases if c["family"] == "attention"}
        if attn_modes != {"prefill", "decode"}:
            raise WorkloadError(f"profile {profile} attention cases require prefill+decode")


def _validate_fixture_case_bindings(
    fixtures: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> None:
    """Require every fixture→case edge to describe a shape produced by that fixture."""
    fixture_names = (
        "short_full_model_fixture",
        "long_full_model_fixture",
        "representative_full_model_fixture",
    )
    by_fixture_id = {fixtures[name]["fixture_id"]: fixtures[name] for name in fixture_names}
    by_case_id = {case["case_id"]: case for case in cases}

    for fixture_id, fixture in by_fixture_id.items():
        for case_id in fixture["candidate_case_ids"]:
            if case_id not in by_case_id:
                raise WorkloadError(f"fixture {fixture_id} references unknown case {case_id!r}")
            if by_case_id[case_id]["fixture_id"] != fixture_id:
                raise WorkloadError(
                    f"fixture {fixture_id} references case {case_id!r} bound to "
                    f"{by_case_id[case_id]['fixture_id']!r}"
                )

    referenced = {
        case_id for fixture in by_fixture_id.values() for case_id in fixture["candidate_case_ids"]
    }
    if referenced != set(by_case_id):
        raise WorkloadError("every representative case must be referenced by its source fixture")

    short = fixtures["short_full_model_fixture"]
    long = fixtures["long_full_model_fixture"]
    primary_total_tokens = sum(int(sample["seq_len"]) for sample in fixtures["samples"])
    primary_max_seq = max(int(sample["seq_len"]) for sample in fixtures["samples"])
    expected_shapes: dict[str, dict[str, dict[str, Any]]] = {
        "short_full_model_seq8": {
            "gemm": {"M": int(short["seq_len"])},
            "logprob": {"B": 1, "T": int(short["seq_len"]) - int(short["prompt_len"])},
            "batch_invariant_logp": {
                "B": 1,
                "T": int(short["seq_len"]) - int(short["prompt_len"]),
            },
            "attention": {
                "B": 1,
                "Sq": int(short["seq_len"]),
                "Skv": int(short["seq_len"]),
                "mode": "prefill",
            },
            "norm": {"T": int(short["seq_len"])},
            "elementwise": {"T": int(short["seq_len"])},
            "embedding": {"T": int(short["seq_len"])},
            "lm_head": {"T": int(short["seq_len"])},
        },
        "long_full_model_seq32": {
            "attention": {"B": 1, "Sq": 1, "Skv": int(long["seq_len"]), "mode": "decode"}
        },
        "rep_full_model_seq16": {
            "gemm": {"M": primary_total_tokens},
            "attention": {
                "B": len(fixtures["samples"]),
                "Sq": primary_max_seq,
                "Skv": primary_max_seq,
                "mode": "prefill",
            },
            "logprob": {
                "B": len(fixtures["samples"]),
                "T": sum(
                    int(sample["seq_len"]) - int(sample["prompt_len"])
                    for sample in fixtures["samples"]
                ),
            },
            "batch_invariant_logp": {
                "B": len(fixtures["samples"]),
                "T": sum(
                    int(sample["seq_len"]) - int(sample["prompt_len"])
                    for sample in fixtures["samples"]
                ),
            },
            "norm": {"T": primary_total_tokens},
            "elementwise": {"T": primary_total_tokens},
            "embedding": {"T": primary_total_tokens},
            "lm_head": {"T": primary_total_tokens},
        },
    }
    for case in cases:
        fixture_shapes = expected_shapes.get(case["fixture_id"])
        if fixture_shapes is None:
            raise WorkloadError(
                f"case {case['case_id']}: unknown fixture_id {case['fixture_id']!r}"
            )
        required = fixture_shapes.get(case["family"])
        if required is None:
            raise WorkloadError(
                f"case {case['case_id']}: fixture {case['fixture_id']!r} does not "
                f"cover family {case['family']!r}"
            )
        mismatched = {
            key: (case["shape"].get(key), value)
            for key, value in required.items()
            if case["shape"].get(key) != value
        }
        if mismatched:
            raise WorkloadError(
                f"case {case['case_id']} shape does not derive from fixture "
                f"{case['fixture_id']}: {mismatched}"
            )


def build_logical_batch(
    manifest: WS1Manifest | None = None,
    *,
    cell_id: str | None = None,
    sample_ids: Sequence[str] | None = None,
) -> LogicalBatch:
    """Build the fixed logical sample multiset for the primary workload."""
    m = manifest if manifest is not None else load_manifest()
    fixtures = m.fixtures
    matrix = m.primary_matrix
    by_id = {s["sample_id"]: s for s in fixtures["samples"]}
    order = list(sample_ids) if sample_ids is not None else list(matrix["sample_ids"])
    samples: list[LogicalSample] = []
    for sid in order:
        if sid not in by_id:
            raise WorkloadError(f"unknown sample_id {sid!r}")
        raw = by_id[sid]
        token_ids = tuple(int(x) for x in raw["token_ids"])
        samples.append(
            LogicalSample(
                sample_id=sid,
                token_ids=token_ids,
                prompt_len=int(raw["prompt_len"]),
                seq_len=int(raw["seq_len"]),
            )
        )
    if cell_id is not None:
        get_matrix_cell(m, cell_id)
    return LogicalBatch(
        workload_id=m.workload_id,
        seed=m.seed,
        samples=tuple(samples),
        cell_id=cell_id,
    )


def get_matrix_cell(manifest: WS1Manifest, cell_id: str) -> dict[str, Any]:
    for cell in manifest.primary_matrix["cells"]:
        if cell["cell_id"] == cell_id:
            return dict(cell)
    raise WorkloadError(f"unknown cell_id {cell_id!r}")


def matrix_cell_ids(manifest: WS1Manifest | None = None) -> tuple[str, ...]:
    m = manifest if manifest is not None else load_manifest()
    return tuple(c["cell_id"] for c in m.primary_matrix["cells"])


def build_chunk_plan(seq_len: int, chunk_size: int) -> ChunkPlan:
    if chunk_size <= 0:
        raise WorkloadError("chunk_size must be positive")
    if seq_len <= 0:
        raise WorkloadError("seq_len must be positive")
    spans: list[tuple[int, int]] = []
    start = 0
    while start < seq_len:
        end = min(start + chunk_size, seq_len)
        spans.append((start, end))
        start = end
    return ChunkPlan(seq_len=seq_len, chunk_size=chunk_size, chunk_spans=tuple(spans))


def chunk_plan_from_manifest(manifest: WS1Manifest | None = None) -> ChunkPlan:
    m = manifest if manifest is not None else load_manifest()
    return build_chunk_plan(
        int(m.fixtures["primary_seq_len"]),
        int(m.primary_matrix["chunk"]["chunk_size_tokens"]),
    )


def apply_chunking(batch: LogicalBatch, *, chunk_size: int) -> PhysicalLayout:
    """Materialize chunked-prefill order for every sample."""
    if chunk_size <= 0:
        raise WorkloadError("chunk_size must be positive")
    ids: list[int] = []
    masks: list[int] = []
    restore: list[tuple[str, int]] = []
    offsets: list[int] = []
    lengths: list[int] = []
    for sample in batch.samples:
        plan = build_chunk_plan(sample.seq_len, chunk_size)
        for start, end in plan.chunk_spans:
            offsets.append(len(ids))
            lengths.append(end - start)
            for pos in range(start, end):
                ids.append(sample.token_ids[pos])
                masks.append(int(pos >= sample.prompt_len))
                restore.append((sample.sample_id, pos))
    return PhysicalLayout(
        layout_kind="chunked",
        physical_token_ids=tuple(ids),
        physical_loss_mask=tuple(masks),
        restore_map=tuple(restore),
        segment_offsets=tuple(offsets),
        segment_lengths=tuple(lengths),
    )


def apply_packing(batch: LogicalBatch) -> PhysicalLayout:
    """Pack variable-length samples in fixed sample/token order."""
    ids: list[int] = []
    masks: list[int] = []
    restore: list[tuple[str, int]] = []
    offsets: list[int] = []
    lengths: list[int] = []
    for sample in batch.samples:
        offsets.append(len(ids))
        lengths.append(sample.seq_len)
        ids.extend(sample.token_ids)
        masks.extend(int(pos >= sample.prompt_len) for pos in range(sample.seq_len))
        restore.extend((sample.sample_id, pos) for pos in range(sample.seq_len))
    return PhysicalLayout(
        layout_kind="packed",
        physical_token_ids=tuple(ids),
        physical_loss_mask=tuple(masks),
        restore_map=tuple(restore),
        segment_offsets=tuple(offsets),
        segment_lengths=tuple(lengths),
    )


def restore_logical_order(
    layout: PhysicalLayout, physical_values: Sequence[Any]
) -> dict[tuple[str, int], Any]:
    if len(physical_values) != len(layout.restore_map):
        raise WorkloadError("physical_values length does not match restore map")
    out: dict[tuple[str, int], Any] = {}
    for key, value in zip(layout.restore_map, physical_values, strict=True):
        if key in out:
            raise WorkloadError(f"duplicate logical key {key}")
        out[key] = value
    return out


def apply_padding(
    batch: LogicalBatch,
    *,
    pad_side: str,
    padded_len: int | None = None,
    pad_token_id: int | None = None,
    manifest: WS1Manifest | None = None,
) -> PaddedBatch:
    """Pad logical sequences; restore_map recovers (sample_id, token_position)."""
    if pad_side not in ("left", "right"):
        raise WorkloadError(f"pad_side must be left or right, got {pad_side!r}")
    m = manifest if manifest is not None else load_manifest()
    pad_id = (
        int(pad_token_id)
        if pad_token_id is not None
        else int(m.fixtures["padding"]["pad_token_id"])
    )
    target_len = (
        int(padded_len)
        if padded_len is not None
        else int(m.fixtures["padding"]["primary_padded_len"])
    )
    max_seq = max(s.seq_len for s in batch.samples)
    if target_len < max_seq:
        raise WorkloadError(f"padded_len {target_len} < max logical seq_len {max_seq}")

    physical_ids: list[tuple[int, ...]] = []
    masks: list[tuple[int, ...]] = []
    loss_masks: list[tuple[int, ...]] = []
    positions: list[tuple[int, ...]] = []
    restore: list[tuple[tuple[str, int] | None, ...]] = []
    for sample in batch.samples:
        pad_count = target_len - sample.seq_len
        pad_tokens = (pad_id,) * pad_count
        pad_restore: tuple[None, ...] = (None,) * pad_count
        logical_restore = tuple((sample.sample_id, pos) for pos in range(sample.seq_len))
        if pad_side == "right":
            ids = sample.token_ids + pad_tokens
            mask = (1,) * sample.seq_len + (0,) * pad_count
            rmap = logical_restore + pad_restore
            loss_mask = (
                tuple(int(pos >= sample.prompt_len) for pos in range(sample.seq_len))
                + (0,) * pad_count
            )
            position_ids = tuple(range(sample.seq_len)) + (0,) * pad_count
        else:
            ids = pad_tokens + sample.token_ids
            mask = (0,) * pad_count + (1,) * sample.seq_len
            rmap = pad_restore + logical_restore
            loss_mask = (0,) * pad_count + tuple(
                int(pos >= sample.prompt_len) for pos in range(sample.seq_len)
            )
            position_ids = (0,) * pad_count + tuple(range(sample.seq_len))
        physical_ids.append(ids)
        masks.append(mask)
        loss_masks.append(loss_mask)
        positions.append(position_ids)
        restore.append(rmap)

    return PaddedBatch(
        physical_token_ids=tuple(physical_ids),
        physical_attention_mask=tuple(masks),
        physical_loss_mask=tuple(loss_masks),
        physical_position_ids=tuple(positions),
        pad_side=pad_side,
        pad_token_id=pad_id,
        padded_len=target_len,
        restore_map=tuple(restore),
        sample_ids=batch.sample_ids,
    )


def restore_logical_order_from_padded(
    padded: PaddedBatch,
    physical_values: Sequence[Sequence[Any]],
) -> dict[tuple[str, int], Any]:
    """Map physical per-position values back to logical (sample_id, token_position)."""
    if len(physical_values) != len(padded.restore_map):
        raise WorkloadError("physical_values batch size mismatch")
    out: dict[tuple[str, int], Any] = {}
    for row_vals, row_map in zip(physical_values, padded.restore_map, strict=True):
        if len(row_vals) != len(row_map):
            raise WorkloadError("physical_values seq length mismatch")
        for val, key in zip(row_vals, row_map, strict=True):
            if key is None:
                continue
            if key in out:
                raise WorkloadError(f"duplicate logical key {key}")
            out[key] = val
    return out


def permute_batch(batch: LogicalBatch, permutation: Sequence[int]) -> LogicalBatch:
    n = len(batch.samples)
    if sorted(permutation) != list(range(n)):
        raise WorkloadError("permutation must be a permutation of sample indices")
    samples = tuple(batch.samples[i] for i in permutation)
    return LogicalBatch(
        workload_id=batch.workload_id,
        seed=batch.seed,
        samples=samples,
        cell_id=batch.cell_id,
    )


def batch_permutation_from_manifest(manifest: WS1Manifest | None = None) -> tuple[int, ...]:
    m = manifest if manifest is not None else load_manifest()
    perm = m.primary_matrix["batch_permutation"]
    return tuple(int(x) for x in perm["permutation"])


def singleton_aggregate_plan(
    batch: LogicalBatch,
    *,
    denominator: str = "active_token_count_across_all_samples",
) -> SingletonAggregatePlan:
    """N× B=1 schedule over the same multiset as one B=N run."""
    if not batch.samples:
        raise WorkloadError("empty batch")
    run_ids = tuple((s.sample_id,) for s in batch.samples)
    return SingletonAggregatePlan(
        sample_ids=batch.sample_ids,
        run_sample_ids=run_ids,
        aggregation_order=batch.sample_ids,
        denominator=denominator,
        token_multiset=batch.token_multiset(active_only=True),
    )


def same_logical_multiset(a: LogicalBatch, b: LogicalBatch, *, active_only: bool = True) -> bool:
    return a.token_multiset(active_only=active_only) == b.token_multiset(active_only=active_only)


def profile_required_nodes(
    manifest: WS1Manifest | None = None, profile_id: str = "cuda_bf16"
) -> list[dict[str, Any]]:
    m = manifest if manifest is not None else load_manifest()
    if profile_id not in m.backend_profiles:
        raise WorkloadError(f"unknown profile_id {profile_id!r}")
    return [dict(n) for n in m.backend_profiles[profile_id]["required_nodes"]]


def profile_missing_required_nodes(
    manifest: WS1Manifest | None = None, profile_id: str = "triton_cuda_bf16"
) -> list[str]:
    nodes = profile_required_nodes(manifest, profile_id)
    return [n["node"] for n in nodes if n.get("status") == "missing_required"]


def get_case(manifest: WS1Manifest | None = None, case_id: str = "") -> dict[str, Any]:
    m = manifest if manifest is not None else load_manifest()
    for case in m.representative_cases:
        if case["case_id"] == case_id:
            return dict(case)
    raise WorkloadError(f"unknown case_id {case_id!r}")


def case_ids(manifest: WS1Manifest | None = None) -> tuple[str, ...]:
    m = manifest if manifest is not None else load_manifest()
    return tuple(c["case_id"] for c in m.representative_cases)


def assert_no_undeclared_randomness(
    *,
    declared_rng_sources: Iterable[str],
    encountered_rng_sources: Iterable[str],
) -> None:
    """Gate helper: any RNG source not declared in the manifest hard-fails."""
    allowed = set(declared_rng_sources)
    bad = [s for s in encountered_rng_sources if s not in allowed]
    if bad:
        raise WorkloadError(f"undeclared stochastic source(s) {bad}; policy is hard_fail")


def fixture_hash(
    manifest: WS1Manifest | None = None,
    *,
    batch: LogicalBatch | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Stable hash of workload identity-defining fields and logical fixtures."""
    m = manifest if manifest is not None else load_manifest()
    logical = batch if batch is not None else build_logical_batch(m)
    payload = _manifest_identity_payload(m.raw)
    payload["selected_logical_batch"] = [list(x) for x in logical.token_multiset(active_only=False)]
    payload["extra"] = dict(extra) if extra else {}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _manifest_identity_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    # Hash every declared section so future manifest keys cannot escape identity.
    return {k: v for k, v in raw.items() if k != "fixture_identity_sha256"}


def manifest_identity_hash(raw: Mapping[str, Any]) -> str:
    blob = json.dumps(
        _manifest_identity_payload(raw), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _sequence_digest(values: Any) -> str:
    blob = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def weight_snapshot_hash(shards: Sequence[Mapping[str, Any]]) -> str:
    """Hash canonical filename/SHA-256/size records for all weight shards."""
    records = sorted((str(s["filename"]), str(s["sha256"]), int(s["size_bytes"])) for s in shards)
    blob = "".join(f"{name}\t{digest}\t{size}\n" for name, digest, size in records)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def reference_payload(
    manifest: WS1Manifest | None = None,
    *,
    cell_id: str | None = None,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Payload emitted by scripts/ws1_reference.py (no full-model forward)."""
    m = manifest if manifest is not None else load_manifest()
    if dtype not in {"bfloat16", "bf16", "float32", "fp32"}:
        raise WorkloadError(f"unsupported dtype {dtype!r}")
    norm_dtype = "bfloat16" if dtype in {"bfloat16", "bf16"} else "float32"
    batch = build_logical_batch(m, cell_id=cell_id)
    cell = get_matrix_cell(m, cell_id) if cell_id else None
    plan = singleton_aggregate_plan(batch)
    chunk = chunk_plan_from_manifest(m)
    chunked = apply_chunking(batch, chunk_size=chunk.chunk_size)
    packed = apply_packing(batch)
    padded_left = apply_padding(batch, pad_side="left", manifest=m)
    padded_right = apply_padding(batch, pad_side="right", manifest=m)
    return {
        "workload_id": m.workload_id,
        "seed": m.seed,
        "dtype": norm_dtype,
        "fixture_hash": fixture_hash(m, batch=batch),
        "clip_interval": list(m.clip_interval),
        "model_id": m.model_identity["model_id"],
        "revision": m.model_identity["revision"],
        "config_fingerprint": m.model_identity["config_fingerprint"],
        "weight_snapshot": m.model_identity["weight_snapshot"],
        "cell_id": cell_id,
        "cell": cell,
        "sample_ids": list(batch.sample_ids),
        "active_token_count": batch.active_token_count(),
        "singleton_aggregate": {
            "aggregation_order": list(plan.aggregation_order),
            "denominator": plan.denominator,
            "num_runs": len(plan.run_sample_ids),
            "token_multiset_len": len(plan.token_multiset),
        },
        "chunk_plan": {
            "seq_len": chunk.seq_len,
            "chunk_size": chunk.chunk_size,
            "num_chunks": chunk.num_chunks,
            "chunk_spans": [list(s) for s in chunk.chunk_spans],
        },
        "backend_profiles": list(m.backend_profiles.keys()),
        "backend_actual_semantics": m.chain_semantics["backend_actual_semantics"],
        "case_ids": list(case_ids(m)),
        "profile_missing_required": {
            pid: profile_missing_required_nodes(m, pid) for pid in m.backend_profiles
        },
        "reference_outputs": {
            "logical_token_ids_sha256": _sequence_digest(
                [list(s.token_ids) for s in batch.samples]
            ),
            "logical_loss_mask_sha256": _sequence_digest(
                [[int(t.is_active) for t in s.tokens()] for s in batch.samples]
            ),
            "padded_left_sha256": _sequence_digest(
                [
                    padded_left.physical_token_ids,
                    padded_left.physical_attention_mask,
                    padded_left.physical_loss_mask,
                    padded_left.physical_position_ids,
                ]
            ),
            "padded_right_sha256": _sequence_digest(
                [
                    padded_right.physical_token_ids,
                    padded_right.physical_attention_mask,
                    padded_right.physical_loss_mask,
                    padded_right.physical_position_ids,
                ]
            ),
            "chunked_sha256": _sequence_digest(
                [
                    chunked.physical_token_ids,
                    chunked.physical_loss_mask,
                    chunked.restore_map,
                    chunked.segment_offsets,
                    chunked.segment_lengths,
                ]
            ),
            "packed_sha256": _sequence_digest(
                [
                    packed.physical_token_ids,
                    packed.physical_loss_mask,
                    packed.restore_map,
                    packed.segment_offsets,
                    packed.segment_lengths,
                ]
            ),
            "short_fixture_sha256": _sequence_digest(m.fixtures["short_full_model_fixture"]),
            "long_fixture_sha256": _sequence_digest(m.fixtures["long_full_model_fixture"]),
        },
    }
