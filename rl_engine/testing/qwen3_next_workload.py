"""Qwen3-Next norm-only C3/C4 workload, explicitly separate from the Dense chain."""

from collections.abc import Mapping
from typing import Any

from rl_engine.testing.ws1_workload import (
    WorkloadError,
    _validate_backend_profiles,
    _validate_fixtures,
    _validate_logical_identity,
    _validate_model_identity,
    _validate_primary_matrix,
    _validate_stochastic_policy,
    manifest_identity_hash,
)

MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"
REVISION = "9c7f2fbe84465e40164a94cc16cd30b6999b0cc7"
FINGERPRINT = {
    "num_hidden_layers": 48,
    "hidden_size": 2048,
    "intermediate_size": 5120,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 151936,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "num_experts": 512,
    "num_experts_per_tok": 10,
}
NORM_OPS = {"qwen3_next_rms_norm": 2048, "rms_norm_gated": 128}


def validate_norm_manifest(raw: Mapping[str, Any]) -> None:
    identity = raw["model_identity"]
    if identity["model_id"] != MODEL_ID or identity["revision"] != REVISION:
        raise WorkloadError("Qwen3-Next norm workload requires the pinned official checkpoint")
    _validate_model_identity(identity, fingerprint=FINGERPRINT, model_label="Qwen3-Next")
    if (
        raw.get("scope") != "qwen3_next_norm_operators"
        or raw.get("full_model_evidence") is not False
    ):
        raise WorkloadError("norm workload cannot claim full-model evidence")
    _validate_stochastic_policy(raw["stochastic_policy"])
    _validate_primary_matrix(raw["primary_matrix"], raw["fixtures"])
    _validate_fixtures(raw["fixtures"], raw["primary_matrix"])
    _validate_logical_identity(raw["logical_identity"])
    caps = raw["capabilities"]
    if {e["op"] for e in caps["required_chain_ops"]} != set(NORM_OPS):
        raise WorkloadError("norm workload must contain exactly the two Qwen3-Next norms")
    if any(e["status"] != "required" for e in caps["required_chain_ops"]):
        raise WorkloadError("both norm operators are required")
    _validate_backend_profiles(raw["backend_profiles"], caps)
    cases = {case["case_id"]: case for case in raw["representative_cases"]}
    if len(cases) != len(raw["representative_cases"]):
        raise WorkloadError("duplicate norm case ID")
    referenced = set()
    for key in (
        "short_full_model_fixture",
        "long_full_model_fixture",
        "representative_full_model_fixture",
    ):
        fixture = raw["fixtures"][key]
        for case_id in fixture["candidate_case_ids"]:
            if case_id not in cases or cases[case_id]["fixture_id"] != fixture["fixture_id"]:
                raise WorkloadError("norm case fixture binding mismatch")
            referenced.add(case_id)
    if referenced != set(cases):
        raise WorkloadError("unreferenced norm case")
    if {case["operator_spec"] for case in cases.values()} != set(NORM_OPS):
        raise WorkloadError("representative cases must cover both norms")
    for case in cases.values():
        if case["hidden"] != NORM_OPS[case["operator_spec"]]:
            raise WorkloadError("norm case hidden dimension does not match checkpoint")
        if case["architecture_identity"] != "qwen3_next_80b_a3b_norm_operators":
            raise WorkloadError("norm cases cannot claim Dense or full-model architecture evidence")
    if raw["fixture_identity_sha256"] != manifest_identity_hash(raw):
        raise WorkloadError("Qwen3-Next fixture identity hash mismatch")


def validate_norm_dimensions(raw: Mapping[str, Any], op: str, hidden: int, head_dim: int) -> None:
    if raw.get("scope") != "qwen3_next_norm_operators":
        return
    if op not in NORM_OPS or hidden != 2048 or head_dim != 128:
        raise WorkloadError(
            "Qwen3-Next norm gate requires its two norm ops, --hidden 2048 --head-dim 128"
        )
