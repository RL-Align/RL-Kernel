# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 C2 (#268) canonical workload / logical identity tests (CPU-only)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rl_engine.kernels.gtest.operator_specs import OP_SPECS

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SCRIPT = REPO_ROOT / "scripts" / "ws1_reference.py"
CANDIDATE_EVIDENCE_SCRIPT = REPO_ROOT / "scripts" / "ws1_candidate_evidence.py"
CONTRACT_PATH = REPO_ROOT / "rl_engine/kernels/gtest/tolerance_contract.json"


def _load_pure_workload_module():
    path = REPO_ROOT / "rl_engine/testing/ws1_workload.py"
    spec = importlib.util.spec_from_file_location("_ws1_workload_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ws1 = _load_pure_workload_module()
WorkloadError = ws1.WorkloadError
WS1Manifest = ws1.WS1Manifest
apply_padding = ws1.apply_padding
apply_chunking = ws1.apply_chunking
apply_packing = ws1.apply_packing
assert_no_undeclared_randomness = ws1.assert_no_undeclared_randomness
batch_permutation_from_manifest = ws1.batch_permutation_from_manifest
build_chunk_plan = ws1.build_chunk_plan
build_logical_batch = ws1.build_logical_batch
case_ids = ws1.case_ids
chunk_plan_from_manifest = ws1.chunk_plan_from_manifest
default_manifest_path = ws1.default_manifest_path
fixture_hash = ws1.fixture_hash
get_case = ws1.get_case
get_matrix_cell = ws1.get_matrix_cell
load_manifest = ws1.load_manifest
matrix_cell_ids = ws1.matrix_cell_ids
permute_batch = ws1.permute_batch
profile_missing_required_nodes = ws1.profile_missing_required_nodes
profile_required_nodes = ws1.profile_required_nodes
reference_payload = ws1.reference_payload
restore_logical_order_from_padded = ws1.restore_logical_order_from_padded
restore_logical_order = ws1.restore_logical_order
same_logical_multiset = ws1.same_logical_multiset
singleton_aggregate_plan = ws1.singleton_aggregate_plan
validate_manifest = ws1.validate_manifest


def load_contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


REQUIRED_CELLS = {
    "B1-singleton_aggregate/full",
    "BN/full",
    "B1-singleton_aggregate/chunked",
    "BN/chunked",
}


@pytest.fixture(scope="module")
def manifest():
    return load_manifest()


def test_default_manifest_path_exists():
    path = default_manifest_path()
    assert path.is_file()
    assert path.name == "ws1_manifest.json"


def test_manifest_loads_and_validates(manifest):
    assert manifest.workload_id.startswith("ws1-qwen3-8b-dense")
    assert manifest.seed == 20260812
    validate_manifest(manifest.raw)


def test_model_identity_is_full_qwen3_8b(manifest):
    fp = manifest.model_identity["config_fingerprint"]
    assert fp["num_hidden_layers"] == 36
    assert fp["hidden_size"] == 4096
    assert fp["num_attention_heads"] == 32
    assert fp["num_key_value_heads"] == 8
    assert fp["head_dim"] == 128
    assert fp["vocab_size"] == 151936
    assert fp["intermediate_size"] == 12288
    assert fp["tie_word_embeddings"] is False
    assert fp["qk_norm"] is True
    assert manifest.model_identity["exit_forbids_architecture_shrink"] is True
    weight = manifest.model_identity["weight_snapshot"]
    assert weight["total_size_bytes"] > 0
    assert weight["pin_method"]
    assert len(weight["shards"]) == 5
    assert weight["content_hash"] == (
        "fc664a19c52c82b6f5ddb33d4fe2723181daeb93a344b16fee6369963e5a13a5"
    )


def test_clip_interval_pinned_and_aligns_with_c1(manifest):
    assert list(manifest.clip_interval) == [0.8, 1.2]
    contract = load_contract()
    # C1 stores default_clip_interval under chain_logprob_aggregates.
    c1_interval = contract["chain_logprob_aggregates"]["default_clip_interval"]
    assert list(c1_interval) == list(manifest.clip_interval)


def test_forbidden_comparison_roles_align_with_c1(manifest):
    forbidden = set(manifest.chain_semantics["forbidden_comparison_roles"])
    assert "baseline" in forbidden
    assert "singleton_aggregate" in forbidden
    contract = load_contract()
    c1_forbidden = set(contract["comparison_roles"]["forbidden"])
    assert forbidden == c1_forbidden


def test_primary_matrix_2x2_and_n(manifest):
    assert set(matrix_cell_ids(manifest)) == REQUIRED_CELLS
    assert int(manifest.primary_matrix["N"]) > 1
    for cell_id in REQUIRED_CELLS:
        cell = get_matrix_cell(manifest, cell_id)
        assert "batch_mode" in cell
        assert cell["batch_mode"] in {"singleton_aggregate", "batched"}
        # Naming boundary: never treat singleton_aggregate as a C1 role field.
        assert "comparison_lhs_role" not in cell
        assert "comparison_rhs_role" not in cell


def test_chunk_plan_multi_chunk_non_divisible(manifest):
    plan = chunk_plan_from_manifest(manifest)
    assert plan.num_chunks >= 2
    assert plan.seq_len % plan.chunk_size != 0
    # Reconstruct full coverage without overlap.
    covered = []
    for start, end in plan.chunk_spans:
        covered.extend(range(start, end))
    assert covered == list(range(plan.seq_len))


def test_logical_batch_reproducible_and_hash_stable(manifest):
    a = build_logical_batch(manifest)
    b = build_logical_batch(manifest)
    assert a.sample_ids == b.sample_ids
    assert a.token_multiset(active_only=False) == b.token_multiset(active_only=False)
    assert fixture_hash(manifest, batch=a) == fixture_hash(manifest, batch=b)
    assert len(fixture_hash(manifest)) == 64


def test_fixture_hash_covers_all_manifest_identity_fields(manifest):
    raw = json.loads(default_manifest_path().read_text(encoding="utf-8"))
    original = fixture_hash(manifest)
    raw["fixtures"]["loss_mask"]["prompt_tokens_active"] = True
    changed = WS1Manifest(raw=raw, path=default_manifest_path())
    assert fixture_hash(changed) != original


def test_same_workload_id_rejects_unversioned_manifest_change():
    raw = json.loads(default_manifest_path().read_text(encoding="utf-8"))
    raw["chain_semantics"]["temperature"] = 0.5
    with pytest.raises(WorkloadError, match="fixture_identity_sha256"):
        validate_manifest(raw)


def test_short_long_and_varlen_fixtures_are_materialized(manifest):
    fixtures = manifest.fixtures
    for name in ("short_full_model_fixture", "long_full_model_fixture"):
        fixture = fixtures[name]
        assert len(fixture["token_ids"]) == fixture["seq_len"]
        assert fixture["candidate_case_ids"]
    batch = build_logical_batch(manifest)
    assert [sample.seq_len for sample in batch.samples] == fixtures["varlen_seq_lens"]
    assert fixtures["prompt_lens"] == [sample.prompt_len for sample in batch.samples]
    assert fixtures["completion_lens"] == [
        sample.seq_len - sample.prompt_len for sample in batch.samples
    ]
    assert fixtures["max_completion_len"] == max(fixtures["completion_lens"])
    assert "primary_completion_len" not in fixtures


def test_chain_semantics_report_and_actual_boundaries(manifest):
    sem = manifest.chain_semantics
    assert "tolerance_contract.json" in sem["tf32_policy_ref"]
    assert set(sem["report_naming"]["forbidden_in_reports"]) >= {
        "baseline",
        "singleton_aggregate",
    }
    assert sem["report_naming"]["singleton_aggregate_is"] == "c2_execution_aggregation_mode_only"
    assert (
        sem["backend_actual_semantics"]["c2_representative_actual_source"]
        == "scripts/ws1_candidate_evidence.py runtime execution"
    )
    assert "C8" in sem["backend_actual_semantics"]["full_model_runtime_observed_actual_owner"]
    boundary = manifest.raw["provenance_boundary"]
    assert "full_model_forward" in boundary["not_in_c2"]


def test_stale_primary_completion_len_rejected():
    raw = json.loads(default_manifest_path().read_text(encoding="utf-8"))
    raw["fixtures"]["primary_completion_len"] = 8
    with pytest.raises(WorkloadError, match="primary_completion_len is forbidden"):
        validate_manifest(raw)


def test_active_tokens_are_completion_only(manifest):
    batch = build_logical_batch(manifest)
    for sample in batch.samples:
        for tok in sample.tokens():
            if tok.token_position < sample.prompt_len:
                assert not tok.is_active
            else:
                assert tok.is_active
    assert batch.active_token_count() == sum(
        sample.seq_len - sample.prompt_len for sample in batch.samples
    )


@pytest.mark.parametrize("pad_side", ["right", "left"])
def test_padding_restores_logical_identity(manifest, pad_side):
    batch = build_logical_batch(manifest)
    padded = apply_padding(batch, pad_side=pad_side, manifest=manifest)
    # Physical values encode a unique marker per logical key.
    physical_values = []
    for row_map in padded.restore_map:
        row = []
        for key in row_map:
            if key is None:
                row.append(None)
            else:
                row.append(f"{key[0]}@{key[1]}")
        physical_values.append(row)
    restored = restore_logical_order_from_padded(padded, physical_values)
    expected_keys = set(batch.logical_keys(active_only=False))
    assert set(restored.keys()) == expected_keys
    for sample in batch.samples:
        for pos in range(sample.seq_len):
            assert restored[(sample.sample_id, pos)] == f"{sample.sample_id}@{pos}"


def test_batch_permutation_restores_multiset(manifest):
    batch = build_logical_batch(manifest)
    perm = batch_permutation_from_manifest(manifest)
    permuted = permute_batch(batch, perm)
    assert permuted.sample_ids != batch.sample_ids

    # Multiset equality is order-sensitive in token_multiset (fixed order).
    # After sorting by sample_id, the pairs must match.
    def sorted_multiset(b):
        return tuple(sorted(b.token_multiset(active_only=True)))

    assert sorted_multiset(batch) == sorted_multiset(permuted)
    # Restoring original order via inverse permutation.
    inverse = [0] * len(perm)
    for new_i, old_i in enumerate(perm):
        inverse[old_i] = new_i
    # samples in permuted are batch.samples[perm[i]]; map back:
    restored_samples = []
    for old_i in range(len(batch.samples)):
        restored_samples.append(permuted.samples[inverse[old_i]])
    restored = ws1.LogicalBatch(
        workload_id=batch.workload_id,
        seed=batch.seed,
        samples=tuple(restored_samples),
    )
    assert same_logical_multiset(batch, restored)


def test_singleton_aggregate_matches_bn_multiset(manifest):
    bn = build_logical_batch(manifest, cell_id="BN/full")
    plan = singleton_aggregate_plan(bn)
    assert plan.sample_ids == bn.sample_ids
    assert len(plan.run_sample_ids) == len(bn.samples)
    assert all(len(run) == 1 for run in plan.run_sample_ids)
    # Rebuild B1 runs and concatenate multiset in fixed order.
    combined = []
    for (sid,) in plan.run_sample_ids:
        run = build_logical_batch(manifest, sample_ids=[sid])
        combined.extend(run.token_multiset(active_only=True))
    assert tuple(combined) == bn.token_multiset(active_only=True)
    assert tuple(combined) == plan.token_multiset


def test_chunk_positions_cover_logical_keys(manifest):
    batch = build_logical_batch(manifest)
    chunk_size = manifest.primary_matrix["chunk"]["chunk_size_tokens"]
    for sample in batch.samples:
        plan = build_chunk_plan(sample.seq_len, chunk_size)
        keys = []
        for start, end in plan.chunk_spans:
            for pos in range(start, end):
                keys.append((sample.sample_id, pos))
        expected = [(sample.sample_id, pos) for pos in range(sample.seq_len)]
        assert keys == expected


def test_chunk_and_pack_layouts_restore_identity(manifest):
    batch = build_logical_batch(manifest)
    chunked = apply_chunking(batch, chunk_size=7)
    packed = apply_packing(batch)
    for layout in (chunked, packed):
        values = [f"{sid}@{pos}" for sid, pos in layout.restore_map]
        restored = restore_logical_order(layout, values)
        assert set(restored) == set(batch.logical_keys())
        assert len(layout.physical_token_ids) == len(layout.restore_map)
    assert chunked.segment_lengths[-1] == 5
    assert packed.segment_lengths == (11, 16, 13, 19)


def test_stochastic_policy_hard_fails_undeclared_rng(manifest):
    policy = manifest.raw["stochastic_policy"]
    assert policy["dropout"] == 0.0
    assert policy["sampling_in_logprob_parity"] is False
    assert policy["undeclared_randomness"] == "hard_fail"
    declared = {policy["rng_source"]}
    assert_no_undeclared_randomness(
        declared_rng_sources=declared,
        encountered_rng_sources=[policy["rng_source"]],
    )
    with pytest.raises(WorkloadError, match="undeclared stochastic"):
        assert_no_undeclared_randomness(
            declared_rng_sources=declared,
            encountered_rng_sources=["torch.randn_unseeded"],
        )


def test_backend_profiles_enumerate_required_nodes(manifest):
    for profile_id in ("cuda_bf16", "triton_cuda_bf16"):
        nodes = profile_required_nodes(manifest, profile_id)
        names = {n["node"] for n in nodes}
        for required in (
            "embedding",
            "rms_norm",
            "det_gemm",
            "attention",
            "rope",
            "swiglu",
            "lm_head",
            "logprob",
            "batch_invariant_logp",
        ):
            assert required in names
        for node in nodes:
            assert node["status"] in {"declared", "missing_required"}
            if node["status"] == "declared":
                assert node["expected_backend_id"]
                assert node["expected_kernel_config_id"]
                assert node["algorithm_property"]


def test_triton_profile_has_all_required_candidates(manifest):
    missing = profile_missing_required_nodes(manifest, "triton_cuda_bf16")
    assert missing == []


def test_representative_cases_stable_ids_and_pins(manifest):
    ids = case_ids(manifest)
    assert len(ids) == len(set(ids))
    families = {get_case(manifest, cid)["family"] for cid in ids}
    assert {"gemm", "attention", "logprob"} <= families
    for cid in ids:
        case = get_case(manifest, cid)
        assert case["architecture_identity"] == "full_qwen3_8b_dense"
        assert case["expected_backend_id"] == case["actual_backend_id"]
        assert case["expected_kernel_config_id"] == case["actual_kernel_config_id"]
        assert case["provenance_status"] == "runtime_evidence_required"
        assert case["provenance_evidence"]["resolved_path"] == case["actual_kernel_config_id"]
        assert case["provenance_evidence"]["runtime_evidence_command"]
        assert case["algorithm_property"]
        assert "shape" in case
    for profile in ("cuda_bf16", "triton_cuda_bf16"):
        cases = [
            get_case(manifest, cid)
            for cid in ids
            if profile in get_case(manifest, cid)["profile_ids"]
        ]
        assert {"gemm", "attention", "logprob"} <= {c["family"] for c in cases}
        assert len({c["shape"]["M"] for c in cases if c["family"] == "gemm"}) >= 2
        attention_modes = {c["shape"]["mode"] for c in cases if c["family"] == "attention"}
        assert attention_modes == {"prefill", "decode"}


def test_fixture_case_shapes_are_derived_from_fixed_fixtures(manifest):
    fixtures = manifest.fixtures
    cases = {case["case_id"]: case for case in manifest.representative_cases}
    for fixture_name in (
        "short_full_model_fixture",
        "long_full_model_fixture",
        "representative_full_model_fixture",
    ):
        fixture = fixtures[fixture_name]
        for case_id in fixture["candidate_case_ids"]:
            assert cases[case_id]["fixture_id"] == fixture["fixture_id"]

    short_cases = [
        cases[case_id] for case_id in fixtures["short_full_model_fixture"]["candidate_case_ids"]
    ]
    assert {case["shape"]["M"] for case in short_cases if case["family"] == "gemm"} == {8}
    assert {case["shape"]["T"] for case in short_cases if case["family"] == "logprob"} == {4}
    primary_cases = [
        cases[case_id]
        for case_id in fixtures["representative_full_model_fixture"]["candidate_case_ids"]
    ]
    assert {case["shape"]["M"] for case in primary_cases if case["family"] == "gemm"} == {59}
    assert {
        (case["shape"]["B"], case["shape"]["Sq"], case["shape"]["Skv"])
        for case in primary_cases
        if case["family"] == "attention"
    } == {(4, 19, 19)}


def test_declared_candidates_resolve_to_real_operator_specs(manifest):
    spec_map = manifest.raw["capabilities"]["operator_spec_map"]
    for node, spec_name in spec_map.items():
        assert spec_name in OP_SPECS, node
    for case in manifest.representative_cases:
        evidence = case["provenance_evidence"]
        spec = OP_SPECS[case["operator_spec"]]
        assert spec.candidate_paths[evidence["candidate_name"]] == evidence["resolved_path"]
        algorithm_path, symbol = evidence["algorithm_source"].rsplit(":", 1)
        algorithm_file = REPO_ROOT / algorithm_path
        assert algorithm_file.is_file()
        assert symbol in algorithm_file.read_text(encoding="utf-8")


def test_capabilities_packing_and_qk_norm(manifest):
    caps = manifest.raw["capabilities"]
    assert caps["qk_norm"]["status"] == "required_on_chain"
    packing = caps["packing"]
    assert packing["status"] == "supported"
    assert manifest.fixtures["packing"]["packed_fixture"]["total_tokens"] == 59


def test_missing_weight_hash_rejected():
    raw = json.loads(default_manifest_path().read_text(encoding="utf-8"))
    del raw["model_identity"]["weight_snapshot"]["content_hash"]
    with pytest.raises(WorkloadError, match="content_hash"):
        validate_manifest(raw)


def test_packing_cannot_be_marked_na_when_supported():
    raw = json.loads(default_manifest_path().read_text(encoding="utf-8"))
    raw["fixtures"]["packing"]["status"] = "n_a_with_capability_proof"
    with pytest.raises(WorkloadError, match="must pin a supported packed fixture"):
        validate_manifest(raw)


def test_architecture_shrink_rejected():
    raw = json.loads(default_manifest_path().read_text(encoding="utf-8"))
    raw["model_identity"]["config_fingerprint"]["num_hidden_layers"] = 2
    with pytest.raises(WorkloadError, match="does not match official"):
        validate_manifest(raw)


def test_missing_matrix_cell_rejected():
    raw = json.loads(default_manifest_path().read_text(encoding="utf-8"))
    raw["primary_matrix"]["cells"] = raw["primary_matrix"]["cells"][:3]
    with pytest.raises(WorkloadError, match=r"primary_matrix\.cells"):
        validate_manifest(raw)


def test_reference_payload_contains_required_fields(manifest):
    payload = reference_payload(manifest, cell_id="BN/full", dtype="bf16")
    assert payload["workload_id"] == manifest.workload_id
    assert payload["seed"] == manifest.seed
    assert payload["dtype"] == "bfloat16"
    assert payload["fixture_hash"] == fixture_hash(manifest)
    assert payload["cell_id"] == "BN/full"
    assert payload["clip_interval"] == [0.8, 1.2]
    assert "c2_representative_actual_source" in payload["backend_actual_semantics"]


def test_ws1_reference_cli_emits_identity():
    proc = subprocess.run(
        [
            sys.executable,
            str(REFERENCE_SCRIPT),
            "--dtype",
            "bf16",
            "--cell-id",
            "BN/full",
            "--emit-json",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "workload_id" in payload
    assert "seed" in payload
    assert payload["dtype"] == "bfloat16"
    assert len(payload["fixture_hash"]) == 64


def test_candidate_evidence_cli_help_is_available():
    proc = subprocess.run(
        [sys.executable, str(CANDIDATE_EVIDENCE_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "representative candidates on a real accelerator" in proc.stdout


def test_build_chunk_plan_edges():
    plan = build_chunk_plan(16, 7)
    assert plan.chunk_spans == ((0, 7), (7, 14), (14, 16))
    with pytest.raises(WorkloadError):
        build_chunk_plan(8, 0)
