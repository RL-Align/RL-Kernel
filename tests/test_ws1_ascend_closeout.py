# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""WS1 #266 closeout wiring for the Ascend BF16 profile (CPU-only).

These tests do not need an NPU. They assert that ``ascend_bf16`` is a
first-class required profile everywhere C1-C11 look, that every declared
candidate names a real importable object, and that the gates fail closed
rather than borrowing another vendor's kernels when no NPU is present.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from rl_engine.kernels.gtest.accelerator import (
    ACCELERATOR_TYPES,
    AcceleratorUnavailable,
    candidate_family,
    device_type_for_profile,
    disable_tf32,
    family_for_profile,
    is_available,
    resolve_device,
)
from rl_engine.kernels.gtest.elementwise_inventory import inventory_items, unresolved_needs_fix
from rl_engine.kernels.gtest.four_judgment_matrix import (
    C8_REQUIRED_OPS,
    JUDGMENTS,
    PROFILES,
    TIERS,
    build_classified_matrix,
    hidden_required_na,
    undefined_cells,
)
from rl_engine.kernels.gtest.gradient_adapters import (
    GRADIENT_ADAPTERS,
    gradient_adapter_status_matrix,
    resolve_profile_candidate,
)
from rl_engine.kernels.gtest.operator_specs import OP_SPECS
from rl_engine.kernels.gtest.tolerance import (
    BackendProvenance,
    ContractResolveError,
    load_contract,
    resolve_dtype_policy,
    validate_backend_provenance,
)
from rl_engine.testing.ws1_workload import (
    load_manifest,
    manifest_identity_hash,
    profile_required_nodes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = "ascend_bf16"
REQUIRED_CHAIN_NODES = (
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


def _load_object(path: str):
    module_path, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), name)


# --------------------------------------------------------------------------
# C1 (#267): contract
# --------------------------------------------------------------------------


def test_c1_contract_declares_ascend_as_a_required_profile():
    contract = load_contract()
    policy = resolve_dtype_policy(contract)
    assert PROFILE in policy.backend_profiles
    contracts = contract["policy"]["backend_profile_contracts"]
    assert contracts[PROFILE]["backend_family"] == "ascend"


def test_c1_ascend_provenance_validates_and_rejects_borrowed_backends():
    contract = load_contract()

    def provenance(actual: str) -> BackendProvenance:
        return BackendProvenance(
            backend_profile=PROFILE,
            requested_backend="ascend",
            actual_backend=actual,
            execution_dtype="bfloat16",
            accumulation_dtype="float32",
            output_dtype="bfloat16",
            reference_dtype="float32",
            candidate_tf32_enabled=False,
            reference_tf32_enabled=False,
        )

    validate_backend_provenance(contract, provenance("ascend"))
    # Reporting a CUDA kernel under the Ascend profile is the undeclared
    # fallback C1 exists to catch.
    with pytest.raises(ContractResolveError):
        validate_backend_provenance(contract, provenance("cuda"))


def test_c1_ascend_has_no_private_tolerance_relaxation():
    policy = resolve_dtype_policy(load_contract())
    assert policy.backend_private_tolerance_relaxation is False
    assert policy.execution_dtype == "bfloat16"
    assert policy.reference_dtype == "float32"


# --------------------------------------------------------------------------
# C2 (#268): workload manifest
# --------------------------------------------------------------------------


def test_c2_manifest_declares_every_required_ascend_node():
    manifest = load_manifest()
    assert PROFILE in manifest.backend_profiles
    profile = manifest.backend_profiles[PROFILE]
    assert profile["backend_family"] == "ascend"
    assert profile["execution_dtype"] == "bfloat16"
    nodes = {n["node"]: n for n in profile_required_nodes(manifest, PROFILE)}
    assert set(nodes) == set(REQUIRED_CHAIN_NODES)
    for node in nodes.values():
        assert node["status"] == "declared", node
        assert node["expected_backend_id"] == "ascend", node
        assert node["expected_kernel_config_id"], node
        assert node["algorithm_property"], node


def test_c2_identity_hash_covers_the_added_profile():
    manifest = load_manifest()
    assert manifest.raw["fixture_identity_sha256"] == manifest_identity_hash(manifest.raw)


def test_c2_ascend_representative_cases_mirror_the_cuda_tiers():
    manifest = load_manifest()
    by_profile: dict[str, set[tuple[str, str]]] = {}
    for case in manifest.representative_cases:
        op = str(case.get("op_name") or case["operator_spec"])
        fixture = str(case["fixture_id"])
        tier = (
            "short"
            if fixture.startswith("short_")
            else "primary" if fixture.startswith("rep_") else fixture
        )
        for profile in case["profile_ids"]:
            by_profile.setdefault(profile, set()).add((op, tier))
    assert by_profile["cuda_bf16"] == by_profile[PROFILE]


def test_c2_ascend_cases_pin_real_ascend_kernels_and_sources():
    manifest = load_manifest()
    cases = [c for c in manifest.representative_cases if PROFILE in c["profile_ids"]]
    assert cases
    for case in cases:
        assert case["expected_backend_id"] == "ascend"
        assert case["actual_backend_id"] == case["expected_backend_id"]
        evidence = case["provenance_evidence"]
        assert evidence["candidate_name"] == "ascend"
        assert evidence["resolved_path"] == case["actual_kernel_config_id"]
        # The algorithm source must be an .asc kernel that exists in-tree.
        source = evidence["algorithm_source"]
        path, _, symbol = source.partition(":")
        assert path.endswith(".asc"), source
        assert (REPO_ROOT / path).is_file(), source
        assert symbol in (REPO_ROOT / path).read_text(encoding="utf-8"), source
        assert "--device npu" in evidence["runtime_evidence_command"]


# --------------------------------------------------------------------------
# C3 / C4 (#269, #270): harness adapters
# --------------------------------------------------------------------------


def test_c3_c4_every_required_adapter_resolves_an_ascend_candidate():
    manifest = load_manifest()
    for name, adapter in GRADIENT_ADAPTERS.items():
        if adapter.requirement not in ("required",):
            continue
        resolved = resolve_profile_candidate(adapter, PROFILE, manifest)
        assert resolved["status"] == "declared", name
        assert resolved["expected_backend_id"] == "ascend", name
        assert resolved["candidate_path"], name
        assert candidate_family(str(resolved["expected_backend_id"])) == "ascend"


def test_c4_adapter_status_matrix_has_no_red_ascend_rows():
    rows = [r for r in gradient_adapter_status_matrix() if r.backend_profile == PROFILE]
    assert rows
    assert not [r for r in rows if r.tracked_red or r.untracked_red]


def test_operator_specs_expose_an_importable_ascend_candidate_per_chain_node():
    manifest = load_manifest()
    spec_map = manifest.raw["capabilities"]["operator_spec_map"]
    for node in REQUIRED_CHAIN_NODES:
        spec = OP_SPECS[spec_map[node]]
        assert "ascend" in spec.candidate_paths, node
        path = spec.candidate_paths["ascend"]
        # Import the class without constructing it: the constructors demand a
        # compiled _C_npu, which a CPU test host does not have.
        assert _load_object(path).__name__, path


# --------------------------------------------------------------------------
# C5 (#271): elementwise / RoPE residual inventory
# --------------------------------------------------------------------------


def test_c5_inventory_carries_an_ascend_verdict_with_no_blockers():
    items = inventory_items()
    assert items
    for item in items:
        assert item.ascend_verdict in (
            "pass",
            "blocker",
            "blocked_hardware",
            "tracked_red",
            "absent_not_required",
        )
        assert "ascend_verdict" in item.to_dict()
    assert unresolved_needs_fix() == ()


# --------------------------------------------------------------------------
# C8 (#274): four-judgment matrix
# --------------------------------------------------------------------------


def test_c8_matrix_includes_ascend_in_the_required_profiles():
    assert PROFILE in PROFILES
    report = build_classified_matrix()
    keys = {
        (c.profile, c.op_name, c.judgment, c.tier) for c in report.cells if c.profile == PROFILE
    }
    assert keys == {
        (PROFILE, op, judgment, tier)
        for op in C8_REQUIRED_OPS
        for judgment in JUDGMENTS
        for tier in TIERS
    }
    assert undefined_cells(report) == ()
    assert not [c for c in hidden_required_na(report) if c.profile == PROFILE]


def test_c8_matrix_can_be_scoped_to_one_hosts_profiles():
    # A GPU host cannot execute the Ascend cells and vice versa, so each
    # vendor's CI job sweeps its own profiles.
    report = build_classified_matrix(profiles=(PROFILE,))
    assert {c.profile for c in report.cells} == {PROFILE}


def test_c8_ascend_cells_are_never_silently_na():
    report = build_classified_matrix(profiles=(PROFILE,))
    for cell in report.cells:
        if cell.op_name == "pack":
            continue
        assert cell.status != "N/A" or "optional_fused" in (cell.detail or "")


# --------------------------------------------------------------------------
# C9-C11: device abstraction and CI wiring
# --------------------------------------------------------------------------


def test_accelerator_maps_the_ascend_profile_to_the_npu():
    assert device_type_for_profile(PROFILE) == "npu"
    assert family_for_profile(PROFILE) == "ascend"
    assert device_type_for_profile("cuda_bf16") == "cuda"
    assert family_for_profile("triton_cuda_bf16") == "triton"
    assert "npu" in ACCELERATOR_TYPES


def test_candidate_family_maps_ascend_ids():
    assert candidate_family("ascend") == "ascend"
    assert candidate_family("npu") == "ascend"
    assert candidate_family("cuda-sm90") == "cuda"
    assert candidate_family("triton") == "triton"


def test_resolve_device_fails_closed_without_an_npu():
    if is_available("npu"):
        pytest.skip("this host has an NPU; the fail-closed path cannot be exercised")
    with pytest.raises(AcceleratorUnavailable):
        resolve_device(None, profile=PROFILE)
    # Pointing an Ascend profile at a CUDA device is the cross-vendor fallback
    # the contract forbids, and is rejected before any device probe.
    with pytest.raises(AcceleratorUnavailable):
        resolve_device("cuda:0", profile=PROFILE)


def test_tf32_policy_holds_on_npu_without_a_tf32_switch():
    # Ascend has no TF32 mode, so the contract's "disabled" clause is satisfied
    # by construction and the reported flag must still be False.
    assert disable_tf32("npu") is False


@pytest.mark.parametrize(
    "script",
    [
        "scripts/check_forward_invariance.py",
        "scripts/check_gradient_invariance.py",
        "scripts/check_decode_prefill.py",
        "scripts/check_stateful_kv.py",
        "scripts/ws1_chain_gate.py",
        "scripts/ws1_chain_fwd_bwd.py",
        "scripts/ws1_candidate_evidence.py",
    ],
)
def test_c3_to_c10_clis_accept_the_ascend_profile(script):
    source = (REPO_ROOT / script).read_text(encoding="utf-8")
    assert PROFILE in source, f"{script} does not offer {PROFILE}"


def test_c11_ascend_ci_entry_points_exist_and_target_the_ascend_profile():
    ci_script = REPO_ROOT / "ci" / "run_ws1_ascend_ci.sh"
    assert ci_script.is_file()
    body = ci_script.read_text(encoding="utf-8")
    for expected in (
        "--backend-profile ascend_bf16",
        "--profile ascend_bf16",
        "KERNEL_ALIGN_FORCE_ASCEND=1",
        "run_ws1_chain_gate.sh",
    ):
        assert expected in body, expected

    workflow = REPO_ROOT / ".github" / "workflows" / "ws1-chain-npu.yml"
    assert workflow.is_file()
    workflow_body = workflow.read_text(encoding="utf-8")
    assert "run_ws1_ascend_ci.sh" in workflow_body
    assert "pull_request_target:" not in workflow_body


def test_c11_chain_gate_script_is_profile_parameterised():
    body = (REPO_ROOT / "ci" / "run_ws1_chain_gate.sh").read_text(encoding="utf-8")
    assert "WS1_PROFILES" in body
    # The embedded verifier must know the Ascend family, or an Ascend run would
    # be checked against the wrong backward provenance.
    assert re.search(r'"ascend_bf16":\s*"ascend"', body)
