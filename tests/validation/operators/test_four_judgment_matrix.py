# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU tests for the WS1 C8 four-judgment matrix schema."""

from __future__ import annotations

import json
from pathlib import Path

from rl_engine.config.workload import load_manifest
from rl_engine.validation.operators.four_judgment_matrix import (
    C8_REQUIRED_OPS,
    JUDGMENTS,
    PROFILES,
    TIERS,
    build_classified_matrix,
    hidden_required_na,
    undefined_cells,
)

_EXECUTE_ARTIFACT = Path(__file__).resolve().parents[3] / "docs/archive/design/ws1-c8-execute.json"


def test_matrix_covers_required_ops_profiles_judgments_and_tiers():
    report = build_classified_matrix()
    keys = {(cell.profile, cell.op_name, cell.judgment, cell.tier) for cell in report.cells}
    expected = {
        (profile, op_name, judgment, tier)
        for profile in PROFILES
        for op_name in C8_REQUIRED_OPS
        for judgment in JUDGMENTS
        for tier in TIERS
    }
    assert keys == expected
    assert undefined_cells(report) == ()


def test_triton_required_candidates_are_declared():
    report = build_classified_matrix()
    declared = [
        cell
        for cell in report.cells
        if cell.profile == "triton_cuda_bf16" and cell.op_name in {"embedding", "lm_head", "logp"}
    ]
    assert declared
    assert all(cell.candidate == "triton" for cell in declared)
    assert all(cell.case_id for cell in declared)
    # Classify-only still paints declared-but-unexecuted required cells red.
    assert all(cell.status == "red" for cell in declared)
    assert hidden_required_na(report) == ()


def test_logp_and_batch_invariant_logp_have_own_case_ids():
    report = build_classified_matrix()
    for op_name in ("logp", "batch_invariant_logp"):
        cells = [cell for cell in report.cells if cell.op_name == op_name and cell.status != "N/A"]
        assert cells
        assert all(cell.case_id for cell in cells if cell.status != "pending_hopper")
        assert not any(
            cell.case_id and "batch-invariant" in cell.case_id and op_name == "logp"
            for cell in cells
        )


def test_pack_is_explicit_na_with_c2_reason():
    report = build_classified_matrix()
    pack = [cell for cell in report.cells if cell.op_name == "pack"]
    assert pack
    assert all(cell.status == "N/A" for cell in pack)
    assert all("profile-independent" in cell.detail for cell in pack)


def test_hidden_required_na_detects_unreasoned_na_status():
    from rl_engine.validation.operators.four_judgment_matrix import MatrixCell, MatrixReport

    report = MatrixReport(
        cells=(
            MatrixCell(
                profile="cuda_bf16",
                op_name="silu",
                judgment="forward_accuracy",
                tier="short",
                case_id=None,
                status="N/A",
                detail="silently skipped without C2 reason",
            ),
            MatrixCell(
                profile="cuda_bf16",
                op_name="linear_logp",
                judgment="forward_accuracy",
                tier="short",
                case_id=None,
                status="N/A",
                detail="optional_fused with no C2 required node",
            ),
        )
    )
    hidden = hidden_required_na(report)
    assert len(hidden) == 1
    assert hidden[0].op_name == "silu"


def test_sm90_declared_cells_are_pending_hopper():
    report = build_classified_matrix()
    hopper = [
        cell
        for cell in report.cells
        if cell.profile == "cuda_bf16"
        and cell.op_name in {"embedding", "lm_head", "rope", "batch_invariant_logp"}
    ]
    assert hopper
    assert all(cell.status == "pending_hopper" for cell in hopper if cell.case_id is not None)
    assert all(cell.candidate == "cuda-sm90" for cell in hopper)


def test_declared_runnable_ops_have_short_and_primary_case_ids():
    manifest = load_manifest()
    report = build_classified_matrix(manifest)
    runnable = {
        "rms_norm",
        "qk_norm",
        "det_gemm",
        "attention",
        "silu",
        "swiglu",
    }
    for cell in report.cells:
        if cell.op_name not in runnable:
            continue
        if cell.status == "pending_hopper":
            continue
        assert cell.case_id, (cell.op_name, cell.profile, cell.tier)
        assert any(case["case_id"] == cell.case_id for case in manifest.representative_cases)


def test_triton_rope_has_case_ids_but_cuda_rope_is_hopper():
    report = build_classified_matrix()
    triton_rope = [
        cell
        for cell in report.cells
        if cell.op_name == "rope" and cell.profile == "triton_cuda_bf16"
    ]
    cuda_rope = [
        cell for cell in report.cells if cell.op_name == "rope" and cell.profile == "cuda_bf16"
    ]
    assert all(cell.case_id for cell in triton_rope)
    assert all(cell.status == "pending_hopper" for cell in cuda_rope)


def test_checked_in_execute_matrix_has_zero_red():
    payload = json.loads(_EXECUTE_ARTIFACT.read_text(encoding="utf-8"))
    cells = payload["cells"]
    assert cells
    statuses = {cell["status"] for cell in cells}
    assert statuses <= {"green", "N/A"}
    assert payload["counts"].get("red", 0) == 0
    assert payload["counts"]["green"] == 176
    assert payload["counts"]["N/A"] == 16
    pack = [cell for cell in cells if cell["op_name"] == "pack"]
    assert pack and all(cell["status"] == "N/A" for cell in pack)
    required = [cell for cell in cells if cell["op_name"] != "pack"]
    assert all(cell["status"] == "green" for cell in required)
    assert all(cell["judgment"] in JUDGMENTS for cell in cells)
    if payload.get("schema_version") == "ws1-c8-execute-v2":
        invariance = [cell for cell in required if cell["judgment"].endswith("invariance")]
        assert invariance
        assert all(cell["actual_backend_id"] for cell in invariance)
        assert all(cell["actual_kernel_config_id"] for cell in invariance)
        assert payload["git"]["commit"]
        assert payload["environment"]["gpu_name"]
        assert payload["workload"]["workload_id"]
