# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The norm workload cannot stand in for Dense or full-checkpoint evidence."""

import copy
from pathlib import Path

import pytest

from rl_engine.config.workload import WorkloadError, load_manifest, manifest_identity_hash
from rl_engine.validation.models.qwen3_next_workload import validate_norm_manifest
from rl_engine.validation.operators.gradient_adapters import get_adapter, resolve_profile_candidate

MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "rl_engine/validation/models/qwen3_next_norm_manifest.json"
)


def test_norm_manifest_pins_real_architecture_and_separate_scope():
    manifest = load_manifest(MANIFEST)
    assert manifest.model_identity["config_fingerprint"]["hidden_size"] == 2048
    assert manifest.raw["full_model_evidence"] is False
    for name, gradients in (
        ("qwen3_next_rms_norm", ("dx", "dweight")),
        ("rms_norm_gated", ("dx", "dweight", "dgate")),
    ):
        adapter = get_adapter(name)
        assert tuple(t.name for t in adapter.tensors) == gradients
        assert resolve_profile_candidate(adapter, "cuda_bf16", manifest)["status"] == "declared"
        assert (
            resolve_profile_candidate(adapter, "triton_cuda_bf16", manifest)["status"]
            == "missing_required"
        )
        assert (
            resolve_profile_candidate(adapter, "cuda_bf16", load_manifest())["status"]
            == "absent_not_required"
        )


@pytest.mark.parametrize("fault", ["architecture", "full_model", "revision", "shape", "binding"])
def test_norm_manifest_rejects_false_evidence_even_with_regenerated_hash(fault):
    raw = copy.deepcopy(load_manifest(MANIFEST).raw)
    if fault == "architecture":
        raw["model_identity"]["config_fingerprint"]["hidden_size"] = 4096
    elif fault == "full_model":
        raw["full_model_evidence"] = True
    elif fault == "revision":
        raw["model_identity"]["revision"] = "main"
    elif fault == "shape":
        raw["representative_cases"][0]["hidden"] = 64
    else:
        raw["representative_cases"][0]["fixture_id"] = "wrong_fixture"
    raw["fixture_identity_sha256"] = manifest_identity_hash(raw)
    with pytest.raises(WorkloadError):
        validate_norm_manifest(raw)


def test_norm_dimension_gate_rejects_shrunk_workload():
    from rl_engine.validation.models.qwen3_next_workload import validate_norm_dimensions

    raw = load_manifest(MANIFEST).raw
    with pytest.raises(WorkloadError, match="hidden 2048"):
        validate_norm_dimensions(raw, "qwen3_next_rms_norm", 64, 128)
    validate_norm_dimensions(raw, "rms_norm_gated", 2048, 128)
