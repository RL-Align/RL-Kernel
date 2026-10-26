# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The norm workload cannot stand in for Dense or full-checkpoint evidence."""

import copy
import json
from pathlib import Path

import pytest

from rl_engine.config.workload import WorkloadError, load_manifest, manifest_identity_hash
from rl_engine.validation.models.qwen3_next.qwen3_next_workload import validate_norm_manifest
from rl_engine.validation.operators.gradient_adapters import get_adapter, resolve_profile_candidate

MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "rl_engine/validation/models/qwen3_next/qwen3_next_norm_manifest.json"
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


@pytest.mark.parametrize(
    "missing", ["model_identity", "stochastic_policy", "primary_matrix", "fixtures", "capabilities"]
)
def test_norm_manifest_missing_block_raises_workload_error(missing):
    """A required block removed from a norm manifest is a WorkloadError, not a KeyError."""
    raw = copy.deepcopy(load_manifest(MANIFEST).raw)
    del raw[missing]
    with pytest.raises(WorkloadError, match="malformed"):
        validate_norm_manifest(raw)


@pytest.mark.parametrize(
    "op,hidden,head_dim",
    [
        ("qwen3_next_rms_norm", 64, 128),
        ("qwen3_next_rms_norm", 2048, 256),
        ("rms_norm_gated", 2048, 256),
        ("rms_norm_gated", 128, 128),
        ("rms_norm", 2048, 128),
    ],
)
def test_norm_dimension_gate_rejects_other_widths(op, hidden, head_dim):
    """Both gates run only at the checkpoint widths: hidden 2048, gated head_dim 128."""
    from rl_engine.validation.models.qwen3_next.qwen3_next_workload import validate_norm_dimensions

    raw = load_manifest(MANIFEST).raw
    with pytest.raises(WorkloadError, match="hidden 2048"):
        validate_norm_dimensions(raw, op, hidden, head_dim)


@pytest.mark.parametrize("op", ["qwen3_next_rms_norm", "rms_norm_gated"])
def test_norm_dimension_gate_accepts_checkpoint_widths(op):
    from rl_engine.validation.models.qwen3_next.qwen3_next_workload import validate_norm_dimensions

    validate_norm_dimensions(load_manifest(MANIFEST).raw, op, 2048, 128)


def test_norm_adapters_are_required_by_the_norm_manifest_only():
    """The Qwen3-Next norms are required where the manifest declares them, not in Dense."""
    from rl_engine.validation.operators.gradient_adapters import required_gradient_adapters

    norm = {spec.op_name for spec in required_gradient_adapters(load_manifest(MANIFEST))}
    assert norm == {"qwen3_next_rms_norm", "rms_norm_gated"}
    dense = {spec.op_name for spec in required_gradient_adapters(load_manifest())}
    assert dense.isdisjoint(norm)


@pytest.mark.parametrize(
    "source,scope",
    [(MANIFEST, "qwen3-next-norm-operators"), (None, "qwen3_next_norm_operator")],
)
def test_unknown_manifest_scope_is_rejected(tmp_path, source, scope):
    """A typo'd scope is a scope error, not a Dense fingerprint mismatch or a silent Dense."""
    raw = copy.deepcopy(load_manifest(source).raw)
    raw["scope"] = scope
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(WorkloadError, match="unknown manifest scope"):
        load_manifest(path)
