# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""A label or environment flag cannot certify the CUSTOM provider."""

import copy
import sys
from types import SimpleNamespace

import pytest

from rl_engine.integrations.orchestrators.vime.experiments.qwen3_next import profile as profile


def test_qualification_never_advertises_batch_invariance(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    assert profile.activate_for_identity({}) == "qualification"
    assert not profile.profile_is_certified()


@pytest.mark.parametrize("mode", ["1", "", "true"])
def test_strict_flag_without_certificate_is_rejected(monkeypatch, mode):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", mode)
    with pytest.raises(ValueError, match="Uncertified"):
        profile.activate_for_identity({})
    assert not profile.profile_is_certified()


def test_passed_label_without_matching_certificate_bytes_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    result = tmp_path / "result.json"
    result.write_text('{"status": "passed"}')
    identity = {"scheduler_profile_certificate": {"path": str(result), "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="content identity"):
        profile.activate_for_identity(identity)
    assert not profile.profile_is_certified()


@pytest.fixture
def updated_identity():
    certified = {
        "checkpoint": "/official/revision",
        "checkpoint_identity": {"revision": "official", "metadata": {"tokenizer": "pinned"}},
        "source_commit": "commit",
        "source_tree": "tree",
        "environment_sha256": "environment",
        "provider_config": {"profile": "qwen3_next_tp4_shared_eager_v1"},
        "topology": {"tp": 4, "pp": 1, "cp": 1, "ep": 1},
        "weight_version": "baseline",
        "architecture_layers": 48,
        "weights_dtype": "bfloat16",
        "recurrent_state_dtype": "float32",
    }
    record = {
        "schema_version": 1,
        "kind": "qwen3_next_native_hf_export",
        "status": "complete",
        "weight_version": "update1",
        "step": 1,
        "export_id": "unit-test-only",
        "base_checkpoint": certified["checkpoint"],
        "base_checkpoint_identity": certified["checkpoint_identity"],
        "training_identity": copy.deepcopy(certified),
        "architecture_layers": 48,
        "weights_dtype": "bfloat16",
        "weights_content_sha256": "new-weight-content",
    }
    identity = {
        **copy.deepcopy(certified),
        "weight_version": "update1",
        "checkpoint": "/run/current-export",
        "checkpoint_identity": record,
        "base_checkpoint": certified["checkpoint"],
        "base_checkpoint_identity": certified["checkpoint_identity"],
    }
    return identity, certified


def test_certificate_rehashes_actual_updated_export_not_old_base(updated_identity, monkeypatch):
    identity, certified = updated_identity
    calls = []

    def check(path, expected):
        calls.append((path, expected))
        return expected

    monkeypatch.setitem(
        sys.modules, "qwen3_next_training_export", SimpleNamespace(validate_training_export=check)
    )
    profile.validate_certified_weights(identity, certified)
    assert calls == [(identity["checkpoint"], identity["checkpoint_identity"])]
    assert calls[0][1] != certified["checkpoint_identity"]
    assert not profile.profile_is_certified()  # A JSON-boundary unit check cannot activate BI.


@pytest.mark.parametrize(
    "field", ["base_checkpoint", "base_checkpoint_identity", "weight_version", "source_tree"]
)
def test_updated_identity_cannot_rebind_base_version_or_source(updated_identity, field):
    identity, certified = updated_identity
    identity[field] = "changed"
    with pytest.raises(ValueError):
        profile.validate_certified_weights(identity, certified)


def test_changed_actual_export_bytes_are_rejected(updated_identity, monkeypatch):
    identity, certified = updated_identity
    monkeypatch.setitem(
        sys.modules,
        "qwen3_next_training_export",
        SimpleNamespace(
            validate_training_export=lambda path, expected: {
                **expected,
                "weights_content_sha256": "different",
            },
        ),
    )
    with pytest.raises(ValueError, match="Actual updated weight files"):
        profile.validate_certified_weights(identity, certified)


def test_updated_export_cannot_be_labeled_as_baseline(updated_identity):
    identity, certified = updated_identity
    identity["weight_version"] = "baseline"
    with pytest.raises(ValueError, match="masquerade"):
        profile.validate_certified_weights(identity, certified)


def test_historical_export_identity_checks_do_not_open_deleted_shards(updated_identity):
    from rl_engine.validation.models.qwen3_next_l2 import validate_weight_identity

    identity, _ = updated_identity
    validate_weight_identity(identity)
    # No export files exist in this unit test; this verifies structural history
    # only. Actual profile activation always invokes the deep file validator.


def test_updated_model_identity_carries_actual_and_base_checkpoint(updated_identity):
    from qwen3_next_l2 import model_identity

    identity, _ = updated_identity
    model = model_identity(identity)
    assert model["checkpoint"] == "/run/current-export"
    assert model["checkpoint_identity"]["weights_content_sha256"] == "new-weight-content"
    assert model["base_checkpoint"] == "/official/revision"
    assert model["base_checkpoint_identity"] == {
        "revision": "official",
        "metadata": {"tokenizer": "pinned"},
    }
