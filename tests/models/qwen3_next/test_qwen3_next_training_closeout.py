# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU recovery/audit regressions; these never substitute for native GPU updates."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rl_engine.models.qwen3_next.qwen3_next_subnode import SUBNODES


def test_requeue_repeats_final_reload_without_reusing_stale_attempt(monkeypatch):
    import qwen3_next_training as runner

    journal = {
        "schema_version": 1,
        "identity": {},
        "events": list(runner.EVENTS),
        "recovery": [],
        "workers": {phase: {"path": phase} for phase in runner.PHASES},
        "l2_reports": {
            phase: {"path": phase} for phase in ("baseline", "update1", "update2", "reload")
        },
    }
    monkeypatch.setattr(runner, "read_record", lambda record: {"attempt": "0"})
    assert runner.repeat_reload_in_current_attempt(journal, "1")
    runner.validate_journal(journal, {}, verify_files=False)
    assert set(journal["workers"]) == {"update1", "update2"}
    assert set(journal["l2_reports"]) == {"baseline", "update1", "update2"}
    retired = journal["recovery"][0]
    assert retired["from_attempt"] == "0" and retired["to_attempt"] == "1"
    assert set(retired["workers"]) == {"reload", "continuous", "resumed"}
    assert not runner.repeat_reload_in_current_attempt(journal, "1")


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models/qwen3_next"))
import qwen3_next_training as runner  # noqa: E402

from rl_engine.integrations.orchestrators.vime.experiments.qwen3_next import (  # noqa: E402
    native_training as native,
)
from rl_engine.integrations.orchestrators.vime.providers import (  # noqa: E402
    qwen3_next_training as observer,
)


def parameter_names():
    return [
        f"chunk0.decoder.layers.{layer}.{component}.weight"
        for layer in range(48)
        for component in ("self_attention", "mlp", "pre_mlp_layernorm")
    ] + [
        f"chunk0.{component}.weight"
        for component in ("embedding", "output_layer", "final_layernorm")
    ]


def audit(step=1):
    names = parameter_names()
    before = {name: "1" * 64 for name in names}
    after = {name: "2" * 64 for name in names}
    return {
        "step": step,
        "sample_position": step * 4,
        "fixture": "a" * 64,
        "policy_loss_calls": 4,
        "grad_norm": 0.25,
        "metrics": {"loss": -0.25},
        "parameter_hashes_before": before,
        "parameter_hashes_after": after,
        "changed_parameters": names,
        "gradients": {
            "parameters": {
                name: {"present": True, "nonzero": True, "dtype": "torch.float32"} for name in names
            },
            "active_groups": {name: [name] for name in names},
        },
        "progress": {
            "consumed_before": (step - 1) * 4,
            "consumed_after": step * 4,
            "scheduler_before": (step - 1) * 4,
            "scheduler_after": step * 4,
            "sample_counter_owner": "qwen3_next_native_training.policy_step",
        },
    }


def state():
    value = {
        "schema_version": 1,
        "scope": "native_vime_training_state",
        "rank": 0,
        "identity": {"fixture_sha256": "a" * 64},
        "step": 1,
        "sample_position": 4,
        "consumed_train_samples": 4,
        "fixture_sha256": "a" * 64,
        "model_sha256": ["b" * 64],
        "optimizer_parameter_sha256": {
            name: {kind: "c" * 64 for kind in ("param", "exp_avg", "exp_avg_sq")}
            for name in parameter_names()
        },
        "optimizer_header_sha256": "d" * 64,
        "scheduler_sha256": "e" * 64,
        "rng_sha256": "f" * 64,
    }
    return seal_state(value)


def seal_state(value):
    value["state_sha256"] = observer.state_digest(
        {key: item for key, item in value.items() if key != "state_sha256"}
    )
    return value


def validate_state(value):
    runner.validate_state(
        value, rank=0, step=1, identity={"fixture_sha256": "a" * 64}, parameters=parameter_names()
    )


def test_full_actor_update_and_recovery_state_contract():
    runner.validate_update(audit(), step=1, fixture="a" * 64)
    validate_state(state())


@pytest.mark.parametrize(
    "damage", ["no_change", "zero_gradient", "missing_layer", "nan_loss", "bad_progress"]
)
def test_update_rejects_fake_or_incomplete_actor_evidence(damage):
    value = audit()
    name = parameter_names()[0]
    if damage == "no_change":
        value["parameter_hashes_after"] = dict(value["parameter_hashes_before"])
        value["changed_parameters"] = []
    elif damage == "zero_gradient":
        value["gradients"]["parameters"][name]["nonzero"] = False
    elif damage == "missing_layer":
        for key in ("parameter_hashes_before", "parameter_hashes_after"):
            value[key] = {
                name: digest for name, digest in value[key].items() if ".layers.47." not in name
            }
    elif damage == "nan_loss":
        value["metrics"]["loss"] = float("nan")
    else:
        value["progress"]["consumed_after"] = 0
    with pytest.raises(ValueError):
        runner.validate_update(value, step=1, fixture="a" * 64)


@pytest.mark.parametrize("field", ["rng_sha256", "scheduler_sha256", "optimizer_header_sha256"])
def test_state_rejects_tampering_even_when_comparison_peer_was_copied(field):
    value = state()
    value[field] = "0" * 64
    copied_peer = copy.deepcopy(value)
    assert value == copied_peer
    with pytest.raises(ValueError, match="digest"):
        validate_state(value)


@pytest.mark.parametrize("damage", ["consumed", "missing_master", "missing_parameter"])
def test_resigned_state_still_requires_complete_progress_and_optimizer(damage):
    value = state()
    if damage == "consumed":
        value["consumed_train_samples"] = 0
    elif damage == "missing_master":
        del value["optimizer_parameter_sha256"][parameter_names()[0]]["param"]
    else:
        del value["optimizer_parameter_sha256"][parameter_names()[0]]
    with pytest.raises(ValueError):
        validate_state(seal_state(value))


@pytest.mark.parametrize("behavior", ["success", "native_failure", "missing_scheduler_step"])
def test_controller_advances_samples_only_after_successful_native_scheduler_step(
    monkeypatch, behavior
):
    args, scheduler = SimpleNamespace(consumed_train_samples=4), SimpleNamespace(num_steps=4)

    def update(*unused, **kwargs):
        if behavior == "native_failure":
            raise RuntimeError("injected native optimizer failure")
        if behavior == "success":
            scheduler.num_steps += 4
        return audit(step=2)

    monkeypatch.setattr(observer, "native_policy_update", update)

    def invoke():
        return native.policy_step(args, [], None, scheduler, None, step=2, required_groups={})

    if behavior == "success":
        result = invoke()
        assert args.consumed_train_samples == scheduler.num_steps == 8
        assert result["progress"]["consumed_before"] == 4
    else:
        with pytest.raises((RuntimeError, ValueError)):
            invoke()
        assert args.consumed_train_samples == 4


def test_journal_requires_l2_between_updates_and_before_reload_controls():
    journal = {
        "schema_version": 1,
        "identity": {},
        "events": list(runner.EVENTS[:3]),
        "workers": {"update1": {}},
        "l2_reports": {"baseline": {}, "update1": {}},
    }
    runner.validate_journal(journal, {}, verify_files=False)
    journal["events"] = ["l2:baseline", "worker:update1", "worker:update2"]
    with pytest.raises(ValueError, match="order"):
        runner.validate_journal(journal, {}, verify_files=False)


@pytest.mark.parametrize("missing", ["update2", "reload", "continuous", "resumed"])
def test_final_acceptance_rejects_missing_phases_before_reading_evidence(missing):
    report = {
        "schema_version": 1,
        "status": "passed",
        "scope": "qwen3_next_training_and_reload",
        "required_sections": list(runner.SECTIONS),
        "events": list(runner.EVENTS),
        "l2_reports": dict.fromkeys(("baseline", "update1", "update2", "reload"), {}),
        "workers": dict.fromkeys(runner.PHASES, {}),
    }
    del report["workers"][missing]
    with pytest.raises(ValueError, match="omitted"):
        runner.validate_training_report(report)


def test_cleanup_rejects_symlink_to_another_directory_in_same_run(tmp_path):
    retained = tmp_path / "retained"
    retained.mkdir()
    (retained / "state").write_text("must survive")
    alias = tmp_path / "alias"
    alias.symlink_to(retained, target_is_directory=True)
    with pytest.raises(ValueError, match="owned"):
        runner._owned_remove(alias, tmp_path)
    assert (retained / "state").read_text() == "must survive"


def test_interrupted_cleanup_requires_matching_run_identity(tmp_path):
    root = tmp_path / "checkpoints"
    identity = {"run_id": "this-run"}
    runner.owned_root(root, identity)
    own = root / "step-1.partial"
    own.mkdir()
    (own / "STAGING.json").write_text(json.dumps({"identity": identity, "status": "staging"}))
    assert len(runner.remove_interrupted(root, identity)) == 1
    assert not own.exists()
    foreign = root / "foreign.partial"
    foreign.mkdir()
    (foreign / "STAGING.json").write_text(
        json.dumps({"identity": {"run_id": "other"}, "status": "staging"})
    )
    with pytest.raises(ValueError, match="another run"):
        runner.remove_interrupted(root, identity)
    assert foreign.exists()


def test_actual_fault_injection_kills_writer_and_rejects_partial_or_corrupt_protocol(tmp_path):
    result = runner.checkpoint_faults(tmp_path / "faults", {"run_id": "cpu-protocol-only"})
    evidence = runner.read_record(result)
    assert set(evidence["cases"]) == set(runner.FAULTS)
    assert all(case["passed"] for case in evidence["cases"].values())
    assert evidence["cases"]["interrupted_writer"]["exit_code"] < 0
    # A passed label cannot survive repair/substitution of the rejected bytes.
    corrupt = Path(evidence["cases"]["hash_error"]["checkpoint"]) / "rank-2-optimizer.bin"
    corrupt.write_bytes(b"2:optimizer")
    with pytest.raises(ValueError, match="no longer reproduces"):
        runner.validate_fault_evidence(evidence, {"run_id": "cpu-protocol-only"})


def test_checkpoint_retention_keeps_successor_and_only_removes_older_owned_state(tmp_path):
    identity = {"run_id": "retention-unit-contract"}
    runner.owned_root(tmp_path, identity)
    checkpoints = []
    for step in (1, 2):
        path = tmp_path / f"step-{step}"
        path.mkdir()
        (path / "COMPLETE.json").write_text(
            json.dumps({"identity": identity, "status": "complete", "step": step})
        )
        checkpoints.append(path)
    retired = native.retire_prior_checkpoints(checkpoints[1], tmp_path, identity)
    assert retired[0]["complete"]["step"] == 1
    assert not checkpoints[0].exists()
    assert checkpoints[1].is_dir()


def test_checkpoint_retention_cannot_delete_a_foreign_run(tmp_path):
    identity = {"run_id": "this-run"}
    runner.owned_root(tmp_path, identity)
    successor, foreign = tmp_path / "step-2", tmp_path / "other-run"
    for path, owner, step in ((successor, identity, 2), (foreign, {"run_id": "other"}, 1)):
        path.mkdir()
        (path / "COMPLETE.json").write_text(
            json.dumps({"identity": owner, "status": "complete", "step": step})
        )
    with pytest.raises(ValueError, match="unrelated"):
        native.retire_prior_checkpoints(successor, tmp_path, identity)
    assert successor.exists() and foreign.exists()


@pytest.mark.parametrize("difference", [None, "signed_zero", "state"])
def test_reload_comparison_reopens_original_tensors_and_rejects_cross_phase_drift(
    tmp_path, difference
):
    # Small tensors test the raw comparison plumbing. Full matrix shape and
    # provenance remain the independent validate_matrix_report prerequisite.
    payload = {
        "response_tokens": torch.ones((1, 16), dtype=torch.int64),
        "logits": torch.zeros((1, 16, 2), dtype=torch.bfloat16),
        "logprobs": torch.zeros((1, 16), dtype=torch.float32),
        "embedding_digest": {"sample_sha256": ["d" * 64], "sha256": "e" * 64},
        "layer_digests": [
            {
                "layer": index,
                "sample_output_sha256": ["a" * 64],
                "sample_state_sha256": ["b" * 64],
                "sample_subnode_sha256": {name: ["f" * 64] for name in SUBNODES},
            }
            for index in range(48)
        ],
    }
    left, right = tmp_path / "update.pt", tmp_path / "reload.pt"
    torch.save(payload, left)
    if difference == "signed_zero":
        payload["logits"][0, 0, 0] = -0.0
    elif difference == "state":
        payload["layer_digests"][3]["sample_state_sha256"] = ["c" * 64]
    torch.save(payload, right)
    ids = [case.case_id for case in runner.case_matrix()]

    def matrix(path):
        return {
            "case_ids": ids,
            "cases": {
                name: {
                    "artifacts": {role: runner.record(path) for role in ("rollout", "recompute")}
                }
                for name in ids
            },
        }

    if difference is None:
        result = runner.compare_reload_matrices(matrix(left), matrix(right))
        assert result["status"] == "passed" and len(result["cases"]) == 16
    else:
        with pytest.raises(ValueError, match="original.*tensor bits/state"):
            runner.compare_reload_matrices(matrix(left), matrix(right))


def test_stage_phase_ranges_and_continuation_identity():
    assert runner.stage_phases("train1") == ("update1",)
    assert runner.stage_phases("train2") == ("update1", "update2")
    assert runner.stage_phases("train") == runner.PHASES
    model = {"source_tree": "t", "environment_sha256": "e"}
    journal_identity = {**model, "run_id": "r1", "execution_id": "1", "fixture_sha256": "f"}
    assert runner.continuation_identity_matches(journal_identity, model, "f")
    assert not runner.continuation_identity_matches(journal_identity, model, "g")
    assert not runner.continuation_identity_matches(
        journal_identity, {**model, "environment_sha256": "x"}, "f"
    )


def test_partial_training_evidence_uses_the_phase_export_identity(tmp_path, monkeypatch):
    import qwen3_next_contract as contract
    import qwen3_next_l2

    from rl_engine.validation.models.qwen3_next import qwen3_next_l2 as l2_testing

    manifest = {"stage": "train1", "checkpoint": "/official"}
    export = {"checkpoint": "/export", "weight_version": "update-1", "base_checkpoint": "/official"}
    monkeypatch.setattr(qwen3_next_l2, "model_identity", lambda m: {"ckpt": m["checkpoint"]})
    monkeypatch.setattr(runner, "read_record", lambda value: {"export": export})
    seen = []
    monkeypatch.setattr(
        l2_testing,
        "validate_matrix_report",
        lambda report, expected_identity, artifact_root: seen.append(expected_identity),
    )
    matrix = tmp_path / "l2-matrix.json"
    matrix.write_text("{}")
    progress = {
        "status": "passed",
        "scope": contract.SCOPES["train1"],
        "identity": {"ckpt": "/official"},
        "phases": ["update1"],
        "workers": {"update1": {"path": "w"}},
        "l2_reports": {"update1": {"sha256": "abc"}},
    }
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(progress))
    evidence = {"updates": {"path": str(path)}, "l2_matrix": {"path": str(matrix), "sha256": "abc"}}
    contract.validate_partial_training_evidence(manifest, tmp_path, evidence, attempt_root=tmp_path)
    assert seen == [{"ckpt": "/export"}]  # not the base model identity
    with pytest.raises(ValueError, match="does not reference"):
        contract.validate_partial_training_evidence(
            manifest,
            tmp_path,
            {**evidence, "l2_matrix": {**evidence["l2_matrix"], "sha256": "x"}},
            attempt_root=tmp_path,
        )
    export["base_checkpoint"] = "/elsewhere"
    with pytest.raises(ValueError, match="pinned base"):
        contract.validate_partial_training_evidence(
            manifest, tmp_path, evidence, attempt_root=tmp_path
        )


def test_continued_storage_follows_the_chain_back_to_train1(tmp_path):
    def run(name, stage, previous=None):
        directory = tmp_path / name
        directory.mkdir()
        manifest = {"stage": stage, "previous": str(previous) if previous else None}
        (directory / "run.json").write_text(json.dumps(manifest))
        return directory

    l2 = run("l2", "l2")
    train1 = run("train1", "train1", l2)
    train2 = run("train2", "train2", train1)
    assert runner.continued_training_storage(train1) == train1 / "training-state"
    assert runner.continued_training_storage(train2) == train1 / "training-state"
    with pytest.raises(ValueError, match="back to a train1 run"):
        runner.continued_training_storage(l2)
