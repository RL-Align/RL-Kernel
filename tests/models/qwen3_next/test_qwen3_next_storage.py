# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Faults at persistent filesystem boundaries, including an actual SIGKILL."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models/qwen3_next"))
import qwen3_next_runtime as storage  # noqa: E402
from qwen3_next_checkpoint import (  # noqa: E402
    newest_complete_checkpoint,
    publish_checkpoint,
    validate_checkpoint,
    write_rank_manifest,
)

IDENTITY = {"run_id": "storage-fault-test"}


class Preempted(BaseException):
    pass


def fail_at(monkeypatch, boundary):
    def crash(name):
        if name == boundary:
            raise Preempted(name)

    monkeypatch.setattr(storage, "_boundary", crash)


def metadata():
    return {"identity": IDENTITY, "status": "staging", "step": 1}


def complete(root, name="step-2", step=2):
    staging, destination = storage.create_staging(root, name, metadata())
    for rank in range(4):
        files = {}
        for role in ("model", "optimizer", "rng", "progress"):
            relative = f"rank-{rank}-{role}.bin"
            (staging / relative).write_bytes(f"{rank}:{role}".encode())
            files[role] = [relative]
        write_rank_manifest(
            staging,
            rank=rank,
            world_size=4,
            identity=IDENTITY,
            step=step,
            sample_position=step * 4,
            files=files,
        )
    marker = publish_checkpoint(staging, destination, world_size=4, identity=IDENTITY)
    return destination, marker


@pytest.mark.parametrize("boundary", ["root_owner_persisted", "root_created"])
def test_root_creation_recovers_without_internal_owner(monkeypatch, tmp_path, boundary):
    root = tmp_path / "checkpoints"
    fail_at(monkeypatch, boundary)
    with pytest.raises(Preempted):
        storage.owned_root(root, IDENTITY)
    monkeypatch.setattr(storage, "_boundary", lambda name: None)
    storage.owned_root(root, IDENTITY)
    assert json.loads((root / "RUN_OWNER.json").read_text()) == IDENTITY
    storage.recover_storage(root, IDENTITY)


@pytest.mark.parametrize(
    "boundary", ["staging_intent_persisted", "staging_created", "staging_marker_persisted"]
)
def test_staging_creation_recovers_every_window_without_loading_partial(
    monkeypatch, tmp_path, boundary
):
    root = storage.owned_root(tmp_path / "checkpoints", IDENTITY)
    retained, marker = complete(root)
    fail_at(monkeypatch, boundary)
    with pytest.raises(Preempted):
        storage.create_staging(root, "next-step", metadata())
    monkeypatch.setattr(storage, "_boundary", lambda name: None)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_checkpoint(root / "next-step.partial", IDENTITY)
    storage.recover_storage(root, IDENTITY)
    assert not (root / "next-step.partial").exists()
    assert validate_checkpoint(retained, IDENTITY) == marker
    assert newest_complete_checkpoint(root, IDENTITY)["checkpoint"] == str(retained)
    assert storage.recover_storage(root, IDENTITY) == []


@pytest.mark.parametrize(
    "boundary",
    [
        "retirement_intent_persisted",
        "retirement_renamed",
        "retirement_removed",
        "retirement_committed",
    ],
)
def test_retirement_resumes_each_window_and_preserves_successor(monkeypatch, tmp_path, boundary):
    root = storage.owned_root(tmp_path / "checkpoints", IDENTITY)
    prior, old_marker = complete(root, "step-1", 1)
    retained, marker = complete(root)
    proof = {"complete": old_marker, "successor": str(retained), "successor_complete": marker}
    fail_at(monkeypatch, boundary)
    with pytest.raises(Preempted):
        storage.retire_directory(prior, root, IDENTITY, proof=proof)
    monkeypatch.setattr(storage, "_boundary", lambda name: None)
    with pytest.raises(ValueError, match="Unpublished"):
        validate_checkpoint(root / ".step-1.deleting", IDENTITY)
    storage.recover_storage(root, IDENTITY)
    assert not prior.exists() and not (root / ".step-1.deleting").exists()
    assert validate_checkpoint(retained, IDENTITY) == marker
    assert storage.recover_storage(root, IDENTITY) == []


def test_preemption_after_internal_marker_deleted_does_not_strand_cleanup(monkeypatch, tmp_path):
    root = storage.owned_root(tmp_path / "exports", IDENTITY)
    stage, _ = storage.create_staging(root, "update-1", metadata())
    (stage / "unfinished-weights").write_bytes(b"never load this")
    remove = storage.shutil.rmtree

    def interrupted(path):
        (path / "STAGING.json").unlink()
        raise Preempted("during rmtree")

    monkeypatch.setattr(storage.shutil, "rmtree", interrupted)
    with pytest.raises(Preempted):
        storage.recover_storage(root, IDENTITY)
    assert not stage.exists()
    assert (root / ".update-1.partial.deleting" / "unfinished-weights").is_file()
    monkeypatch.setattr(storage.shutil, "rmtree", remove)
    storage.recover_storage(root, IDENTITY)
    assert not (root / ".update-1.partial.deleting").exists()


def test_pending_retirement_rechecks_successor_before_deleting_last_good_state(
    monkeypatch, tmp_path
):
    root = storage.owned_root(tmp_path / "checkpoints", IDENTITY)
    prior, marker = complete(root, "step-1", 1)
    successor, next_marker = complete(root)
    fail_at(monkeypatch, "retirement_intent_persisted")
    with pytest.raises(Preempted):
        storage.retire_directory(
            prior,
            root,
            IDENTITY,
            proof={
                "complete": marker,
                "successor": str(successor),
                "successor_complete": next_marker,
            },
        )
    monkeypatch.setattr(storage, "_boundary", lambda name: None)
    (successor / "rank-0-model.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum|size mismatch"):
        storage.recover_storage(root, IDENTITY)
    assert validate_checkpoint(prior, IDENTITY) == marker


@pytest.mark.parametrize("foreign", ["missing_owner", "wrong_owner", "symlink"])
def test_unrelated_partial_directories_are_never_deleted(tmp_path, foreign):
    root = storage.owned_root(tmp_path / "exports", IDENTITY)
    target = root / "unrelated.partial"
    if foreign == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "STAGING.json").write_text(json.dumps(metadata()))
        target.symlink_to(outside, target_is_directory=True)
    else:
        target.mkdir()
        if foreign == "wrong_owner":
            (target / "STAGING.json").write_text(
                json.dumps({"status": "staging", "identity": {"run_id": "other"}})
            )
    with pytest.raises(ValueError):
        storage.recover_storage(root, IDENTITY)
    assert target.exists()


def test_real_sigkill_between_mkdir_and_internal_marker_is_recoverable(tmp_path):
    root = tmp_path / "exports"
    script = """
import os, signal, sys
import qwen3_next_runtime as storage
identity = {'run_id': 'storage-fault-test'}
root = storage.owned_root(sys.argv[1], identity)
def stop(name):
    if name == 'staging_created':
        os.kill(os.getpid(), signal.SIGKILL)
storage._boundary = stop
storage.create_staging(
    root, 'crashed-export', {'identity': identity, 'status': 'staging', 'step': 1}
)
"""
    environment = {**os.environ, "PYTHONPATH": str(Path(storage.__file__).parent)}
    child = subprocess.run([sys.executable, "-c", script, str(root)], env=environment, timeout=30)
    assert child.returncode == -signal.SIGKILL
    assert (root / "crashed-export.partial").is_dir()
    assert not (root / "crashed-export.partial" / "STAGING.json").exists()
    storage.owned_root(root, IDENTITY)
    storage.recover_storage(root, IDENTITY)
    assert not (root / "crashed-export.partial").exists()


@pytest.mark.parametrize("changed", ["identity", "target", "filename", "published", "status"])
def test_malformed_parent_staging_intent_never_deletes_its_target(tmp_path, changed):
    root = storage.owned_root(tmp_path / "exports", IDENTITY)
    staging, _ = storage.create_staging(root, "unfinished", metadata())
    intent = root / ".unfinished.staging.json"
    record = json.loads(intent.read_text())
    if changed == "identity":
        record["identity"] = {"run_id": "another"}
    elif changed == "target":
        record["staging"] = "../outside.partial"
    elif changed == "filename":
        intent.rename(root / ".wrong-name.staging.json")
        intent = root / ".wrong-name.staging.json"
    elif changed == "published":
        record["published"] = ".unfinished"
        record["staging"] = ".unfinished.partial"
    else:
        record["metadata"]["status"] = "complete"
    intent.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        storage.recover_storage(root, IDENTITY)
    assert staging.is_dir()


def test_replaced_target_marker_stops_pending_retirement(monkeypatch, tmp_path):
    root = storage.owned_root(tmp_path / "checkpoints", IDENTITY)
    prior, previous = complete(root, "step-1", 1)
    successor, current = complete(root)
    fail_at(monkeypatch, "retirement_intent_persisted")
    with pytest.raises(Preempted):
        storage.retire_directory(
            prior,
            root,
            IDENTITY,
            proof={
                "complete": previous,
                "successor": str(successor),
                "successor_complete": current,
            },
        )
    monkeypatch.setattr(storage, "_boundary", lambda name: None)
    (prior / "COMPLETE.json").write_text(json.dumps({**previous, "step": 99}))
    with pytest.raises(ValueError, match="differs from its durable ownership"):
        storage.recover_storage(root, IDENTITY)
    assert prior.is_dir()


@pytest.mark.parametrize("proof", [{}, {"status": "staging", "identity": IDENTITY}])
def test_unsupported_proof_cannot_retire_unrelated_published_directory(tmp_path, proof):
    root = storage.owned_root(tmp_path / "checkpoints", IDENTITY)
    unrelated = root / "unrelated"
    unrelated.mkdir()
    with pytest.raises(ValueError):
        storage.retire_directory(unrelated, root, IDENTITY, proof=proof)
    assert unrelated.is_dir()
    assert not (root / ".unrelated.delete.json").exists()


@pytest.mark.parametrize("record_name", ["parent_owner", "root_owner", "intent", "marker"])
def test_interrupted_atomic_creation_record_does_not_strand_recovery(
    monkeypatch, tmp_path, record_name
):
    root = tmp_path / "exports"
    atomic = storage.atomic_json
    targets = {
        "parent_owner": root.parent / ".exports.owner.json",
        "root_owner": root / "RUN_OWNER.json",
        "intent": root / ".unfinished.staging.json",
        "marker": root / "unfinished.partial" / "STAGING.json",
    }

    def interrupted(path, value):
        if path == targets[record_name]:
            path.with_suffix(path.suffix + ".partial").write_text('{"interrupted":')
            raise Preempted("during atomic record write")
        atomic(path, value)

    monkeypatch.setattr(storage, "atomic_json", interrupted)
    with pytest.raises(Preempted):
        storage.owned_root(root, IDENTITY)
        storage.create_staging(root, "unfinished", metadata())
    monkeypatch.setattr(storage, "atomic_json", atomic)
    storage.owned_root(root, IDENTITY)
    storage.recover_storage(root, IDENTITY)
    assert not (root / "unfinished.partial").exists()
    assert storage.recover_storage(root, IDENTITY) == []


@pytest.mark.parametrize("status", ["retiring", "retired"])
def test_interrupted_atomic_retirement_record_is_recoverable(monkeypatch, tmp_path, status):
    root = storage.owned_root(tmp_path / "exports", IDENTITY)
    staging, _ = storage.create_staging(root, "unfinished", metadata())
    atomic = storage.atomic_json

    def interrupted(path, value):
        if path.name.endswith(".delete.json") and value["status"] == status:
            path.with_suffix(path.suffix + ".partial").write_text('{"interrupted":')
            raise Preempted("during atomic retirement record")
        atomic(path, value)

    monkeypatch.setattr(storage, "atomic_json", interrupted)
    with pytest.raises(Preempted):
        storage.recover_storage(root, IDENTITY)
    monkeypatch.setattr(storage, "atomic_json", atomic)
    storage.recover_storage(root, IDENTITY)
    assert not staging.exists()
    assert not (root / ".unfinished.partial.deleting").exists()
    assert storage.recover_storage(root, IDENTITY) == []


@pytest.mark.parametrize("name", ["foo.a", "foo.b", "foo.partial", ".hidden"])
def test_published_names_cannot_alias_one_staging_directory(tmp_path, name):
    root = storage.owned_root(tmp_path / "exports", IDENTITY)
    with pytest.raises(ValueError, match="visible and have no suffix"):
        storage.create_staging(root, name, metadata())
    assert list(root.iterdir()) == [root / "RUN_OWNER.json"]


def test_training_storage_root_is_the_declared_root_or_inside_the_run(tmp_path):
    assert storage.training_storage_root({}, tmp_path) == tmp_path / "training-state"
    shared = tmp_path / "shared" / "run-1"
    assert storage.training_storage_root({"training_storage": str(shared)}, tmp_path) == shared
