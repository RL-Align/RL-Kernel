# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Crash boundaries for four-rank atomic checkpoint publication."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models"))
from qwen3_next_checkpoint import (  # noqa: E402
    newest_complete_checkpoint,
    publish_checkpoint,
    resume_interrupted_publications,
    validate_checkpoint,
    write_rank_manifest,
)

IDENTITY = {"provider": "explicit", "tp": 4, "environment": "pinned", "weight_revision": "official"}


def staged(root, step=1):
    directory = root / f"step-{step}.partial"
    directory.mkdir()
    for rank in range(4):
        rank_dir = directory / f"rank-{rank}"
        rank_dir.mkdir()
        files = {}
        for role in ("model", "optimizer", "rng", "progress"):
            path = rank_dir / f"{role}.json"
            path.write_text(json.dumps({"rank": rank, "role": role, "step": step}))
            files[role] = [str(path.relative_to(directory))]
        write_rank_manifest(
            directory,
            rank=rank,
            world_size=4,
            identity=IDENTITY,
            step=step,
            sample_position=step * 4,
            files=files,
        )
    return directory


def test_complete_publication_and_recovery(tmp_path):
    partial = staged(tmp_path)
    destination = tmp_path / "step-1"
    publish_checkpoint(partial, destination, world_size=4, identity=IDENTITY)
    assert not partial.exists()
    assert validate_checkpoint(destination, IDENTITY)["step"] == 1
    assert json.loads((tmp_path / "LATEST.json").read_text())["path"] == "step-1"
    recovery = newest_complete_checkpoint(tmp_path, IDENTITY)
    assert recovery["checkpoint"] == str(destination)
    assert recovery["sample_position"] == 4


def test_missing_rank_prevents_completion_marker(tmp_path):
    partial = staged(tmp_path)
    (partial / "rank-3.manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        publish_checkpoint(partial, tmp_path / "step-1", world_size=4, identity=IDENTITY)
    assert not (partial / "COMPLETE.json").exists()
    assert not (tmp_path / "step-1").exists()


def test_corruption_prevents_publication(tmp_path):
    partial = staged(tmp_path)
    (partial / "rank-2/optimizer.json").write_text("damaged")
    with pytest.raises(ValueError, match="checksum|size mismatch"):
        publish_checkpoint(partial, tmp_path / "step-1", world_size=4, identity=IDENTITY)
    assert not (partial / "COMPLETE.json").exists()


def test_only_completed_checkpoints_can_load(tmp_path):
    partial = staged(tmp_path)
    with pytest.raises(ValueError, match="Unpublished"):
        validate_checkpoint(partial, IDENTITY)
    assert newest_complete_checkpoint(tmp_path, IDENTITY)["checkpoint"] is None


def test_corrupt_latest_falls_back_and_records_reason(tmp_path):
    for step in (1, 2):
        partial = staged(tmp_path, step)
        publish_checkpoint(partial, tmp_path / f"step-{step}", world_size=4, identity=IDENTITY)
    (tmp_path / "step-2/rank-0/rng.json").write_text("corrupt")
    staged(tmp_path, 3)
    recovery = newest_complete_checkpoint(tmp_path, IDENTITY)
    assert recovery["step"] == 1
    assert recovery["rejected"][0]["path"] == "step-2"
    assert recovery["interrupted"] == ["step-3.partial"]


def test_identity_mismatch_never_resumes_fresh_silently(tmp_path):
    partial = staged(tmp_path)
    publish_checkpoint(partial, tmp_path / "step-1", world_size=4, identity=IDENTITY)
    with pytest.raises(ValueError, match="No valid"):
        newest_complete_checkpoint(tmp_path, {**IDENTITY, "tp": 2})


def test_rank_progress_mismatch_prevents_publication(tmp_path):
    partial = staged(tmp_path)
    path = partial / "rank-3.manifest.json"
    payload = json.loads(path.read_text())
    payload["step"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different progress"):
        publish_checkpoint(partial, tmp_path / "step-1", world_size=4, identity=IDENTITY)


def test_rank_cannot_omit_optimizer_or_reuse_rng_as_model(tmp_path):
    partial = tmp_path / "new.partial"
    partial.mkdir()
    (partial / "state").write_text("state")
    with pytest.raises(ValueError, match="Every rank"):
        write_rank_manifest(
            partial,
            rank=0,
            world_size=4,
            identity=IDENTITY,
            step=1,
            sample_position=4,
            files={"model": ["state"]},
        )
    with pytest.raises(ValueError, match="multiple state roles"):
        write_rank_manifest(
            partial,
            rank=0,
            world_size=4,
            identity=IDENTITY,
            step=1,
            sample_position=4,
            files={role: ["state"] for role in ("model", "optimizer", "rng", "progress")},
        )


def test_crash_after_marker_before_rename_stays_unloadable(tmp_path, monkeypatch):
    partial = staged(tmp_path)

    def crash(*args, **kwargs):
        raise OSError("simulated interrupted rename")

    monkeypatch.setattr(Path, "rename", crash)
    with pytest.raises(OSError):
        publish_checkpoint(partial, tmp_path / "step-1", world_size=4, identity=IDENTITY)
    assert (partial / "COMPLETE.json").exists()
    assert newest_complete_checkpoint(tmp_path, IDENTITY)["checkpoint"] is None


def test_published_directory_cannot_be_overwritten(tmp_path):
    partial = staged(tmp_path)
    destination = tmp_path / "step-1"
    destination.mkdir()
    with pytest.raises(ValueError, match="new sibling"):
        publish_checkpoint(partial, destination, world_size=4, identity=IDENTITY)


def test_disk_budget_counts_last_next_exports_and_records_failure(tmp_path, monkeypatch):
    from collections import namedtuple

    import qwen3_next_checkpoint as checkpoint

    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(checkpoint.shutil, "disk_usage", lambda path: usage(1000, 750, 250))
    with pytest.raises(RuntimeError, match="insufficient_disk"):
        checkpoint.require_checkpoint_disk_space(
            tmp_path, checkpoint_bytes=100, export_bytes=40, headroom_bytes=20
        )
    record = json.loads((tmp_path / "disk-budget.json").read_text())
    assert record["required_free_bytes"] == 260 and record["status"] == "insufficient_disk"
    record = checkpoint.require_checkpoint_disk_space(
        tmp_path,
        checkpoint_bytes=100,
        export_bytes=40,
        retained_complete_bytes=100,
        headroom_bytes=20,
    )
    assert record["required_free_bytes"] == 160 and record["status"] == "passed"


def test_discovery_without_hashing_checks_sizes_and_defers_content_to_the_loader(tmp_path):
    partial = staged(tmp_path)
    publish_checkpoint(partial, tmp_path / "step-1", world_size=4, identity=IDENTITY)
    target = tmp_path / "step-1/rank-0/rng.json"
    original = target.read_text()
    target.write_text("X" * len(original))  # same size, different bytes
    fast = newest_complete_checkpoint(tmp_path, IDENTITY, verify_files=False)
    assert fast["checkpoint"] == str(tmp_path / "step-1") and fast["rejected"] == []
    with pytest.raises(ValueError, match="checksum mismatch"):
        newest_complete_checkpoint(tmp_path, IDENTITY)  # hashed discovery still rejects it
    target.write_text(original + "!")  # size change is still caught without hashing
    with pytest.raises(ValueError, match="size mismatch"):
        newest_complete_checkpoint(tmp_path, IDENTITY, verify_files=False)


def test_interrupted_but_complete_partial_is_published_on_resume(tmp_path):
    partial = staged(tmp_path, 1)  # all four ranks and manifests written, never published
    (partial / "STAGING.json").write_text(json.dumps({"identity": IDENTITY, "status": "staging"}))
    records = resume_interrupted_publications(tmp_path, IDENTITY)
    assert records == [
        {"reason": "interrupted_publication_resumed", "checkpoint": "step-1", "step": 1}
    ]
    assert not partial.exists()
    assert newest_complete_checkpoint(tmp_path, IDENTITY)["checkpoint"] == str(tmp_path / "step-1")


def test_incomplete_or_corrupt_partial_is_not_resumed(tmp_path):
    missing = staged(tmp_path, 1)
    (missing / "rank-3.manifest.json").unlink()
    corrupt = staged(tmp_path, 2)
    (corrupt / "rank-0/rng.json").write_text("corrupt")
    for partial in (missing, corrupt):
        (partial / "STAGING.json").write_text(
            json.dumps({"identity": IDENTITY, "status": "staging"})
        )
    records = resume_interrupted_publications(tmp_path, IDENTITY)
    assert [r["reason"] for r in records] == ["interrupted_publication_not_resumable"]
    assert records[0]["staging"] == "step-2.partial"
    assert missing.exists() and corrupt.exists()
    assert not (corrupt / "COMPLETE.json").exists()
    assert newest_complete_checkpoint(tmp_path, IDENTITY)["checkpoint"] is None
