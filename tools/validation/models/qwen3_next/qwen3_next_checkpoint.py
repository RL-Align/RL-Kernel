# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Publish and recover complete all-rank checkpoints on the shared filesystem.

Training owns serialization and barriers. This module never loads tensor files;
all four ranks must finish model, optimizer, RNG and progress before publishing.
"""

import json
import os
import shutil
from pathlib import Path

from qwen3_next_contract import atomic_json
from qwen3_next_runtime import file_digest

ROLES = frozenset(("model", "optimizer", "rng", "progress"))


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Checkpoint file must be relative to its checkpoint directory")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or external checkpoint file: {relative}")
    return path


def write_rank_manifest(staging, *, rank, world_size, identity, step, sample_position, files):
    staging = Path(staging)
    if world_size != 4 or not 0 <= rank < world_size or step < 0 or sample_position < 0:
        raise ValueError("Checkpoint requires four ranks and nonnegative progress")
    if staging.suffix != ".partial" or (staging / "COMPLETE.json").exists():
        raise ValueError("Rank output requires an unpublished .partial checkpoint")
    if set(files) != ROLES or any(not paths for paths in files.values()):
        raise ValueError("Every rank must save model, optimizer, RNG and progress")
    records = {}
    seen = set()
    for role, paths in files.items():
        records[role] = []
        for relative in paths:
            if str(relative) in seen:
                raise ValueError("A checkpoint file cannot fulfill multiple state roles")
            seen.add(str(relative))
            path = _file(staging, relative)
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            records[role].append(
                {
                    "path": str(relative),
                    "size": path.stat().st_size,
                    "sha256": file_digest(path),
                }
            )
    manifest = {
        "schema_version": 1,
        "rank": rank,
        "world_size": world_size,
        "identity": identity,
        "step": step,
        "sample_position": sample_position,
        "files": records,
    }
    atomic_json(staging / f"rank-{rank}.manifest.json", manifest)
    return manifest


def _validate_ranks(root, *, world_size, identity, verify_files=True):
    if world_size != 4:
        raise ValueError("Checkpoint topology must use four ranks")
    manifests = []
    used_paths = set()
    progress = None
    for rank in range(world_size):
        manifest = json.loads((root / f"rank-{rank}.manifest.json").read_text())
        if (manifest.get("schema_version"), manifest.get("rank"), manifest.get("world_size")) != (
            1,
            rank,
            world_size,
        ):
            raise ValueError("Checkpoint rank topology mismatch")
        if manifest.get("identity") != identity:
            raise ValueError("Checkpoint model/provider/environment identity mismatch")
        current = (manifest.get("step"), manifest.get("sample_position"))
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in current
        ):
            raise ValueError("Invalid checkpoint progress")
        if progress is not None and current != progress:
            raise ValueError("Checkpoint ranks have different progress")
        progress = current
        files = manifest.get("files", {})
        if set(files) != ROLES or any(not paths for paths in files.values()):
            raise ValueError("Checkpoint omits required optimizer/RNG/model/progress state")
        for records in files.values():
            for record in records:
                if record["path"] in used_paths:
                    raise ValueError("Checkpoint ranks must own distinct state files")
                used_paths.add(record["path"])
                path = _file(root, record["path"])
                if path.stat().st_size != record["size"]:
                    raise ValueError(f"Checkpoint state size mismatch: {record['path']}")
                if verify_files and file_digest(path) != record["sha256"]:
                    raise ValueError(f"Checkpoint state checksum mismatch: {record['path']}")
        manifests.append(manifest)
    return manifests


def publish_checkpoint(staging, destination, *, world_size, identity):
    staging, destination = Path(staging), Path(destination)
    if (
        staging.suffix != ".partial"
        or destination.suffix == ".partial"
        or destination.name.startswith(".")
    ):
        raise ValueError("Publish from a .partial directory to a visible completed directory")
    if staging.parent.resolve() != destination.parent.resolve() or destination.exists():
        raise ValueError("Checkpoint publication needs a new sibling destination")
    manifests = _validate_ranks(staging, world_size=world_size, identity=identity)
    complete = {
        "schema_version": 1,
        "status": "complete",
        "world_size": world_size,
        "identity": identity,
        "step": manifests[0]["step"],
        "sample_position": manifests[0]["sample_position"],
        "rank_manifests": {
            str(rank): file_digest(staging / f"rank-{rank}.manifest.json")
            for rank in range(world_size)
        },
    }
    atomic_json(staging / "COMPLETE.json", complete)
    # Sync nested directory entries as well as file contents before publication.
    directories = sorted(
        (path for path in staging.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        _sync_directory(directory)
    _sync_directory(staging)
    staging.rename(destination)
    _sync_directory(destination.parent)
    atomic_json(
        destination.parent / "LATEST.json",
        {
            "path": destination.name,
            "step": complete["step"],
            "complete_sha256": file_digest(destination / "COMPLETE.json"),
        },
    )
    return complete


def validate_checkpoint(path, expected_identity, *, verify_files=True):
    """Validate a published checkpoint's markers, manifests and files.

    ``verify_files=False`` checks every state file's presence and size but not its
    content digest; callers use it where the content is verified anyway when the
    files are read (the native restore compares every loaded tensor against the
    saved digests), so a 1.1 TB checkpoint is not rehashed before each phase.
    """
    path = Path(path)
    if path.suffix == ".partial" or path.name.startswith("."):
        raise ValueError("Unpublished checkpoint cannot be loaded")
    complete = json.loads((path / "COMPLETE.json").read_text())
    if complete.get("schema_version") != 1 or complete.get("status") != "complete":
        raise ValueError("Checkpoint has no supported completion marker")
    if complete.get("identity") != expected_identity:
        raise ValueError("Checkpoint model/provider/environment identity mismatch")
    manifests = _validate_ranks(
        path,
        world_size=complete["world_size"],
        identity=expected_identity,
        verify_files=verify_files,
    )
    if complete.get("rank_manifests") != {
        str(rank): file_digest(path / f"rank-{rank}.manifest.json") for rank in range(4)
    }:
        raise ValueError("Checkpoint rank manifest checksum mismatch")
    if (complete.get("step"), complete.get("sample_position")) != (
        manifests[0]["step"],
        manifests[0]["sample_position"],
    ):
        raise ValueError("Completion marker progress mismatch")
    return complete


def resume_interrupted_publications(root, expected_identity):
    """Publish every fully written ``.partial`` checkpoint a killed job left unpublished.

    A preempted job can die after all four ranks wrote their files and manifests
    but before rank 0 validated and renamed the directory. Such a directory is
    publishable as is: publication rehashes every file against the manifests,
    so an incomplete or damaged one is left for the normal retirement path.
    Returns journal recovery records.
    """
    root = Path(root)
    records = []
    for staging in sorted(root.glob("*.partial")):
        if not staging.is_dir() or staging.is_symlink() or (staging / "COMPLETE.json").exists():
            continue
        marker = staging / "STAGING.json"
        if not marker.is_file():
            continue
        metadata = json.loads(marker.read_text())
        if metadata.get("identity") != expected_identity or metadata.get("status") != "staging":
            continue
        if not all((staging / f"rank-{rank}.manifest.json").is_file() for rank in range(4)):
            continue
        destination = staging.with_suffix("")
        if destination.exists():
            continue
        try:
            complete = publish_checkpoint(
                staging, destination, world_size=4, identity=expected_identity
            )
        except (OSError, ValueError, KeyError) as exc:
            records.append(
                {
                    "reason": "interrupted_publication_not_resumable",
                    "staging": staging.name,
                    "error": str(exc)[:300],
                }
            )
            continue
        records.append(
            {
                "reason": "interrupted_publication_resumed",
                "checkpoint": destination.name,
                "step": complete["step"],
            }
        )
    return records


def newest_complete_checkpoint(root, expected_identity, *, verify_files=True):
    """Return a recovery record, retaining rejected and interrupted candidates."""
    root = Path(root)
    candidates, rejected, interrupted = [], [], []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if path.suffix == ".partial" or path.name.startswith("."):
            interrupted.append(path.name)
            continue
        try:
            complete = validate_checkpoint(path, expected_identity, verify_files=verify_files)
        except (OSError, ValueError, KeyError) as exc:
            rejected.append({"path": path.name, "reason": str(exc)})
        else:
            candidates.append((complete["step"], path, complete))
    if not candidates and rejected:
        raise ValueError(f"No valid published checkpoint remains: {rejected}")
    chosen = max(candidates, key=lambda item: (item[0], item[1].name)) if candidates else None
    return {
        "checkpoint": str(chosen[1]) if chosen else None,
        "step": chosen[0] if chosen else None,
        "sample_position": chosen[2]["sample_position"] if chosen else None,
        "rejected": rejected,
        "interrupted": interrupted,
    }


def require_checkpoint_disk_space(
    root, *, checkpoint_bytes, export_bytes, retained_complete_bytes=0, headroom_bytes=16 * 1024**3
):
    """Budget last+next checkpoint and exports before serializing, without deletion.

    checkpoint_bytes includes model, optimizer/master weights, RNG and progress.
    retained_complete_bytes discounts only already retained complete checkpoint
    bytes; unfinished directories are never counted as available or reusable.
    """
    if (
        any(
            type(value) is not int or value < 0
            for value in (checkpoint_bytes, export_bytes, retained_complete_bytes, headroom_bytes)
        )
        or checkpoint_bytes == 0
    ):
        raise ValueError("Checkpoint disk estimates must be nonnegative integer byte counts")
    retained = min(retained_complete_bytes, checkpoint_bytes)
    required = 2 * checkpoint_bytes + export_bytes + headroom_bytes - retained
    usage = shutil.disk_usage(root)
    record = {
        "scope": "checkpoint_disk_budget",
        "checkpoint_bytes": checkpoint_bytes,
        "last_plus_next_bytes": 2 * checkpoint_bytes,
        "export_bytes": export_bytes,
        "retained_complete_bytes": retained,
        "headroom_bytes": headroom_bytes,
        "required_free_bytes": required,
        "actual_free_bytes": usage.free,
        "status": "passed" if usage.free >= required else "insufficient_disk",
    }
    atomic_json(Path(root) / "disk-budget.json", record)
    if usage.free < required:
        raise RuntimeError(
            f"insufficient_disk: need {required} free bytes for last+next checkpoint "
            f"and exports, have {usage.free}"
        )
    return record
