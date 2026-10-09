# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Bind each native model execution to its freshly built and archived CUDA binary."""

import hashlib
import json
import os
import shutil
from pathlib import Path


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def archive_provider_binary(source, output, *, source_tree):
    source, output = Path(source).absolute(), Path(output).absolute()
    binaries = list((source / "rl_engine").glob("_C.cpython*.so"))
    if len(binaries) != 1 or not source_tree:
        raise ValueError("Expected one freshly built CUDA provider and its source tree")
    binary = binaries[0]
    digest = _digest(binary)
    archive = output / "provider-binary.so"
    temporary = archive.with_suffix(".partial")
    shutil.copyfile(binary, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    if _digest(temporary) != digest:
        raise ValueError("CUDA provider changed while archiving its execution identity")
    temporary.replace(archive)
    payload = {
        "schema_version": 1,
        "source_tree": source_tree,
        "actual": {"module": "rl_engine._C", "path": str(binary), "sha256": digest},
        "archive": {"path": str(archive), "sha256": digest},
    }
    path = output / "provider-binary.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": _digest(path)}


def validate_binary_record(record, *, source_tree=None, artifact_root=None):
    """Check archived bytes, without depending on a subsequently rebuilt live file."""
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("Native execution requires a pinned CUDA binary evidence record")

    def checked(item):
        path = Path(item["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("CUDA binary evidence requires an existing absolute path")
        if artifact_root is not None and not path.resolve().is_relative_to(
            Path(artifact_root).resolve()
        ):
            raise ValueError("CUDA binary evidence escaped its owning stage")
        if _digest(path) != item["sha256"]:
            raise ValueError("CUDA provider evidence content changed")
        return path

    payload = json.loads(checked(record).read_text())
    if payload.get("schema_version") != 1 or not payload.get("source_tree"):
        raise ValueError("Unsupported CUDA provider evidence")
    if source_tree is not None and payload["source_tree"] != source_tree:
        raise ValueError("CUDA provider belongs to another source tree")
    actual = payload.get("actual", {})
    if set(actual) != {"module", "path", "sha256"} or actual["module"] != "rl_engine._C":
        raise ValueError("The actual native CUDA extension identity is missing")
    if not Path(actual["path"]).is_absolute() or not Path(actual["path"]).name.startswith(
        "_C.cpython"
    ):
        raise ValueError("Provider identity does not name the actual native extension")
    archive = payload.get("archive", {})
    if archive.get("sha256") != actual["sha256"]:
        raise ValueError("Archived binary differs from the binary imported by workers")
    checked(archive)
    return actual


def verify_imported_provider(record, *, source_tree):
    from rl_engine.integrations.qwen3_next_binary import imported_provider_identity

    expected = validate_binary_record(record, source_tree=source_tree)
    actual = imported_provider_identity()
    if actual != expected:
        raise ValueError("Worker imported a different CUDA binary than its stage built")
    return actual
