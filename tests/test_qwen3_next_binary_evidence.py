# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Loaded provider identity must match immutable build evidence, including on reload."""

import json
import sys
from types import SimpleNamespace

import pytest

from rl_engine.testing.qwen3_next_binary import (
    archive_provider_binary,
    validate_binary_record,
    verify_imported_provider,
)


@pytest.fixture
def archived(tmp_path):
    source, output = tmp_path / "source", tmp_path / "attempt"
    (source / "rl_engine").mkdir(parents=True)
    output.mkdir()
    binary = source / "rl_engine/_C.cpython-test.so"
    binary.write_bytes(b"built CUDA extension")
    return binary, archive_provider_binary(source, output, source_tree="original-tree")


def test_archive_survives_later_live_binary_rebuild(archived):
    binary, record = archived
    actual = validate_binary_record(record, source_tree="original-tree")
    binary.write_bytes(b"later build")
    assert validate_binary_record(record) == actual
    with pytest.raises(ValueError, match="source tree"):
        validate_binary_record(record, source_tree="different-tree")


def test_imported_binary_cannot_substitute_another_provider(archived, monkeypatch):
    _, record = archived
    expected = validate_binary_record(record)
    loaded = dict(expected)
    monkeypatch.setitem(
        sys.modules,
        "rl_engine.integrations.qwen3_next_binary",
        SimpleNamespace(imported_provider_identity=lambda: loaded),
    )
    assert verify_imported_provider(record, source_tree="original-tree") == expected
    loaded["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different CUDA binary"):
        verify_imported_provider(record, source_tree="original-tree")


def test_archived_binary_corruption_and_attempt_escape_are_rejected(archived, tmp_path):
    from pathlib import Path

    _, record = archived
    with pytest.raises(ValueError, match="escaped"):
        validate_binary_record(record, artifact_root=tmp_path / "different-attempt")
    metadata = json.loads(Path(record["path"]).read_text())
    Path(metadata["archive"]["path"]).write_bytes(b"corrupted original binary")
    with pytest.raises(ValueError, match="content changed"):
        validate_binary_record(record)


@pytest.mark.parametrize("record", [None, {}, {"path": "/tmp/unpinned"}])
def test_missing_binary_record_cannot_close_native_acceptance(record):
    with pytest.raises(ValueError, match="pinned CUDA binary"):
        validate_binary_record(record)
