# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import hashlib

import pytest

from rl_engine.validation.common.file_digest import file_sha256


@pytest.mark.parametrize("size", [0, 17, 8 * 1024 * 1024 + 17])
def test_artifact_sha256_without_python311_api(tmp_path, monkeypatch, size):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    content = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    path = tmp_path / "artifact.bin"
    path.write_bytes(content)
    assert file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_missing_artifact_is_not_a_valid_digest(tmp_path):
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "missing.bin")
