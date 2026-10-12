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


def test_runtime_digest_does_not_import_torch(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    runtime = Path(__file__).resolve().parents[3] / "tools/validation/models/qwen3_next_runtime.py"
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"standalone evidence runtime")
    program = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('runtime', sys.argv[1])
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
assert 'torch' not in sys.modules
assert runtime.file_digest(sys.argv[2]) == runtime.digest(sys.argv[2])
print(runtime.digest(sys.argv[2]))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(runtime), str(artifact)],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.stdout.strip() == hashlib.sha256(artifact.read_bytes()).hexdigest()
