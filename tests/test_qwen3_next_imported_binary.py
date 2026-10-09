# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU integrity checks; stub loader metadata does not claim CUDA execution."""

import hashlib
import importlib.machinery
from types import SimpleNamespace as Namespace

import pytest

from rl_engine.integrations import qwen3_next_binary as binary


def extension_metadata(tmp_path, monkeypatch):
    path = tmp_path / ("_C" + importlib.machinery.EXTENSION_SUFFIXES[0])
    path.write_bytes(b"controlled-native-loader-metadata")
    loader = importlib.machinery.ExtensionFileLoader("rl_engine._C", str(path))
    module = Namespace(
        __name__="rl_engine._C",
        __file__=str(path),
        __spec__=Namespace(loader=loader, origin=str(path)),
    )
    imports = []

    def import_module(name):
        imports.append(name)
        return module

    monkeypatch.setattr(binary.importlib, "import_module", import_module)
    return path, module, imports


def test_import_is_required_and_content_is_rehashed_after_same_size_replacement(
    tmp_path, monkeypatch
):
    path, _module, imports = extension_metadata(tmp_path, monkeypatch)
    first = binary.imported_provider_identity()
    assert first == {
        "module": "rl_engine._C",
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.write_bytes(b"X" * path.stat().st_size)
    second = binary.imported_provider_identity()
    assert first["sha256"] != second["sha256"]
    assert imports == ["rl_engine._C", "rl_engine._C"]


def test_extension_import_failure_does_not_use_python_fallback(monkeypatch):
    def unavailable(name):
        raise ImportError("controlled missing extension")

    monkeypatch.setattr(binary.importlib, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="cannot be imported"):
        binary.imported_provider_identity()


@pytest.mark.parametrize("drift", ["name", "loader", "suffix", "origin"])
def test_native_identity_rejects_fallback_or_disagreeing_import_metadata(
    tmp_path, monkeypatch, drift
):
    path, module, _imports = extension_metadata(tmp_path, monkeypatch)
    if drift == "name":
        module.__name__ = "fallback"
    elif drift == "loader":
        module.__spec__.loader = importlib.machinery.SourceFileLoader("rl_engine._C", str(path))
    elif drift == "suffix":
        module.__file__ = str(tmp_path / "fallback.py")
    else:
        other = tmp_path / "other.so"
        other.write_bytes(path.read_bytes())
        module.__spec__.origin = str(other)
    with pytest.raises(RuntimeError, match="native extension|import origin"):
        binary.imported_provider_identity()


def test_file_write_during_hash_is_rejected(tmp_path, monkeypatch):
    path, _module, _imports = extension_metadata(tmp_path, monkeypatch)
    original = binary.os.fstat
    calls = 0

    def modified_fstat(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_bytes(b"changed-during-read")
        return original(fd)

    monkeypatch.setattr(binary.os, "fstat", modified_fstat)
    with pytest.raises(RuntimeError, match="changed while"):
        binary.imported_provider_identity()


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"module": "fallback", "path": "/a.so", "sha256": "a" * 64},
        {"module": "rl_engine._C", "path": "relative.so", "sha256": "a" * 64},
        {"module": "rl_engine._C", "path": "/a.so", "sha256": "unknown"},
    ],
)
def test_serialized_binary_identity_is_mandatory_and_complete(value):
    with pytest.raises(ValueError, match="binary evidence"):
        binary.validate_provider_identity(value)
