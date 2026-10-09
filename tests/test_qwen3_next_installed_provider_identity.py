# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Wheel metadata alone cannot detect a locally patched numerical provider."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen3_next_runtime import installed_provider_sources  # noqa: E402


def test_same_version_provider_patch_changes_content_identity(tmp_path):
    name = "vllm/model_executor/determinism/batch_invariant.py"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text("KERNEL = 'original'\n")
    dist = SimpleNamespace(
        metadata={"Name": "vllm"}, files=[name], locate_file=lambda name: tmp_path / name
    )
    first = installed_provider_sources([dist])
    path.write_text("KERNEL = 'patched'\n")
    second = installed_provider_sources([dist])
    assert first[name]["sha256"] != second[name]["sha256"]
    assert first[name]["path"] == second[name]["path"]


@pytest.mark.parametrize("files", [None, [], ["vllm/unrelated.py"]])
def test_missing_provider_inventory_is_rejected(files):
    dist = SimpleNamespace(metadata={"Name": "vllm"}, files=files)
    with pytest.raises(ValueError):
        installed_provider_sources([dist])


def test_provider_file_deleted_after_install_is_rejected(tmp_path):
    dist = SimpleNamespace(
        metadata={"Name": "vllm"},
        files=["vllm/model_executor/determinism/batch_invariant.py"],
        locate_file=lambda name: tmp_path / name,
    )
    with pytest.raises(FileNotFoundError):
        installed_provider_sources([dist])
