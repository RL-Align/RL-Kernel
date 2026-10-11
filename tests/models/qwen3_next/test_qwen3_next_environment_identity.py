# SPDX-License-Identifier: Apache-2.0
"""The environment identity must not depend on what the process has imported."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models"))

from qwen3_next_runtime import current_identity, installation_roots  # noqa: E402


def _fake_distribution(root: Path, name: str, version: str) -> None:
    info = root / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")


def test_identity_ignores_sys_path_entries_added_after_startup(tmp_path, monkeypatch):
    before = current_identity()
    extra = tmp_path / "late-site-packages"
    _fake_distribution(extra, "late_arrival", "1.0")
    monkeypatch.syspath_prepend(str(extra))
    after = current_identity()
    assert after["sha256"] == before["sha256"]
    assert all(item["name"] != "late_arrival" for item in after["identity"]["distributions"])


def test_identity_survives_importing_setuptools():
    before = current_identity()
    import setuptools  # noqa: F401  # appends setuptools/_vendor to sys.path

    after = current_identity()
    assert after["sha256"] == before["sha256"]
    assert all(item["name"] != "typeguard" for item in after["identity"]["distributions"])


def test_installation_roots_are_site_directories_only():
    roots = installation_roots()
    assert roots
    assert all(root.is_dir() for root in roots)
    assert not any("_vendor" in str(root) for root in roots)
