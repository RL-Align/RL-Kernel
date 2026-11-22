# SPDX-License-Identifier: Apache-2.0
"""The environment identity must not depend on what the process has imported."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models/qwen3_next"))

from qwen3_next_runtime import (  # noqa: E402
    current_identity,
    failure_kind,
    installation_roots,
    user_site_shadowing,
)


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


def test_user_site_copy_of_an_environment_package_is_reported(tmp_path):
    env_site, user_site = tmp_path / "site-packages", tmp_path / "user-site"
    _fake_distribution(env_site, "apache_tvm_ffi", "0.1.11")
    _fake_distribution(env_site, "torch", "2.13.0")
    _fake_distribution(user_site, "apache-tvm-ffi", "0.1.16")
    _fake_distribution(user_site, "only_in_user_site", "1.0")
    roots = [env_site, user_site]
    assert user_site_shadowing(roots, user_site) == ["apache-tvm-ffi"]
    assert user_site_shadowing([env_site], user_site) == []


def test_identity_refuses_a_shadowing_user_site(tmp_path, monkeypatch):
    import qwen3_next_runtime as runtime

    env_site, user_site = tmp_path / "site-packages", tmp_path / "user-site"
    _fake_distribution(env_site, "apache_tvm_ffi", "0.1.11")
    _fake_distribution(user_site, "apache-tvm-ffi", "0.1.16")
    monkeypatch.setattr(runtime, "installation_roots", lambda: [env_site, user_site])
    monkeypatch.setattr(runtime.site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(runtime.site, "getusersitepackages", lambda: str(user_site))
    try:
        current_identity()
    except RuntimeError as error:
        assert "apache-tvm-ffi" in str(error) and "PYTHONNOUSERSITE=1" in str(error)
        assert failure_kind(error) == "environment_mismatch"
    else:
        raise AssertionError("a shadowing user site must be refused")


def test_probe_keeps_the_child_refusal_and_its_failure_kind(tmp_path):
    """A refusing probe interpreter surfaces as environment_mismatch with its message."""
    import qwen3_next_runtime as runtime

    message = "RuntimeError: environment_mismatch: the user site-packages shadows apache-tvm-ffi"
    fake = tmp_path / "python"
    fake.write_text(f"#!/bin/sh\necho '{message}' >&2\nexit 1\n")
    fake.chmod(0o755)
    try:
        runtime.probe_environment(fake)
    except runtime.ProcessFailure as error:
        assert "shadows apache-tvm-ffi" in str(error)
        assert failure_kind(error) == "environment_mismatch"
    else:
        raise AssertionError("a refusing probe must raise")
