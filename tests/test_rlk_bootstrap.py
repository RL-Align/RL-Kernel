# SPDX-License-Identifier: Apache-2.0
"""Exercise the checkout launcher before any GPU runtime is imported."""

import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest


@pytest.fixture
def checkout(tmp_path):
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "checkout"
    for relative in (
        "rlk",
        "rl_engine/repro.py",
        "examples/vime_qwen3_8b_tp4_cp2_200/profiles/qwen3-8b-tp4-cp2.json",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    return root


def write_profile(path, **paths):
    path.write_text(
        json.dumps(
            {
                "schema_version": "rlkernel.repro.profile.v1",
                "name": path.stem,
                "paths": {"workspace": str(path.parent), **paths},
                "modes": {"consistency": {}},
                "ray_address": "http://selected-profile:8265",
            }
        ),
        encoding="utf-8",
    )
    return path


def launch(checkout, *args, **overrides):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("RLK_REPRO_")
        and key not in {"PYTHONPATH", "PYTHONHOME", "RAY_API_SERVER_ADDRESS"}
    }
    env.update(overrides)
    return subprocess.run(
        [sys.executable, str(checkout / "rlk"), *args],
        cwd=checkout.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("selection", ["option", "equals", "environment", "local"])
def test_selected_profile_configures_bootstrap_runtime(checkout, tmp_path, selection):
    runtime = tmp_path / "selected runtime"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(runtime)
    python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    selected = write_profile(tmp_path / "selected profile.json", python=str(python))
    env = {}
    args = ["plan"]
    if selection == "local":
        shutil.copy2(selected, checkout / ".rlk-profile.json")
    else:
        write_profile(checkout / ".rlk-profile.json", python=str(tmp_path / "missing-python"))
        if selection == "environment":
            env["RLK_REPRO_PROFILE"] = str(selected)
        else:
            stale = write_profile(tmp_path / "stale.json", python=str(tmp_path / "missing-python"))
            env["RLK_REPRO_PROFILE"] = str(stale)
            args += (
                ["--profile", str(selected)] if selection == "option" else [f"--profile={selected}"]
            )

    result = launch(checkout, *args, **env)

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["paths"]["runtime_root"] == str(runtime.resolve())
    assert plan["paths"]["python"] == str(python)
    command = plan["runner_command"]
    assert command[command.index("--ray-address") + 1] == "http://selected-profile:8265"


def test_explicit_profile_bypasses_malformed_environment_profile(checkout, tmp_path):
    stale = tmp_path / "stale.json"
    stale.write_text("{", encoding="utf-8")
    selected = write_profile(tmp_path / "selected.json")

    result = launch(checkout, "plan", "--profile", str(selected), RLK_REPRO_PROFILE=str(stale))

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["profile"] == "selected"


def test_selected_profile_sets_pythonpath_before_runtime_startup(checkout, tmp_path):
    site = tmp_path / "selected-site"
    site.mkdir()
    capture = tmp_path / "pythonpath.txt"
    (site / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\n"
        f"Path({str(capture)!r}).write_text(os.environ['PYTHONPATH'])\n",
        encoding="utf-8",
    )
    selected = write_profile(tmp_path / "selected.json", runtime_site=str(site))
    inherited = str(tmp_path / "inherited-site")

    result = launch(checkout, "plan", "--profile", str(selected), PYTHONPATH=inherited)

    assert result.returncode == 0, result.stderr
    assert capture.read_text().split(os.pathsep) == [str(checkout), str(site), inherited]


def test_profile_relative_to_checkout_is_used_from_another_directory(checkout):
    selected = write_profile(checkout / "machine.json")

    result = launch(checkout, "plan", "--profile", selected.name)

    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["runner_command"]
    assert command[command.index("--ray-address") + 1] == "http://selected-profile:8265"


@pytest.mark.parametrize("content", ["{", "[]"])
def test_invalid_selected_profile_reports_cli_error(checkout, tmp_path, content):
    selected = tmp_path / "invalid.json"
    selected.write_text(content, encoding="utf-8")

    result = launch(checkout, "plan", RLK_REPRO_PROFILE=str(selected))

    assert result.returncode == 2
    assert "rlk-repro:" in result.stderr
    assert "Traceback" not in result.stderr


def test_explicit_ray_address_overrides_profile(checkout, tmp_path):
    selected = write_profile(tmp_path / "selected.json")

    result = launch(
        checkout, "plan", "--profile", str(selected), "--ray-address", "http://explicit:8265"
    )

    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["runner_command"]
    assert command[command.index("--ray-address") + 1] == "http://explicit:8265"


def test_no_machine_profile_uses_default_profile(checkout, tmp_path):
    result = launch(checkout, "plan", "--workspace", str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["profile"] == "qwen3-8b-tp4-cp2"
