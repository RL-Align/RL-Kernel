# SPDX-License-Identifier: Apache-2.0
import json
import sys

import pytest

from rl_engine import repro


def test_profile_keeps_virtual_environment_python_symlink(tmp_path):
    executable = tmp_path / "system-python"
    executable.touch()
    venv_python = tmp_path / "venv-python"
    try:
        venv_python.symlink_to(executable)
    except OSError:
        if sys.platform == "win32":
            pytest.skip("Windows account cannot create symbolic links")
        raise
    profile = {"paths": {"python": str(venv_python)}}
    paths = repro.resolve_paths(profile, workspace=str(tmp_path), arm="consistency")
    assert paths.python == venv_python


def test_machine_profile_and_short_options_override_defaults(tmp_path, monkeypatch, capsys):
    profile = tmp_path / "machine.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "rlkernel.repro.profile.v1",
                "paths": {"workspace": str(tmp_path)},
                "defaults": {
                    "backend": "rocm",
                    "mode": "consistency",
                    "rollouts": 1,
                    "tp": 4,
                    "cp": 2,
                    "top-p": 0.95,
                    "temperature": 0.7,
                },
            }
        )
    )
    monkeypatch.setenv("RLK_REPRO_PROFILE", str(profile))
    assert (
        repro.main(
            [
                "plan",
                "--tp",
                "2",
                "--cp",
                "4",
                "--rollout-tp",
                "8",
                "--top-p",
                "0.8",
                "--temperature",
                "1.3",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    command = plan["runner_command"]
    for option, expected in {
        "tp-size": "2",
        "cp-size": "4",
        "rollout-tp-size": "8",
        "rollout-top-p": "0.8",
        "rollout-temperature": "1.3",
        "mode": "consistency",
        "rollouts": "1",
    }.items():
        assert command[command.index(f"--{option}") + 1] == expected


@pytest.mark.parametrize(
    "options",
    [
        ["--tp", "3"],
        ["--temperature", "0"],
        ["--temperature", "nan"],
        ["--top-p", "0"],
        ["--top-p", "1.01"],
    ],
)
def test_invalid_rocm_options_fail_before_launch(tmp_path, options):
    assert repro.main(["plan", "--backend", "rocm", "--workspace", str(tmp_path), *options]) == 2
