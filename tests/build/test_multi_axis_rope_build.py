# SPDX-License-Identifier: Apache-2.0
"""The explicit operator wheel profile must build native code or fail closed."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from build_tools import extensions


def test_rope_only_wheel_selects_production_sources(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_ROPE_ONLY", "1")
    fake_torch = SimpleNamespace(
        __file__=torch.__file__, version=SimpleNamespace(cuda="12.8", hip=None)
    )
    monkeypatch.setattr(
        extensions,
        "_load_torch_extension_tools",
        lambda: (fake_torch, None, lambda **kwargs: kwargs),
    )
    (extension,) = extensions.get_extensions()
    assert extension["name"] == "rl_engine._rope_cuda"
    assert set(extension["sources"]) == {
        "csrc/bindings/rope_module.cpp",
        "csrc/cuda/rope/multi_axis_rope.cu",
    }
    assert "--fmad=false" in extension["extra_compile_args"]["nvcc"]


@pytest.mark.parametrize("hip", [None, "6.2"])
def test_rope_only_wheel_rejects_non_cuda_builds(monkeypatch, hip):
    monkeypatch.setenv("RL_KERNEL_ROPE_ONLY", "1")
    fake_torch = SimpleNamespace(
        __file__=torch.__file__, version=SimpleNamespace(cuda=None, hip=hip)
    )
    monkeypatch.setattr(extensions, "_load_torch_extension_tools", lambda: (fake_torch, None, None))
    with pytest.raises(RuntimeError, match="requires CUDA-enabled PyTorch"):
        extensions.get_extensions()


def test_rope_only_wheel_rejects_missing_torch(monkeypatch):
    monkeypatch.setenv("RL_KERNEL_ROPE_ONLY", "1")
    monkeypatch.setattr(extensions, "_load_torch_extension_tools", lambda: (None, None, None))
    with pytest.raises(RuntimeError, match="PyTorch is unavailable"):
        extensions.get_extensions()


@pytest.fixture
def workflow_scope(tmp_path):
    """Execute the workflow's actual script against a multi-commit Git history."""
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/qwenimage-rope.yml"
    script = textwrap.dedent(
        workflow.read_text(encoding="utf-8")
        .split("python - <<'PY'\n", 1)[1]
        .split("\n          PY", 1)[0]
    )

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "--quiet")
    git("config", "user.name", "Workflow test")
    git("config", "user.email", "workflow@example.invalid")
    (tmp_path / "unchanged.py").write_text("# base\n", encoding="utf-8")
    git("add", ".")
    git("commit", "--quiet", "-m", "base")
    base = git("rev-parse", "HEAD")
    for name in ("earlier.py", "space name.py", "-leading.py"):
        (tmp_path / name).write_text("# earlier commit\n", encoding="utf-8")
    git("add", ".")
    git("commit", "--quiet", "-m", "earlier changes")
    (tmp_path / "latest.py").write_text("# latest commit\n", encoding="utf-8")
    git("add", ".")
    git("commit", "--quiet", "-m", "latest change")
    # Substitute only the hook runner; Git diff and workflow event handling are real.
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "pre_commit.py").write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['HOOK_ARGS']).write_text(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    event_path, record = tmp_path / "event.json", tmp_path / "hook-args.json"

    def run(event):
        event_path.write_text(json.dumps(event), encoding="utf-8")
        env = dict(os.environ, GITHUB_EVENT_PATH=str(event_path), HOOK_ARGS=str(record))
        env["PYTHONPATH"] = str(runner)
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=tmp_path, env=env, capture_output=True, text=True
        )
        return result, json.loads(record.read_text()) if record.exists() else None

    return base, run


@pytest.mark.parametrize("event_kind", ["pull_request", "manual_base", "manual_all"])
def test_workflow_checks_complete_scope(workflow_scope, event_kind):
    base, run = workflow_scope
    events = {
        "pull_request": {"pull_request": {"base": {"sha": base}}},
        "manual_base": {"inputs": {"base_sha": base}},
        "manual_all": {},
    }
    result, args = run(events[event_kind])
    assert result.returncode == 0, result.stderr
    if event_kind == "manual_all":
        assert args == ["run", "--all-files"]
    else:
        assert args[:2] == ["run", "--files"]
        assert set(args[2:]) == {"./-leading.py", "earlier.py", "space name.py", "latest.py"}


def test_workflow_rejects_invalid_manual_base(workflow_scope):
    _, run = workflow_scope
    result, args = run({"inputs": {"base_sha": "HEAD^"}})
    assert result.returncode != 0
    assert args is None
