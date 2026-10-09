# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The scheduler-independent Qwen3-Next chain launcher."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import qwen3_next_run as run  # noqa: E402


def manifest(tmp_path, stage="layers", script=None):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True, exist_ok=True)
    if script is not None:
        (source / run.STAGES[stage][3]).write_text(script)
    return {
        "run_id": "r",
        "execution_id": "e" * 32,
        "stage": stage,
        "source": str(source),
        "source_tree": "tree",
        "python": sys.executable,
        "cuda_home": str(tmp_path),
        "cuda_identity": {"nvcc_sha256": "nvcc"},
        "memory": "512G",
        "environment": {"identity": {"base_prefix": sys.prefix, "build_tools": {"ninja": "1"}}},
        "provider_config": {"runtime_env": {"TRITON_F32_DEFAULT": "ieee"}},
        "scheduler_profile_certificate": {"path": "/certificate"},
    }


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(run, "verify_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "file_digest", lambda path: "nvcc")
    monkeypatch.setattr(run, "_git", lambda *args, cwd=None: "tree" if "rev-parse" in args else "")


def test_prepare_refuses_a_dirty_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "_git", lambda *args, cwd=None: " M scripts/qwen3_next_run.py")
    with pytest.raises(SystemExit, match="Commit the source"):
        run.prepare("preflight", None, None)


@pytest.mark.parametrize(
    "stage,bi", [("preflight", "0"), ("layers", "0"), ("l2", "1"), ("train", "1")]
)
def test_stage_environment_pins_batch_invariance_and_the_declared_memory(tmp_path, stage, bi):
    env = run.stage_environment(manifest(tmp_path, stage), tmp_path, "0")
    assert env["VLLM_BATCH_INVARIANT"] == bi
    assert env["QWEN3_NEXT_ALLOCATED_MEMORY"] == "512G"
    assert env["QWEN3_NEXT_ATTEMPT"] == "0"
    assert env["TRITON_F32_DEFAULT"] == "ieee"
    assert env["PYTHONPATH"].endswith("source")
    assert ("QWEN_SHARED_PROFILE_RESULT" in env) == (bi == "1")


def test_execute_validates_and_publishes_a_passed_result(tmp_path, frozen, monkeypatch):
    script = (
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--output') + 1]\n"
        "json.dump({'status': 'passed', 'attempt': '0'}, open(out + '/result.json', 'w'))\n"
    )
    current = manifest(tmp_path, script=script)
    seen = []
    monkeypatch.setattr(run, "validate_result", lambda *args, **kwargs: seen.append(args))
    status = run.execute(tmp_path, current)
    assert status["status"] == "passed" and len(seen) == 1
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "passed"
    assert not (tmp_path / "RUNNING").exists()


def test_execute_keeps_the_stage_failure_kind(tmp_path, frozen):
    script = (
        "import json, sys\n"
        "out = sys.argv[sys.argv.index('--output') + 1]\n"
        "json.dump({'failure_kind': 'insufficient_disk', 'error': 'full'},"
        " open(out + '/result.json', 'w'))\n"
        "sys.exit(3)\n"
    )
    status = run.execute(tmp_path, manifest(tmp_path, script=script))
    assert status["status"] == "failed" and status["exit_code"] == 3
    assert status["failure_kind"] == "insufficient_disk"
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "RUNNING").exists()


def test_execute_refuses_a_changed_source(tmp_path, monkeypatch, frozen):
    monkeypatch.setattr(run, "_git", lambda *args, cwd=None: "other-tree")
    with pytest.raises(RuntimeError, match="source_mismatch"):
        run.execute(tmp_path, manifest(tmp_path, script="raise SystemExit(0)\n"))


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--stage", "all", "--from", "layers"], "needs --previous"),
        (["--stage", "l2"], "needs --previous"),
        (["--stage", "preflight", "--previous", "."], "needs --previous"),
        (["--stage", "all", "--from", "train", "--until", "l2", "--previous", "."], "--from"),
    ],
)
def test_stage_selection_requires_a_predecessor_exactly_after_preflight(
    monkeypatch, tmp_path, arguments, message
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run", "--runs-root", str(tmp_path), "--checkpoint", ".", "--cuda-home", ".", *arguments],
    )
    with pytest.raises(SystemExit, match=message):
        run.main()
