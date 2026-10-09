# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Failure classification must retain evidence and never guess from signals."""

import errno
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen3_next_runtime import failure_kind, subprocess_failure  # noqa: E402


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Killed", "unknown"),
        ("exit 137", "unknown"),
        ("MemoryError: allocation failed", "unknown"),
        ("torch.OutOfMemoryError: allocation failed", "out_of_memory"),
        ("RuntimeError: CUDA out of memory", "out_of_memory"),
        ("OSError: [Errno 28] No space left on device", "insufficient_disk"),
        ("RuntimeError: insufficient_disk: budget", "insufficient_disk"),
        ("ModuleNotFoundError: No module named 'vime'", "environment_mismatch"),
        ("RuntimeError: environment_mismatch: changed", "environment_mismatch"),
        ("RuntimeError: source_mismatch: changed", "source_mismatch"),
        ("RuntimeError: checkpoint_mismatch: changed", "checkpoint_mismatch"),
        ("subprocess.TimeoutExpired: command exceeded timeout", "time_limit"),
    ],
)
def test_tail_classification_is_explicit(tmp_path, text, expected):
    log = tmp_path / "worker.log"
    log.write_text(text)
    failure = subprocess_failure(-9, log)
    assert failure_kind(failure) == expected
    assert text in str(failure)
    assert str(log) in str(failure)


def test_exception_classification_and_actual_sigkill(tmp_path):
    from qwen3_next_l2 import run_process

    assert failure_kind(OSError(errno.ENOSPC, "disk full")) == "insufficient_disk"
    assert failure_kind(subprocess.TimeoutExpired("worker", 1)) == "time_limit"
    with pytest.raises(RuntimeError) as caught:
        run_process(
            [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
            log_path=tmp_path / "killed.log",
        )
    assert failure_kind(caught.value) == "unknown"
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            log_path=tmp_path / "timeout.log",
            timeout=0.02,
            cleanup_grace_seconds=0.05,
        )
    assert failure_kind(caught.value) == "time_limit"


def test_timeout_kills_ignoring_grandchild_even_after_leader_exits(tmp_path):
    import os
    import signal
    import time

    from qwen3_next_l2 import run_process

    pid_file = tmp_path / "grandchild.pid"
    grandchild = (
        "import os,pathlib,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    leader = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(60)"
    )
    pid = None
    try:
        with pytest.raises(subprocess.TimeoutExpired) as caught:
            run_process(
                [sys.executable, "-c", leader],
                log_path=tmp_path / "family.log",
                timeout=0.5,
                cleanup_grace_seconds=0.15,
            )
        assert failure_kind(caught.value) == "time_limit"
        pid = int(pid_file.read_text())
        stat = Path(f"/proc/{pid}/stat")
        # An orphan may briefly remain as a zombie until init reaps it, but must
        # exit once the kernel schedules its pending SIGKILL. We cannot waitpid
        # an orphan that has already been adopted by init.
        deadline = time.monotonic() + 2
        while stat.exists() and stat.read_text().split(") ", 1)[1].split()[0] != "Z":
            if time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert not stat.exists() or stat.read_text().split(") ", 1)[1].split()[0] == "Z"
    finally:
        if pid is None and pid_file.exists():
            pid = int(pid_file.read_text())
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_training_failure_binds_run_attempt_and_error(tmp_path, monkeypatch):
    import qwen3_next_training as training

    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps({"run_id": "controlled-run"}))
    output = tmp_path / "attempts/2"
    monkeypatch.setenv("QWEN3_NEXT_ATTEMPT", "2")
    monkeypatch.setattr(
        sys, "argv", ["training", "--manifest", str(manifest), "--output", str(output)]
    )

    def fail(*args):
        raise OSError(errno.ENOSPC, "controlled full disk")

    monkeypatch.setattr(training, "execute", fail)
    with pytest.raises(OSError):
        training.main()
    result = json.loads((output / "result.json").read_text())
    assert (result["run_id"], result["attempt"], result["failure_kind"]) == (
        "controlled-run",
        "2",
        "insufficient_disk",
    )
    assert "controlled full disk" in result["error"]
