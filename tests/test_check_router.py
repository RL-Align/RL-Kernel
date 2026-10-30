# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Command-line tests for the router validation entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_ROUTER = REPO_ROOT / "scripts" / "check_router.py"


@pytest.mark.parametrize(
    "args,expected_code",
    [
        (["--cases", "smoke"], 67),
        (["--cases", "learned_basic", "--stages", "L1"], 0),
        (["--cases", "hash_basic", "--stages", "L2,L3a"], 0),
        (["--cases", "learned_basic", "--stages", "WS2-rank,WS2-cross"], 0),
        (["--stages", "L9"], 2),
        (["--cases", "nope"], 2),
        (["--cases", "", "--stages", "L1"], 2),
        (["--cases", "smoke", "--stages", ""], 2),
    ],
)
def test_check_router_cli(args: list[str], expected_code: int) -> None:
    proc = subprocess.run(
        [sys.executable, str(CHECK_ROUTER), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == expected_code, proc.stdout + proc.stderr
    if expected_code == 2:
        assert "error:" in proc.stderr
    else:
        assert "check_router:" in proc.stdout


def test_check_router_json_report_uses_stage_name() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(CHECK_ROUTER),
            "--cases",
            "learned_basic",
            "--stages",
            "L1",
            "--json",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert '"stage": "L1"' in proc.stdout
