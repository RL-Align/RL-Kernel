#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a repository CI suite from any working directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "layout": [
        sys.executable,
        "-m",
        "pytest",
        "tests/build",
        "tests/runtime/test_layout_compatibility.py",
    ],
    "ws1": ["bash", "ci/scripts/run_ws1_gtest.sh"],
    "ws1-chain": ["bash", "ci/scripts/run_ws1_chain_gate.sh"],
    "ws1-ascend": ["bash", "ci/scripts/run_ws1_ascend_ci.sh"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=sorted(SUITES))
    args, extra = parser.parse_known_args()
    return subprocess.call([*SUITES[args.suite], *extra], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
