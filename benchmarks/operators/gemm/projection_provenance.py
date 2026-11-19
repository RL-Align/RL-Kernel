# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU-only Git metadata for the projection benchmark; no build attestation."""

import hashlib
import subprocess
from pathlib import Path


def source_provenance(cwd: Path | None = None) -> dict[str, str]:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=cwd)

    paths = ("csrc", "rl_engine")
    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "committed_tree_sha": git("rev-parse", "HEAD^{tree}").decode().strip(),
        "unstaged_source_diff_sha256": hashlib.sha256(
            git("diff", "--no-ext-diff", "--no-textconv", "--binary", "--", *paths)
        ).hexdigest(),
        "staged_source_diff_sha256": hashlib.sha256(
            git(
                "diff",
                "--cached",
                "--no-ext-diff",
                "--no-textconv",
                "--binary",
                "--",
                *paths,
            )
        ).hexdigest(),
        "source_status": git(
            "status", "--porcelain=v1", "--untracked-files=all", "--", *paths
        ).decode(),
    }
