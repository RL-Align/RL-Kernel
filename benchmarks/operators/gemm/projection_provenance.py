# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU-only Git metadata for the projection benchmark; no build attestation."""

import hashlib
import os
import subprocess
from pathlib import Path


def git_environment() -> dict[str, str]:
    """Keep caller settings except overrides that select a different Git repository/index."""
    locations = {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_SHALLOW_FILE",
        "GIT_IMPLICIT_WORK_TREE",
        "GIT_PREFIX",
    }
    return {key: value for key, value in os.environ.items() if key not in locations}


def source_provenance(cwd: Path | None = None) -> dict[str, str]:
    env = git_environment()

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=cwd, env=env)

    paths = (":(top)csrc", ":(top)rl_engine")
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
