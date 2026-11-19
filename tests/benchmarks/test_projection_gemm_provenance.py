# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import hashlib
import subprocess

import pytest

from benchmarks.operators.gemm.projection_provenance import source_provenance

EMPTY_HASH = hashlib.sha256(b"").hexdigest()


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path)

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "csrc").mkdir()
    (tmp_path / "csrc/kernel.cu").write_text("before\n")
    git("add", ".")
    git("commit", "-qm", "baseline")
    return tmp_path, git


def test_clean_revisions_have_distinct_committed_trees(repo):
    path, git = repo
    before = source_provenance(path)
    (path / "csrc/kernel.cu").write_text("after\n")
    git("add", ".")
    git("commit", "-qm", "change kernel")
    after = source_provenance(path)
    assert before["commit"] != after["commit"]
    assert before["committed_tree_sha"] != after["committed_tree_sha"]
    for record in (before, after):
        assert record["unstaged_source_diff_sha256"] == EMPTY_HASH
        assert record["staged_source_diff_sha256"] == EMPTY_HASH
        assert record["source_status"] == ""


def test_staged_and_unstaged_changes_are_separate(repo):
    path, git = repo
    clean = source_provenance(path)
    (path / "csrc/kernel.cu").write_text("unstaged\n")
    unstaged = source_provenance(path)
    assert unstaged["unstaged_source_diff_sha256"] != EMPTY_HASH
    assert unstaged["staged_source_diff_sha256"] == EMPTY_HASH
    git("add", ".")
    staged = source_provenance(path)
    assert staged["unstaged_source_diff_sha256"] == EMPTY_HASH
    assert staged["staged_source_diff_sha256"] != EMPTY_HASH
    assert staged["committed_tree_sha"] == clean["committed_tree_sha"]
    assert staged["commit"] == clean["commit"]


def test_untracked_source_is_visible_without_claiming_content_hash(repo):
    path, _ = repo
    (path / "csrc/new.cu").write_text("untracked\n")
    record = source_provenance(path)
    assert "?? csrc/new.cu" in record["source_status"]
    assert record["unstaged_source_diff_sha256"] == EMPTY_HASH
    assert record["staged_source_diff_sha256"] == EMPTY_HASH
