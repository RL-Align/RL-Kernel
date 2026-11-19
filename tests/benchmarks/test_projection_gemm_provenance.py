# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import hashlib
import subprocess

import pytest

from benchmarks.operators.gemm.projection_provenance import git_environment, source_provenance

EMPTY_HASH = hashlib.sha256(b"").hexdigest()


def _make_repo(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, env=git_environment())

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "csrc").mkdir()
    (tmp_path / "csrc/kernel.cu").write_text("before\n")
    git("add", ".")
    git("commit", "-qm", "baseline")
    return tmp_path, git


@pytest.fixture
def repo(tmp_path):
    return _make_repo(tmp_path)


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


def test_source_changes_are_found_from_a_subdirectory(repo):
    path, git = repo
    subdir = path / "nested"
    subdir.mkdir()
    (path / "csrc/kernel.cu").write_text("unstaged\n")
    assert source_provenance(subdir) == source_provenance(path)
    assert source_provenance(subdir)["unstaged_source_diff_sha256"] != EMPTY_HASH
    git("add", "csrc")
    assert source_provenance(subdir) == source_provenance(path)
    assert source_provenance(subdir)["staged_source_diff_sha256"] != EMPTY_HASH


@pytest.mark.parametrize(
    "variable",
    ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR"),
)
def test_inherited_git_locations_do_not_change_provenance(repo, monkeypatch, variable):
    path, _ = repo
    expected = source_provenance(path)
    monkeypatch.setenv(variable, str(path / "must-not-be-used"))
    monkeypatch.setenv("TASK12_UNRELATED_SETTING", "preserved")
    assert source_provenance(path) == expected
    assert git_environment()["TASK12_UNRELATED_SETTING"] == "preserved"


def test_fixture_cannot_write_an_inherited_index(tmp_path, monkeypatch):
    # Exercise isolation starting with git init, not only after fixture creation.
    caller_index = tmp_path / "caller-index"
    caller_index.write_bytes(b"caller index must remain untouched")
    monkeypatch.setenv("GIT_INDEX_FILE", str(caller_index))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "unrelated.git"))
    repository = tmp_path / "repository"
    repository.mkdir()
    path, git = _make_repo(repository)
    (path / "csrc/kernel.cu").write_text("changed\n")
    git("add", ".")
    assert source_provenance(path)["staged_source_diff_sha256"] != EMPTY_HASH
    assert caller_index.read_bytes() == b"caller index must remain untouched"
