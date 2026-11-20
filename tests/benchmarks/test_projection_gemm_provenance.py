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
    (tmp_path / "rl_engine").mkdir()
    (tmp_path / "rl_engine/operator.py").write_text("before\n")
    git("add", ".")
    git("commit", "-qm", "baseline")
    return tmp_path, git


def _foreign_git_locations(path):
    return {
        "GIT_DIR": str(path / ".git"),
        "GIT_WORK_TREE": str(path),
        "GIT_INDEX_FILE": str(path / ".git/index"),
        "GIT_OBJECT_DIRECTORY": str(path / ".git/objects"),
        "GIT_COMMON_DIR": str(path / ".git"),
    }


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


@pytest.mark.parametrize("source", ("csrc/kernel.cu", "rl_engine/operator.py"))
def test_source_changes_are_found_from_a_subdirectory(repo, source):
    path, git = repo
    subdir = path / "nested/deeper"
    subdir.mkdir(parents=True)
    (path / source).write_text("unstaged\n")
    assert source_provenance(subdir) == source_provenance(path)
    assert source_provenance(subdir)["unstaged_source_diff_sha256"] != EMPTY_HASH
    git("add", source)
    assert source_provenance(subdir) == source_provenance(path)
    assert source_provenance(subdir)["staged_source_diff_sha256"] != EMPTY_HASH
    untracked = (path / source).with_name("untracked" + (path / source).suffix)
    untracked.write_text("untracked\n")
    assert source_provenance(subdir) == source_provenance(path)
    assert (
        f"?? {untracked.relative_to(path).as_posix()}" in source_provenance(subdir)["source_status"]
    )


@pytest.mark.parametrize(
    "variable",
    ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR", "all"),
)
def test_inherited_git_locations_do_not_change_provenance(
    repo, tmp_path_factory, monkeypatch, variable
):
    path, git = repo
    # The external locations are valid: a leak could silently return real but
    # wrong metadata, rather than merely fail on a nonexistent directory.
    foreign, foreign_git = _make_repo(tmp_path_factory.mktemp("foreign"))
    (foreign / "csrc/kernel.cu").write_text("foreign source\n")
    foreign_git("add", ".")
    foreign_git("commit", "-qm", "foreign revision")
    foreign_index = foreign / ".git/index"
    original_index = foreign_index.read_bytes()
    (path / "csrc/kernel.cu").write_text("local unstaged\n")
    (path / "rl_engine/operator.py").write_text("local staged\n")
    git("add", "rl_engine")
    expected = source_provenance(path)
    assert expected["commit"] != source_provenance(foreign)["commit"]
    locations = _foreign_git_locations(foreign)
    for name, value in locations.items():
        if variable in (name, "all"):
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("TASK12_UNRELATED_SETTING", "preserved")
    assert source_provenance(path) == expected
    assert not locations.keys() & git_environment().keys()
    assert git_environment()["TASK12_UNRELATED_SETTING"] == "preserved"
    assert foreign_index.read_bytes() == original_index


def test_fixture_cannot_write_an_inherited_index(tmp_path, monkeypatch):
    # Exercise isolation starting with git init, not only after fixture creation.
    caller = tmp_path / "caller"
    caller.mkdir()
    _, caller_git = _make_repo(caller)
    (caller / "csrc/kernel.cu").write_text("uncommitted caller staging\n")
    caller_git("add", ".")
    caller_index = caller / ".git/index"
    original_index = caller_index.read_bytes()
    original_stat = caller_index.stat()
    original_head = caller_git("rev-parse", "HEAD")
    for name, value in _foreign_git_locations(caller).items():
        monkeypatch.setenv(name, value)
    repository = tmp_path / "repository"
    repository.mkdir()
    path, git = _make_repo(repository)
    (path / "csrc/kernel.cu").write_text("changed\n")
    git("add", ".")
    assert source_provenance(path)["staged_source_diff_sha256"] != EMPTY_HASH
    assert (repository / ".git").is_dir()
    assert caller_index.read_bytes() == original_index
    assert caller_index.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert caller_index.stat().st_ino == original_stat.st_ino
    assert caller_git("rev-parse", "HEAD") == original_head
