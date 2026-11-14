# SPDX-License-Identifier: Apache-2.0
"""Reject the diagnostic image's upstream source patches for pure-RLK runs."""

import base64
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
from pathlib import Path


def audit(root=Path("/opt/dsv4")):
    expected = {
        "verl": "8c94a12d7811030413cd85e2bba6db863e0913c8",
        "Megatron-LM": "97b50bdd936022ef58031c35db6d10581c181be0",
        "vllm": "c07d58dc82d75f7d1a947b3264b460384ec3e18d",
    }
    result = {}
    for name, sha in expected.items():
        repo = root / name

        def git(*args, _repo=repo):
            return subprocess.check_output(["git", "-C", str(_repo), *args], text=True).strip()

        actual = git("rev-parse", "HEAD")
        assert actual == sha, (name, actual, sha)
        assert not git("diff", "--name-only", "HEAD"), (name, "patched source")
        assert not git("ls-files", "--others", "--exclude-standard"), (name, "unrecorded source")
        result[name] = {"sha": actual, "pristine": True, "path": str(repo)}
    if os.environ.get("RLK_ASSEMBLY_SHA"):
        repository = root / "RL-Kernel"
        actual = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        assert actual == os.environ["RLK_ASSEMBLY_SHA"]
        assert not subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True
        ).strip(), "RLK assembled checkout is dirty"
        result["RL-Kernel"] = dict(sha=actual, pristine_assembly=True, path=str(repository))
    dist = md.distribution("megatron-bridge")
    assert dist.version == "0.7.0"
    checked = 0
    for file in dist.files:
        if str(file).startswith("megatron/bridge/") and str(file).endswith(".py"):
            assert file.hash and file.hash.mode == "sha256", str(file)
            digest = hashlib.sha256(Path(dist.locate_file(file)).read_bytes()).digest()
            actual = base64.urlsafe_b64encode(digest).decode().rstrip("=")
            assert actual == file.hash.value, (str(file), "Bridge differs from wheel RECORD")
            checked += 1
    result["megatron-bridge"] = {"version": dist.version, "pristine_python_files": checked}
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
