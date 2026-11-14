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
    for name in (*expected, "RL-Kernel"):
        repo = root / name

        def git(*args, _repo=repo):
            return subprocess.check_output(["git", "-C", str(_repo), *args], text=True).strip()

        actual = git("rev-parse", "HEAD")
        wanted = expected.get(name, os.environ.get("RLK_ASSEMBLY_SHA"))
        if wanted is not None and actual != wanted:
            raise RuntimeError(f"{name} revision mismatch: {actual} != {wanted}")
        if git("diff", "--name-only", "HEAD"):
            raise RuntimeError(f"{name} has patched source")
        if git("ls-files", "--others", "--exclude-standard"):
            raise RuntimeError(f"{name} has unrecorded source")
        result[name] = {"sha": actual, "expected_sha": wanted, "pristine": True, "path": str(repo)}
    try:
        dist = md.distribution("megatron-bridge")
    except md.PackageNotFoundError as error:
        raise RuntimeError("megatron-bridge 0.7.0 wheel is required") from error
    if dist.version != "0.7.0" or not dist.files:
        raise RuntimeError("megatron-bridge requires the pinned wheel and readable RECORD")
    checked = 0
    for file in dist.files:
        if str(file).startswith("megatron/bridge/") and str(file).endswith(".py"):
            if not file.hash or file.hash.mode != "sha256":
                raise RuntimeError(f"Bridge source has no SHA256 RECORD entry: {file}")
            digest = hashlib.sha256(Path(dist.locate_file(file)).read_bytes()).digest()
            actual = base64.urlsafe_b64encode(digest).decode().rstrip("=")
            if actual != file.hash.value:
                raise RuntimeError(f"Bridge differs from wheel RECORD: {file}")
            checked += 1
    if not checked:
        raise RuntimeError("Bridge RECORD contains no verifiable Python sources")
    result["megatron-bridge"] = {"version": dist.version, "pristine_python_files": checked}
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
