# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Content identities for validation artifacts, including large tensor files."""

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Hash a file in bounded chunks on every supported Python version."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
