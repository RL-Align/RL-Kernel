# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Identity of the native provider actually imported by a model worker."""

import hashlib
import importlib
import importlib.machinery
import os
import re
from pathlib import Path


def validate_provider_identity(identity):
    """Validate serialized execution evidence without importing GPU code."""
    if (
        not isinstance(identity, dict)
        or set(identity) != {"module", "path", "sha256"}
        or identity["module"] != "rl_engine._C"
        or not isinstance(identity["path"], str)
        or not Path(identity["path"]).is_absolute()
        or not isinstance(identity["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None
    ):
        raise ValueError("Provider binary evidence must identify the actual native rl_engine._C")
    return dict(identity)


def imported_provider_identity():
    """Really import the extension, hash its current bytes, and reject fallbacks.

    Call at worker initialization and on both sides of each case. No cached file
    digest can hide a replacement made between cases or while a case executes.
    """
    try:
        module = importlib.import_module("rl_engine._C")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "The required native rl_engine._C provider cannot be imported"
        ) from error
    spec = getattr(module, "__spec__", None)
    filename = getattr(module, "__file__", None)
    if (
        getattr(module, "__name__", None) != "rl_engine._C"
        or not isinstance(getattr(spec, "loader", None), importlib.machinery.ExtensionFileLoader)
        or not isinstance(filename, str)
        or not isinstance(getattr(spec, "origin", None), str)
        or not any(filename.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    ):
        raise RuntimeError(
            "rl_engine._C is not an imported native extension; fallback is forbidden"
        )
    path = Path(filename).resolve(strict=True)
    if Path(spec.origin).resolve(strict=True) != path:
        raise RuntimeError("Provider module file and import origin disagree")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())

    # Detect replacement of the pathname as well as writes to the opened file.
    def fingerprint(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    if fingerprint(before) != fingerprint(after) or fingerprint(after) != fingerprint(path.stat()):
        raise RuntimeError("Provider binary changed while its content identity was being recorded")
    return validate_provider_identity(
        {"module": "rl_engine._C", "path": str(path), "sha256": digest.hexdigest()}
    )
