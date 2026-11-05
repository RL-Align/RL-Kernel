# SPDX-License-Identifier: Apache-2.0
"""Validate immutable reference inventories before consuming any tensor evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path) -> str:
    """sha256 of a file's bytes (the hash the evidence inventories record)."""

    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def verify_inventory(root: Path, manifest: Path):
    records = json.loads(manifest.read_text())["files"]
    for name, expected in records.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("evidence path escapes root")
        if digest(path) != expected:
            raise ValueError(f"reference evidence changed: {name}")


def verify_generation(root):
    report = json.loads((root / "report.json").read_text())
    if report["status"] != "generation_complete_comparison_pending":
        raise ValueError("generation did not complete")
    required = {"initial.pt", "video.npy", "audio.npy"} | {
        f"step-{i:03}.pt" for i in range(report["steps"])
    }
    if not required.issubset(report["artifacts"]):
        raise ValueError("incomplete generation artifact inventory")
    for name, expected in report["artifacts"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"generation artifact changed: {name}")
    return report
