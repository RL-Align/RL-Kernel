#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Write ``pinned_checkpoint.json``: the Hub facts the H3 fingerprint is built on.

Reads, at the pinned revision of ``MiniMaxAI/MiniMax-H3`` and without downloading
any weights:

* the file tree (path, size, LFS sha256) of the variant transformers, the two
  schedulers and the FL2VA / Ref2VA wrapper directories;
* ``config.json`` / ``scheduler_config.json`` of each;
* the safetensors headers (name, dtype, shape) of every transformer shard, reduced to a
  digest, a parameter count and a tensor count per variant.

    python tools/weights/snapshot_h3_checkpoint.py [--out PATH] [--token-file PATH]

The token comes from ``HF_TOKEN`` or the Hugging Face token file. The output is
committed; ``tests/models/minimax_h3/test_h3_arch_fingerprint.py`` checks it
against the analytic tensor table so a stale or edited file fails closed.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.models.minimax_h3 import fingerprint as fp  # noqa: E402

DIRECTORIES = ("transformer", "transformer_ref", "scheduler", "audio_scheduler", "FL2VA", "Ref2VA")
CONFIG_FILES = {
    "transformer": "config.json",
    "transformer_ref": "config.json",
    "scheduler": "scheduler_config.json",
    "audio_scheduler": "scheduler_config.json",
}


def _token(token_file: Path | None) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        path = token_file or Path.home() / ".cache" / "huggingface" / "token"
        token = path.read_text().strip() if path.is_file() else ""
    if not token:
        raise SystemExit("set HF_TOKEN or pass --token-file (the repo requires access)")
    return token


class Hub:
    """Minimal authenticated reader for one pinned revision."""

    def __init__(self, token: str):
        self.token = token

    def _open(self, url: str, headers: dict[str, str] | None = None):
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.token}", **(headers or {})}
        )
        return urllib.request.urlopen(request, timeout=60)

    def tree(self, directory: str) -> list[dict]:
        url = (
            f"https://huggingface.co/api/models/{fp.REPO}/tree/{fp.REVISION}/{directory}"
            "?recursive=true"
        )
        entries: list[dict] = []
        while url:
            response = self._open(url)
            entries += json.load(response)
            link = response.headers.get("Link") or ""
            url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None
        return entries

    def resolve(self, path: str, headers: dict[str, str] | None = None) -> bytes:
        url = f"https://huggingface.co/{fp.REPO}/resolve/{fp.REVISION}/{path}"
        return self._open(url, headers).read()

    def safetensors_header(self, path: str) -> dict:
        (length,) = struct.unpack("<Q", self.resolve(path, {"Range": "bytes=0-7"}))
        header = json.loads(self.resolve(path, {"Range": f"bytes=8-{7 + length}"}))
        header.pop("__metadata__", None)
        return header


def build_snapshot(hub: Hub) -> dict:
    snapshot: dict = {"repo": fp.REPO, "revision": fp.REVISION, "dirs": {}, "configs": {}}
    for directory in DIRECTORIES:
        files = [
            {
                "path": entry["path"],
                "size": entry["size"],
                "sha256": (entry.get("lfs") or {}).get("oid"),
                "git_oid": entry["oid"],
            }
            for entry in hub.tree(directory)
            if entry["type"] == "file"
        ]
        snapshot["dirs"][directory] = sorted(files, key=lambda f: f["path"])
        if directory in CONFIG_FILES:
            raw = hub.resolve(f"{directory}/{CONFIG_FILES[directory]}")
            snapshot["configs"][directory] = json.loads(raw)
    tables: dict = {}
    for directory in ("transformer", "transformer_ref"):
        table: dict[str, tuple[str, tuple[int, ...]]] = {}
        for entry in snapshot["dirs"][directory]:
            if entry["path"].endswith(".safetensors"):
                for name, item in hub.safetensors_header(entry["path"]).items():
                    table[name] = (item["dtype"], tuple(item["shape"]))
        tables[directory] = {
            "digest": fp.tensor_table_digest(table),
            "parameters": fp.parameter_count(table),
            "tensors": len(table),
        }
    snapshot["tensor_tables"] = tables
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=fp.PINNED_CHECKPOINT_PATH)
    parser.add_argument("--token-file", type=Path, default=None)
    args = parser.parse_args()
    snapshot = build_snapshot(Hub(_token(args.token_file)))
    args.out.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    for directory, info in snapshot["tensor_tables"].items():
        print(f"{directory}: {info['tensors']} tensors, {info['parameters']:,} parameters")


if __name__ == "__main__":
    main()
