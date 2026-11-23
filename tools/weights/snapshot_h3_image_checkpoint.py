#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Write ``minimax_h3_image/pinned_checkpoint.json``: the Hub facts the H3-Image fingerprint uses.

Reads, at the pinned revision of ``MiniMaxAI/MiniMax-H3-Image``, without downloading whole
weight files:

* the file tree (path, size, LFS sha256) of every component directory and of the root PDD files;
* every ``config.json`` / ``scheduler_config.json`` / index JSON the fingerprint checks;
* the safetensors headers (name, dtype, shape) of the two transformer layouts, the two VAE
  layouts and ``pdd_fused_heads.safetensors``, each reduced to a digest, a parameter count
  and a tensor count;
* the bytes of the output heads only (range reads, about 15 MB in total): the six fused PDD
  heads, the six Diffusers ``pdd_proj_out.<i>`` heads, and the upstream layout's single
  ``final_layer.video_out``. Their sha256 values let the tests prove that the two PDD layouts
  hold the same heads and that ``video_out`` is not one of them;
* the ``text_encoder/`` file tree of ``MiniMaxAI/MiniMax-H3`` at the H3 pinned revision, to
  check the inherited text encoder file by file.

    python tools/weights/snapshot_h3_image_checkpoint.py [--out PATH] [--token-file PATH]

The token comes from ``HF_TOKEN`` or the Hugging Face token file. The output is committed;
``tests/models/minimax_h3_image/test_h3_image_arch_fingerprint.py`` checks it against the
analytic tensor table, so a stale or edited file fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rl_engine.models.minimax_h3 import fingerprint as h3_fp  # noqa: E402
from rl_engine.models.minimax_h3_image import fingerprint as fp  # noqa: E402

DIRECTORIES = (
    "transformer",
    "transformer_diffusers",
    "vae",
    "vae_diffusers",
    "scheduler",
    "processor",
    "tokenizer",
    "text_encoder",
)
ROOT_FILES = ("model_index.json", "modular_model_index.json", "pdd_config.json")
CONFIG_FILES = {
    "transformer": "transformer/config.json",
    "transformer_diffusers": "transformer_diffusers/config.json",
    "vae": "vae/config.json",
    "vae_diffusers": "vae_diffusers/config.json",
    "scheduler": "scheduler/scheduler_config.json",
    "model_index": "model_index.json",
    "modular_model_index": "modular_model_index.json",
    "pdd_config": "pdd_config.json",
}
TABLES = ("transformer", "transformer_diffusers", "vae", "vae_diffusers")


def _token(token_file: Path | None) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        path = token_file or Path.home() / ".cache" / "huggingface" / "token"
        token = path.read_text().strip() if path.is_file() else ""
    if not token:
        raise SystemExit("set HF_TOKEN or pass --token-file")
    return token


class Hub:
    """Minimal authenticated reader for one pinned revision of one repo."""

    def __init__(self, token: str, repo: str, revision: str):
        self.token, self.repo, self.revision = token, repo, revision

    def _open(self, url: str, headers: dict[str, str] | None = None):
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.token}", **(headers or {})}
        )
        return urllib.request.urlopen(request, timeout=120)

    def tree(self, directory: str = "") -> list[dict]:
        suffix = f"/{directory}" if directory else ""
        url = (
            f"https://huggingface.co/api/models/{self.repo}/tree/{self.revision}{suffix}"
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
        url = f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{path}"
        return self._open(url, headers).read()

    def header(self, path: str) -> tuple[int, dict]:
        (length,) = struct.unpack("<Q", self.resolve(path, {"Range": "bytes=0-7"}))
        header = json.loads(self.resolve(path, {"Range": f"bytes=8-{7 + length}"}))
        header.pop("__metadata__", None)
        return length, header

    def tensor_sha256(self, path: str, length: int, item: dict) -> str:
        start, end = item["data_offsets"]
        base = 8 + length
        data = self.resolve(path, {"Range": f"bytes={base + start}-{base + end - 1}"})
        if len(data) != end - start:
            raise SystemExit(f"short range read for {path}")
        return hashlib.sha256(data).hexdigest()


def _files(entries: list[dict]) -> list[dict]:
    return sorted(
        (
            {
                "path": entry["path"],
                "size": entry["size"],
                "sha256": (entry.get("lfs") or {}).get("oid"),
                "git_oid": entry["oid"],
            }
            for entry in entries
            if entry["type"] == "file"
        ),
        key=lambda f: f["path"],
    )


def build_snapshot(hub: Hub, h3_hub: Hub) -> dict:
    snapshot: dict = {
        "repo": fp.REPO,
        "revision": fp.REVISION,
        "dirs": {},
        "configs": {},
        "tensor_tables": {},
        "head_sha256": {},
    }
    for directory in DIRECTORIES:
        snapshot["dirs"][directory] = _files(hub.tree(directory))
    root = {entry["path"]: entry for entry in hub.tree() if entry["type"] == "file"}
    snapshot["dirs"]["pdd"] = _files(
        [root[name] for name in ("pdd_config.json", "pdd_fused_heads.safetensors")]
    )
    snapshot["dirs"]["pipeline"] = _files([root[name] for name in ROOT_FILES[:2]])
    for key, path in CONFIG_FILES.items():
        snapshot["configs"][key] = json.loads(hub.resolve(path))

    headers: dict[str, list[tuple[str, int, dict]]] = {}
    for directory in TABLES:
        table: dict[str, tuple[str, tuple[int, ...]]] = {}
        headers[directory] = []
        for entry in snapshot["dirs"][directory]:
            if entry["path"].endswith(".safetensors"):
                length, header = hub.header(entry["path"])
                headers[directory].append((entry["path"], length, header))
                for name, item in header.items():
                    table[name] = (item["dtype"], tuple(item["shape"]))
        snapshot["tensor_tables"][directory] = {
            "digest": h3_fp.tensor_table_digest(table),
            "parameters": h3_fp.parameter_count(table),
            "tensors": len(table),
        }
    length, pdd_header = hub.header(fp.PDD_FILE)
    pdd_table = {name: (item["dtype"], tuple(item["shape"])) for name, item in pdd_header.items()}
    snapshot["tensor_tables"]["pdd"] = {
        "digest": h3_fp.tensor_table_digest(pdd_table),
        "parameters": h3_fp.parameter_count(pdd_table),
        "tensors": len(pdd_table),
        "table": {name: [dtype, list(shape)] for name, (dtype, shape) in sorted(pdd_table.items())},
    }

    # Head bytes. The fused file stores all six heads in one tensor; hash each head's slice.
    weight, bias = pdd_header[fp.PDD_WEIGHT], pdd_header[fp.PDD_BIAS]
    w_start, w_end = weight["data_offsets"]
    b_start, b_end = bias["data_offsets"]
    base = 8 + length
    w_raw = hub.resolve(fp.PDD_FILE, {"Range": f"bytes={base + w_start}-{base + w_end - 1}"})
    b_raw = hub.resolve(fp.PDD_FILE, {"Range": f"bytes={base + b_start}-{base + b_end - 1}"})
    heads = fp.NUM_PDD_HEADS
    w_step, b_step = len(w_raw) // heads, len(b_raw) // heads
    snapshot["head_sha256"]["pdd_fused"] = [
        {
            "weight": hashlib.sha256(w_raw[i * w_step : (i + 1) * w_step]).hexdigest(),
            "bias": hashlib.sha256(b_raw[i * b_step : (i + 1) * b_step]).hexdigest(),
        }
        for i in range(heads)
    ]

    def head(directory: str, weight_name: str, bias_name: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for path, length, header in headers[directory]:
            for key, name in (("weight", weight_name), ("bias", bias_name)):
                if name in header:
                    found[key] = hub.tensor_sha256(path, length, header[name])
        if set(found) != {"weight", "bias"}:
            raise SystemExit(f"{directory}: {weight_name} / {bias_name} not found")
        return found

    snapshot["head_sha256"]["diffusers_pdd_proj_out"] = [
        head("transformer_diffusers", f"pdd_proj_out.{i}.weight", f"pdd_proj_out.{i}.bias")
        for i in range(heads)
    ]
    snapshot["head_sha256"]["upstream_video_out"] = head(
        "transformer", "final_layer.video_out.weight", "final_layer.video_out.bias"
    )
    snapshot["inherited_text_encoder"] = {
        "repo": h3_hub.repo,
        "revision": h3_hub.revision,
        "files": _files(h3_hub.tree("text_encoder")),
    }
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=fp.PINNED_CHECKPOINT_PATH)
    parser.add_argument("--token-file", type=Path, default=None)
    args = parser.parse_args()
    token = _token(args.token_file)
    snapshot = build_snapshot(
        Hub(token, fp.REPO, fp.REVISION), Hub(token, fp.TEXT_ENCODER_REPO, fp.TEXT_ENCODER_REVISION)
    )
    args.out.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    for directory, info in snapshot["tensor_tables"].items():
        print(f"{directory}: {info['tensors']} tensors, {info['parameters']:,} parameters")


if __name__ == "__main__":
    main()
