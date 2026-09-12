# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Download and convert DAPO-Math-17k to VIME prompt/label JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_URL = (
    "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/"
    "resolve/main/data/train-00000-of-00001.parquet"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
    partial.replace(destination)


def _index(row: dict[str, Any]) -> str:
    extra = row.get("extra_info") or {}
    value = extra.get("index") if isinstance(extra, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError("DAPO row is missing extra_info.index")
    return value


def convert(source: Path, output: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("prepare_dapo_data.py requires pyarrow") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    seen: set[str] = set()
    rows_read = 0
    rows_written = 0
    with partial.open("w", encoding="utf-8") as destination:
        parquet_file = parquet.ParquetFile(source)
        for batch in parquet_file.iter_batches(
            batch_size=4096,
            columns=["data_source", "prompt", "ability", "reward_model", "extra_info"],
        ):
            for row in batch.to_pylist():
                rows_read += 1
                row_id = _index(row)
                if row_id in seen:
                    continue
                seen.add(row_id)
                reward_model = row.get("reward_model") or {}
                label = reward_model.get("ground_truth")
                prompt = row.get("prompt")
                if not isinstance(prompt, list) or not isinstance(label, str) or not label:
                    raise ValueError(f"invalid prompt or ground truth for row {row_id}")
                record = {
                    "prompt": prompt,
                    "label": label,
                    "data_source": row.get("data_source"),
                    "ability": row.get("ability"),
                    "source_index": row_id,
                    "reward_style": reward_model.get("style"),
                }
                destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                destination.write("\n")
                rows_written += 1
    partial.replace(output)
    return {
        "schema_version": "rlkernel.dapo_math_17k.v1",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "rows_read": rows_read,
        "rows_written": rows_written,
        "duplicates_removed": rows_read - rows_written,
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-url", default=DEFAULT_URL)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download and not args.source.exists():
        _download(args.download_url, args.source)
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    manifest = convert(args.source, args.output)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
