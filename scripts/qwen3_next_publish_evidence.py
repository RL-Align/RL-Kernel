#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Copy a finished Qwen3-Next chain's evidence into the repository, without site details.

    python scripts/qwen3_next_publish_evidence.py --last <train run dir> \\
        --chain-report chain-report.json --out docs/usage/evidence/qwen3-next-chain-<date>

For every stage run on the chain (preflight -> ... -> the given run) this copies
``run.json``, ``result.json`` and every evidence file the result names (JSON
and XML only; tensor artifacts stay with the run), plus the chain report. Each
absolute root is replaced by a placeholder - ``<CHECKPOINT>``, ``<RUN>``,
``<PYTHON>``, ``<CUDA_HOME>``, ``<SITE>``, ``<HOME>`` - and host and user names
by ``<HOST>`` / ``<USER>``. The copy fails if any file still contains a
site-specific path or name afterwards. ``redaction.json`` records what was
replaced and ``SHA256SUMS`` the published bytes; ``measured-commit.txt`` names
the commit the stages ran.
"""

import argparse
import getpass
import hashlib
import json
import re
import shutil
import socket
from pathlib import Path

FORBIDDEN = (
    r"/data/",
    r"/home/",
    r"/raid/",
    r"/scratch",
    r"\bdgx-",
    r"SLURM_",
)
TEXT_SUFFIXES = (".json", ".xml", ".txt", ".log")


def chain_runs(last):
    runs, run = [], Path(last).resolve()
    while run is not None:
        manifest = json.loads((run / "run.json").read_text())
        runs.append((manifest, run))
        run = Path(manifest["previous"]) if manifest.get("previous") else None
    if runs[-1][0]["stage"] != "preflight":
        raise SystemExit("The chain does not lead back to a preflight run")
    if len({manifest["source_commit"] for manifest, _ in runs}) != 1:
        raise SystemExit("The chain mixes source commits")
    return list(reversed(runs))


def replacements(runs):
    """Longest-first (path, placeholder) pairs for every root the chain recorded."""
    pairs = {}
    for manifest, run in runs:
        pairs[str(run)] = "<RUN>"
        pairs[str(run.parent)] = "<RUNS>"
        pairs[manifest["checkpoint"]] = "<CHECKPOINT>"
        pairs[manifest["cuda_home"]] = "<CUDA_HOME>"
        pairs[manifest["python"]] = "<PYTHON>"
        python_root = Path(manifest["python"]).parents[1]
        pairs[str(python_root)] = "<PYTHON_ENV>"
        if manifest.get("training_storage"):
            pairs[manifest["training_storage"]] = "<TRAINING_STORAGE>"
    pairs[str(Path.home())] = "<HOME>"
    return sorted(pairs.items(), key=lambda item: -len(item[0]))


def redact(text, pairs, counts):
    for value, placeholder in pairs:
        if value and value in text:
            counts[placeholder] = counts.get(placeholder, 0) + text.count(value)
            text = text.replace(value, placeholder)
    for pattern, placeholder in (
        (r"/[^\s\"']*/site-packages", "<SITE>"),
        (r"\bdgx-[A-Za-z0-9-]+", "<HOST>"),
        (rf"\b{re.escape(socket.gethostname())}\b", "<HOST>"),
        (rf"\b{re.escape(getpass.getuser())}\b", "<USER>"),
    ):
        text, count = re.subn(pattern, placeholder, text)
        if count:
            counts[placeholder] = counts.get(placeholder, 0) + count
    return text


def leftovers(text):
    return [pattern for pattern in FORBIDDEN if re.search(pattern, text)]


def publish(runs, chain_report, out):
    out.mkdir(parents=True, exist_ok=False)
    pairs, counts, files = replacements(runs), {}, []
    for manifest, run in runs:
        stage_dir = out / manifest["stage"]
        stage_dir.mkdir()
        result = json.loads((run / "result.json").read_text())
        sources = {"run.json": run / "run.json", "result.json": run / "result.json"}
        for name, record in result.get("evidence", {}).items():
            path = run / record["path"]
            if path.suffix in TEXT_SUFFIXES and path.is_file():
                sources[f"{name}{path.suffix}"] = path
        for name, source in sources.items():
            target = stage_dir / name
            target.write_text(redact(source.read_text(), pairs, counts))
            files.append(target)
    if chain_report is not None:
        target = out / "chain-report.json"
        target.write_text(redact(Path(chain_report).read_text(), pairs, counts))
        files.append(target)
    (out / "measured-commit.txt").write_text(runs[0][0]["source_commit"] + "\n")
    (out / "redaction.json").write_text(
        json.dumps({"placeholders": dict(sorted(counts.items()))}, indent=2) + "\n"
    )
    files += [out / "measured-commit.txt", out / "redaction.json"]
    dirty = {str(f.relative_to(out)): leftovers(f.read_text()) for f in files}
    dirty = {name: found for name, found in dirty.items() if found}
    if dirty:
        shutil.rmtree(out)
        raise SystemExit(f"Site-specific text survived redaction: {dirty}")
    sums = [
        f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out)}" for f in sorted(files)
    ]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last", type=Path, required=True)
    parser.add_argument("--chain-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = chain_runs(args.last)
    if any(not (run / "result.json").is_file() for _, run in runs):
        raise SystemExit("Every stage on the chain must have finished")
    publish(runs, args.chain_report, args.out)
    print(f"published {len(runs)} stages to {args.out}")


if __name__ == "__main__":
    main()
