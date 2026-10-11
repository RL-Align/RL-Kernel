#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Copy distributed-logprob parity outputs into the repository without site details.

    python tools/validation/models/qwen3_next_publish_parity_evidence.py \\
        --parity <one sub-directory per NCCL setting> --parity-report report.json \\
        --prior-art <prior-art output dir> \\
        --out docs/usage/evidence/qwen3-next-distributed-logprob-b200

Writes ``parity/runs/<setting>/rank-*.json``, ``parity/report.json``,
``prior-art/report.json`` and ``prior-art/runs/<launch>/rank-0.json``, redacted
with the chain publisher's rules, plus ``redaction.json`` and ``SHA256SUMS``.
Refuses to publish anything a forbidden pattern still matches.
"""

import argparse
import hashlib
import json
from pathlib import Path

from qwen3_next_publish_evidence import leftovers, redact


def publish(parity: Path, parity_report: Path, prior_art: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    sources = {Path("parity/report.json"): parity_report}
    for run in sorted(p for p in parity.iterdir() if p.is_dir()):
        for rank in sorted(run.glob("rank-*.json")):
            sources[Path("parity/runs") / run.name / rank.name] = rank
    sources[Path("prior-art/report.json")] = prior_art / "report.json"
    for run in sorted(p for p in prior_art.iterdir() if p.is_dir()):
        sources[Path("prior-art/runs") / run.name / "rank-0.json"] = run / "rank-0.json"
    counts: dict[str, int] = {}
    for relative, source in sources.items():
        text = redact(source.read_text(), [], counts)
        found = leftovers(text)
        if found:
            raise SystemExit(f"{relative} still matches {found}")
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (out / "redaction.json").write_text(json.dumps({"placeholders": counts}, indent=2) + "\n")
    sums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(out)}"
        for path in sorted(out.rglob("*"))
        if path.is_file()
    ]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--prior-art", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    counts = publish(args.parity, args.parity_report, args.prior_art, args.out)
    print(f"published to {args.out}; placeholders {counts}")


if __name__ == "__main__":
    main()
