# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Published Qwen3-Next evidence carries no site paths, hosts, users or scheduler details."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qwen3_next_publish_evidence as publisher  # noqa: E402

PATTERNS = (
    r"(?<![\w.>-])/data/",
    r"(?<![\w.>-])/home/",
    r"(?<![\w.>-])/raid/",
    r"(?<![\w.>-])/scratch",
    r"\bdgx-",
    r"(?i)slurm",
    r"\bsbatch\b",
    r"\bsrun\b",
    r"\bwrd\b",
    r"\boverflow\b",
    r"\bjob_id\b",
)


def published_files():
    roots = [
        *ROOT.glob("docs/usage/evidence/qwen3-next-*"),
        *ROOT.glob("docs/operators/qwen3-next-*.md"),
        *ROOT.glob("docs/usage/qwen3-next-*.md"),
    ]
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if path.is_file() and path.suffix in (".md", ".json", ".txt", ".xml", ""):
                yield path


@pytest.mark.parametrize("path", sorted(published_files()), ids=lambda p: str(p.relative_to(ROOT)))
def test_published_qwen3_next_evidence_is_redacted(path):
    text = path.read_text(errors="replace")
    assert not [pattern for pattern in PATTERNS if re.search(pattern, text)]


def chain(tmp_path):
    runs_root = tmp_path / "data" / "runs"
    checkpoint = tmp_path / "data" / "ckpt" / "rev"
    previous = None
    for stage in ("preflight", "layers"):
        run = runs_root / f"20261009T000000Z-{stage}"
        (run / "attempts" / "0").mkdir(parents=True)
        manifest = {
            "stage": stage,
            "source_commit": "c" * 40,
            "checkpoint": str(checkpoint),
            "cuda_home": "/usr/local/cuda-13.0",
            "python": str(tmp_path / "data" / "env" / "bin" / "python"),
            "previous": str(previous) if previous else None,
        }
        (run / "run.json").write_text(json.dumps(manifest))
        evidence = run / "attempts" / "0" / "environment.json"
        evidence.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint / "config.json"),
                    "executable": manifest["python"],
                    "host": "dgx-1a-02-03",
                    "site": "/usr/lib/python3/site-packages/vllm/__init__.py",
                }
            )
        )
        result = {
            "status": "passed",
            "evidence": {"environment": {"path": "attempts/0/environment.json"}},
        }
        (run / "result.json").write_text(json.dumps(result))
        previous = run
    return previous


def test_publisher_replaces_every_recorded_root(tmp_path):
    out = tmp_path / "published"
    publisher.publish(publisher.chain_runs(chain(tmp_path)), None, out)
    text = (out / "layers" / "environment.json").read_text()
    assert "<CHECKPOINT>/config.json" in text and "<PYTHON>" in text
    assert "<HOST>" in text and "<SITE>" in text and str(tmp_path) not in text
    assert (out / "measured-commit.txt").read_text() == "c" * 40 + "\n"
    sums = (out / "SHA256SUMS").read_text().splitlines()
    assert any(line.endswith("preflight/run.json") for line in sums)


def test_publisher_refuses_text_that_survives_redaction(tmp_path):
    last = chain(tmp_path)
    leak = last / "attempts" / "0" / "environment.json"
    leak.write_text(json.dumps({"note": "SLURM_JOB_ID=123"}))
    with pytest.raises(SystemExit, match="survived redaction"):
        publisher.publish(publisher.chain_runs(last), None, tmp_path / "published")
    assert not (tmp_path / "published").exists()


def test_relative_package_paths_are_not_site_paths(tmp_path):
    counts = {}
    text = publisher.redact(
        '"vllm/ec_connector/cpu/data/base.py" "file:///data/x/editable/vime" "/home/b/env/bin/py"'
        ' "/sys/fs/cgroup/memory/system.slice/slurmd.service/memory.limit_in_bytes"',
        [],
        counts,
    )
    assert "cpu/data/base.py" in text and "<PATH>/vime" in text and "<PATH>/py" in text
    assert "<CGROUP>/memory.limit_in_bytes" in text
    assert not publisher.leftovers(text)
