#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Write the Qwen3-Next routed-MoE prior-art report (BI + accuracy + latency) as JSON.

Run it on a clean tree with ``TRITON_F32_DEFAULT=ieee`` (the FP32 router needs
IEEE Triton dots from process start). The report records the commit, whether
the tracked tree was dirty and the version of every library compared.
``scripts/plot_qwen3_next_moe_prior_art.py`` turns the report into a figure.

    TRITON_F32_DEFAULT=ieee python scripts/qwen3_next_moe_prior_art.py \\
        --out docs/usage/evidence/qwen3-next-moe-route-b200/report.json
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rl_engine.testing.qwen3_next_moe_prior_art import moe_report  # noqa: E402

LIBRARIES = (
    "vllm",
    "flashinfer-python",
    "transformers",
    "triton",
    "sglang",
    "megatron-core",
    "transformer_engine",
    "vime",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {
        "kind": "qwen3_next_prior_art_report",
        "rfc": 428,
        "rl_kernel_commit": _git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "libraries": {name: _version(name) for name in LIBRARIES},
        },
        **moe_report(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"wrote {args.out} (commit {report['rl_kernel_commit'][:7]}, "
        f"dirty={report['tracked_tree_dirty']})"
    )


if __name__ == "__main__":
    main()
