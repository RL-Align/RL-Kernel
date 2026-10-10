# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Launch the PR230 Attention P/R matrix through a real Vime ROCm job."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rl_engine.validation.ablation.rocm import ROCM_ATTENTION_CASE_IDS, run_rocm_attention_ablation


def _case_id(value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in ROCM_ATTENTION_CASE_IDS:
        raise argparse.ArgumentTypeError(
            f"case must be one of {', '.join(ROCM_ATTENTION_CASE_IDS)}"
        )
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/rocm-attention-ablation"),
        help="case logs, framework readbacks, and summary location",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=_case_id,
        dest="cases",
        help="run only one matrix cell; repeat to select multiple cells",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="execute the orchestration command (default: review-only dry run)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Vime command after '--', for example: -- bash scripts/run-qwen3.sh",
    )
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a Vime orchestration command is required after '--'")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_rocm_attention_ablation(
        args.command,
        output_dir=args.output_dir.resolve(),
        base_environment=os.environ,
        case_ids=args.cases or ROCM_ATTENTION_CASE_IDS,
        execute=args.run,
    )
    for result in results:
        print(
            f"[{result.status.upper()}] Attention={result.case_id} "
            f"log={result.log_path} readbacks={result.readback_dir}"
        )
        for error in result.errors:
            print(f"  - {error}")
    print(f"summary={args.output_dir.resolve() / 'summary.md'}")
    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
