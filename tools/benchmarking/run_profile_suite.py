# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""
Automated Performance Profiling Suite — CLI Orchestrator.

Wraps ``benchmarks.common.profiler.run_automated_suite()`` with contributor-friendly
report management:

    --output-dir          Directory for emitted reports
    --json / --csv        Emit one or both formats (default: both)
    --no-summary          Skip the terminal table

Usage:
    python tools/benchmarking/run_profile_suite.py --smoke --workloads logp-native
    python tools/benchmarking/run_profile_suite.py --output-dir reports/ --csv
    python tools/benchmarking/run_profile_suite.py --device cuda --workloads logp-fused
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.profiler import build_arg_parser, run_automated_suite  # noqa: E402


def build_script_parser():
    """Extend the core profiler CLI with report-management flags."""
    parser = build_arg_parser()
    for action in list(parser._actions):
        if action.dest == "format":
            parser._remove_action(action)
            break

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to store timestamped JSON/CSV reports",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Emit a JSON report (default: True)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        default=True,
        help="Emit a CSV report (default: True)",
    )
    return parser


def main() -> None:
    args = build_script_parser().parse_args()

    if args.smoke:
        args.dtype = "float32"
        args.warmup = 0
        args.repeat = 1

    profiler = run_automated_suite(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_gpu_name = profiler.gpu_info.name.replace(" ", "_").replace("/", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.json:
        json_path = args.output_dir / f"perf_report_{safe_gpu_name}_{timestamp}.json"
        profiler.save_json(json_path)

    if args.csv:
        csv_path = args.output_dir / f"perf_report_{safe_gpu_name}.csv"
        profiler.save_csv(csv_path)

    if args.output is not None:
        if args.output.suffix.lower() == ".json":
            profiler.save_json(args.output)
        else:
            profiler.save_csv(args.output)

    if not args.no_summary:
        profiler.print_summary()


if __name__ == "__main__":
    main()
