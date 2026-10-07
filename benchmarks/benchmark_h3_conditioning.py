# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Benchmark the MiniMax-H3 conditioning ops (RFC #420) against the provider path.

CUDA-event medians, bandwidth and peak memory per case, plus the backend
the registry dispatched. Timings alternate candidate/provider execution order;
backward timings exclude forward setup. Cases live in ``rl_engine/testing/h3_report.py``.

    python benchmarks/benchmark_h3_conditioning.py --op timestep_sinusoid_h3
    python benchmarks/benchmark_h3_conditioning.py --op all --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rl_engine.kernels.registry import KernelRegistry  # noqa: E402
from rl_engine.testing.h3_chain import environment  # noqa: E402
from rl_engine.testing.h3_report import PERF_CASES, TIMED_KEYS, measure  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", default="all", choices=["all", *PERF_CASES])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False

    registry = KernelRegistry()
    env = environment()
    print(f"device={env['gpu']} torch={env['torch']} cuda={env['cuda']}")
    print("timing order alternates each iteration; backward timings exclude forward setup")
    results = []
    for name in list(PERF_CASES) if args.op == "all" else [args.op]:
        for case in PERF_CASES[name](registry):
            row = measure(case, args.warmup, args.iters)
            results.append(row)
            parts = [
                f"{key}={row[f'{key}_us']:.2f}us ({row[f'{key}_gbps']:.1f} GB/s, "
                f"peak {row[f'{key}_peak_mib']:.2f} MiB)"
                for key in TIMED_KEYS
                if f"{key}_us" in row
            ]
            order = row["execution_order"]
            phases = [" -> ".join(order[f"iteration_{i}"]) for i in (0, 1)]
            print(
                f"{name} {row['case']} [{row['backend']}]: "
                + "; ".join(parts)
                + "; alternating order: "
                + " / ".join(phases)
            )
    if args.json is not None:
        args.json.write_text(json.dumps({"environment": env, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
