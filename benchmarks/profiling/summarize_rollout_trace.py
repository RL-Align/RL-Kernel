# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Aggregate GPU kernel time from torch.profiler traces exported by vLLM workers.

Example::

    python benchmarks/profiling/summarize_rollout_trace.py /tmp/rollout-trace --top 40
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path


def load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def summarize(path: Path, top: int, name_width: int = 110) -> None:
    trace = load_trace(path)
    events = trace["traceEvents"]
    agg: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0])
    gpu_busy = 0.0
    first = float("inf")
    last = 0.0
    kernel_events = []
    for event in events:
        if event.get("ph") != "X":
            continue
        cat = event.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset", "Kernel"):
            name = re.sub(r"\(.*$", "", event["name"])[:name_width]
            agg[name][0] += event["dur"]
            agg[name][1] += 1
            gpu_busy += event["dur"]
            first = min(first, event["ts"])
            last = max(last, event["ts"] + event["dur"])
            kernel_events.append((event["ts"], event["dur"]))
    if not kernel_events:
        print(f"{path.name}: no GPU kernel events")
        return
    kernel_events.sort()
    idle = 0.0
    cursor = kernel_events[0][0]
    for ts, dur in kernel_events:
        if ts > cursor:
            idle += ts - cursor
        cursor = max(cursor, ts + dur)
    wall = last - first
    print(
        f"{path.name}: wall={wall / 1e3:.1f}ms gpu_busy={gpu_busy / 1e3:.1f}ms "
        f"({100 * gpu_busy / wall:.1f}%) idle_gaps={idle / 1e3:.1f}ms kernels={len(kernel_events)}"
    )
    for name, (dur, count) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"  {dur / 1e3:9.2f}ms {100 * dur / gpu_busy:5.1f}% {count:7d}  {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--rank", type=int, default=None, help="only summarize this rank")
    args = parser.parse_args()
    paths = sorted(args.trace_dir.rglob("*.pt.trace.json*"))
    if not paths:
        raise SystemExit(f"no traces under {args.trace_dir}")
    for path in paths:
        if (
            args.rank is not None
            and f"rank{args.rank}" not in path.name
            and f"_{args.rank}" not in path.name
        ):
            continue
        summarize(path, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
