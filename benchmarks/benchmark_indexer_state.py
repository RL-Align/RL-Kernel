# SPDX-License-Identifier: Apache-2.0
"""Reuse the state benchmark shape: one synchronous Indexer cache append."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_engine.backends.rocm.attention.indexer_state import AiterIndexerStateBackend
from tests.models.deepseek_v4.indexer_state_helpers import make


def case(backend, graph, rows, repeats):
    arena, x, _, _ = make(
        backend,
        graph=graph,
        step=rows,
        tokens=max(4, rows * (5 * repeats + 3)),
        max_requests=1,
        device="cuda",
    )
    h = arena.create_request(0, version="v1", writer_rank=0)
    start = 0
    for _ in range(3):
        arena.append(h, x[start : start + rows], start_position=start)
        start += rows
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(repeats):
            arena.append(h, x[start : start + rows], start_position=start)
            start += rows
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - begin) * 1e6 / repeats)
    snapshot = arena.snapshot(h)
    arena.release(h)
    return {
        "graph": graph,
        "rows_per_append": rows,
        "append_us": statistics.median(samples),
        "processed_tokens": start,
        "complete_rows": snapshot["packed"].shape[0],
        "resident_buffer_bytes": arena.contract.resident_bytes,
        "implementation": arena.implementation_fingerprint,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    backend = AiterIndexerStateBackend()
    report = {
        "provider": backend.provenance,
        "metric": "median of five synchronized loops, validation/staging/commit included",
        "scope": "single-writer complete cache API; not kernel-only latency or model throughput",
        "cases": [
            case(backend, graph, rows, args.repeats) for graph in (False, True) for rows in (1, 256)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
