# SPDX-License-Identifier: Apache-2.0
"""Stateful append timing includes validation, staging and synchronous commit."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_engine.backends.rocm.attention.compressor_state import AiterCompressorStateBackend
from rl_engine.contracts.operators.compressor_state import CompressorStateContract
from rl_engine.models.deepseek_v4.compressor_state import CompressorStateCache
from tests.models.deepseek_v4.compressor_state_helpers import tensors


def case(backend, ratio, graph, rows, repeats):
    c = CompressorStateContract(
        ratio=ratio,
        graph=graph,
        max_requests=1,
        max_tokens_per_step=rows,
        max_sequence_tokens=max(ratio, rows * (5 * repeats + 3)),
    )
    x, a, n, co, si = tensors(c, device="cuda")
    arena = CompressorStateCache(
        c,
        ape=a,
        norm=n,
        cos=co,
        sin=si,
        version="benchmark",
        writer_rank=0,
        expected_arithmetic=c.arithmetic_id,
        backend=backend,
    )
    h = arena.create_request(0, version="benchmark", writer_rank=0)
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
        "ratio": ratio,
        "graph": graph,
        "rows_per_append": rows,
        "append_us": statistics.median(samples),
        "processed_tokens": start,
        "complete_cache_rows": len(snapshot["rows"]),
        "resident_buffer_bytes": c.resident_bytes,
        "arithmetic_id": c.arithmetic_id,
        "implementation": arena.implementation_fingerprint,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    backend = AiterCompressorStateBackend()
    report = {
        "provider": backend.provenance,
        "metric": "median of five synchronized loops; validation and device completion included",
        "scope": "single writer; not kernel-only latency or full model throughput",
        "cases": [
            case(backend, ratio, graph, rows, args.repeats)
            for ratio in (4, 128)
            for graph in (False, True)
            for rows in (1, 256)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
