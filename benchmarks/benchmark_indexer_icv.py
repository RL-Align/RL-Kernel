# SPDX-License-Identifier: Apache-2.0
"""Raw ICV-only microbenchmarks. Fused MQA logits are NOT equivalent outputs."""

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import statistics
import time

import torch

from rl_engine.backends.rocm.attention.indexer_icv import AiterIndexerICVBackend
from rl_engine.contracts.operators.indexer_icv import IndexerICVContract
from rl_engine.integrations.common.indexer_icv import bind_icv_queries, bind_global_icv_keys
from rl_engine.models.deepseek_v4.indexer_icv import IndexerICV


def wall_us(fn, repeats):
    for _ in range(5):
        fn()
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / repeats)
    return statistics.median(samples)


def make_values(c, m, n):
    if c.precision == "mxfp4":
        q = torch.randint(
            0, 256, (m, c.num_heads, c.padded_dim // 2), dtype=torch.uint8, device="cuda"
        )
        k = torch.randint(0, 256, (n, c.padded_dim // 2), dtype=torch.uint8, device="cuda")
        qs = torch.full((m, c.num_heads, c.padded_dim // 32), 125, dtype=torch.uint8, device="cuda")
        ks = torch.full((n, c.padded_dim // 32), 125, dtype=torch.uint8, device="cuda")
    else:
        q = torch.randn((m, c.num_heads, c.head_dim), device="cuda").to(torch.float8_e4m3fnuz)
        k = torch.randn((n, c.head_dim), device="cuda").to(torch.float8_e4m3fnuz)
        qs = torch.full((m, c.num_heads, 1), 0.5, dtype=torch.float32, device="cuda")
        ks = torch.full((n, 1), 0.5, dtype=torch.float32, device="cuda")
    qi = torch.stack(
        (
            torch.zeros(m, dtype=torch.int64, device="cuda"),
            torch.arange(m, dtype=torch.int64, device="cuda"),
        ),
        dim=1,
    )
    ki = torch.stack(
        (
            torch.zeros(n, dtype=torch.int64, device="cuda"),
            torch.arange(n, dtype=torch.int64, device="cuda"),
        ),
        dim=1,
    )
    return q, qs, k, ks, qi, ki


def consume_tiles(op, queries, keys):
    for tile in op.iter_tiles(queries, keys):
        # Do not retain tiles; simulates a downstream consumer without its cost.
        last = tile.values
    return last


def run_case(backend, precision, m, n, repeats):
    c = IndexerICVContract(
        precision=precision, fp8_encoding="e4m3fnuz" if precision == "fp8" else None
    )
    op = IndexerICV(c, backend=backend)
    q, qs, k, ks, qi, ki = make_values(c, m, n)
    torch.cuda.synchronize()
    start = time.perf_counter()
    queries = bind_icv_queries(op, q, qs, qi, layout=c.layout, version="bench-v1")
    torch.cuda.synchronize()
    bind_us = (time.perf_counter() - start) * 1e6
    start = time.perf_counter()
    keys = bind_global_icv_keys(
        op, k, ks, ki, layout=c.layout, version="bench-v1", request_key_counts={0: n}
    )
    torch.cuda.synchronize()
    prepare_us = (time.perf_counter() - start) * 1e6
    q_prepared, qsc = backend.prepare(q, qs, c)
    key = keys.requests[0]
    qref = q_prepared.float() if qsc is None else q_prepared.float() * qsc
    kref = key.values.float() if key.scales is None else key.values.float() * key.scales
    return {
        "precision": precision,
        "M": m,
        "H": c.num_heads,
        "N": n,
        "D": c.head_dim,
        "raw_output_bytes": m * c.num_heads * n * 4,
        "max_tile_bytes": min(m, c.query_tile_rows) * c.num_heads * min(n, c.key_tile_rows) * 4,
        "query_validation_bind_cold_us": bind_us,
        "global_key_validation_prepare_cold_us": prepare_us,
        "query_validation_bind_steady_us": wall_us(
            partial(bind_icv_queries, op, q, qs, qi, layout=c.layout, version="bench-v1"),
            min(repeats, 10),
        ),
        "global_key_validation_prepare_steady_us": wall_us(
            partial(
                bind_global_icv_keys,
                op,
                k,
                ks,
                ki,
                layout=c.layout,
                version="bench-v1",
                request_key_counts={0: n},
            ),
            min(repeats, 10),
        ),
        "upstream_dot_prepared_us": wall_us(
            partial(backend.dot, q_prepared, key.values, qsc, key.scales, c),
            repeats,
        ),
        "bounded_stream_us": wall_us(partial(consume_tiles, op, queries, keys), repeats),
        "materialized_pipeline_us": wall_us(partial(op.materialize, queries, keys), repeats),
        "torch_fp32_einsum_predecoded_us": wall_us(
            partial(torch.einsum, "mhd,nd->mhn", qref, kref), repeats
        ),
        "contract": asdict(c),
        "implementation": op.implementation_fingerprint,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.manual_seed(1616)
    backend = AiterIndexerICVBackend()
    result = {
        "gpu": torch.cuda.get_device_name(),
        "backend": backend.provenance,
        "metric": "median of 5 synchronized wall-time loops; Python launch/dispatch included",
        "scope": (
            "raw independent FP32 ICV, no score fusion, no backward, not full model throughput"
        ),
        "reference_caveat": (
            "torch.einsum is a predecoded FP32 accuracy/performance reference, "
            "not the same BI arithmetic"
        ),
        "cases": [
            run_case(backend, precision, m, n, args.repeats)
            for precision in ("mxfp4", "fp8")
            for m, n in ((1, 128), (8, 512), (32, 2048))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
