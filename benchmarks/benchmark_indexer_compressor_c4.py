# SPDX-License-Identifier: Apache-2.0
"""Separate binding, native pipeline and validation timing; no speedup claims."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_engine.backends.rocm.attention.indexer_compressor_c4 import AiterIndexerC4Backend
from rl_engine.contracts.operators.indexer_compressor_c4 import IndexerC4Contract
from rl_engine.models.deepseek_v4.indexer_compressor_c4 import IndexerC4Compressor
from tests.models.deepseek_v4.indexer_c4_helpers import fixture


def time_us(fn, repeats):
    for _ in range(3):
        fn()
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - begin) * 1e6 / repeats)
    return statistics.median(samples)


def run_case(backend, precision, groups, repeats):
    op = IndexerC4Compressor(IndexerC4Contract(precision=precision), backend=backend)
    x, ids, a, n, co, si = fixture(groups, device="cuda")
    bound = op.bind(x, ids, version="v1")
    rotated = backend.forward(bound, a, n, co, si, op.contract)
    return {
        "precision": precision,
        "groups": groups,
        "tokens": groups * 4,
        "bind_us": time_us(lambda: op.bind(x, ids, version="v1"), repeats),
        "native_compress_rotate_us": time_us(
            lambda: backend.forward(bound, a, n, co, si, op.contract), repeats
        ),
        "quant_only_us": time_us(lambda: backend.quantize(rotated, op.contract), repeats),
        "full_operator_us": time_us(lambda: op(bound, a, n, co, si, version="v1"), repeats),
        "implementation": op.implementation_fingerprint,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    backend = AiterIndexerC4Backend()
    result = {
        "backend": backend.provenance,
        "metric": "median of five synchronized wall-time loops; Python dispatch included",
        "scope": "no cache commit, full framework, full model, quantized VJP or speedup claim",
        "cases": [
            run_case(backend, p, g, args.repeats) for p in ("mxfp4", "fp8") for g in (1, 8, 64, 128)
        ],
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
