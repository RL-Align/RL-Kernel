# SPDX-License-Identifier: Apache-2.0
"""Indexer score (items 10/14) microbenchmarks. Inputs are raw FP32 ICV and head weights."""

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import statistics
import time

import torch

from rl_engine.backends.rocm.attention.indexer_score import AiterIndexerScoreBackend
from rl_engine.contracts.operators.indexer_score import IndexerScoreContract
from rl_engine.models.deepseek_v4.indexer_score import IndexerScore


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


def prepare(icv, weights, c):
    relu = torch.relu(icv * c.softmax_scale)
    return relu.transpose(1, 2).contiguous(), (weights * c.head_scale)[:, None, :].expand(
        -1, icv.shape[2], -1
    )


def torch_sum(icv, weights, c):
    return (torch.relu(icv * c.softmax_scale) * (weights * c.head_scale)[:, :, None]).sum(1)


def run_case(op, m, n, repeats):
    c = op.contract
    icv = torch.randn(m, c.local_heads, n, device="cuda") * 4
    weights = torch.randn(m, c.local_heads, device="cuda")
    grad = torch.randn(m, n, device="cuda")
    record = op.forward(icv, weights)
    a, b = prepare(icv, weights, c)
    return {
        "M": m,
        "H": c.local_heads,
        "N": n,
        "icv_bytes": icv.numel() * 4,
        "forward_us": wall_us(partial(op.forward, icv, weights), repeats),
        "input_validation_us": wall_us(partial(op._check, icv, weights), repeats),
        "scale_relu_transpose_us": wall_us(partial(prepare, icv, weights, c), repeats),
        "row_dot_us": wall_us(partial(op.backend.row_dot, a, b), repeats),
        "backward_us": wall_us(partial(op.backward, record, grad), repeats),
        "torch_sum_reference_us": wall_us(partial(torch_sum, icv, weights, c), repeats),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.manual_seed(1010)
    backend = AiterIndexerScoreBackend()
    op = IndexerScore(IndexerScoreContract(num_heads=64, head_dim=128), backend=backend)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "backend": backend.provenance,
        "contract": asdict(op.contract),
        "implementation": op.implementation_fingerprint,
        "metric": "median of 5 synchronized wall-time loops; Python launch/dispatch included",
        "reference_caveat": "torch_sum is a speed reference only; its head-sum order is not BI",
        "cases": [
            run_case(op, m, n, args.repeats)
            for m, n in ((1, 512), (32, 2048), (128, 4096), (512, 4096))
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
