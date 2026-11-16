# SPDX-License-Identifier: Apache-2.0
"""Indexer stable Top-512 (item 17) microbenchmarks against torch.topk."""

import argparse
from functools import partial
import json
from pathlib import Path

import torch

from benchmarks.benchmark_indexer_score import wall_us
from rl_engine.models.deepseek_v4.indexer_topk import IndexerTopK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(1717)
    op = IndexerTopK()
    cases = []
    for m, n in ((1, 4096), (32, 4096), (128, 16384), (512, 16384), (16, 131072)):
        s = torch.randn(m, n, device="cuda")
        lengths = torch.full((m,), n, dtype=torch.int32, device="cuda")
        cases.append(
            {
                "M": m,
                "N": n,
                "op_us": wall_us(partial(op, s, lengths), args.repeats),
                "kernel_us": wall_us(partial(op.backend.select, s, lengths, 512), args.repeats),
                "torch_topk_us": wall_us(partial(torch.topk, s, 512, 1), args.repeats),
            }
        )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "backend": op.backend.provenance,
        "implementation": op.implementation_fingerprint,
        "metric": "median of 5 synchronized wall-time loops; Python launch included",
        "reference_caveat": "torch.topk has no tie rule and returns score order",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
