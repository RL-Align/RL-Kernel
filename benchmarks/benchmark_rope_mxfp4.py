# SPDX-License-Identifier: Apache-2.0
"""Items 11/15 microbenchmarks: partial RoPE alone and the staged Indexer Q path."""

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import statistics
import time

import torch

from rl_engine.backends.rocm.attention.partial_rope import (
    AiterIndexerQBackend,
    AiterPartialRoPEBackend,
)
from rl_engine.contracts.operators.indexer_q_preprocess import IndexerQPreprocessContract
from rl_engine.contracts.operators.partial_rope import PartialRoPEContract
from rl_engine.models.deepseek_v4.indexer_q_preprocess import IndexerQPreprocess
from rl_engine.models.deepseek_v4.partial_rope import PartialRoPE, select_angles


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


def tables(n):
    angle = torch.arange(n, device="cuda").double()[:, None] * (
        10000.0 ** (-torch.arange(0, 64, 2, device="cuda").double() / 64)
    )
    return angle.cos().bfloat16(), angle.sin().bfloat16()


def run_case(rope, q_op, t, repeats):
    backend = q_op.backend
    cos, sin = tables(8192)
    positions = torch.randint(0, 8192, (t,), device="cuda")
    angles = (
        cos.index_select(0, positions).float().reshape(-1, 1, 1, 32),
        sin.index_select(0, positions).float().reshape(-1, 1, 1, 32),
    )
    q = torch.randn(t, q_op.contract.num_heads, 128, device="cuda").bfloat16()
    main = torch.randn(t, 128, 512, device="cuda").bfloat16()
    rotated = backend.rotate(q, *angles)
    hadamard = backend.codec._rotate(rotated.reshape(-1, 128))
    grad = torch.randn_like(q)
    return {
        "T": t,
        "rope128_us": wall_us(partial(rope[128], q, positions, cos, sin), repeats),
        "rope512_us": wall_us(partial(rope[512], main, positions, cos, sin), repeats),
        "rope128_kernel_us": wall_us(partial(backend.rotate, q, *angles), repeats),
        "rope512_kernel_us": wall_us(partial(backend.rotate, main, *angles), repeats),
        "validation_us": wall_us(
            partial(
                select_angles,
                q,
                positions,
                cos,
                sin,
                contract=q_op.contract.rope,
                backend=backend,
            ),
            repeats,
        ),
        "hadamard_us": wall_us(partial(backend.codec._rotate, rotated.reshape(-1, 128)), repeats),
        "mxfp4_us": wall_us(partial(backend.codec.quantize, hadamard, q_op.contract), repeats),
        "indexer_q_us": wall_us(partial(q_op, q, positions, cos, sin), repeats),
        "indexer_q_vjp_us": wall_us(partial(backend.vjp, grad, *angles), repeats),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    torch.manual_seed(1115)
    backend = AiterIndexerQBackend()
    rope_backend = AiterPartialRoPEBackend()
    rope = {
        dim: PartialRoPE(PartialRoPEContract(head_dim=dim, rope_dim=64), backend=rope_backend)
        for dim in (128, 512)
    }
    q_op = IndexerQPreprocess(IndexerQPreprocessContract(num_heads=64), backend=backend)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "backend": backend.provenance,
        "contract": asdict(q_op.contract),
        "implementation": q_op.implementation_fingerprint,
        "metric": "median of 5 synchronized wall-time loops; Python launch/dispatch included",
        "heads": {"indexer": 64, "main": 128},
        "cases": [run_case(rope, q_op, t, args.repeats) for t in (1, 64, 512, 4096)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["cases"], indent=2))


if __name__ == "__main__":
    main()
