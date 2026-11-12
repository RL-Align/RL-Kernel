# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark sharing the deterministic ROCm RoPE table across vLLM Q/K.

The baseline reproduces the pre-optimization strict vLLM adapter: convert the
flattened Q/K tensors to head-major layout and invoke the deterministic RoPE
operator twice.  The candidate keeps the same layout transforms and HIP kernel
launches, but builds the FP32 cos/sin table once through ``forward_pair``.

Example on one MI300X:

    HIP_VISIBLE_DEVICES=2 CUDA_VISIBLE_DEVICES=2 \
      python benchmarks/backends/rocm/benchmark_rocm_rope_pair.py \
      --tokens 2,32 --blocks 10 --iterations 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable

import torch

from rl_engine.backends.rocm.rope.rope import RocmDeterministicRoPEOp


def _parse_dtype(value: str) -> torch.dtype:
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        return dtypes[value]
    except KeyError as error:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {value}") from error


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _elapsed_ms(function: Callable[[], object], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _benchmark_shape(
    *,
    tokens: int,
    query_heads: int,
    key_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    blocks: int,
    iterations: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(390_000 + tokens)
    query = torch.randn(
        tokens,
        query_heads * head_dim,
        generator=generator,
        dtype=dtype,
    ).cuda()
    key = torch.randn(
        tokens,
        key_heads * head_dim,
        generator=generator,
        dtype=dtype,
    ).cuda()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    operator = RocmDeterministicRoPEOp()

    def head_major(value: torch.Tensor, heads: int) -> torch.Tensor:
        return value.view(tokens, heads, head_dim).permute(1, 0, 2).contiguous()

    def restore(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return value.permute(1, 0, 2).reshape_as(reference).contiguous()

    @torch.inference_mode()
    def legacy() -> tuple[torch.Tensor, torch.Tensor]:
        # Keep the same left-to-right evaluation order as the old adapter's
        # ``return apply(query), apply(key)`` implementation.
        query_major = head_major(query, query_heads)
        query_out = operator(query_major, positions)
        query_out = restore(query_out, query)
        key_major = head_major(key, key_heads)
        key_out = operator(key_major, positions)
        return query_out, restore(key_out, key)

    @torch.inference_mode()
    def shared_table() -> tuple[torch.Tensor, torch.Tensor]:
        query_major = head_major(query, query_heads)
        key_major = head_major(key, key_heads)
        query_out, key_out = operator.forward_pair(query_major, key_major, positions)
        return restore(query_out, query), restore(key_out, key)

    for _ in range(warmup):
        legacy()
        shared_table()
    torch.cuda.synchronize()

    legacy_out = legacy()
    shared_out = shared_table()
    torch.cuda.synchronize()
    exact = all(
        torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        for expected, actual in zip(legacy_out, shared_out, strict=True)
    )
    if not exact:
        raise AssertionError("shared-table Q/K output differs from the two-call baseline")

    samples: dict[str, list[float]] = {"legacy_two_tables": [], "shared_table": []}
    functions = {"legacy_two_tables": legacy, "shared_table": shared_table}
    # Alternating A/B then B/A blocks balances launch-order and clock drift.
    for block in range(blocks):
        order = (
            ("legacy_two_tables", "shared_table")
            if block % 2 == 0
            else ("shared_table", "legacy_two_tables")
        )
        for name in order:
            samples[name].append(_elapsed_ms(functions[name], iterations))

    baseline_ms = statistics.median(samples["legacy_two_tables"])
    candidate_ms = statistics.median(samples["shared_table"])
    return {
        "tokens": tokens,
        "query_heads": query_heads,
        "key_heads": key_heads,
        "head_dim": head_dim,
        "dtype": str(dtype).removeprefix("torch."),
        "legacy_two_tables_ms": baseline_ms,
        "shared_table_ms": candidate_ms,
        "latency_reduction_percent": 100.0 * (baseline_ms - candidate_ms) / baseline_ms,
        "speedup": baseline_ms / candidate_ms,
        "raw_bytes_equal": exact,
        "query_sha256": _digest(shared_out[0]),
        "key_sha256": _digest(shared_out[1]),
        "legacy_samples_ms": samples["legacy_two_tables"],
        "shared_samples_ms": samples["shared_table"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="2,32")
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--key-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", type=_parse_dtype, default=torch.bfloat16)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=400)
    args = parser.parse_args()

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires ROCm PyTorch and a visible GPU")
    token_counts = [int(value) for value in args.tokens.split(",")]
    results = [
        _benchmark_shape(
            tokens=tokens,
            query_heads=args.query_heads,
            key_heads=args.key_heads,
            head_dim=args.head_dim,
            dtype=args.dtype,
            warmup=args.warmup,
            blocks=args.blocks,
            iterations=args.iterations,
        )
        for tokens in token_counts
    ]
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "blocks": args.blocks,
                "iterations_per_block": args.iterations,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
