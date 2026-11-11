# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Eager per-rank timing for complete replicated Qwen-Image module boundaries."""

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from benchmarks.operators.gemm.benchmark_tp_attn_out_bias_gemm import equal_bytes, measure
from rl_engine.distributed import create_deterministic_collective
from rl_engine.validation.fixtures.qwenimage_qkv import QKV_WORKLOADS, qkv_workload_rows
from rl_engine.validation.fixtures.qwenimage_replicated import (
    MODULES,
    evaluate_replicated,
    make_replicated_workload,
    replicated_calls,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modules", nargs="+", choices=MODULES, default=list(MODULES))
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=QKV_WORKLOADS,
        default=["image-1024-text17", "batch2-cfg-text1024"],
    )
    parser.add_argument(
        "--dtypes", nargs="+", choices=["bfloat16", "float32"], default=["bfloat16", "float32"]
    )
    parser.add_argument("--seed", type=int, default=386)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be nonnegative and samples must be positive")
    world, rank = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0"))
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if world > 1:
        dist.init_process_group("nccl")
    context = (
        create_deterministic_collective(device=torch.device("cuda", torch.cuda.current_device()))
        if world > 1
        else nullcontext(None)
    )
    rows = []
    with context as collective:
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            for name in args.workloads:
                for kind in args.modules:
                    stream = "text" if kind in ("text_input", "qk_norm") else "image"
                    fixture = make_replicated_workload(kind, name, dtype, "cuda", args.seed, stream)
                    x, grad, keys, cuts = qkv_workload_rows(fixture, "batched")
                    parameters = fixture["parameters"]
                    metadata = dict(fixture["metadata"])
                    del fixture
                    c = None
                    if kind == "qk_norm":
                        x, grad = x.reshape(-1, 24, 128), grad.reshape(-1, 24, 128)
                    gold = evaluate_replicated(kind, x, parameters, grad, keys, cuts)
                    if kind == "qk_norm" and world > 1:
                        owned = slice(rank * 24 // world, (rank + 1) * 24 // world)
                        x, grad = x[:, owned].contiguous(), grad[:, owned].contiguous()
                        gold = (gold[0][:, owned], gold[1][:, owned], gold[2])
                        c = collective
                    result = evaluate_replicated(
                        kind, x, parameters, grad, keys, cuts, collective=c
                    )
                    assert all(equal_bytes(a, b) for a, b in zip(gold, result))
                    del result, gold
                    backends = ("cuda", "pytorch") if world == 1 else ("cuda",)
                    for backend in backends:
                        calls = replicated_calls(
                            kind, x, parameters, grad, keys, cuts, backend=backend, collective=c
                        )
                        forward, total = [
                            measure(call, args.warmup, args.samples, distributed=world > 1)
                            for call in calls
                        ]
                        if world > 1:
                            for metric in (forward, total):
                                memory = {k: v for k, v in metric.items() if "mib" in k}
                                peers = [None] * world
                                dist.all_gather_object(peers, memory)
                                metric["per_rank_memory"] = peers
                        record = dict(
                            metadata,
                            dtype=dtype_name,
                            tp=world,
                            backend=backend,
                            forward=forward,
                            forward_backward=total,
                            tp1_byte_equal=True if backend == "cuda" else None,
                            effective_rows_per_second=metadata["active_rows"]
                            * 1000
                            / total["median_ms"],
                        )
                        if rank == 0:
                            rows.append(record)
                            print(
                                f"TP{world} {dtype_name} {kind} {name} {backend}: "
                                f"F+B={total['median_ms']:.3f} ms",
                                flush=True,
                            )
                            args.output.parent.mkdir(parents=True, exist_ok=True)
                            sources = [
                                (
                                    "rl_engine/backends/cuda/model_specific/qwenimage/"
                                    "replicated_modules.py"
                                ),
                                "rl_engine/validation/fixtures/qwenimage_replicated.py",
                                (
                                    "benchmarks/distributed/tp/"
                                    "benchmark_qwenimage_replicated_modules.py"
                                ),
                            ]
                            document = {
                                "seed": args.seed,
                                "warmup": args.warmup,
                                "samples": args.samples,
                                "torch": torch.__version__,
                                "cuda": torch.version.cuda,
                                "gpu": torch.cuda.get_device_name(),
                                "threads": torch.get_num_threads(),
                                "timing": (
                                    "eager synchronized wall time; slowest rank; "
                                    "complete forward/autograd"
                                ),
                                "sha256": {
                                    p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                    for p in sources
                                },
                                "results": rows,
                            }
                            args.output.write_text(json.dumps(document, indent=2) + "\n")
                    del x, grad, keys, parameters, calls
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
