# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Real TP attention output timing with model-derived image/text/CFG layouts."""

import argparse
import hashlib
import json
import platform
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton

from benchmarks.operators.gemm.benchmark_tp_attn_out_bias_gemm import (
    equal_bytes,
    measure,
    native_fb,
    tp_calls,
)
from rl_engine.backends.cuda.model_specific.qwenimage.tp_attn_out_bias_gemm import (
    QwenImageTPAttnOutBiasGemmCudaOp,
    shard_attn_out_parameters,
)
from rl_engine.distributed import create_deterministic_collective
from rl_engine.validation.fixtures.qwenimage_attn_out import (
    ATTN_OUT_WORKLOADS,
    attn_out_workload_rows,
    canonical_attn_out_indices,
    make_attn_out_workload,
)


def worker(rank, args, directory):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group(
        "nccl",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=args.tp,
        timeout=timedelta(minutes=5),
    )
    collective = (
        create_deterministic_collective(device=torch.device("cuda", rank)) if args.tp > 1 else None
    )
    results = []
    try:
        op = QwenImageTPAttnOutBiasGemmCudaOp(collective)
        width = 3072 // args.tp
        owned = slice(rank * width, (rank + 1) * width)
        for workload in args.workloads:
            fixture = make_attn_out_workload(
                workload, getattr(torch, args.dtype), "cuda", args.seed
            )
            gold = {}
            for name, stream in fixture["streams"].items():
                x, grad, keys, _ = attn_out_workload_rows(stream, "batched")
                expected = native_fb(x, stream["weight"], stream["bias"], grad, keys)
                order = canonical_attn_out_indices(keys)
                gold[name] = (
                    expected[0][order],
                    expected[1][order][:, owned],
                    expected[2][:, owned],
                    expected[3],
                )
                local_w, local_b = shard_attn_out_parameters(
                    stream["weight"], stream["bias"], tp_world=args.tp, tp_rank=rank
                )
                stream["weight"] = local_w.detach().requires_grad_()
                stream["bias"] = local_b.detach().requires_grad_()
                stream["x"] = stream["x"][..., owned].contiguous()
                del expected, x, grad, keys, order, local_w, local_b
            layouts = [
                layout
                for layout in args.layouts
                if layout != "sequential_cfg" or fixture["metadata"]["cfg_branches"] == 2
            ]
            for name, stream in fixture["streams"].items():
                for layout in layouts:
                    x, grad, keys, cuts = attn_out_workload_rows(stream, layout)
                    x = x.detach().requires_grad_()
                    f, fb = tp_calls(op, x, stream["weight"], stream["bias"], grad, keys, cuts)
                    actual = fb()
                    order = canonical_attn_out_indices(keys)
                    restored = torch.cat(actual[0])[order], actual[1][order], *actual[2:]
                    assert all(equal_bytes(a, b) for a, b in zip(restored, gold[name])), (
                        workload,
                        name,
                        layout,
                        rank,
                    )
                    del actual, restored, order, f, fb, x, grad, keys
            del gold
            for name, stream in fixture["streams"].items():
                for layout in layouts:
                    x, grad, keys, cuts = attn_out_workload_rows(stream, layout)
                    x = x.detach().requires_grad_()
                    f, fb = tp_calls(op, x, stream["weight"], stream["bias"], grad, keys, cuts)
                    forward = measure(f, args.warmup, args.samples, distributed=True)
                    total = measure(fb, args.warmup, args.samples, distributed=True)
                    active = int((keys[:, 0] >= 0).sum().item())
                    results.append(
                        {
                            "rank": rank,
                            "tp": args.tp,
                            "dtype": args.dtype,
                            "workload": workload,
                            "stream": name,
                            "layout": layout,
                            "rows": keys.shape[0],
                            "active_rows": active,
                            "geometry": fixture["metadata"],
                            "forward": forward,
                            "forward_backward": total,
                            "effective_rows_per_second": active * 1000 / total["median_ms"],
                            "ws1_reference_bytes_equal": True,
                        }
                    )
                    if rank == 0:
                        print(
                            f"TP{args.tp} {args.dtype} {workload} {name} {layout}: "
                            f"F+B={total['median_ms']:.3f} ms",
                            flush=True,
                        )
                    del f, fb, x, grad, keys
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        if collective is not None:
            collective.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--workloads", nargs="+", choices=list(ATTN_OUT_WORKLOADS), default=list(ATTN_OUT_WORKLOADS)
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["batched", "sequential_cfg"],
        default=["batched", "sequential_cfg"],
    )
    parser.add_argument("--seed", type=int, default=386)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error("samples must be positive and warmup nonnegative")
    if torch.cuda.device_count() < args.tp:
        parser.error(f"requires {args.tp} NVIDIA GPUs")
    root = Path(__file__).resolve().parents[3]
    sources = [
        "csrc/cuda/gemm/attn_out_bias_gemm.cu",
        "rl_engine/backends/cuda/model_specific/qwenimage/tp_attn_out_bias_gemm.py",
        "rl_engine/distributed/algorithms/collectives.py",
        "rl_engine/validation/fixtures/qwenimage_attn_out.py",
        "benchmarks/distributed/tp/benchmark_qwenimage_attn_out.py",
        "benchmarks/operators/gemm/benchmark_tp_attn_out_bias_gemm.py",
    ]
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(worker, args=(args, directory), nprocs=args.tp, join=True)
        records = [
            json.loads(Path(directory, f"rank-{rank}.json").read_text()) for rank in range(args.tp)
        ]
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "nccl": torch.cuda.nccl.version(),
        "gpu": torch.cuda.get_device_name(),
        "tp": args.tp,
        "dtype": args.dtype,
        "seed": args.seed,
        "warmup": args.warmup,
        "samples": args.samples,
        "scope": (
            "Eager wall-clock latency including canonical collection and IPC; each timing "
            "sample is the maximum rank latency. Memory counters are per rank and include "
            "allocator-visible collective staging in the base allocation."
        ),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
        },
        "results": records[0],
        "rank_results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
