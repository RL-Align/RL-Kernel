# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qwen-Image MLP performance and memory at seven model-derived TP layouts."""

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

from benchmarks.operators.gemm.benchmark_tp_mlp_gemm_gelu import (
    equal_bytes,
    measure,
    mlp_calls,
    native_mlp,
    pytorch_mlp_calls,
)
from rl_engine.backends.cuda.model_specific.qwenimage.tp_mlp_gemm_gelu import (
    QwenImageTPMlpGemmGeluCudaOp,
    shard_mlp_parameters,
)
from rl_engine.distributed import create_deterministic_collective
from rl_engine.validation.fixtures.qwenimage_mlp import (
    MLP_WORKLOADS,
    canonical_mlp_indices,
    make_mlp_workload,
    mlp_workload_rows,
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
        timeout=timedelta(minutes=10),
    )
    collective = (
        create_deterministic_collective(device=torch.device("cuda", rank)) if args.tp > 1 else None
    )
    results = []
    try:
        op = QwenImageTPMlpGemmGeluCudaOp(collective)
        for workload in args.workloads:
            fixture = make_mlp_workload(workload, "cuda", args.seed)
            gold = {}
            layouts = ["batched"] + (
                ["sequential_cfg"] if fixture["metadata"]["cfg_branches"] == 2 else []
            )
            parameter_names = ("up_weight", "up_bias", "down_weight", "down_bias")
            for name, stream in fixture["streams"].items():
                parameters = tuple(stream[k] for k in parameter_names)
                x, grad, keys, _ = mlp_workload_rows(stream, "batched")
                full = native_mlp(x, *parameters, grad, keys)
                width = 12288 // args.tp
                own = slice(rank * width, (rank + 1) * width)
                gold[name] = (
                    full[0],
                    full[1],
                    full[2][own].clone(),
                    full[3][own].clone(),
                    full[4][:, own].clone(),
                    full[5],
                )
                local = shard_mlp_parameters(*parameters, tp_world=args.tp, tp_rank=rank)
                for key, value in zip(parameter_names, local):
                    stream[key] = value.detach().requires_grad_()
                del parameters, local, full, x, grad, keys
            for name, stream in fixture["streams"].items():
                parameters = tuple(stream[k] for k in parameter_names)
                for layout in layouts:
                    x, grad, keys, cuts = mlp_workload_rows(stream, layout)
                    x = x.detach().requires_grad_()
                    pair = (
                        pytorch_mlp_calls(x, parameters, grad, cuts)
                        if args.backend == "pytorch"
                        else mlp_calls(op, x, parameters, grad, keys, cuts)
                    )
                    actual = pair[1]()
                    order = canonical_mlp_indices(keys)
                    restored = (torch.cat(actual[0])[order], actual[1][order], *actual[2:])
                    if args.backend == "tp":
                        assert all(equal_bytes(a, b) for a, b in zip(restored, gold[name])), (
                            workload,
                            name,
                            layout,
                            rank,
                        )
                    else:
                        for i, (a, b) in enumerate(zip(restored, gold[name])):
                            torch.testing.assert_close(
                                a.float(), b.float(), atol=0.05 if i == 0 else 0.1, rtol=0.02
                            )
                    del pair, actual, restored, x, grad, keys, order
                del parameters
            # Reference results and full parameter copies are outside the memory measurement.
            del gold
            for name, stream in fixture["streams"].items():
                parameters = tuple(stream[k] for k in parameter_names)
                for layout in layouts:
                    x, grad, keys, cuts = mlp_workload_rows(stream, layout)
                    x = x.detach().requires_grad_()
                    pair = (
                        pytorch_mlp_calls(x, parameters, grad, cuts)
                        if args.backend == "pytorch"
                        else mlp_calls(op, x, parameters, grad, keys, cuts)
                    )
                    forward, total = [
                        measure(call, args.warmup, args.samples, distributed=True) for call in pair
                    ]
                    active = int((keys[:, 0] >= 0).sum())
                    results.append(
                        {
                            "rank": rank,
                            "tp": args.tp,
                            "backend": args.backend,
                            "workload": workload,
                            "stream": name,
                            "layout": layout,
                            "rows": keys.shape[0],
                            "active_rows": active,
                            "geometry": fixture["metadata"],
                            "forward": forward,
                            "forward_backward": total,
                            "effective_rows_per_second": active * 1000 / total["median_ms"],
                            "reference_check": "bytes" if args.backend == "tp" else "tolerance",
                            "passed": True,
                        }
                    )
                    if rank == 0:
                        print(
                            args.backend,
                            args.tp,
                            workload,
                            name,
                            layout,
                            total["median_ms"],
                            flush=True,
                        )
                    del pair, x, grad, keys
                del parameters
            del fixture
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        if collective is not None:
            collective.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument("--backend", choices=["tp", "pytorch"], default="tp")
    parser.add_argument(
        "--workloads", nargs="+", choices=list(MLP_WORKLOADS), default=list(MLP_WORKLOADS)
    )
    parser.add_argument("--seed", type=int, default=386)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "pytorch" and args.tp != 1:
        parser.error("ordinary PyTorch requires --tp 1")
    if args.samples < 1 or args.warmup < 0:
        parser.error("samples must be positive and warmup nonnegative")
    if torch.cuda.device_count() < args.tp:
        parser.error(f"requires {args.tp} NVIDIA GPUs")
    root = Path(__file__).resolve().parents[3]
    sources = [
        "csrc/cuda/gemm/mlp_up_gemm_gelu.cu",
        "csrc/cuda/gemm/mlp_up_gemm_gelu_math.cuh",
        "rl_engine/backends/cuda/model_specific/qwenimage/tp_mlp_gemm_gelu.py",
        "rl_engine/reference/model_specific/qwenimage/tp_mlp_gemm_gelu.py",
        "rl_engine/validation/fixtures/qwenimage_mlp.py",
        "benchmarks/distributed/tp/benchmark_qwenimage_mlp.py",
        "benchmarks/operators/gemm/benchmark_tp_mlp_gemm_gelu.py",
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
        "backend": args.backend,
        "tp": args.tp,
        "seed": args.seed,
        "warmup": args.warmup,
        "samples": args.samples,
        "cpu_threads": 1,
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        "results": records[0],
        "rank_results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
