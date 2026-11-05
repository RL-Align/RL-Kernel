# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Real CUDA TP QKV timing, including IPC transport and canonical backward."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton

from rl_engine.backends.cuda.model_specific.qwenimage.tp_qkv_bias_gemm import (
    QwenImageTPQKVBiasGemmCudaOp,
    shard_qkv_parameters,
)
from rl_engine.backends.extension import _C
from rl_engine.distributed import create_deterministic_collective
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session
from rl_engine.validation.fixtures.qwenimage_qkv import (
    QKV_WORKLOADS,
    canonical_qkv_indices,
    make_qkv_workload,
    qkv_workload_rows,
)


def _equal_bytes(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.detach().contiguous().view(torch.uint8), b.detach().contiguous().view(torch.uint8)
        )
    )


def _measure(fn, warmup, samples):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(samples):
        dist.barrier()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        held = fn()
        torch.cuda.synchronize()
        elapsed = torch.tensor((time.perf_counter() - begin) * 1000, device="cuda")
        # Statistics only: this scalar is outside the timed operator computation.
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        times.append(elapsed.item())
        del held
    dist.barrier()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    held = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del held
    dist.barrier()
    return {
        "median_ms": statistics.median(times),
        "global_max_samples_ms": times,
        "base_allocated_mib": before / 1024**2,
        "peak_delta_mib": (peak - before) / 1024**2,
        "peak_total_allocated_mib": peak / 1024**2,
    }


def _workload_calls(stream, layout, op, parameter_id):
    x, grad, keys, cuts = qkv_workload_rows(stream, layout)
    x = x.detach().requires_grad_()
    weight, bias = stream["weight"], stream["bias"]

    def forward():
        with torch.no_grad():
            return tuple(op(x[s:e], weight, bias) for s, e in cuts)

    def forward_backward():
        with canonical_backward_session() as session:
            ys = [
                op(x[s:e], weight, bias, logical_keys=keys[s:e], parameter_id=parameter_id)
                for s, e in cuts
            ]
            grads = torch.autograd.grad(ys, (x, weight, bias), [grad[s:e] for s, e in cuts])
        session.validate_complete()
        return tuple(ys), *grads

    return forward, forward_backward, keys


def _measure_workloads(rank, world, args, op):
    results = []
    dtype = getattr(torch, args.dtype)
    idx = torch.cat(
        [
            torch.arange(
                q * 3072 + rank * (3072 // world),
                q * 3072 + (rank + 1) * (3072 // world),
                device="cuda",
            )
            for q in range(3)
        ]
    )
    for workload in args.workloads:
        fixture = make_qkv_workload(workload, dtype, torch.device("cuda"), args.seed)
        # Replicate prepared activations and controls before validation or timing.
        for stream in fixture["streams"].values():
            for name in ("x", "grad", "keys", "weight", "bias"):
                dist.broadcast(stream[name], src=0)
        gold = {}
        for name, stream in fixture["streams"].items():
            x, grad, keys, _ = qkv_workload_rows(stream, "batched")
            order = canonical_qkv_indices(keys)
            w, b = stream["weight"], stream["bias"]
            y = _C.qwenimage_det_linear_forward(x, w, b, 0, False)[0][order][:, idx]
            dx = torch.empty_like(x)
            for start in range(0, x.shape[0], 128):
                dx[start : start + 128] = _C.qwenimage_det_linear_backward_input(
                    grad[start : start + 128].float().contiguous(), w, dtype
                )
            dw, db = _C.qwenimage_det_linear_backward_weight(
                grad[order].float().contiguous(), x[order].contiguous(), dtype, True
            )
            gold[name] = y, dx[order], dw[idx], db[idx]
            local_w, local_b = shard_qkv_parameters(w, b, tp_world=world, tp_rank=rank)
            stream["weight"] = local_w.detach().requires_grad_()
            stream["bias"] = local_b.detach().requires_grad_()
            stream["grad"] = stream["grad"].index_select(-1, idx)
            del x, grad, keys, order, w, b, y, dx, dw, db, local_w, local_b
        layouts = [
            layout
            for layout in args.layouts
            if layout != "sequential_cfg" or fixture["metadata"]["cfg_branches"] == 2
        ]
        # Check every stream/layout before timing; full parameters and gold are then released.
        for name, stream in fixture["streams"].items():
            for layout in layouts:
                _, fb, keys = _workload_calls(stream, layout, op, name + ".qkv")
                actual = fb()
                order = canonical_qkv_indices(keys)
                actual = torch.cat(actual[0])[order], actual[1][order], *actual[2:]
                assert all(_equal_bytes(a, b) for a, b in zip(actual, gold[name]))
                del actual, fb, keys, order, _
        del gold
        for name, stream in fixture["streams"].items():
            for layout in layouts:
                f, fb, keys = _workload_calls(stream, layout, op, name + ".qkv")
                active_rows = int((keys[:, 0] >= 0).sum().item())
                forward = _measure(f, args.warmup, args.samples)
                backward = _measure(fb, args.warmup, args.samples)
                results.append(
                    {
                        "rank": rank,
                        "tp": world,
                        "rows": keys.shape[0],
                        "active_rows": active_rows,
                        "dtype": args.dtype,
                        "workload": workload,
                        "stream": name,
                        "layout": layout,
                        "geometry": fixture["metadata"],
                        "forward": forward,
                        "forward_backward": backward,
                        "effective_rows_per_second": active_rows * 1000 / backward["median_ms"],
                        "native_reference_bytes_equal": True,
                    }
                )
                if rank == 0:
                    print(
                        f"TP{world} {args.dtype} {workload} {name} {layout}: "
                        f"F+B={backward['median_ms']:.3f} ms",
                        flush=True,
                    )
                del f, fb, keys
    return results


def _worker(rank, world, args, directory):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=5),
    )
    collective = (
        create_deterministic_collective(device=torch.device("cuda", rank)) if world > 1 else None
    )
    results = []
    try:
        dtype = getattr(torch, args.dtype)
        op = QwenImageTPQKVBiasGemmCudaOp(collective)
        for rows in [] if args.workloads else args.rows:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)

            def rand(shape, scale=1):
                return (torch.randn(shape, generator=generator) * scale).to(
                    device="cuda", dtype=dtype
                )

            x, full_w, full_b, full_grad = (
                rand((rows, 3072)),
                rand((9216, 3072), 3072**-0.5),
                rand((9216,), 0.1),
                rand((rows, 9216), 9216**-0.5),
            )
            idx = torch.cat(
                [
                    torch.arange(
                        q * 3072 + rank * (3072 // world),
                        q * 3072 + (rank + 1) * (3072 // world),
                        device=x.device,
                    )
                    for q in range(3)
                ]
            )
            weight, bias = shard_qkv_parameters(full_w, full_b, tp_world=world, tp_rank=rank)
            grad = full_grad.index_select(1, idx)
            x.requires_grad_()
            weight = weight.detach().requires_grad_()
            bias = bias.detach().requires_grad_()
            keys = torch.stack(
                (
                    torch.zeros(rows, dtype=torch.int64, device=x.device),
                    torch.arange(rows, device=x.device),
                ),
                1,
            )

            def forward():
                with torch.no_grad():
                    return op(x, weight, bias)

            def forward_backward():
                with canonical_backward_session() as session:
                    y = op(x, weight, bias, logical_keys=keys, parameter_id="qkv")
                    grads = torch.autograd.grad(y, (x, weight, bias), grad)
                session.validate_complete()
                return y, *grads

            gold_y = _C.qwenimage_det_linear_forward(x, full_w, full_b, 0, False)[0][:, idx]
            gold_dx = _C.qwenimage_det_linear_backward_input(full_grad.float(), full_w, dtype)
            gold_dw, gold_db = _C.qwenimage_det_linear_backward_weight(
                full_grad.float(), x, dtype, True
            )
            actual = forward_backward()
            assert all(
                _equal_bytes(a, b)
                for a, b in zip(actual, (gold_y, gold_dx, gold_dw[idx], gold_db[idx]))
            )
            del actual, gold_y, gold_dx, gold_dw, gold_db, full_w, full_b, full_grad
            # Only local parameter and upstream shards remain resident for measurement.
            f = _measure(forward, args.warmup, args.samples)
            fb = _measure(forward_backward, args.warmup, args.samples)
            results.append(
                {
                    "rank": rank,
                    "tp": world,
                    "rows": rows,
                    "dtype": args.dtype,
                    "forward": f,
                    "forward_backward": fb,
                    "effective_rows_per_second": rows * 1000 / fb["median_ms"],
                    "native_reference_bytes_equal": True,
                }
            )
            if rank == 0:
                print(
                    f"TP{world} M={rows} {args.dtype}: F={f['median_ms']:.3f} ms, "
                    f"F+B={fb['median_ms']:.3f} ms",
                    flush=True,
                )
        if args.workloads:
            results = _measure_workloads(rank, world, args, op)
        Path(directory, f"rank-{rank}.json").write_text(json.dumps(results, indent=2))
    finally:
        if collective is not None:
            collective.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, nargs="+", choices=[1, 2, 4, 8], default=[1, 2, 4, 8])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 129, 4096, 6032, 6889])
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=386)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--workloads", nargs="+", choices=list(QKV_WORKLOADS))
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["batched", "sequential_cfg", "requests"],
        default=["batched", "sequential_cfg"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows) < 1 or args.samples < 1 or args.warmup < 0:
        parser.error("rows/samples must be positive and warmup must be nonnegative")
    if torch.version.hip is not None or torch.cuda.device_count() < max(args.tp):
        parser.error("the requested TP sizes require that many visible NVIDIA GPUs")
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "seed": args.seed,
        "scope": "Real per-rank GPUs, NCCL process group and native CUDA IPC AllGather; "
        "wall time uses the slowest rank per sample; replicated rows count once",
        "results": [],
    }
    for world in args.tp:
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(world, args, directory), nprocs=world, join=True)
            ranks = [
                json.loads(Path(directory, f"rank-{rank}.json").read_text())
                for rank in range(world)
            ]
        for index in range(len(ranks[0])):
            per_rank = [r[index] for r in ranks]
            result = dict(per_rank[0])
            result.pop("rank")
            result["per_rank"] = per_rank
            result["peak_delta_mib_max_rank"] = max(
                r["forward_backward"]["peak_delta_mib"] for r in per_rank
            )
            result["peak_total_allocated_mib_max_rank"] = max(
                r["forward_backward"]["peak_total_allocated_mib"] for r in per_rank
            )
            report["results"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
