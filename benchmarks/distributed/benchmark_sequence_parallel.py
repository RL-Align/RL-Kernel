# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Measure image sharding and deterministic gather on an explicit SP group."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.distributed.algorithms.sequence_parallel import gather_sequence, shard_image_sequence
from rl_engine.distributed.transports.rccl import TorchDistributedDeterministicCollective


def _measure(function, device, warmup, iterations):
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = torch.tensor((time.perf_counter() - start) * 1000 / iterations, device=device)
    if dist.is_initialized():
        # Report the slowest rank. This reduction is timing metadata only.
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def _gather_forward_backward(local, layout, gradient):
    restored = gather_sequence(local, layout)
    return torch.autograd.grad(restored, local, gradient)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[4096, 6032, 6889])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch, args.hidden, args.iterations, *args.seq_lengths) < 1 or args.warmup < 0:
        parser.error("sizes and iterations must be positive; warmup must be non-negative")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        device = torch.device("cuda", torch.cuda.current_device())
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        world_size = dist.get_world_size() if distributed else 1
        dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
        results = []
        for length in args.seq_lengths:
            generator = torch.Generator().manual_seed(386)
            image = torch.randn(args.batch, length, args.hidden, generator=generator)
            image = image.to(device=device, dtype=dtype)
            token = torch.arange(length, device=device)
            positions = torch.stack((torch.zeros_like(token), token // 83, token % 83), dim=-1)
            with torch.no_grad():
                local, _, layout = shard_image_sequence(image, positions)
                restored = gather_sequence(local, layout)
                assert torch.equal(restored.view(torch.uint8), image.view(torch.uint8))
                shard_ms = _measure(
                    partial(shard_image_sequence, image, positions),
                    device,
                    args.warmup,
                    args.iterations,
                )
                gather_ms = _measure(
                    partial(gather_sequence, local, layout), device, args.warmup, args.iterations
                )

            local = local.detach().requires_grad_(True)
            gradient = torch.ones_like(image)

            gather_forward_backward = partial(_gather_forward_backward, local, layout, gradient)
            assert torch.equal(gather_forward_backward(), torch.full_like(local, world_size))
            forward_backward_ms = _measure(
                gather_forward_backward, device, args.warmup, args.iterations
            )
            results.append(
                {
                    "sequence_length": length,
                    "shard_ms": shard_ms,
                    "gather_ms": gather_ms,
                    "gather_forward_backward_ms": forward_backward_ms,
                }
            )

        root = Path(__file__).resolve().parents[2]
        source = root / "rl_engine/distributed/algorithms/sequence_parallel.py"
        report = {
            "operator": "sp_shard_gather / sp_arbitrary_length",
            "collective_backend_id": (
                TorchDistributedDeterministicCollective.backend_id if world_size > 1 else None
            ),
            "transport": dist.get_backend() if world_size > 1 else "local_copy",
            "gradient_accumulation": "float32_balanced_rank_tree" if world_size > 1 else "identity",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root, text=True
                ).strip()
            ),
            "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version() if device.type == "cuda" else None,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "rank0_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "world_size": world_size,
            "batch": args.batch,
            "hidden": args.hidden,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "results": results,
        }
        if not distributed or dist.get_rank() == 0:
            content = json.dumps(report, indent=2) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(content)
            print(content, end="")
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
