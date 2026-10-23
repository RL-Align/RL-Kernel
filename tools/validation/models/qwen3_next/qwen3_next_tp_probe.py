#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Four independent NCCL ranks must communicate exactly before model loading."""

import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from qwen3_next_runtime import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 4:
        raise RuntimeError("This probe requires TP=4")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=3), device_id=torch.device("cuda", rank)
    )
    try:
        value = torch.arange(1024, device="cuda", dtype=torch.int64) + rank
        gathered = [torch.empty_like(value) for _ in range(4)]
        dist.all_gather(gathered, value)
        for peer, actual in enumerate(gathered):
            assert torch.equal(actual, torch.arange(1024, device="cuda") + peer)
        dist.all_reduce(value)
        assert torch.equal(value, 4 * torch.arange(1024, device="cuda") + 6)
        torch.cuda.synchronize()
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output / f"rank-{rank}.json",
            {
                "status": "passed",
                "rank": rank,
                "world_size": 4,
                "device": torch.cuda.get_device_name(),
                "nccl": torch.cuda.nccl.version(),
            },
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
