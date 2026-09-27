# SPDX-License-Identifier: Apache-2.0
"""Real NCCL transport gate for the P4 ordered-collective mock, not live P2 attention.

Run from the repository with PYTHONPATH=.:
  torchrun --standalone --nproc-per-node=8 scripts/p2_distributed_smoke.py
"""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from rl_engine.p2.oracle import fixed_sum
from rl_engine.p2.planner import topology, validate_topology


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        world = dist.get_world_size()
        assert world in (1, 2, 4, 8), world
        reference = torch.stack(
            (
                torch.tensor([1e20, 1.0, -1e20, 1.0] * 16),
                torch.arange(64, dtype=torch.float32),
                torch.arange(64, dtype=torch.float32).square() / 16,
            ),
            -1,
        )
        local = reference.chunk(world)[rank].cuda()
        received = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(received, local)
        gathered = torch.cat(received, 0)
        assert torch.equal(gathered.cpu().view(torch.uint8), reference.view(torch.uint8))
        actual = fixed_sum(gathered, 0).cpu()
        assert torch.equal(actual.view(torch.uint8), fixed_sum(reference, 0).view(torch.uint8))
        for cp in (1, 2, 4):
            validate_topology(topology(1025, cp, world), 1025)
        dist.barrier()
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "scope": "nccl_transport_and_global_tree_mock",
                        "ranks": world,
                        "backend": dist.get_backend(),
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "nccl": list(torch.cuda.nccl.version()),
                        "device": torch.cuda.get_device_name(rank),
                        "configured_nccl_nvls_enable": os.environ.get("NCCL_NVLS_ENABLE"),
                        "actual_nccl_algorithm": None,
                        "live_ws2": "UNSUPPORTED_CAPABILITY",
                    },
                    sort_keys=True,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
