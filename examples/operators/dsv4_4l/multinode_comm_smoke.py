# SPDX-License-Identifier: Apache-2.0
"""Check multi-node ROCm collectives and P2P; this is not DSV4 acceptance."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-nodes", required=True, type=int)
    parser.add_argument("--timeout-seconds", default=180, type=int)
    args = parser.parse_args()
    if args.expected_nodes < 2 or args.timeout_seconds < 1:
        parser.error("require at least two nodes and a positive timeout")

    import torch
    import torch.distributed as dist

    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("a ROCm PyTorch build and visible AMD GPUs are required")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        identity = dict(
            rank=rank, local_rank=local_rank, hostname=socket.gethostname(),
            node_id=os.environ["RLK_NODE_ID"],
            device=torch.cuda.get_device_name(local_rank), hip=torch.version.hip,
        )
        identities = [None] * world
        dist.all_gather_object(identities, identity)
        if len({item["node_id"] for item in identities}) != args.expected_nodes:
            raise RuntimeError("observed node IDs differ from --expected-nodes")
        if len({(item["node_id"], item["local_rank"]) for item in identities}) != world:
            raise RuntimeError("duplicate node/local-rank assignment")

        # Integer arithmetic checks transport only, not FP reduction ordering.
        total = torch.tensor([rank + 1], device="cuda", dtype=torch.int64)
        dist.all_reduce(total)
        if total.item() != world * (world + 1) // 2:
            raise RuntimeError("all-reduce payload differs")

        # Rank-ordered forward and backward chains cross node boundaries under
        # torchrun's contiguous per-node rank allocation. No model is loaded.
        payload = torch.tensor([314159 if rank == 0 else 0], device="cuda", dtype=torch.int64)
        if rank:
            dist.recv(payload, src=rank - 1)
        if rank + 1 < world:
            dist.send(payload, dst=rank + 1)
        if payload.item() != 314159:
            raise RuntimeError("forward P2P payload differs")
        dist.barrier()
        payload.fill_(271828 if rank == world - 1 else 0)
        if rank + 1 < world:
            dist.recv(payload, src=rank + 1)
        if rank:
            dist.send(payload, dst=rank - 1)
        if payload.item() != 271828:
            raise RuntimeError("backward P2P payload differs")
        dist.barrier()
        args.output.mkdir(parents=True, exist_ok=True)
        record = dict(
            status="PASS", scope="communication smoke only; no model acceptance",
            rank=rank, world_size=world, identities=identities,
            checks=["integer_all_reduce", "forward_p2p", "backward_p2p"],
        )
        with (args.output / f"comm-rank-{rank}.json").open("x") as output:
            output.write(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
