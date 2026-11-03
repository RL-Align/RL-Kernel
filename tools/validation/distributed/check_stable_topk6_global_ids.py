# SPDX-License-Identifier: Apache-2.0
"""Two-GPU RCCL materialization with one common global expert numbering contract."""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.ops.moe.stable_topk6 import stable_topk6, tensor_sha256
from rl_engine.reference.moe.stable_topk6 import stable_topk6_reference


def worker(rank, devices, rendezvous, output_dir):
    torch.cuda.set_device(devices[rank])
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    cases = []
    try:
        local_ids = torch.arange(rank, 256, 2, device="cuda", dtype=torch.int64).flip(0)
        for m in (1, 16, 17, 257):
            for pattern in ("ties", "random_ties", "close_scores", "signed_zero"):
                generator = torch.Generator().manual_seed(3100 + m)
                canonical = torch.randint(-5, 6, (m, 256), generator=generator).float()
                if pattern == "ties":
                    canonical.fill_(1.0)
                if pattern == "close_scores":
                    canonical.fill_(-1.0)
                    value = torch.tensor(1.0)
                    for expert in range(256):
                        canonical[:, expert] = value
                        value = torch.nextafter(value, torch.tensor(torch.inf))
                if pattern == "signed_zero":
                    canonical.fill_(0.0)
                    canonical[:, 1::2] = -0.0
                local_scores = canonical[:, local_ids.cpu()].contiguous().cuda()
                gathered_scores = [torch.empty_like(local_scores) for _ in range(2)]
                gathered_ids = [torch.empty_like(local_ids) for _ in range(2)]
                dist.all_gather(gathered_scores, local_scores)
                dist.all_gather(gathered_ids, local_ids)
                # Each rank uses a different materialization order. Only global
                # IDs, not rank/local column offsets, may break score ties.
                order = (rank, 1 - rank)
                scores = torch.cat([gathered_scores[i] for i in order], dim=1).requires_grad_()
                ids = torch.cat([gathered_ids[i] for i in order])
                result = stable_topk6(scores, global_expert_ids=ids, model_version="task31-rccl-v1")
                values, reference_ids, _ = stable_topk6_reference(canonical)
                assert torch.equal(result.expert_ids.cpu(), reference_ids)
                assert torch.equal(
                    result.selected_scores.detach().cpu().view(torch.int32),
                    values.view(torch.int32),
                )
                grad = torch.arange(1.0, 7.0, device="cuda")[None, :].repeat(m, 1)
                dx = torch.autograd.grad(result.selected_scores, scores, grad)[0]
                global_dx = torch.zeros_like(dx).scatter_(1, ids[None, :].expand(m, -1), dx)
                other_ids = [torch.empty_like(result.expert_ids) for _ in range(2)]
                other_values = [torch.empty_like(result.selected_scores) for _ in range(2)]
                other_dx = [torch.empty_like(global_dx) for _ in range(2)]
                dist.all_gather(other_ids, result.expert_ids)
                dist.all_gather(other_values, result.selected_scores.detach())
                dist.all_gather(other_dx, global_dx)
                assert torch.equal(other_ids[0], other_ids[1])
                assert torch.equal(
                    other_values[0].view(torch.int32), other_values[1].view(torch.int32)
                )
                assert torch.equal(other_dx[0].view(torch.int32), other_dx[1].view(torch.int32))
                cases.append(
                    {
                        "m": m,
                        "pattern": pattern,
                        "input_sha256": tensor_sha256(canonical),
                        "expert_ids_sha256": tensor_sha256(result.expert_ids),
                        "selected_scores_sha256": tensor_sha256(result.selected_scores),
                        "global_local_vjp_sha256": tensor_sha256(global_dx),
                        "discrete_and_payload_mismatches": 0,
                    }
                )
        record = {
            "rank": rank,
            "device": devices[rank],
            "passed": True,
            "cases": cases,
            "execution": result.execution,
        }
        (Path(output_dir) / f"rank{rank}.json").write_text(json.dumps(record, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.devices[0] == args.devices[1]:
        raise ValueError("two distinct ROCm devices required")
    with tempfile.TemporaryDirectory(prefix="rlk-task31-rccl-") as directory:
        rendezvous = "file://" + str(Path(directory) / "rendezvous")
        mp.spawn(worker, args=(args.devices, rendezvous, directory), nprocs=2, join=True)
        ranks = [json.loads((Path(directory) / f"rank{i}.json").read_text()) for i in range(2)]
    assert ranks[0]["cases"] == ranks[1]["cases"]
    report = {
        "schema": "rlkernel.stable_topk6.two_gpu_global_ids.v1",
        "passed": True,
        "transport": "PyTorch NCCL backend on ROCm (RCCL) all_gather",
        "node_count": 1,
        "world_size": 2,
        "devices": args.devices,
        "scope": (
            "gather complete global scores/IDs, stable selection on each rank; local value VJP only"
        ),
        "not_validated": ["multi-node", "framework hooks", "distributed autograd", "full model"],
        "ranks": ranks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "world_size": 2,
                "cases_per_rank": len(ranks[0]["cases"]),
                "devices": args.devices,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
