# SPDX-License-Identifier: Apache-2.0
"""Two-MI300X/RCCL complete-domain capacity admission, including empty ranks."""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.ops.moe.learned_route import (
    _digest,
    prepare_capacity_domain,
    prepare_learned_route_plan,
    tensor_sha256,
)
from rl_engine.reference.moe.learned_route import learned_route_reference, vjp_fp64
from rl_engine.ops.moe.router_training import (
    normal_learned_route,
    record_route_vjp,
    diagnostic_fixed_route,
    compare_vjp_records,
)


def worker(rank, devices, rendezvous, output_dir):
    torch.cuda.set_device(devices[rank])
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    cases = []
    try:
        configurations = [
            ("overflow", (9, 8), 2),
            ("empty_rank", (0, 17), 1),
            ("all_drop", (8, 0), 0),
            ("empty_domain", (0, 0), 0),
            ("unbounded", (16, 17), None),
        ]
        for case, (name, counts, capacity) in enumerate(configurations):
            count = counts[rank]
            width = max(1, max(counts))
            gen = torch.Generator().manual_seed(3200 + case * 10 + rank)
            x = torch.randn(count, 256, generator=gen).cuda().requires_grad_()
            tokens = (torch.arange(count, dtype=torch.int64) * 2 + rank) * 10 + 100
            tokens = tokens.flip(0)
            padded_x = torch.zeros(width, 256, device="cuda")
            padded_x[:count] = x.detach()
            padded_ids = torch.full((width,), -1, device="cuda", dtype=torch.int64)
            padded_ids[:count] = tokens.cuda()
            gathered_x = [torch.empty_like(padded_x) for _ in range(2)]
            gathered_ids = [torch.empty_like(padded_ids) for _ in range(2)]
            dist.all_gather(gathered_x, padded_x)
            dist.all_gather(gathered_ids, padded_ids)
            # Different row assembly order must not change global token priority.
            order = (rank, 1 - rank)
            full = torch.cat([gathered_x[r][: counts[r]] for r in order])
            full_ids = torch.cat([gathered_ids[r][: counts[r]] for r in order]).cpu()
            bias = torch.zeros(256, device="cuda")
            bias[:8] = torch.linspace(8.0, 7.3, 8, device="cuda")
            domain = prepare_capacity_domain(
                full_ids,
                capacity,
                absolute_layer=3,
                model_version="task33-rccl-v1",
                domain_version=name,
            )
            plan = prepare_learned_route_plan(
                full, full_ids, bias, domain, bias_version="b1", plan_version="p1"
            )
            digest = torch.tensor(
                list(bytes.fromhex(plan.sha256)), dtype=torch.uint8, device="cuda"
            )
            digests = [torch.empty_like(digest) for _ in range(2)]
            dist.all_gather(digests, digest)
            assert torch.equal(digests[0], digests[1])
            actual = normal_learned_route(
                x,
                tokens,
                bias,
                plan,
                record_version="task33.r1",
                rank=rank,
                world_size=2,
                absolute_layer=3,
                model_version="task33-rccl-v1",
                bias_version="b1",
                domain_version=name,
                plan_version="p1",
            )
            payload = actual.record.payload()
            assert payload["rank"] == rank and payload["world_size"] == 2
            by_token = {row["logical_id"]: row for row in payload["tokens"]}
            actual_mask = torch.tensor(
                [by_token[int(t)]["keep"] for t in tokens], dtype=torch.bool
            ).reshape(-1, 6)
            actual_positions = torch.tensor(
                [by_token[int(t)]["positions"] for t in tokens], dtype=torch.int64
            ).reshape(-1, 6)
            ref, e, c, k, p = learned_route_reference(full, full_ids, bias, domain.capacities)
            index = {int(token): i for i, token in enumerate(full_ids.tolist())}
            rows = torch.tensor([index[int(token)] for token in tokens], dtype=torch.int64)
            assert torch.equal(actual.expert_ids.cpu(), e[rows])
            assert torch.equal(actual_mask, k[rows])
            assert torch.equal(actual_positions, p[rows])
            torch.testing.assert_close(
                actual.weights.detach().cpu().double(), ref[rows], rtol=2e-6, atol=1e-8
            )
            upstream = torch.randn(count, 6, generator=gen).cuda()
            dx, vjp_record = record_route_vjp(
                actual, upstream, loss_version="global-domain.loss.v1"
            )
            reference_dx = vjp_fp64(x.detach().cpu(), c[rows], k[rows], upstream.cpu())
            torch.testing.assert_close(dx.cpu().double(), reference_dx, rtol=3e-4, atol=2e-7)
            fixed = diagnostic_fixed_route(
                x.detach().clone().requires_grad_(),
                tokens,
                actual.record,
                absolute_layer=3,
                versions=payload["versions"],
            )
            _, replay_vjp = record_route_vjp(fixed, upstream, loss_version="global-domain.loss.v1")
            compared = compare_vjp_records(replay_vjp, vjp_record, require_normal=False)
            assert (
                compared["status"] == ("match" if count else "empty") and not compared["acceptance"]
            )
            cases.append(
                {
                    "case": name,
                    "rank": rank,
                    "local_tokens": count,
                    "global_tokens": sum(counts),
                    "capacity": capacity,
                    "plan_sha256": plan.sha256,
                    "domain_sha256": domain.sha256,
                    "selected_loads_sha256": _digest(list(plan.selected_loads)),
                    "accepted_loads_sha256": _digest(list(plan.accepted_loads)),
                    "normal_forward_record_sha256": actual.record.sha256,
                    "normal_vjp_record_sha256": vjp_record.sha256,
                    "diagnostic_acceptance": False,
                    "local_normal_acceptance_sample": count > 0,
                    "expert_ids_sha256": tensor_sha256(actual.expert_ids),
                    "mask_sha256": tensor_sha256(actual_mask),
                    "weights_sha256": tensor_sha256(actual.weights),
                    "discrete_mask_position_mismatches": 0,
                    "fp64_weight_and_vjp_passed": True,
                }
            )
        (Path(output_dir) / f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "device": devices[rank],
                    "passed": True,
                    "cases": cases,
                    "execution": {
                        key: payload["execution"][key]
                        for key in [
                            "actual_backend",
                            "source_sha256",
                            "torch",
                            "rocm",
                            "gpu",
                            "arch",
                            "backward_kernel",
                        ]
                    },
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.devices[0] == args.devices[1]:
        raise ValueError("two distinct devices required")
    with tempfile.TemporaryDirectory(prefix="rlk-task33-rccl-") as directory:
        mp.spawn(
            worker,
            args=(args.devices, "file://" + str(Path(directory) / "rendezvous"), directory),
            nprocs=2,
            join=True,
        )
        ranks = [json.loads((Path(directory) / f"rank{i}.json").read_text()) for i in range(2)]
    for a, b in zip(ranks[0]["cases"], ranks[1]["cases"], strict=True):
        assert (
            a["plan_sha256"] == b["plan_sha256"]
            and a["accepted_loads_sha256"] == b["accepted_loads_sha256"]
        )
    report = {
        "schema": "rlkernel.router_training.two_gpu_records.v1",
        "passed": True,
        "transport": "PyTorch NCCL backend on ROCm (RCCL) all_gather",
        "node_count": 1,
        "world_size": 2,
        "scope": (
            "global capacity admission over gathered domain; normal local routing and local VJP"
        ),
        "not_validated": [
            "distributed autograd",
            "multi-node",
            "framework integration",
            "full model",
        ],
        "ranks": ranks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": True, "world_size": 2, "cases_per_rank": 5, "devices": args.devices},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
