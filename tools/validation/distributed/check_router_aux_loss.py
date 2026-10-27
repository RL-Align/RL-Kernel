# SPDX-License-Identifier: Apache-2.0
"""Two-MI300X P4/RCCL global auxiliary loss and local backward acceptance."""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.distributed.transports.rccl import RCCLDeterministicCollective
from rl_engine.ops.moe.learned_route import (
    prepare_capacity_domain,
    prepare_learned_route_plan,
    tensor_sha256,
)
from rl_engine.ops.moe.router_aux_loss import (
    AuxLossConfig,
    prepare_router_aux_context,
    router_aux_loss,
)
from rl_engine.reference.moe.router_aux_loss import router_aux_loss_fp64


def bits(value):
    return value.detach().contiguous().view(torch.int32)


class RCCLTensorTransport(RCCLDeterministicCollective):
    """Explicitly exercise P4's existing RCCL transport without HIP IPC.

    Only transport selection changes; no floating reduction runs in transport.
    """

    backend_id = "rocm_rccl_tensor_transport"

    def _initialize_ipc_transport(self):
        pass


def worker(rank, devices, rendezvous, output_dir, transport):
    torch.cuda.set_device(devices[rank])
    dist.init_process_group(
        "nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60)
    )
    factory = RCCLTensorTransport if transport == "rccl" else RCCLDeterministicCollective
    collective = factory(device=devices[rank], max_size_bytes=2 * 1024 * 1024)
    cases = []
    try:
        configurations = [
            ("overflow", (9, 8), 2),
            ("empty_rank", (0, 17), 2),
            ("all_drop", (8, 0), 0),
            ("empty_domain", (0, 0), 0),
            ("unbounded", (16, 17), None),
        ]
        for case, (name, counts, capacity) in enumerate(configurations):
            count = counts[rank]
            width = max(1, max(counts))
            gen = torch.Generator().manual_seed(3400 + case * 10 + rank)
            local = torch.randn(count, 256, generator=gen).cuda().requires_grad_()
            tokens = ((torch.arange(count, dtype=torch.int64) * 2 + rank) * 10 + 100).flip(0)
            padded = torch.zeros(width, 256, device="cuda")
            padded[:count] = local.detach()
            padded_ids = torch.full((width,), -1, device="cuda", dtype=torch.int64)
            padded_ids[:count] = tokens.cuda()
            gathered_ids, gathered = collective.all_gather_many((padded_ids, padded))
            full_ids = gathered_ids[gathered_ids >= 0].cpu()
            full = gathered[gathered_ids >= 0].detach().requires_grad_()
            ownership = tuple(
                tuple(int(t) for t in gathered_ids[r * width : r * width + counts[r]].cpu())
                for r in range(2)
            )
            # Different global row order on each rank; plan and loss must agree.
            if rank:
                full = full.flip(0).detach().requires_grad_()
                full_ids = full_ids.flip(0)
            bias = torch.zeros(256, device="cuda")
            bias[:8] = torch.linspace(8.0, 7.3, 8, device="cuda")
            domain = prepare_capacity_domain(
                full_ids, capacity, absolute_layer=3, model_version="m1", domain_version=name
            )
            plan = prepare_learned_route_plan(
                full, full_ids, bias, domain, bias_version="b1", plan_version="p1"
            )
            config = AuxLossConfig(True, 0.03, "aux.v1")
            context = prepare_router_aux_context(
                plan, config=config, ownership=ownership, collective=collective, version="ctx.v1"
            )
            versions = dict(
                absolute_layer=3,
                model_version="m1",
                bias_version="b1",
                domain_version=name,
                plan_version="p1",
            )
            result = router_aux_loss(
                local,
                tokens,
                bias,
                plan,
                config=config,
                context=context,
                microbatch_size=3,
                **versions,
            )
            dx = torch.autograd.grad(result.loss, local)[0]
            # Full-domain single-device baseline avoids any distributed VJP.
            single_context = prepare_router_aux_context(plan, config=config, version="single.v1")
            single = router_aux_loss(
                full, full_ids, bias, plan, config=config, context=single_context, **versions
            )
            single_dx = torch.autograd.grad(single.loss, full)[0]
            row_for = {int(t): i for i, t in enumerate(full_ids)}
            rows = torch.tensor([row_for[int(t)] for t in tokens], device="cuda", dtype=torch.int64)
            assert torch.equal(bits(result.loss), bits(single.loss))
            assert torch.equal(bits(dx), bits(single_dx[rows]))
            assert torch.equal(result.expert_ids, single.expert_ids[rows])
            cpu = full.detach().cpu().double().requires_grad_()
            reference, experts, _, _, _, ref_counts, _ = router_aux_loss_fp64(
                cpu, full_ids, bias.cpu(), domain.capacities, config.validate()
            )
            assert torch.equal(single.expert_ids.cpu(), experts) and torch.equal(
                result.counts.cpu(), ref_counts
            )
            torch.testing.assert_close(
                result.loss.cpu().double(), reference.detach(), rtol=3e-6, atol=2e-8
            )
            mathdx = torch.autograd.grad(reference, cpu)[0]
            torch.testing.assert_close(dx.cpu().double(), mathdx[rows.cpu()], rtol=5e-4, atol=2e-8)
            digest = collective.all_gather(result.loss.detach().reshape(1)).view(torch.int32)
            assert torch.equal(digest, digest[0].expand_as(digest))
            cases.append(
                {
                    "case": name,
                    "rank": rank,
                    "local_valid_tokens": count,
                    "global_valid_tokens": sum(counts),
                    "plan_sha256": plan.sha256,
                    "loss_sha256": tensor_sha256(result.loss.reshape(1)),
                    "dlogits_sha256": tensor_sha256(dx),
                    "counts_sha256": tensor_sha256(result.counts),
                    "single_device_loss_bytes_equal": True,
                    "single_device_local_vjp_bytes_equal": True,
                    "normal_discrete_fp64_loss_vjp_passed": True,
                    "no_world_multiplier": True,
                    "acceptance_sample": sum(counts) > 0,
                }
            )
        # Same sequence on both ranks: mismatched coefficients are rejected by
        # a tensor-only metadata handshake rather than silently mixing formulas.
        bad_config = AuxLossConfig(True, 0.03 + rank * 0.01, "aux.v1")
        bad_context = prepare_router_aux_context(
            plan, config=bad_config, ownership=ownership, collective=collective, version="ctx.v1"
        )
        try:
            router_aux_loss(
                local, tokens, bias, plan, config=bad_config, context=bad_context, **versions
            )
            raise AssertionError("mismatched rank config was accepted")
        except ValueError as exc:
            assert "ranks disagree" in str(exc)
        final = router_aux_loss(
            local, tokens, bias, plan, config=config, context=context, **versions
        )
        try:
            final.loss.backward(torch.tensor(float(rank + 1), device="cuda"))
            raise AssertionError("mismatched replicated upstream was accepted")
        except ValueError as exc:
            assert "identical upstream" in str(exc)
        report = {
            "rank": rank,
            "device": devices[rank],
            "gpu": torch.cuda.get_device_name(),
            "passed": True,
            "p4_backend": collective.backend_id,
            "p4_hip_ipc_active": bool(collective._ipc_handle),
            "world_size": 2,
            "cases": cases,
            "rank_config_mismatch_rejected": True,
            "replicated_upstream_mismatch_rejected": True,
        }
        (Path(output_dir) / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        collective.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=["rccl", "p4-auto"], default="rccl")
    args = parser.parse_args()
    if args.devices[0] == args.devices[1]:
        raise ValueError("two distinct GPU devices required")
    with tempfile.TemporaryDirectory(prefix="rlk-task34-rccl-") as directory:
        mp.spawn(
            worker,
            args=(
                args.devices,
                "file://" + str(Path(directory) / "rendezvous"),
                directory,
                args.transport,
            ),
            nprocs=2,
            join=True,
        )
        ranks = [json.loads((Path(directory) / f"rank{i}.json").read_text()) for i in range(2)]
    for a, b in zip(ranks[0]["cases"], ranks[1]["cases"], strict=True):
        assert a["plan_sha256"] == b["plan_sha256"] and a["loss_sha256"] == b["loss_sha256"]
    report = {
        "schema": "rlkernel.router_aux_loss.two_gpu_p4.v1",
        "passed": True,
        "node_count": 1,
        "world_size": 2,
        "transport_profile": args.transport,
        "transport": (
            "P4 RCCLDeterministicCollective all_gather/all_gather_many; "
            "no floating transport reduction"
        ),
        "ranks": ranks,
        "scope": (
            "replicated global auxiliary scalar, local-only VJP; no framework gradient averaging"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases_per_rank": 5, "world_size": 2}, indent=2))


if __name__ == "__main__":
    main()
