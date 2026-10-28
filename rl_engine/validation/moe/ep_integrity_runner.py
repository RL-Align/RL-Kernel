# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Short torchrun transport baseline; explicitly not the unavailable P4 adapter.

Example: torchrun --standalone --nproc-per-node=2 -m
rl_engine.validation.moe.ep_integrity_runner --backend gloo --output report.json
For AMD use --backend rccl and explicitly assigned ROCR_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from rl_engine.validation.moe.ep_integrity import EPRankResults, audit_ep_results


def _exchange(
    routes: torch.Tensor, values: torch.Tensor, destinations: torch.Tensor, world: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Send independent rows; no float reduction or implicit local fallback."""

    order = torch.argsort(destinations, stable=True)
    send_counts = torch.bincount(destinations, minlength=world).to(torch.int64)
    receive_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(receive_counts, send_counts)
    send = send_counts.cpu().tolist()
    receive = receive_counts.cpu().tolist()
    rows = sum(receive)
    output_routes = torch.empty((rows, routes.shape[1]), dtype=routes.dtype, device=routes.device)
    output_values = torch.empty((rows, values.shape[1]), dtype=values.dtype, device=values.device)
    dist.all_to_all_single(output_routes, routes[order].contiguous(), receive, send)
    dist.all_to_all_single(output_values, values[order].contiguous(), receive, send)
    return output_routes, output_values


def run_case(world: int, rank: int, device: torch.device, *, skew: bool) -> dict:
    """Dispatch, perform synthetic expert work, return, and audit every slot."""

    top_k, hidden = 2, 4
    # Include a source with no tokens; skew also makes all but expert rank 0 empty.
    counts = tuple(0 if world > 1 and source == world - 1 else 2 for source in range(world))
    owners = tuple(range(world))
    planned = [
        [
            source,
            token,
            slot,
            0 if skew else (source + token + slot) % world,
            0 if skew else (source + token + slot) % world,
        ]
        for source in range(world)
        for token in range(counts[source])
        for slot in range(top_k)
    ]
    expected_routes = torch.tensor(planned, dtype=torch.int64)
    # Exactly representable small BF16 values distinguish source/token/slot/hidden.
    original = torch.tensor(
        [
            [source * 16 + token * 4 + slot * 2 + column for column in range(hidden)]
            for source, token, slot, _, _ in planned
        ],
        dtype=torch.bfloat16,
    )
    expected_values = original + expected_routes[:, 3:4].to(torch.bfloat16)
    local = expected_routes[:, 0] == rank
    routes = expected_routes[local].to(device)
    values = original[local].to(device)
    routed, expert_inputs = _exchange(routes, values, routes[:, 4], world)
    # Record the actual receiving/compute rank, not the planned owner field.
    routed[:, 4] = rank
    # Synthetic arithmetic is an integrity sentinel, not an MoE kernel claim.
    expert_values = expert_inputs + routed[:, 3:4].to(torch.bfloat16)
    # Deliberately reverse physical arrival order before returning to each source.
    routed = routed.flip(0)
    expert_values = expert_values.flip(0)
    returned, outputs = _exchange(routed, expert_values, routed[:, 0], world)
    # Gather only small diagnostic payloads after the actual device transfers.
    received: list = [None] * world
    dist.all_gather_object(received, (returned.cpu(), outputs.cpu()))
    returned_to_source = all(
        bool((item[0][:, 0] == source).all()) for source, item in enumerate(received)
    )
    all_routes = torch.cat([item[0] for item in received])
    all_values = torch.cat([item[1] for item in received])
    batches = []
    for owner in range(world):
        selector = all_routes[:, 4] == owner
        batches.append(
            EPRankResults(
                rank=owner,
                mapping_run_id="transport-baseline-skew" if skew else "transport-baseline-balanced",
                producer_backend=f"torch.distributed.{dist.get_backend()}",
                producer_build_fingerprint=torch.__version__,
                routes=all_routes[selector, :4],
                values=all_values[selector],
            )
        )
    report = audit_ep_results(
        expected_routes=expected_routes,
        expected_values=expected_values,
        rank_results=batches,
        source_num_tokens=counts,
        top_k=top_k,
        expert_to_rank=owners,
        mapping_run_id=batches[0].mapping_run_id,
    )
    payload = report.to_dict()
    payload["returned_to_source"] = returned_to_source
    payload["passed"] = report.passed and returned_to_source
    payload["case"] = "skew" if skew else "balanced"
    payload["source_num_tokens"] = counts
    payload["expert_rank_rows"] = [len(batch.routes) for batch in batches]
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gloo", "rccl"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    world = int(os.environ["WORLD_SIZE"])
    if world not in (1, 2, 4, 8):
        raise ValueError("transport baseline declares only EP=1/2/4/8")
    if args.backend == "rccl":
        if not any(
            os.environ.get(key)
            for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
        ):
            raise RuntimeError("RCCL baseline requires an explicit assigned-device visibility mask")
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("RCCL baseline requires ROCm PyTorch and assigned AMD devices")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"  # PyTorch exposes RCCL under the NCCL API.
    else:
        backend, device = "gloo", torch.device("cpu")
    dist.init_process_group(backend, timeout=timedelta(seconds=45))
    try:
        rank = dist.get_rank()
        device_record = {
            "rank": rank,
            "device": str(device),
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
        }
        rank_devices: list = [None] * world
        dist.all_gather_object(rank_devices, device_record)
        cases = [run_case(world, rank, device, skew=skew) for skew in (False, True)]
        report = {
            "schema_version": "p6.ep_transport_baseline.v1",
            "passed": all(case["passed"] for case in cases),
            "execution_scope": "rccl_device_transport" if args.backend == "rccl" else "cpu_gloo",
            "p4_integration_verified": False,
            "p4_gap": "P4 expert dispatch/return adapter is not available on this branch",
            "world_size": world,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "transport_backend": dist.get_backend(),
            "device": str(device),
            "rank_devices": rank_devices,
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
                + Path(__file__).with_name("ep_integrity.py").read_bytes()
            ).hexdigest(),
            "cases": cases,
        }
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
        if not report["passed"]:
            raise RuntimeError("EP integrity check failed; see report")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
