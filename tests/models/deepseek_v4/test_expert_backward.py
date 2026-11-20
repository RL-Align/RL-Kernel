# SPDX-License-Identifier: Apache-2.0
"""Saved-plan expert dy/dX round trip with empty-rank and stale-plan cases."""

from pathlib import Path
from dataclasses import replace
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.models.deepseek_v4.expert_backward import (
    ExpertBackwardError,
    ExpertBackwardRoundTrip,
    ForwardSlot,
    SavedForwardRoute,
)


def _receipt(rank):
    slots = (
        (
            ForwardSlot(0, 0, 11, 0),
            ForwardSlot(0, 1, 13, 1),
            ForwardSlot(1, 0, 13, 1),
        )
        if rank == 0
        else ()
    )
    return SavedForwardRoute("run-7", "v1", (0, 1), 2 if rank == 0 else 0, 2, 2, slots)


def test_saved_plan_rejects_duplicate_or_unsorted_slots():
    route = _receipt(0)
    with pytest.raises(ExpertBackwardError, match="duplicate"):
        SavedForwardRoute(**{**vars(route), "slots": (route.slots[0], route.slots[0])})
    with pytest.raises(ExpertBackwardError, match="order"):
        SavedForwardRoute(**{**vars(route), "slots": route.slots[::-1]})


def test_real_p3_sealed_route_adapter():
    pytest.importorskip("rl_engine.p3.recordings")
    from rl_engine.p3.assembler import assemble, seal_forward_saved
    from rl_engine.p3.contract import ENVELOPE_SCHEMA_V2, P3OpCtxHost, P3OpResult, P3Verdict
    from rl_engine.p3.fixtures import catalog
    from rl_engine.p3.mocks import OWNERSHIP
    from rl_engine.p3.recordings import record_case

    case = catalog()[0]
    recording = record_case(case)
    provenance = recording["provenance"]
    score = P3OpResult(
        P3Verdict.PASS, recording["operators"]["router_sqrt_softplus_fwd"]["payload"], provenance
    )
    route = P3OpResult(
        P3Verdict.PASS,
        recording["operators"][case.router_mode + "_route_fwd"]["payload"],
        provenance,
    )
    token_set = sorted(case.global_token_id[case.row_active].tolist())
    bundle = assemble(
        case,
        score,
        route,
        run_id="run-7",
        rank=0,
        envelope_version=ENVELOPE_SCHEMA_V2,
        runtime={
            "group": "cp1.ep2",
            "topology": "two-rank",
            "token_ownership": dict(OWNERSHIP),
            "expected_token_set": token_set,
        },
    )
    artifact = bundle["RoutePlan"]["route_artifact_fingerprint"]
    ctx = P3OpCtxHost(
        "run-7", "recorded", 0, case.row_active.copy(), artifact, invocation_id=(1 << 32) | 2
    )
    score_ctx = replace(ctx, invocation_id=(1 << 32) | 1)
    sealed = seal_forward_saved(case, bundle, score, route, score_ctx, ctx)["route"]
    owners = tuple(expert % 2 for expert in range(256))
    kwargs = dict(
        expert_owners=owners,
        global_ranks=(0, 1),
        source_rank=0,
        local_tokens=len(case.z),
        hidden_size=4096,
        expected_token_set=token_set,
        placement_version="identity.v1",
    )
    receipt = SavedForwardRoute.from_p3_forward(bundle, sealed, ctx, **kwargs)
    assert receipt.run_id == "run-7"
    assert receipt.topk_width == 6
    assert receipt.forward_artifact_fingerprint == artifact
    assert len(receipt.slots) == int(case.row_active.sum()) * 6
    assert all(s.destination_rank == s.expert % 2 for s in receipt.slots)
    with pytest.raises(ExpertBackwardError, match="invocation"):
        SavedForwardRoute.from_p3_forward(bundle, sealed, replace(ctx, invocation_id=3), **kwargs)
    changed = deepcopy(bundle)
    changed["RoutePlan"]["core"][0]["route_weight"] += 1
    with pytest.raises(ExpertBackwardError, match="invalid"):
        SavedForwardRoute.from_p3_forward(changed, sealed, ctx, **kwargs)


def test_strict_p4_returned_route_adapter():
    pytest.importorskip("rl_engine.distributed.ep")
    from rl_engine.distributed.ep import EpExchange, EpIdentity, EpPlacement
    from tests.distributed.ep.test_ep_exchange import run_ranks

    placement = EpPlacement.from_owners([expert // 128 for expert in range(256)], 2)
    identity = EpIdentity("run-7", "v1", "placement-v1")

    def body(rank, transport):
        exchange = EpExchange(None, placement, strict=True, transport=transport)
        ids = torch.tensor(
            [[0, 1, 2, 128, 129, 130]] if rank == 0 else [[128, 129, 130, 0, 1, 2]],
            dtype=torch.int64,
        )
        route = exchange.build_route(ids, identity)
        with pytest.raises(ExpertBackwardError, match="strict returned"):
            SavedForwardRoute.from_p4_forward(route, global_ranks=(0, 1), hidden_size=8)
        rows = exchange.dispatch_tokens(route, torch.ones((1, 8), dtype=torch.bfloat16))
        exchange.return_outputs(route, rows)
        receipt = SavedForwardRoute.from_p4_forward(route, global_ranks=(0, 1), hidden_size=8)
        sealed = replace(receipt, forward_artifact_fingerprint="sealed-p3")
        assert (
            SavedForwardRoute.from_p4_forward(
                route, global_ranks=(0, 1), hidden_size=8, p3_receipt=sealed
            )
            is sealed
        )
        with pytest.raises(ExpertBackwardError, match="P3 sealed route differs"):
            SavedForwardRoute.from_p4_forward(
                route,
                global_ranks=(0, 1),
                hidden_size=8,
                p3_receipt=replace(sealed, model_version="stale"),
            )
        return receipt

    receipts = run_ranks(2, body)
    assert all(isinstance(receipt, SavedForwardRoute) for receipt in receipts), receipts
    assert all(len(receipt.slots) == 6 for receipt in receipts)
    assert all(receipt.topk_width == 6 for receipt in receipts)


def _worker(rank, rendezvous, scenario):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        route = _receipt(rank)
        if scenario == "construct_partition":
            if rank == 1:
                route = replace(route, global_ranks=(0, 2))
            message = "saved route group" if rank == 1 else "invalid saved route"
            with pytest.raises(ExpertBackwardError, match=message):
                ExpertBackwardRoundTrip(route, group=dist.group.WORLD, allow_cpu_reference=True)
            return
        if scenario == "placement_version":
            route = replace(route, placement_version="stale") if rank == 1 else route
            with pytest.raises(ExpertBackwardError, match="metadata mismatch"):
                ExpertBackwardRoundTrip(route, group=dist.group.WORLD, allow_cpu_reference=True)
            return
        runtime = ExpertBackwardRoundTrip(route, group=dist.group.WORLD, allow_cpu_reference=True)
        dy = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) if rank == 0 else torch.empty((0, 2))
        if scenario == "stale":
            with pytest.raises(ExpertBackwardError, match="mismatch"):
                runtime.dispatch_dy(
                    dy,
                    run_id="wrong" if rank == 0 else "run-7",
                    model_version="v1",
                    forward_fingerprint=route.fingerprint,
                )
            return
        if scenario == "fingerprint":
            with pytest.raises(ExpertBackwardError, match="stale"):
                runtime.dispatch_dy(
                    dy,
                    run_id="run-7",
                    model_version="v1",
                    forward_fingerprint="old" if rank == 0 else route.fingerprint,
                )
            return
        if scenario == "recompute_mismatch":
            changed = replace(
                route,
                slots=tuple(
                    replace(s, expert=s.expert + 1) if rank == 0 and index == 0 else s
                    for index, s in enumerate(route.slots)
                ),
            )
            with pytest.raises(ExpertBackwardError, match="stale"):
                runtime.dispatch_dy(
                    dy,
                    run_id="run-7",
                    model_version="v1",
                    forward_fingerprint=route.fingerprint,
                    recomputed_route=changed,
                )
            return
        received = runtime.dispatch_dy(
            dy,
            run_id="run-7",
            model_version="v1",
            forward_fingerprint=route.fingerprint,
            recomputed_route=replace(route),
        )
        assert len(runtime.inbound_slots) == (1 if rank == 0 else 2)
        expected = (
            torch.tensor([[1.0, 2.0]]) if rank == 0 else dy.new_tensor([[1.0, 2.0], [3.0, 4.0]])
        )
        torch.testing.assert_close(received, expected)
        expert_dx = received * 2 + rank
        dx = runtime.return_dx(
            expert_dx, run_id="run-7", model_version="v1", forward_fingerprint=route.fingerprint
        )
        torch.testing.assert_close(
            dx, torch.tensor([[5.0, 9.0], [7.0, 9.0]]) if rank == 0 else torch.empty((0, 2))
        )
        outputs = (
            torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]) if rank == 0 else torch.empty((0, 2))
        )
        dp_s = runtime.route_weight_gradient(dy, outputs)
        assert dp_s.dtype == torch.float32
        torch.testing.assert_close(
            dp_s, torch.tensor([[3.0, 6.0], [21.0, 0.0]]) if rank == 0 else torch.empty((0, 2))
        )
        with pytest.raises(ExpertBackwardError, match="stale"):
            runtime.dispatch_dy(
                dy, run_id="run-7", model_version="v1", forward_fingerprint=route.fingerprint
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario",
    [
        "valid",
        "stale",
        "fingerprint",
        "recompute_mismatch",
        "construct_partition",
        "placement_version",
    ],
)
def test_two_rank_saved_round_trip(tmp_path: Path, scenario):
    mp.spawn(_worker, args=(str(tmp_path / f"{scenario}-init"), scenario), nprocs=2, join=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="ROCm GPU required"
)
def test_rocm_one_rank_saved_plan(tmp_path: Path):
    if dist.is_initialized():
        pytest.skip("isolated process group required")
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rocm-init'}", rank=0, world_size=1
    )
    try:
        route = SavedForwardRoute("run", "v1", (0,), 1, 2, 1, (ForwardSlot(0, 0, 1, 0),))
        runtime = ExpertBackwardRoundTrip(route, group=dist.group.WORLD)
        dy = torch.tensor([[2.0, 3.0]], device="cuda", dtype=torch.bfloat16)
        received = runtime.dispatch_dy(
            dy, run_id="run", model_version="v1", forward_fingerprint=route.fingerprint
        )
        torch.testing.assert_close(received, dy)
        dx = runtime.return_dx(
            received, run_id="run", model_version="v1", forward_fingerprint=route.fingerprint
        )
        torch.testing.assert_close(dx, dy)
        assert runtime.route_weight_gradient(dy, dy).dtype == torch.float32
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="ROCm GPU required"
)
def test_rocm_one_rank_strict_p4_forward_to_backward(tmp_path: Path):
    pytest.importorskip("rl_engine.distributed.ep")
    from rl_engine.distributed.ep import EpExchange, EpIdentity, EpPlacement

    if dist.is_initialized():
        pytest.skip("isolated process group required")
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'p4-rocm-init'}", rank=0, world_size=1
    )
    try:
        placement = EpPlacement.from_owners([0] * 256, 1)
        exchange = EpExchange(dist.group.WORLD, placement, strict=True)
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]], device="cuda")
        route = exchange.build_route(ids, EpIdentity("run", "v1", "place-v1"))
        x = torch.ones((1, 8), device="cuda", dtype=torch.bfloat16)
        received = exchange.dispatch_tokens(route, x)
        exchange.return_outputs(route, received)
        receipt = SavedForwardRoute.from_p4_forward(route, global_ranks=(0,), hidden_size=8)
        runtime = ExpertBackwardRoundTrip(receipt, group=dist.group.WORLD)
        dy = torch.ones_like(x)
        expert_dy = runtime.dispatch_dy(
            dy, run_id="run", model_version="v1", forward_fingerprint=receipt.fingerprint
        )
        assert expert_dy.shape == (6, 8)
        dx = runtime.return_dx(
            expert_dy * 2,
            run_id="run",
            model_version="v1",
            forward_fingerprint=receipt.fingerprint,
        )
        torch.testing.assert_close(dx, dy * 12)
    finally:
        dist.destroy_process_group()


def _rocm_two_rank_p4_worker(rank: int, rendezvous: str):
    from rl_engine.distributed.ep import EpExchange, EpIdentity, EpPlacement

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        placement = EpPlacement.from_owners([expert // 128 for expert in range(256)], 2)
        exchange = EpExchange(dist.group.WORLD, placement, strict=True)
        ids = (
            torch.tensor([[0, 1, 2, 128, 129, 130]], device="cuda")
            if rank == 0
            else torch.empty((0, 6), dtype=torch.int64, device="cuda")
        )
        x = torch.ones((ids.shape[0], 8), device="cuda", dtype=torch.bfloat16)
        route = exchange.build_route(ids, EpIdentity("run", "v1", "place-v1"))
        received = exchange.dispatch_tokens(route, x)
        exchange.return_outputs(route, received)
        receipt = SavedForwardRoute.from_p4_forward(route, global_ranks=(0, 1), hidden_size=8)
        runtime = ExpertBackwardRoundTrip(receipt, group=dist.group.WORLD)
        expert_dy = runtime.dispatch_dy(
            x, run_id="run", model_version="v1", forward_fingerprint=receipt.fingerprint
        )
        torch.testing.assert_close(expert_dy, torch.ones((3, 8), device="cuda", dtype=x.dtype))
        dx = runtime.return_dx(
            expert_dy * 2,
            run_id="run",
            model_version="v1",
            forward_fingerprint=receipt.fingerprint,
        )
        torch.testing.assert_close(dx, x * 12)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip or torch.cuda.device_count() < 2,
    reason="two ROCm GPUs required",
)
def test_rocm_two_rank_strict_p4_forward_to_backward(tmp_path: Path):
    pytest.importorskip("rl_engine.distributed.ep")
    if dist.is_initialized():
        pytest.skip("isolated process group required")
    mp.spawn(
        _rocm_two_rank_p4_worker,
        args=(str(tmp_path / "p4-rocm-two-rank-init"),),
        nprocs=2,
        join=True,
    )


def _rocm_four_rank_p4_worker(rank: int, rendezvous: str):
    from rl_engine.distributed.ep import EpExchange, EpIdentity, EpPlacement

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        placement = EpPlacement.from_owners([expert // 64 for expert in range(256)], 4)
        exchange = EpExchange(dist.group.WORLD, placement, strict=True)
        experts = {
            0: [0, 1, 64, 65, 128, 129],
            2: [0, 64, 128, 1, 65, 129],
            3: [0, 1, 2, 64, 65, 66],
        }
        ids = (
            torch.tensor([experts[rank]], device="cuda", dtype=torch.int64)
            if rank in experts
            else torch.empty((0, 6), device="cuda", dtype=torch.int64)
        )
        dy = torch.full((ids.shape[0], 8), rank + 1, device="cuda", dtype=torch.bfloat16)
        route = exchange.build_route(ids, EpIdentity("ep4-run", "v1", "place-v1"))
        received = exchange.dispatch_tokens(route, dy)
        exchange.return_outputs(route, received)
        receipt = SavedForwardRoute.from_p4_forward(route, global_ranks=(0, 1, 2, 3), hidden_size=8)
        runtime = ExpertBackwardRoundTrip(receipt, group=dist.group.WORLD)
        expert_dy = runtime.dispatch_dy(
            dy,
            run_id="ep4-run",
            model_version="v1",
            forward_fingerprint=receipt.fingerprint,
        )
        assert expert_dy.shape == (len(runtime.inbound_slots), 8)
        for row, (source, _, _, expert) in enumerate(runtime.inbound_slots):
            assert expert // 64 == rank
            torch.testing.assert_close(expert_dy[row], torch.full_like(expert_dy[row], source + 1))
        if rank == 3:
            assert expert_dy.numel() == 0
        dx = runtime.return_dx(
            expert_dy * 2 + rank,
            run_id="ep4-run",
            model_version="v1",
            forward_fingerprint=receipt.fingerprint,
        )
        destinations = sum(expert // 64 for expert in experts.get(rank, ()))
        torch.testing.assert_close(dx, dy * 12 + destinations)
        dp_s = runtime.route_weight_gradient(dy, torch.ones((len(receipt.slots), 8), device="cuda"))
        torch.testing.assert_close(
            dp_s, torch.full((len(ids), 6), 8 * (rank + 1), device="cuda", dtype=torch.float32)
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip or torch.cuda.device_count() < 4,
    reason="four ROCm GPUs required",
)
def test_rocm_four_rank_strict_p4_forward_to_backward(tmp_path: Path):
    pytest.importorskip("rl_engine.distributed.ep")
    if dist.is_initialized():
        pytest.skip("isolated process group required")
    mp.spawn(
        _rocm_four_rank_p4_worker,
        args=(str(tmp_path / "p4-rocm-four-rank-init"),),
        nprocs=4,
        join=True,
    )
