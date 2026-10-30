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


def _worker(rank, rendezvous, scenario):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        route = _receipt(rank)
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


@pytest.mark.parametrize("scenario", ["valid", "stale", "fingerprint", "recompute_mismatch"])
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
