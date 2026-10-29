# SPDX-License-Identifier: Apache-2.0
"""Saved-plan expert dy/dX round trip with empty-rank and stale-plan cases."""

from pathlib import Path
from dataclasses import replace

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
