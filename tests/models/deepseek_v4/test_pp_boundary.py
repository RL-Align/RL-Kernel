# SPDX-License-Identifier: Apache-2.0
"""DSV4 PP2 contract and stage handoff with a CPU test transport."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.models.deepseek_v4.pp_boundary import Dsv4Pp2Boundary, Dsv4Pp2Plan, PpBoundaryError


def _plan():
    return Dsv4Pp2Plan((0, 1), ((0, 2), (2, 4)), (2, 3), torch.float32, "v1")


class _ReferenceP2P:
    def send_forward(self, value, is_last_stage):
        assert not is_last_stage
        dist.send(value, dst=1)

    def recv_forward(self, shape, is_first_stage):
        assert not is_first_stage
        value = torch.empty(shape)
        dist.recv(value, src=0)
        return value

    def send_backward(self, value, is_first_stage):
        assert not is_first_stage
        dist.send(value, dst=0)

    def recv_backward(self, shape, is_last_stage):
        assert not is_last_stage
        value = torch.empty(shape)
        dist.recv(value, src=1)
        return value


def test_global_layer_boundaries():
    plan = _plan()
    assert [plan.stage_for_layer(layer) for layer in range(4)] == [0, 0, 1, 1]
    with pytest.raises(PpBoundaryError, match="contiguous"):
        Dsv4Pp2Plan((0, 1), ((0, 2), (3, 4)), (2, 3), torch.float32, "v1")


def _worker(rank, rendezvous, scenario):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        bridge = Dsv4Pp2Boundary(
            _plan(), group=dist.group.WORLD, communicator=_ReferenceP2P(), allow_cpu_reference=True
        )
        value = torch.arange(6.0, dtype=torch.float32).view(2, 3)
        if scenario == "stale":
            with pytest.raises(PpBoundaryError):
                bridge.forward(
                    value if rank == 0 else None,
                    microbatch_id=4,
                    model_version="old" if rank == 0 else "v1",
                )
            return
        received = bridge.forward(value if rank == 0 else None, microbatch_id=4, model_version="v1")
        if rank == 1:
            torch.testing.assert_close(received, value)
        if scenario == "wrong_microbatch":
            with pytest.raises(PpBoundaryError):
                bridge.backward(
                    value * 2 if rank == 1 else None,
                    microbatch_id=5 if rank == 1 else 4,
                    model_version="v1",
                )
            return
        returned = bridge.backward(
            value * 2 if rank == 1 else None, microbatch_id=4, model_version="v1"
        )
        if rank == 0:
            torch.testing.assert_close(returned, value * 2)
        with pytest.raises(PpBoundaryError):
            bridge.forward(value if rank == 0 else None, microbatch_id=4, model_version="v1")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["valid", "stale", "wrong_microbatch"])
def test_pp2_stage_round_trip(tmp_path: Path, scenario):
    mp.spawn(_worker, args=(str(tmp_path / f"{scenario}-init"), scenario), nprocs=2, join=True)
