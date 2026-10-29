# SPDX-License-Identifier: Apache-2.0
"""DSV4 sealed collective plan and executable Gloo reference tests."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.contracts.operators.dsv4_collectives import (
    CollectivePhase,
    Dsv4CollectiveContract,
    Dsv4CollectiveError,
)
from rl_engine.models.deepseek_v4.parallel_collectives import Dsv4ParallelCollectives


def _contract():
    return Dsv4CollectiveContract(
        global_ranks=(0, 1, 2, 3),
        tp_groups=((0, 1), (2, 3)),
        sp_groups=((0,), (1,), (2,), (3,)),
        dp_groups=((0, 2), (1, 3)),
        phases=(
            CollectivePhase("tp_gather", "tp", "all_gather", (1,), torch.float32),
            CollectivePhase("sp_sum", "sp", "all_reduce", (1,), torch.float32),
            CollectivePhase("dp_mean", "dp", "all_reduce", (1,), torch.float32, 2),
            CollectivePhase("dp_shard", "dp", "reduce_scatter", (2,), torch.float32),
        ),
        model_version="checkpoint-42",
    )


def test_plan_rejects_invalid_groups_shapes_and_graph():
    plan = _contract()
    assert plan.fingerprint == _contract().fingerprint
    with pytest.raises(Dsv4CollectiveError, match="Graph"):
        Dsv4CollectiveContract(**{**vars(plan), "graph": True})
    with pytest.raises(Dsv4CollectiveError, match="partition"):
        Dsv4CollectiveContract(**{**vars(plan), "dp_groups": ((0, 2), (1, 2))})
    with pytest.raises(Dsv4CollectiveError, match="divide"):
        Dsv4CollectiveContract(
            **{
                **vars(plan),
                "phases": (
                    *plan.phases[:-1],
                    CollectivePhase("dp_shard", "dp", "reduce_scatter", (3,), torch.float32),
                ),
            }
        )
    with pytest.raises(Dsv4CollectiveError, match="normalization"):
        CollectivePhase("bad", "tp", "all_reduce", (1,), torch.float32, 2)


def _rank_worker(rank, rendezvous, scenario):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        plan = _contract()
        groups = {}
        for dimension in ("tp", "sp", "dp"):
            for ranks in getattr(plan, f"{dimension}_groups"):
                group = dist.new_group(ranks=list(ranks), backend="gloo")
                if rank in ranks:
                    groups[dimension] = group
        runtime = Dsv4ParallelCollectives(
            plan,
            world_group=dist.group.WORLD,
            groups=groups,
            allow_cpu_reference=True,
        )
        try:
            if scenario == "version_mismatch":
                with pytest.raises(Dsv4CollectiveError, match="mismatch"):
                    runtime.begin_step(7, model_version="stale" if rank == 0 else "checkpoint-42")
                with pytest.raises(Dsv4CollectiveError, match="poisoned"):
                    runtime.begin_step(8, model_version="checkpoint-42")
                return
            runtime.begin_step(7, model_version="checkpoint-42")
            if scenario == "phase_mismatch":
                with pytest.raises(Dsv4CollectiveError, match="mismatch"):
                    runtime.execute(
                        "wrong" if rank == 0 else "tp_gather", torch.tensor([rank + 1.0])
                    )
                return
            if scenario == "shape_mismatch":
                with pytest.raises(Dsv4CollectiveError, match="mismatch"):
                    runtime.execute("tp_gather", torch.ones(2 if rank == 0 else 1))
                return
            gathered = runtime.execute("tp_gather", torch.tensor([rank + 1.0]))
            torch.testing.assert_close(
                gathered, torch.tensor([rank // 2 * 2 + 1.0, rank // 2 * 2 + 2.0])
            )
            torch.testing.assert_close(
                runtime.execute("sp_sum", torch.tensor([rank + 1.0])), torch.tensor([rank + 1.0])
            )
            torch.testing.assert_close(
                runtime.execute("dp_mean", torch.tensor([rank + 1.0])),
                torch.tensor([2.0 if rank % 2 == 0 else 3.0]),
            )
            shard = runtime.execute("dp_shard", torch.tensor([rank + 1.0, rank + 11.0]))
            expected = (
                (4.0 if rank == 0 else 24.0) if rank % 2 == 0 else (6.0 if rank == 1 else 26.0)
            )
            torch.testing.assert_close(shard, torch.tensor([expected]))
            runtime.end_step()
        finally:
            runtime.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario", ["valid", "version_mismatch", "phase_mismatch", "shape_mismatch"]
)
def test_four_rank_gloo_plan(tmp_path: Path, scenario: str):
    rendezvous = str(tmp_path / f"{scenario}-rendezvous")
    mp.spawn(_rank_worker, args=(rendezvous, scenario), nprocs=4, join=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="ROCm GPU required"
)
def test_rocm_world_one(tmp_path: Path):
    if dist.is_initialized():
        pytest.skip("requires an isolated process group")
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rocm-rendezvous'}", rank=0, world_size=1
    )
    try:
        plan = Dsv4CollectiveContract(
            global_ranks=(0,),
            tp_groups=((0,),),
            sp_groups=((0,),),
            dp_groups=((0,),),
            phases=(
                CollectivePhase("tp", "tp", "all_gather", (2,), torch.float32),
                CollectivePhase("sp", "sp", "reduce_scatter", (2,), torch.float32),
                CollectivePhase("dp", "dp", "all_reduce", (2,), torch.float32, 2),
            ),
            model_version="v1",
        )
        with Dsv4ParallelCollectives(
            plan,
            world_group=dist.group.WORLD,
            groups={key: dist.group.WORLD for key in ("tp", "sp", "dp")},
        ) as runtime:
            runtime.begin_step(0, model_version="v1")
            value = torch.tensor([2.0, 4.0], device="cuda")
            torch.testing.assert_close(runtime.execute("tp", value), value)
            torch.testing.assert_close(runtime.execute("sp", value), value)
            torch.testing.assert_close(runtime.execute("dp", value), value / 2)
            runtime.end_step()
    finally:
        dist.destroy_process_group()
