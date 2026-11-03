# SPDX-License-Identifier: Apache-2.0
"""DSV4 sealed collective plan and executable Gloo reference tests."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

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
from rl_engine.distributed.transports.rccl import RCCLDeterministicCollective


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
        if scenario == "empty":
            plan = replace(
                plan,
                phases=(
                    CollectivePhase("tp_empty", "tp", "all_gather", (0,), torch.float32),
                    CollectivePhase("sp_empty", "sp", "all_reduce", (0,), torch.float32),
                    CollectivePhase("dp_empty", "dp", "all_reduce", (0,), torch.float32),
                ),
            )
        groups = {}
        for dimension in ("tp", "sp", "dp"):
            for ranks in getattr(plan, f"{dimension}_groups"):
                group = dist.new_group(ranks=list(ranks), backend="gloo")
                if rank in ranks:
                    groups[dimension] = group
        if scenario == "ipc_metadata_failure":
            import rl_engine

            class FakeNative:
                def __getattr__(self, name):
                    if name == "deterministic_collective_rocm_ipc_allocate":

                        def allocate(_size):
                            if rank == 0:
                                raise RuntimeError("injected IPC allocation failure")
                            return torch.empty(1)

                        return allocate
                    if name == "deterministic_collective_rocm_ipc_meta":
                        return lambda _staging: ([0], 0)
                    return lambda *args: 0

            transport = RCCLDeterministicCollective.__new__(RCCLDeterministicCollective)
            transport.world_size = 4
            transport.rank = rank
            transport.group = dist.group.WORLD
            transport.max_size_bytes = 1024
            with patch.object(rl_engine, "_C", FakeNative(), create=True):
                with pytest.raises(RuntimeError, match="IPC metadata"):
                    transport._initialize_ipc_transport()
            return
        if scenario == "transport_failure":
            import rl_engine.models.deepseek_v4.parallel_collectives as module

            class FailingTransport(module.TorchDistributedDeterministicCollective):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    if rank == 0:
                        raise RuntimeError("injected transport failure")

            with patch.object(module, "TorchDistributedDeterministicCollective", FailingTransport):
                error_type = RuntimeError if rank == 0 else Dsv4CollectiveError
                with pytest.raises(error_type, match="injected transport|invalid DSV4"):
                    Dsv4ParallelCollectives(
                        plan,
                        world_group=dist.group.WORLD,
                        groups=groups,
                        allow_cpu_reference=True,
                    )
            return
        if scenario == "transport_preflight":
            import rl_engine.models.deepseek_v4.parallel_collectives as module

            original = module.TorchDistributedDeterministicCollective._normalize_device

            def normalize(device):
                if rank == 0:
                    raise RuntimeError("injected device validation failure")
                return original(device)

            with patch.object(
                module.TorchDistributedDeterministicCollective,
                "_normalize_device",
                staticmethod(normalize),
            ):
                with pytest.raises((RuntimeError, ValueError, Dsv4CollectiveError)) as error:
                    Dsv4ParallelCollectives(
                        plan,
                        world_group=dist.group.WORLD,
                        groups=groups,
                        allow_cpu_reference=True,
                    )
            if rank == 0:
                assert "injected device validation failure" in str(error.value)
            return
        if scenario == "construct_partition" and rank == 0:
            groups["tp"] = groups["dp"]
        if scenario == "construct_partition":
            with pytest.raises(Dsv4CollectiveError) as error:
                Dsv4ParallelCollectives(
                    plan,
                    world_group=dist.group.WORLD,
                    groups=groups,
                    allow_cpu_reference=True,
                )
            assert ("tp rank partition" if rank == 0 else "invalid DSV4") in str(error.value)
            return
        runtime = Dsv4ParallelCollectives(
            plan,
            world_group=dist.group.WORLD,
            groups=groups,
            allow_cpu_reference=True,
        )
        try:
            if scenario == "begin_active":
                if rank == 0:
                    runtime._active = True
                with pytest.raises(Dsv4CollectiveError, match="invalid"):
                    runtime.begin_step(7, model_version="checkpoint-42")
                return
            if scenario == "version_mismatch":
                with pytest.raises(Dsv4CollectiveError, match="mismatch"):
                    runtime.begin_step(7, model_version="stale" if rank == 0 else "checkpoint-42")
                with pytest.raises(Dsv4CollectiveError, match="poisoned"):
                    runtime.begin_step(8, model_version="checkpoint-42")
                return
            runtime.begin_step(7, model_version="checkpoint-42")
            if scenario == "execute_inactive":
                if rank == 0:
                    runtime._active = False
                with pytest.raises(Dsv4CollectiveError, match="invalid"):
                    runtime.execute("tp_gather", torch.tensor([rank + 1.0]))
                return
            if scenario == "empty":
                for phase in plan.phases:
                    result = runtime.execute(phase.name, torch.empty(0))
                    assert tuple(result.shape) == (0,)
                runtime.end_step()
                return
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
    "scenario",
    [
        "valid",
        "empty",
        "version_mismatch",
        "phase_mismatch",
        "shape_mismatch",
        "begin_active",
        "execute_inactive",
        "construct_partition",
        "transport_failure",
        "transport_preflight",
        "ipc_metadata_failure",
    ],
)
def test_four_rank_gloo_plan(tmp_path: Path, scenario: str):
    rendezvous = str(tmp_path / f"{scenario}-rendezvous")
    mp.spawn(_rank_worker, args=(rendezvous, scenario), nprocs=4, join=True)


def test_empty_plan_rejects_reduce_scatter():
    with pytest.raises(Dsv4CollectiveError, match="empty reduce-scatter"):
        CollectivePhase("empty_shard", "dp", "reduce_scatter", (0,), torch.float32)


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


def _rocm_four_rank_worker(rank: int, rendezvous: str):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    try:
        plan = _contract()
        groups = {}
        for dimension in ("tp", "sp", "dp"):
            for ranks in getattr(plan, f"{dimension}_groups"):
                group = dist.new_group(ranks=list(ranks), backend="nccl")
                if rank in ranks:
                    groups[dimension] = group
        with Dsv4ParallelCollectives(plan, world_group=dist.group.WORLD, groups=groups) as runtime:
            for step in (0, 1):
                runtime.begin_step(step, model_version="checkpoint-42")
                value = torch.tensor([float(rank + 1 + step)], device="cuda")
                torch.testing.assert_close(
                    runtime.execute("tp_gather", value),
                    torch.tensor(
                        [rank // 2 * 2 + 1.0 + step, rank // 2 * 2 + 2.0 + step], device="cuda"
                    ),
                )
                torch.testing.assert_close(runtime.execute("sp_sum", value), value)
                torch.testing.assert_close(
                    runtime.execute("dp_mean", value),
                    torch.tensor([2.0 + rank % 2 + step], device="cuda"),
                )
                shard = runtime.execute(
                    "dp_shard", torch.tensor([rank + 1.0 + step, rank + 11.0 + step], device="cuda")
                )
                expected = (
                    (4.0 + 2 * step if rank == 0 else 24.0 + 2 * step)
                    if rank % 2 == 0
                    else (6.0 + 2 * step if rank == 1 else 26.0 + 2 * step)
                )
                torch.testing.assert_close(shard, torch.tensor([expected], device="cuda"))
                runtime.end_step()
            with pytest.raises(Dsv4CollectiveError, match="mismatch"):
                runtime.begin_step(2, model_version="stale" if rank == 0 else "checkpoint-42")
            with pytest.raises(Dsv4CollectiveError, match="poisoned"):
                runtime.begin_step(3, model_version="checkpoint-42")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip or torch.cuda.device_count() < 4,
    reason="four ROCm GPUs required",
)
def test_rocm_four_rank_tp_sp_dp_plan(tmp_path: Path):
    if dist.is_initialized():
        pytest.skip("requires an isolated process group")
    mp.spawn(
        _rocm_four_rank_worker,
        args=(str(tmp_path / "rocm-four-rank-rendezvous"),),
        nprocs=4,
        join=True,
    )
