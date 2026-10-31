# SPDX-License-Identifier: Apache-2.0
"""DSV4 contiguous CP ownership, KV gather and fixed-tree dKV tests."""

from pathlib import Path
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.models.deepseek_v4.cp_ownership import (
    CpOwnershipError,
    Dsv4CpKvExchange,
    Dsv4CpPlan,
)


def _plan():
    return Dsv4CpPlan(7, 256, (0, 1), 128, 3, "checkpoint-42")


def test_compressed_row_uses_global_end_position():
    plan = _plan()
    assert plan.bounds(0) == (0, 128)
    assert plan.bounds(1) == (128, 256)
    assert plan.owner((7, 0, 127)) == 0
    assert plan.owner((7, 0, 255)) == 1
    assert plan.owner((7, 1, 128)) == 1
    assert plan.query_keys(1, device="cpu")[0].tolist() == [7, 128]
    with pytest.raises(CpOwnershipError, match="complete"):
        plan.owner((7, 0, 128))


def _worker(rank, rendezvous, scenario):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        plan = _plan()
        if scenario in ("empty_rank", "no_rows"):
            plan = replace(plan, total_tokens=1)
        exchange = Dsv4CpKvExchange(plan, group=dist.group.WORLD, allow_cpu_reference=True)
        try:
            if scenario in ("empty_rank", "no_rows"):
                keys = (
                    torch.tensor([[7, 1, 0]])
                    if rank == 0 and scenario == "empty_rank"
                    else torch.empty((0, 3), dtype=torch.int64)
                )
                values = (
                    torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
                    if keys.shape[0]
                    else torch.empty((0, 2), dtype=torch.bfloat16)
                )
                global_keys, global_kv = exchange.gather_kv(
                    keys, values, model_version="checkpoint-42"
                )
                assert global_keys.shape[0] == (1 if scenario == "empty_rank" else 0)
                local = exchange.return_dkv(
                    torch.ones_like(global_kv) * (rank + 1), model_version="checkpoint-42"
                )
                assert local.shape[0] == keys.shape[0]
                if rank == 0 and scenario == "empty_rank":
                    torch.testing.assert_close(local, torch.full_like(local, 3))
                return
            positions = (127, 126) if rank == 0 else (255, 128)
            keys = torch.tensor([[7, 0, positions[0]], [7, 1, positions[1]]])
            if scenario == "wrong_owner" and rank == 0:
                keys[1, 2] = 128
            values = torch.tensor(
                [[rank + 1.0, rank + 2.0], [rank + 3.0, rank + 4.0]], dtype=torch.bfloat16
            )
            if scenario == "wrong_owner":
                with pytest.raises(CpOwnershipError):
                    exchange.gather_kv(keys, values, model_version="checkpoint-42")
                return
            if scenario == "stale":
                with pytest.raises(CpOwnershipError):
                    exchange.gather_kv(
                        keys, values, model_version="stale" if rank == 0 else "checkpoint-42"
                    )
                return
            global_keys, global_kv = exchange.gather_kv(keys, values, model_version="checkpoint-42")
            assert global_keys.tolist() == [[7, 0, 127], [7, 0, 255], [7, 1, 126], [7, 1, 128]]
            expected = torch.tensor(
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=torch.bfloat16
            )
            torch.testing.assert_close(global_kv, expected)
            local_dkv = exchange.return_dkv(
                torch.ones_like(global_kv) * (rank + 1), model_version="checkpoint-42"
            )
            torch.testing.assert_close(local_dkv, torch.full_like(values, 3))
            with pytest.raises(CpOwnershipError):
                exchange.return_dkv(torch.ones_like(global_kv), model_version="checkpoint-42")
        finally:
            exchange.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scenario", ["valid", "empty_rank", "no_rows", "wrong_owner", "stale"])
def test_two_rank_cp_exchange(tmp_path: Path, scenario):
    mp.spawn(_worker, args=(str(tmp_path / f"{scenario}-init"), scenario), nprocs=2, join=True)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="ROCm GPU required"
)
def test_rocm_one_rank_cp_exchange(tmp_path: Path):
    if dist.is_initialized():
        pytest.skip("isolated process group required")
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rocm-init'}", rank=0, world_size=1
    )
    try:
        plan = Dsv4CpPlan(7, 128, (0,), 128, 2, "v1")
        exchange = Dsv4CpKvExchange(plan, group=dist.group.WORLD)
        try:
            keys = torch.tensor([[7, 0, 127]], dtype=torch.int64, device="cuda")
            values = torch.tensor([[2.0, 4.0]], dtype=torch.bfloat16, device="cuda")
            global_keys, global_values = exchange.gather_kv(keys, values, model_version="v1")
            torch.testing.assert_close(global_keys, keys)
            torch.testing.assert_close(global_values, values)
            torch.testing.assert_close(exchange.return_dkv(values, model_version="v1"), values)
        finally:
            exchange.close()
    finally:
        dist.destroy_process_group()
