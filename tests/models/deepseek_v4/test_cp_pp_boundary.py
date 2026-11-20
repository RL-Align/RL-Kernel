# SPDX-License-Identifier: Apache-2.0
"""Four-rank CP2 x PP2 boundary integration on real RCCL groups."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.models.deepseek_v4.cp_ownership import Dsv4CpKvExchange, Dsv4CpPlan
from rl_engine.models.deepseek_v4.pp_boundary import Dsv4Pp2Boundary, Dsv4Pp2Plan


def _worker(rank: int, rendezvous: str):
    from megatron.core.model_parallel_config import ModelParallelConfig

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        groups = [dist.new_group(ranks) for ranks in ((0, 1), (2, 3), (0, 2), (1, 3))]
        stage, lane = divmod(rank, 2)
        cp_ranks = ((0, 1), (2, 3))[stage]
        pp_ranks = (lane, lane + 2)
        cp = Dsv4CpKvExchange(
            Dsv4CpPlan(7, 256, cp_ranks, 128, 2, "checkpoint-42"), group=groups[stage]
        )
        try:
            pp = Dsv4Pp2Boundary(
                Dsv4Pp2Plan(pp_ranks, ((0, 2), (2, 4)), (4, 2), torch.bfloat16, "checkpoint-42"),
                group=groups[2 + lane],
                config=ModelParallelConfig(
                    pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16
                ),
            )
            expected = torch.tensor(
                [[1, 2], [2, 3], [3, 4], [4, 5]], device="cuda", dtype=torch.bfloat16
            )
            positions = ((127, 126), (255, 128))[lane]
            keys = torch.tensor(
                [[7, 0, positions[0]], [7, 1, positions[1]]], device="cuda", dtype=torch.int64
            )
            local = (
                torch.tensor(
                    [[lane + 1, lane + 2], [lane + 3, lane + 4]],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                if stage == 0
                else expected.index_select(0, torch.tensor([lane, 2 + lane], device="cuda"))
            )
            if stage == 0:
                global_keys, global_kv = cp.gather_kv(keys, local, model_version="checkpoint-42")
                torch.testing.assert_close(global_kv, expected, rtol=0, atol=0)
                assert global_keys.tolist() == [[7, 0, 127], [7, 0, 255], [7, 1, 126], [7, 1, 128]]
            received = pp.forward(
                global_kv if stage == 0 else None,
                microbatch_id=3,
                model_version="checkpoint-42",
            )
            if stage == 1:
                torch.testing.assert_close(received, expected, rtol=0, atol=0)
                global_keys, global_kv = cp.gather_kv(keys, local, model_version="checkpoint-42")
                torch.testing.assert_close(global_kv, received, rtol=0, atol=0)
                local_dkv = cp.return_dkv(
                    torch.ones_like(global_kv) * (lane + 1), model_version="checkpoint-42"
                )
                torch.testing.assert_close(local_dkv, torch.full_like(local, 3), rtol=0, atol=0)
            gradient = pp.backward(
                received * (lane + 1) if stage == 1 else None,
                microbatch_id=3,
                model_version="checkpoint-42",
            )
            if stage == 0:
                local_dkv = cp.return_dkv(gradient, model_version="checkpoint-42")
                torch.testing.assert_close(local_dkv, local * 3, rtol=0, atol=0)
        finally:
            cp.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip or torch.cuda.device_count() < 4,
    reason="four ROCm GPUs required",
)
def test_rocm_cp2_pp2_boundary_round_trip(tmp_path: Path):
    if dist.is_initialized():
        pytest.skip("isolated process group required")
    mp.spawn(_worker, args=(str(tmp_path / "cp2-pp2-init"),), nprocs=4, join=True)
