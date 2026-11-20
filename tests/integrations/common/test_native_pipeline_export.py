# SPDX-License-Identifier: Apache-2.0
"""Real CPU/Gloo PP and DP state transfers; not MI300X model acceptance."""

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, root):
    from rl_engine.integrations.engines.train.megatron.pipeline_export import export_pipeline_state

    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=4
    )
    try:
        groups = [
            dist.new_group(ranks=ranks, backend="gloo")
            for ranks in ([0, 2], [1, 3], [0, 1], [2, 3])
        ]
        pp = groups[rank % 2]
        dp = groups[2 + rank // 2]
        stage = rank // 2
        state = (
            {"embed.weight": torch.tensor([[0.125, -0.25]], dtype=torch.bfloat16)}
            if stage == 0
            else {"layers.3.ffn.gate.bias": torch.tensor([0.25, 0.5])}
        )
        expected = {
            "embed.weight": torch.tensor([[0.125, -0.25]], dtype=torch.bfloat16),
            "layers.3.ffn.gate.bias": torch.tensor([0.25, 0.5]),
        }

        def exported(version=0):
            return dict(
                export_pipeline_state(state, pp_group=pp, dp_group=dp, weight_version=version)
            )

        for update in [False, True]:
            if update:
                if stage == 1:
                    state["layers.3.ffn.gate.bias"].add_(0.001)
                expected["layers.3.ffn.gate.bias"].add_(0.001)
            actual = exported(version=int(update))
            assert set(actual) == set(expected)
            for key in actual:
                assert actual[key].dtype == expected[key].dtype
                assert torch.equal(actual[key].view(torch.uint8), expected[key].view(torch.uint8))
        # One replica alone changes state; every PP/DP rank must reject it.
        if rank == 0:
            state["embed.weight"].add_(1)
        with pytest.raises(RuntimeError, match="DP replicas differ"):
            exported()
        if rank == 0:
            state["embed.weight"].sub_(1)
        # DP replicas agree, but stages disagree on the live version.
        with pytest.raises(RuntimeError, match="pipeline ranks disagree"):
            exported(version=stage)
        # Both stages claim one key. Reject instead of silently taking one copy.
        state["duplicate"] = torch.zeros(1)
        with pytest.raises(ValueError, match="duplicate ownership"):
            exported()
        Path(root, f"rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    rank=rank,
                    pp_ranks=dist.get_process_group_ranks(pp),
                    dp_ranks=dist.get_process_group_ranks(dp),
                    original_dtypes_preserved=True,
                    updated_buffer_transferred=True,
                    dp_divergence_rejected=True,
                    duplicate_ownership_rejected=True,
                    cross_stage_version_mismatch_rejected=True,
                )
            )
        )
    finally:
        dist.destroy_process_group()


def test_actual_four_rank_pipeline_transfer_and_dp_replica_validation(tmp_path):
    mp.spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    records = [json.loads((tmp_path / f"rank-{r}.json").read_text()) for r in range(4)]
    assert [x["pp_ranks"] for x in records] == [[0, 2], [1, 3], [0, 2], [1, 3]]
    assert [x["dp_ranks"] for x in records] == [[0, 1], [0, 1], [2, 3], [2, 3]]
