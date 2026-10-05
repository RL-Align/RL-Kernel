# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.ws2.dkv_ordered_reduce import ordered_dkv_reduce
from rl_engine.kernels.p2.ws2.o_proj_shard import OProjShardPlan, tp1_plan


def test_tp1_dkv_is_identity():
    dkv = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = ordered_dkv_reduce(dkv, torch.arange(64), tp_world_size=1)
    assert torch.equal(out, dkv)


def test_tp_gt1_without_collective_is_unsupported():
    dkv = torch.zeros(2, 4)
    with pytest.raises(P2FailClosedError) as exc:
        ordered_dkv_reduce(dkv, torch.tensor([0, 1]), tp_world_size=2, collective=None)
    assert exc.value.status is P2Status.UNSUPPORTED_CAPABILITY


def test_unsorted_head_tags_fail_closed():
    dkv = torch.zeros(2, 3, 4)

    def ident(x):
        return x

    with pytest.raises(P2FailClosedError) as exc:
        ordered_dkv_reduce(
            dkv, torch.tensor([3, 1]), tp_world_size=2, collective=ident
        )
    assert exc.value.status is P2Status.FORBIDDEN_ATOMIC_REDUCTION


def test_mock_collective_preserves_shape():
    dkv = torch.ones(2, 4)

    def twice(x):
        return x * 2

    out = ordered_dkv_reduce(dkv, torch.tensor([0, 1]), tp_world_size=2, collective=twice)
    assert torch.equal(out, dkv * 2)


def test_tp1_o_proj_plan():
    plan = tp1_plan()
    assert plan.tp_world_size == 1
    assert plan.group_ids == tuple(range(8))
    assert plan.actual_collective is None


def test_tp_o_proj_requires_provenance():
    with pytest.raises(P2FailClosedError) as exc:
        OProjShardPlan(
            tp_rank=0,
            tp_world_size=2,
            group_ids=(0, 1, 2, 3),
            wo_b_split="row",
            merge_order="group_index_ascending",
            actual_collective=None,
        )
    assert exc.value.status is P2Status.MISSING_PROVENANCE
