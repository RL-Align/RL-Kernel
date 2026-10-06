# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.dsv4.attention.errors import DSv4FailClosedError, DSv4Status
from rl_engine.kernels.dsv4.attention.ws2.dkv_ordered_reduce import ordered_dkv_reduce
from rl_engine.kernels.dsv4.attention.ws2.o_proj_shard import OProjShardPlan, tp1_plan


def _head_contributions():
    contributions = torch.zeros(64, 2, 3)
    contributions[0].fill_(2**24)
    contributions[1].fill_(-(2**24))
    contributions[2].fill_(1)
    return contributions


def test_tp1_packed_dkv_is_identity():
    dkv = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = ordered_dkv_reduce(dkv, torch.arange(64), tp_world_size=1)
    assert out is dkv


def test_tp1_per_head_dkv_folds_to_complete_mqa_gradient():
    out = ordered_dkv_reduce(_head_contributions(), torch.arange(64))
    assert out.shape == (2, 3)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.ones(2, 3))


@pytest.mark.parametrize("tp_world_size", [2, 4, 8])
def test_tagged_gather_folds_global_heads_independently_of_arrival_order(tp_world_size):
    contributions = _head_contributions()
    local_ids = torch.arange(64 // tp_world_size)
    local = contributions[local_ids].clone()
    arrival_ids = torch.tensor([0, 2, 1, *range(3, 64)])
    calls = []

    def gather(x, tags):
        assert x is local
        assert tags is local_ids
        calls.append(True)
        return contributions[arrival_ids], arrival_ids

    out = ordered_dkv_reduce(local, local_ids, tp_world_size=tp_world_size, collective=gather)
    assert calls == [True]
    assert torch.equal(out, torch.ones(2, 3))
    # The same three finite contributions produce zero in arrival order.
    arrival_sum = (contributions[0] + contributions[2]) + contributions[1]
    assert torch.equal(arrival_sum, torch.zeros(2, 3))
    assert not torch.equal(out, arrival_sum)


def test_rank_local_packing_counterexample_fails_before_collective():
    contributions = _head_contributions()
    local_ids = torch.arange(0, 64, 2)
    local_packed = contributions[0] + contributions[2]
    remote_packed = contributions[1]
    assert torch.equal(local_packed + remote_packed, torch.zeros(2, 3))
    calls = []

    def shape_preserving_allreduce(x):
        calls.append(True)
        return x + remote_packed

    with pytest.raises(DSv4FailClosedError) as exc:
        ordered_dkv_reduce(
            local_packed, local_ids, tp_world_size=2, collective=shape_preserving_allreduce
        )
    assert exc.value.status is DSv4Status.UNSUPPORTED_CAPABILITY
    assert calls == []


def test_tp_gt1_without_collective_is_unsupported():
    with pytest.raises(DSv4FailClosedError) as exc:
        ordered_dkv_reduce(torch.zeros(2, 3, 4), torch.tensor([0, 1]), tp_world_size=2)
    assert exc.value.status is DSv4Status.UNSUPPORTED_CAPABILITY


@pytest.mark.parametrize(
    "tags, status",
    [
        (torch.tensor([0]), DSv4Status.SCHEMA_MISMATCH),
        (torch.tensor([[0, 1]]), DSv4Status.SCHEMA_MISMATCH),
        (torch.tensor([0.0, 1.0]), DSv4Status.SCHEMA_MISMATCH),
        (torch.tensor([0, 0]), DSv4Status.DUPLICATE_LOGICAL_OWNER),
        (torch.tensor([3, 1]), DSv4Status.FORBIDDEN_ATOMIC_REDUCTION),
        (torch.tensor([-1, 1]), DSv4Status.AMBIGUOUS_LOGICAL_INDEX),
        (torch.tensor([0, 64]), DSv4Status.AMBIGUOUS_LOGICAL_INDEX),
    ],
)
def test_invalid_local_head_tags_fail_before_collective(tags, status):
    def gather(*_args):
        pytest.fail("invalid local ownership reached the collective")

    with pytest.raises(DSv4FailClosedError) as exc:
        ordered_dkv_reduce(torch.zeros(2, 3, 4), tags, tp_world_size=2, collective=gather)
    assert exc.value.status is status


@pytest.mark.parametrize("packed", [False, True])
def test_tp1_requires_complete_global_head_ownership(packed):
    dkv = torch.zeros(2, 3) if packed else torch.zeros(2, 2, 3)
    with pytest.raises(DSv4FailClosedError) as exc:
        ordered_dkv_reduce(dkv, torch.arange(2))
    expected = DSv4Status.SCHEMA_MISMATCH if packed else DSv4Status.MISSING_GLOBAL_VISIBILITY
    assert exc.value.status is expected


@pytest.mark.parametrize(
    "fault, status",
    [
        ("missing_head", DSv4Status.MISSING_GLOBAL_VISIBILITY),
        ("duplicate_head", DSv4Status.DUPLICATE_LOGICAL_OWNER),
        ("out_of_range_head", DSv4Status.AMBIGUOUS_LOGICAL_INDEX),
        ("missing_tag", DSv4Status.SCHEMA_MISMATCH),
        ("tag_shape", DSv4Status.SCHEMA_MISMATCH),
        ("tag_dtype", DSv4Status.SCHEMA_MISMATCH),
        ("tag_device", DSv4Status.SCHEMA_MISMATCH),
        ("shape", DSv4Status.SCHEMA_MISMATCH),
        ("dtype", DSv4Status.SCHEMA_MISMATCH),
        ("device", DSv4Status.SCHEMA_MISMATCH),
        ("untagged", DSv4Status.SCHEMA_MISMATCH),
        ("mixed_heads", DSv4Status.CORRUPT_ARTIFACT),
    ],
)
def test_gather_cannot_claim_complete_order_by_preserving_only_shape(fault, status):
    contributions = _head_contributions()
    local_ids = torch.arange(32)
    local = contributions[:32].clone()
    gathered = contributions.clone()
    tags = torch.arange(64)
    if fault == "missing_head":
        gathered, tags = gathered[:-1], tags[:-1]
    elif fault == "duplicate_head":
        tags[-1] = 0
    elif fault == "out_of_range_head":
        tags[-1] = 64
    elif fault == "missing_tag":
        tags = tags[:-1]
    elif fault == "tag_shape":
        tags = tags.reshape(8, 8)
    elif fault == "tag_dtype":
        tags = tags.float()
    elif fault == "tag_device":
        tags = tags.to("meta")
    elif fault == "shape":
        gathered = torch.zeros(64, 3, 3)
    elif fault == "dtype":
        gathered = gathered.double()
    elif fault == "device":
        gathered = gathered.to("meta")
    elif fault == "mixed_heads":
        gathered = gathered.flip(0)

    def gather(_x, _tags):
        return gathered if fault == "untagged" else (gathered, tags)

    with pytest.raises(DSv4FailClosedError) as exc:
        ordered_dkv_reduce(local, local_ids, tp_world_size=2, collective=gather)
    assert exc.value.status is status


def _plan_kwargs():
    return dict(
        tp_rank=0,
        tp_world_size=2,
        group_ids=(0, 1, 2, 3),
        wo_b_split="row",
        merge_order="group_index_ascending",
        actual_collective="tagged_group_gather",
    )


def test_tp1_o_proj_plan():
    plan = tp1_plan()
    assert plan.tp_world_size == 1
    assert plan.group_ids == tuple(range(8))
    assert plan.wo_b_split == "none"
    assert plan.actual_collective is None


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_contiguous_o_proj_plans_partition_all_groups(world_size):
    plans = []
    for rank in range(world_size):
        kwargs = _plan_kwargs()
        width = 8 // world_size
        kwargs.update(
            tp_rank=rank,
            tp_world_size=world_size,
            group_ids=tuple(range(rank * width, (rank + 1) * width)),
            wo_b_split="none" if world_size == 1 else "row",
            actual_collective=None if world_size == 1 else "tagged_group_gather",
        )
        plans.append(OProjShardPlan(**kwargs))
    assert tuple(group for plan in plans for group in plan.group_ids) == tuple(range(8))


@pytest.mark.parametrize(
    "groups_by_rank",
    [
        ((0, 2, 4, 6), (1, 3, 5, 7)),
        ((0, 1, 2), (3, 4, 5, 6, 7)),
        ((4, 5, 6, 7), (0, 1, 2, 3)),
        ((0, 1, 2, 7), (3, 4, 5, 6)),
    ],
    ids=["interleaved", "uneven", "reversed-ranks", "whole-groups"],
)
def test_two_rank_accepts_ordered_whole_group_partitions(groups_by_rank):
    plans = []
    for rank, group_ids in enumerate(groups_by_rank):
        kwargs = _plan_kwargs()
        kwargs.update(tp_rank=rank, group_ids=group_ids)
        plans.append(OProjShardPlan(**kwargs))
        assert plans[-1].group_ids == group_ids
    assert sorted(group for plan in plans for group in plan.group_ids) == list(range(8))


@pytest.mark.parametrize(
    "changes, status",
    [
        ({"group_ids": ()}, DSv4Status.MISSING_GLOBAL_VISIBILITY),
        ({"group_ids": (0, 1, 2, 2)}, DSv4Status.DUPLICATE_LOGICAL_OWNER),
        ({"group_ids": (1, 0, 2, 3)}, DSv4Status.INVALID_CANDIDATE_ORDER),
        ({"group_ids": (0, 1, 2, 8)}, DSv4Status.SCHEMA_MISMATCH),
        ({"group_ids": (0, 1, 2, 3.0)}, DSv4Status.SCHEMA_MISMATCH),
        ({"wo_b_split": "unknown"}, DSv4Status.UNSUPPORTED_CAPABILITY),
        ({"wo_b_split": "column"}, DSv4Status.UNSUPPORTED_CAPABILITY),
        ({"wo_b_split": "none"}, DSv4Status.SCHEMA_MISMATCH),
        ({"actual_collective": None}, DSv4Status.MISSING_PROVENANCE),
        ({"merge_order": "rank_order"}, DSv4Status.INVALID_CANDIDATE_ORDER),
        ({"tp_world_size": 3}, DSv4Status.UNSUPPORTED_CAPABILITY),
        ({"tp_rank": 2}, DSv4Status.MISSING_RANK),
        ({"tp_world_size": 1}, DSv4Status.MISSING_GLOBAL_VISIBILITY),
        (
            {"tp_world_size": 1, "group_ids": tuple(range(7)), "wo_b_split": "none"},
            DSv4Status.MISSING_GLOBAL_VISIBILITY,
        ),
        (
            {"tp_world_size": 1, "group_ids": tuple(range(8)), "wo_b_split": "row"},
            DSv4Status.SCHEMA_MISMATCH,
        ),
    ],
)
def test_invalid_o_proj_ownership_and_split_fail_closed(changes, status):
    kwargs = _plan_kwargs()
    kwargs.update(changes)
    with pytest.raises(DSv4FailClosedError) as exc:
        OProjShardPlan(**kwargs)
    assert exc.value.status is status
