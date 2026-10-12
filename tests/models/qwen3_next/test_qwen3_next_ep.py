# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks of the EP4 expert placement used by both engines."""

import pytest
import torch

from rl_engine.models.qwen3_next.qwen3_next_ep import (
    EP,
    LOCAL_EXPERTS,
    _slots_by_owner,
    ep_local_hf_names,
    expert_map,
    expert_owner,
    owns_expert,
)
from rl_engine.models.qwen3_next.qwen3_next_tp_blocks import moe_hf_shapes


def test_every_expert_has_exactly_one_owner():
    owners = expert_owner(torch.arange(512))
    assert torch.equal(owners, torch.arange(512) // 128)
    names = [n for n in moe_hf_shapes() if n.startswith("experts.")]
    held = [n for rank in range(EP) for n in ep_local_hf_names(rank) if n.startswith("experts.")]
    assert sorted(held) == sorted(names) and len(held) == len(set(held)) == 512 * 3


def test_shared_tensors_are_on_every_rank():
    shared = [n for n in moe_hf_shapes() if not n.startswith("experts.")]
    for rank in range(EP):
        assert all(owns_expert(n, rank) for n in shared)
        assert len(ep_local_hf_names(rank)) == LOCAL_EXPERTS * 3 + len(shared)


@pytest.mark.parametrize("rank", range(EP))
def test_expert_map_matches_vllm_convention(rank):
    mapping = expert_map(rank, "cpu")
    assert mapping.dtype == torch.int32 and mapping.shape == (512,)
    local = torch.arange(rank * 128, (rank + 1) * 128)
    assert torch.equal(mapping[local], torch.arange(128, dtype=torch.int32))
    assert int((mapping == -1).sum()) == 384


def test_expert_map_rejects_bad_rank():
    with pytest.raises(ValueError):
        expert_map(4, "cpu")


def test_slots_partition_every_route_once_in_ascending_order():
    indices = torch.tensor(
        [[0, 130, 260, 390, 5, 135, 265, 395, 10, 140], [511, 1, 2, 3, 4, 6, 7, 8, 9, 12]]
    )
    slots = _slots_by_owner(indices)
    flat = torch.cat(slots)
    assert sorted(flat.tolist()) == list(range(20))
    assert all(torch.equal(s, torch.sort(s).values) for s in slots)
    for rank, owned in enumerate(slots):
        assert bool((expert_owner(indices.reshape(-1)[owned]) == rank).all())
    empty = _slots_by_owner(torch.randint(0, 384, (4, 10)))
    assert empty[3].numel() == 0
