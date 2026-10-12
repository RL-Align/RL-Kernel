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


def test_layer_gates_follow_expert_parallelism():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/validation/models"))
    from qwen3_next_layer_contract import layer_requirements

    root = Path(__file__).resolve().parents[3]
    tp = layer_requirements(root, 1)["gates"]
    ep = layer_requirements(root, 4)["gates"]
    assert "moe" in tp and "moe_ep" not in tp
    assert "moe_ep" in ep and "moe" not in ep
    assert ep["moe_ep"]["scope"] == "ep4_moe_block_layer0"
    assert set(tp) - {"moe"} == set(ep) - {"moe_ep"}
    with pytest.raises(ValueError):
        layer_requirements(root, 2)


@pytest.mark.parametrize(
    ("ep", "etp", "accepted"),
    [(1, 4, True), (4, 1, True), (4, 4, False), (2, 2, False), (1, 1, False)],
)
def test_training_topology_accepts_only_ep1_etp4_or_ep4_etp1(ep, etp, accepted):
    from types import SimpleNamespace

    from rl_engine.models.qwen3_next.qwen3_next_engine_weights import validate_training_topology

    config = SimpleNamespace(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    if accepted:
        validate_training_topology(config, config)
    else:
        with pytest.raises(ValueError):
            validate_training_topology(config, config)
