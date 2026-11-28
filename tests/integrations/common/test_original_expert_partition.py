# SPDX-License-Identifier: Apache-2.0
"""Actual packed-byte ownership checks; not model EP4/communication acceptance."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.engines.train.megatron.expert_partition import (
    OriginalExpertPartition,
    load_original_experts,
)


def test_ep4_covers_every_original_expert_once_and_records_dense_dp4():
    parts = [OriginalExpertPartition(256, 4, r) for r in range(4)]
    assert [e for p in parts for e in p.expert_ids] == list(range(256))
    for p in parts:
        p.verify_native(p.expert_ids, etp=1, dense_dp=4, expert_dp=1)
        with pytest.raises(ValueError, match="dense"):
            p.verify_native(p.expert_ids, etp=1, dense_dp=1, expert_dp=1)
        with pytest.raises(ValueError, match="indices"):
            p.verify_native(reversed(p.expert_ids), etp=1, dense_dp=4, expert_dp=1)


def test_loader_reads_only_local_original_experts_and_preserves_packed_bytes():
    seen = []

    def get_tensor(name):
        seen.append(name)
        expert = int(name.split(".")[4])
        shape = (32, 16) if name.endswith("weight") else (32, 1)
        return torch.full(shape, expert, dtype=torch.uint8)

    part = OriginalExpertPartition(256, 4, 2)
    gate_up, down, receipt = load_original_experts(
        SimpleNamespace(get_tensor=get_tensor), "layers.4", part, "cpu"
    )
    assert len(seen) == 64 * 3 * 2
    assert {int(n.split(".")[4]) for n in seen} == set(range(128, 192))
    assert receipt["original_expert_ids"] == tuple(range(128, 192))
    for local, original in enumerate(part.expert_ids):
        assert bool((gate_up.codes[local] == original).all())
        assert bool((gate_up.scales[local] == original).all())
        assert bool((down.codes[local] == original).all())
        assert bool((down.scales[local] == original).all())
