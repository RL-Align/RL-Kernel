# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import torch
from torch import nn

from rl_engine.validation.models.inkling2.miles_golden import (
    _hardware,
    capture_block_visits,
    checkpoint_digest,
)
from rl_engine.validation.models.inkling2.visit_trace import VisitTrace


class Block(nn.Module):
    def forward(self, partial):
        return [], partial * 1.01


class LoopFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({str(i): Block() for i in range(24)})
        self.core = range(6, 18)
        self.num_passes = 3
        self.activation_checkpoint = False

    def forward(self, x):
        # Deliberately tests hook mechanics, not the released model arithmetic.
        layers = list(self.layers.values())
        for block in layers[:6] + layers[6:18] * 3 + layers[18:]:
            _, x = block(x)
        return x


def test_real_module_hooks_and_backward_cleanup():
    model = LoopFixture()
    trace = VisitTrace({"capture_mode": "fixture"})
    with capture_block_visits(model, trace):
        model(torch.ones(2, 4, requires_grad=True)).sum().backward()
    records = trace.report()["records"]
    assert len(records) == 96
    assert {r["logical"] for r in records if r["direction"] == "backward"} == set(range(48))
    assert all(not block._forward_hooks for block in model.layers.values())


def test_incomplete_and_wrong_order_fail_closed():
    model = LoopFixture()
    with pytest.raises(RuntimeError, match="incomplete forward"):
        with capture_block_visits(model, VisitTrace({})):
            model.layers["0"](torch.ones(1, requires_grad=True))
    with pytest.raises(RuntimeError, match="unexpected physical"):
        with capture_block_visits(model, VisitTrace({})):
            model.layers["1"](torch.ones(1, requires_grad=True))
    assert all(not block._forward_hooks for block in model.layers.values())


def test_checkpoint_hash_reads_shards(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    shard = tmp_path / "model-00001.safetensors"
    index.write_text(json.dumps({"weight_map": {"weight": shard.name}}))
    shard.write_bytes(b"first")
    first, _ = checkpoint_digest(tmp_path)
    shard.write_bytes(b"second")
    second, _ = checkpoint_digest(tmp_path)
    assert first != second
    with pytest.raises(RuntimeError, match="actual GPU"):
        _hardware(torch.device("cpu"))
