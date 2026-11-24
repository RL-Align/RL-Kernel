# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU protocol doubles for framework primitives, not architecture adapters."""

import copy

import pytest
import torch
from torch import nn

from rl_engine.alignment.debug import DiscoverCapture
from rl_engine.alignment.debug.discovery_weights import (
    align_primitive_views,
    parameter_inventory,
    scope_inventory,
)
from rl_engine.alignment.debug.entry import analyze


class UnquantizedLinearMethod:
    pass


class LinearBase(nn.Module):
    __module__ = "vllm.model_executor.layers.linear"


class QKVParallelLinear(LinearBase):
    __module__ = "vllm.model_executor.layers.linear"

    def __init__(self, projections):
        super().__init__()
        self.weight = nn.Parameter(torch.cat([m.weight.detach() for m in projections]))
        self.weight.input_dim, self.weight.output_dim = 1, 0
        self.bias = nn.Parameter(torch.cat([m.bias.detach() for m in projections]))
        self.output_sizes = [m.out_features for m in projections]
        self.quant_method = UnquantizedLinearMethod()
        self.skip_bias_add = False

    def forward(self, value):
        return torch.nn.functional.linear(value, self.weight, self.bias), None


class Conv1D(nn.Module):
    __module__ = "transformers.pytorch_utils"

    def __init__(self, projection):
        super().__init__()
        self.weight = nn.Parameter(projection.weight.detach().T.contiguous())
        self.bias = nn.Parameter(projection.bias.detach().clone())

    def forward(self, value):
        return value @ self.weight + self.bias


def test_explicit_primitive_transpose_and_whole_fused_view_match():
    torch.manual_seed(23)
    projections = [nn.Linear(5, 5) for _ in range(3)]
    fused = QKVParallelLinear(projections)
    whole = nn.Linear(5, 15)
    with torch.no_grad():
        whole.weight.copy_(fused.weight)
        whole.bias.copy_(fused.bias)
    left, right = parameter_inventory(Conv1D(whole)), parameter_inventory(fused)
    assert scope_inventory(left) != scope_inventory(right)
    align_primitive_views(left, right)
    assert scope_inventory(left) == scope_inventory(right)
    assert right["owners"][""]["selected_view"] == "verified_whole_projection"
    assert len(right["owners"][""]["atoms"]) == 1


def test_separate_projections_keep_declared_fused_fragments():
    projections = nn.ModuleList([nn.Linear(5, 5) for _ in range(3)])
    left, right = (
        parameter_inventory(projections),
        parameter_inventory(QKVParallelLinear(projections)),
    )
    align_primitive_views(left, right)
    assert scope_inventory(left) == scope_inventory(right)
    assert len(right["owners"][""]["atoms"]) == 3


@pytest.mark.parametrize("mutation", ["sizes", "axes", "quantization"])
def test_invalid_primitive_contract_is_not_guessed(mutation):
    model = QKVParallelLinear([nn.Linear(5, 5) for _ in range(3)])
    if mutation == "sizes":
        model.output_sizes = [5, 5]
    elif mutation == "axes":
        model.weight.input_dim, model.weight.output_dim = 0, 1
    else:
        model.quant_method = object()
    with pytest.raises(ValueError):
        parameter_inventory(model)


def test_fusion_mapping_uses_actual_parameter_bytes():
    projections = nn.ModuleList([nn.Linear(5, 5) for _ in range(3)])
    native = QKVParallelLinear(projections)
    with torch.no_grad():
        native.weight[0, 0] += 1
    left, right = parameter_inventory(projections), parameter_inventory(native)
    align_primitive_views(left, right)
    assert scope_inventory(left) != scope_inventory(right)


def test_separate_and_fused_graphs_are_discovered_without_bindings(tmp_path):
    class Separate(nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v = (nn.Linear(5, 5) for _ in range(3))
            self.down = nn.Linear(5, 5)

        def forward(self, hidden):
            return hidden + self.down(self.q(hidden) + self.k(hidden) + self.v(hidden))

    class Fused(nn.Module):
        def __init__(self, source):
            super().__init__()
            self.projection = QKVParallelLinear([source.q, source.k, source.v])
            self.finish = copy.deepcopy(source.down)

        def forward(self, hidden):
            q, k, v = self.projection(hidden)[0].chunk(3, dim=-1)
            return hidden + self.finish(q + k + v)

    class Model(nn.Module):
        def __init__(self, source=None):
            super().__init__()
            self.embedding = (
                nn.Embedding(16, 5) if source is None else copy.deepcopy(source.embedding)
            )
            self.blocks = nn.ModuleList(
                [Separate() for _ in range(3)]
                if source is None
                else [Fused(block) for block in source.blocks]
            )

        def forward(self, input_ids, position_ids):
            value = self.embedding(input_ids)
            for block in self.blocks:
                value = block(value)
            return value

    left = Model().eval()
    right = Model(left).eval()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    positions = torch.arange(3).expand(2, -1)
    for side, model in (("training", left), ("rollout", right)):
        with torch.no_grad(), DiscoverCapture(tmp_path, side, model, tokens=tokens):
            model(input_ids=tokens, position_ids=positions)
    report = analyze(tmp_path)
    assert report["status"] in {"equal", "diverged"}, report
    assert report["automatic_mapping"]["status"] == "validated"
    assert report["automatic_mapping"]["blocks"][0]["anchors"][0]["fused_components"] == 3
