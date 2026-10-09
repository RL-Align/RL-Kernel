# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""The sub-node recorder used for first-drift localisation inside a decoder layer."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations import qwen3_next_subnode as subnode


def test_buffer_requires_each_sub_node_once_with_its_original_storage():
    buffer = subnode.SubnodeBuffer(batch=2)
    row = torch.arange(2048, dtype=torch.float32).to(torch.bfloat16)
    buffer.put(3, "mixer", 1, 15, row)
    assert torch.equal(buffer.values["mixer"][3, 1, 15], row)
    with pytest.raises(ValueError, match="twice"):
        buffer.put(3, "mixer", 1, 15, row)
    with pytest.raises(ValueError, match="int64"):
        buffer.put(3, "router", 1, 15, torch.zeros(10, dtype=torch.int32))
    with pytest.raises(ValueError, match="bfloat16"):
        buffer.put(4, "mlp", 0, 0, row.float())
    with pytest.raises(ValueError, match="not all observed"):
        buffer.layer(3)
    with pytest.raises(ValueError, match="Unknown"):
        subnode.subnode_layout("gate")


def test_complete_layer_and_embedding_are_returned_in_order():
    buffer = subnode.SubnodeBuffer(batch=1)
    for name in subnode.SUBNODES:
        width, dtype = subnode.subnode_layout(name)
        buffer.put_rows(0, name, torch.zeros(1, 16, width, dtype=dtype))
    assert tuple(buffer.layer(0)) == subnode.SUBNODES
    with pytest.raises(ValueError, match="every response position"):
        buffer.embedding_rows()
    for response in range(16):
        buffer.put_embedding(0, response, torch.zeros(2048, dtype=torch.bfloat16))
    assert buffer.embedding_rows().shape == (1, 16, 2048)


def test_install_binds_the_layer_index_and_remove_clears_every_hook():
    layers = [SimpleNamespace(mlp=SimpleNamespace()) for _ in range(3)]
    seen = []
    subnode.install(layers, subnode.vllm_modules, lambda *args: seen.append(args))
    layers[2].subnode_hook("mixer", "rows")
    layers[1].mlp.subnode_hook("router", "ids")
    assert seen == [(2, "mixer", "rows"), (1, "router", "ids")]
    subnode.remove(layers, subnode.vllm_modules)
    assert all(layer.subnode_hook is None and layer.mlp.subnode_hook is None for layer in layers)
