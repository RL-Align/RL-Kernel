# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Sub-node observations inside each Qwen3-Next decoder layer, for first-drift localisation.

Both the rollout model (vLLM) and the replay model (VIME) call a module's
``subnode_hook(name, packed_rows)`` at the same five points of every decoder
layer, in this order:

* ``input_layernorm`` - the normed mixer input;
* ``mixer`` - the GDN or full-attention output after the TP reduction;
* ``post_attention_layernorm`` - the normed MLP input;
* ``router`` - the ten selected expert ids per token;
* ``mlp`` - the MoE output after the TP reduction.

The embedding output is observed once per model as ``embedding`` (on the
embedding module itself) so a drift there is not attributed to layer 0's norm.

A trace owner binds the layer index, maps packed token rows to the 16
response positions of each sample, and stores them in :class:`SubnodeBuffer`.
Observing copies values and never changes one.
"""

from functools import partial

import torch

SUBNODES = ("input_layernorm", "mixer", "post_attention_layernorm", "router", "mlp")
EMBEDDING = "embedding"
LAYERS, RESPONSE_LENGTH, HIDDEN, TOP_K = 48, 16, 2048, 10


def subnode_layout(name):
    if name == "router":
        return TOP_K, torch.int64
    if name in SUBNODES or name == EMBEDDING:
        return HIDDEN, torch.bfloat16
    raise ValueError(f"Unknown Qwen3-Next sub-node: {name}")


class SubnodeBuffer:
    """[48, B, 16, width] host copies of every sub-node, each written exactly once."""

    def __init__(self, batch):
        self.batch = batch
        self.values, self.written = {}, {}
        for name in SUBNODES:
            width, dtype = subnode_layout(name)
            self.values[name] = torch.zeros(LAYERS, batch, RESPONSE_LENGTH, width, dtype=dtype)
            self.written[name] = torch.zeros(LAYERS, batch, RESPONSE_LENGTH, dtype=torch.bool)
        self.embedding = torch.zeros(batch, RESPONSE_LENGTH, HIDDEN, dtype=torch.bfloat16)
        self.embedding_written = torch.zeros(batch, RESPONSE_LENGTH, dtype=torch.bool)

    def put_embedding(self, row, response, value):
        if value.dtype != torch.bfloat16 or value.shape != (HIDDEN,):
            raise ValueError("The embedding must keep its original BF16 [2048] storage")
        if self.embedding_written[row, response]:
            raise ValueError("The embedding of one response position was observed twice")
        self.embedding[row, response].copy_(value.detach().cpu())
        self.embedding_written[row, response] = True

    def embedding_rows(self):
        if not bool(self.embedding_written.all()):
            raise ValueError("The embedding was not observed at every response position")
        return self.embedding

    def put(self, layer, name, row, response, value):
        width, dtype = subnode_layout(name)
        if value.dtype != dtype or value.shape != (width,):
            raise ValueError(f"Sub-node {name} must keep its original {dtype} [{width}] storage")
        if self.written[name][layer, row, response]:
            raise ValueError(f"Sub-node {name} of layer {layer} was observed twice")
        self.values[name][layer, row, response].copy_(value.detach().cpu())
        self.written[name][layer, row, response] = True

    def put_rows(self, layer, name, values):
        """Store a complete [B, 16, width] block, as the replay model produces it."""
        for row in range(self.batch):
            for response in range(RESPONSE_LENGTH):
                self.put(layer, name, row, response, values[row, response])

    def layer(self, layer):
        missing = [name for name in SUBNODES if not bool(self.written[name][layer].all())]
        if missing:
            raise ValueError(f"Layer {layer} sub-nodes were not all observed: {missing}")
        return {name: self.values[name][layer] for name in SUBNODES}


def install(layers, modules_of, record):
    """Bind ``record(layer_index, name, packed_rows)`` to every hook-bearing module."""
    for index, layer in enumerate(layers):
        for module in modules_of(layer):
            module.subnode_hook = partial(record, index)


def remove(layers, modules_of):
    for layer in layers:
        for module in modules_of(layer):
            module.subnode_hook = None


def vllm_modules(layer):
    """vLLM ``SharedVLLMDecoderLayer``: the layer itself and its MoE."""
    return (layer, layer.mlp)


def vime_modules(layer):
    """MCore layer with the shared VIME providers: mixer, pre-MLP norm and MoE."""
    return (layer.self_attention, layer.pre_mlp_layernorm, layer.mlp)
