# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU checks through VIME's real parameter shard and HF export entrypoints."""

from types import SimpleNamespace

import torch
from megatron.core import mpu
from vime.backends.megatron_utils import hf_to_megatron as load_api
from vime.backends.megatron_utils import megatron_to_hf as export_api
from vime.backends.megatron_utils.hf_to_megatron.common import shard_mcore_tensor

from rl_engine.integrations.orchestrators.vime.providers.qwen3_next_vime import (
    SharedInputNorm,
    install_vime_gdn_weight_hooks,
)
from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal as exact


class Reader(dict):
    def get_tensor(self, key):
        return self[key]


def test_vime_hf_load_shard_gather_export_roundtrip(monkeypatch):
    monkeypatch.setitem(load_api._LOADERS, "qwen3_next", load_api._LOADERS["qwen3_next"])
    monkeypatch.setattr(export_api, "convert_qwen3_next_to_hf", export_api.convert_qwen3_next_to_hf)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 4)
    install_vime_gdn_weight_hooks()
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn((8192, 1, 4), generator=generator).bfloat16()
    name = "module.module.decoder.layers.0.self_attention.linear_attn.conv1d.weight"
    reader = Reader({"model.layers.0.linear_attn.conv1d.weight": weight})
    packed = load_api._LOADERS["qwen3_next"](name, reader, SimpleNamespace())
    parameter = torch.nn.Parameter(torch.empty(2048, 1, 4, dtype=torch.bfloat16))
    parameter.tensor_model_parallel = True
    parameter.partition_dim = 0
    parameter.partition_stride = 1
    local = []
    for rank in range(4):
        monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda rank=rank: rank)
        shard = shard_mcore_tensor(name, packed, parameter)
        exact(shard, torch.cat([part.chunk(4)[rank] for part in weight.split((2048, 2048, 4096))]))
        local.append(shard)
    args = SimpleNamespace(
        hidden_size=2048, num_attention_heads=16, num_query_groups=2, kv_channels=256
    )
    converted = export_api.convert_qwen3_next_to_hf(args, name, torch.cat(local))
    assert converted[0][0] == "model.layers.0.linear_attn.conv1d.weight"
    exact(converted[0][1], weight)


def test_shared_norm_keeps_existing_parameter_name_and_identity():
    weight = torch.nn.Parameter(torch.zeros(2048, dtype=torch.bfloat16))
    norm = SharedInputNorm(1e-6, torch.device("cpu"), weight=weight)
    assert dict(norm.named_parameters()) == {"weight": weight}
    assert norm.weight is weight
    assert norm.weight.tensor_model_parallel is False


def test_vime_kv_load_shard_export_uses_two_replica_pairs(monkeypatch):
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 4)
    install_vime_gdn_weight_hooks()
    weight = torch.randn(512, 2048).bfloat16()
    name = "module.module.decoder.layers.3.self_attention.self_attn.k_proj.weight"
    reader = Reader({"model.layers.3.self_attn.k_proj.weight": weight})
    packed = load_api._LOADERS["qwen3_next"](name, reader, SimpleNamespace())
    parameter = torch.nn.Parameter(torch.empty(256, 2048, dtype=torch.bfloat16))
    parameter.tensor_model_parallel = True
    parameter.partition_dim = 0
    parameter.partition_stride = 1
    shards = []
    for rank in range(4):
        monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda rank=rank: rank)
        shard = shard_mcore_tensor(name, packed, parameter)
        exact(shard, weight.chunk(2)[rank // 2])
        shards.append(shard)
    args = SimpleNamespace(
        hidden_size=2048, num_attention_heads=16, num_query_groups=2, kv_channels=256
    )
    outputs = export_api.convert_qwen3_next_to_hf(args, name, torch.cat(shards))
    assert outputs[0][0] == "model.layers.3.self_attn.k_proj.weight"
    exact(outputs[0][1], weight)


def test_vime_shared_expert_direct_names_roundtrip(monkeypatch):
    install_vime_gdn_weight_hooks()
    weight = torch.randn(512, 2048).bfloat16()
    hf_name = "model.layers.0.mlp.shared_expert.gate_proj.weight"
    name = "module.module.decoder.layers.0.mlp.shared_expert.gate_proj.weight"
    loaded = load_api._LOADERS["qwen3_next"](name, Reader({hf_name: weight}), SimpleNamespace())
    outputs = export_api.convert_qwen3_next_to_hf(SimpleNamespace(), name, loaded)
    assert outputs[0][0] == hf_name
    exact(outputs[0][1], weight)
