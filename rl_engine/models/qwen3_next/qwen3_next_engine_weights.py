# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Explicit TP metadata and convolution layout conversion at engine boundaries."""

import torch


def conv_hf_to_rank_major(weight):
    """Arrange HF [Q(all ranks), K(all ranks), V(all ranks)] for a dim-0 TP split."""
    if weight.shape != (8192, 1, 4) or weight.dtype != torch.bfloat16:
        raise ValueError("Qwen3-Next convolution must be BF16 [8192, 1, 4]")
    q, k, v = weight.split((2048, 2048, 4096))
    return torch.cat([part.chunk(4)[rank] for rank in range(4) for part in (q, k, v)])


def conv_rank_major_to_hf(weight):
    """Undo VIME's rank-major TP gather before exporting the official HF layout."""
    if weight.shape != (8192, 1, 4) or weight.dtype != torch.bfloat16:
        raise ValueError("Gathered Qwen3-Next convolution must be BF16 [8192, 1, 4]")
    shards = [part.split((512, 512, 1024)) for part in weight.chunk(4)]
    return torch.cat([shards[rank][segment] for segment in range(3) for rank in range(4)])


def kv_hf_to_rank_major(weight):
    if weight.shape != (512, 2048) or weight.dtype != torch.bfloat16:
        raise ValueError("Official KV projection requires BF16 [512, 2048]")
    first, second = weight.chunk(2)
    return torch.cat((first, first, second, second))


def kv_rank_major_to_hf(weight):
    from rl_engine.validation.common.tensor_identity import assert_tensor_bitwise_equal

    if weight.shape != (1024, 2048) or weight.dtype != torch.bfloat16:
        raise ValueError("Gathered pair-replicated KV requires BF16 [1024, 2048]")
    parts = weight.chunk(4)
    assert_tensor_bitwise_equal(parts[0], parts[1], name="KV replica pair 0")
    assert_tensor_bitwise_equal(parts[2], parts[3], name="KV replica pair 1")
    return torch.cat((parts[0], parts[2]))


def expert_hf_to_rank_major(gate, up):
    if any(value.shape != (512, 2048) or value.dtype != torch.bfloat16 for value in (gate, up)):
        raise ValueError("Official expert gate/up require BF16 [512, 2048]")
    return torch.cat([value.chunk(4)[rank] for rank in range(4) for value in (gate, up)])


def expert_rank_major_to_hf(weight):
    if weight.shape != (1024, 2048) or weight.dtype != torch.bfloat16:
        raise ValueError("Gathered fused expert requires BF16 [1024, 2048]")
    ranks = [rank.chunk(2) for rank in weight.chunk(4)]
    return tuple(torch.cat([ranks[rank][segment] for rank in range(4)]) for segment in range(2))


def set_gdn_tp_metadata(module):
    """Tell VIME/Megatron which weights are replicated and how TP gathers them.

    Convolution uses stride one because import/export hooks handle its unequal
    Q/K/V segments. The shared norm's gradient is already reduced by TP4GDN.
    """
    for name, parameter in module.named_parameters():
        replicated = name == "norm.weight"
        parameter.tensor_model_parallel = not replicated
        parameter.partition_dim = -1 if replicated else (1 if name == "out_proj.weight" else 0)
        parameter.partition_stride = 1
        parameter.sequence_parallel = False
        parameter.allreduce = True


def validate_training_topology(config, args):
    for name, expected in (
        ("tensor_model_parallel_size", 4),
        ("pipeline_model_parallel_size", 1),
        ("context_parallel_size", 1),
    ):
        if getattr(config, name, None) != expected:
            raise ValueError(f"Shared Qwen3-Next requires {name}={expected}")
    expert = (
        getattr(config, "expert_model_parallel_size", None),
        getattr(config, "expert_tensor_parallel_size", 4),
    )
    if expert not in ((1, 4), (4, 1)):
        raise ValueError("Shared Qwen3-Next requires EP1 x ETP4 or EP4 x ETP1 over TP4")
    if getattr(config, "sequence_parallel", False) or getattr(args, "sequence_parallel", False):
        raise ValueError("Shared Qwen3-Next requires sequence parallelism disabled")
    if getattr(config, "fp32_residual_connection", False):
        raise ValueError("Shared Qwen3-Next requires BF16 residual additions")
    if any(
        getattr(config, name, 0)
        for name in ("attention_dropout", "hidden_dropout", "mtp_num_layers")
    ):
        raise ValueError("Shared Qwen3-Next requires dropout and MTP disabled")
