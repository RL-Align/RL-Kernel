# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Explicit shared Qwen TP4 logprob/entropy boundary for native VIME policy loss."""

import torch
import torch.distributed as dist

from rl_engine.contracts.operators.logprob import (
    LogprobContract,
    LogprobDType,
    LogprobRole,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.models.qwen3_next.qwen3_next_forward import shared_logprob


def qwen_logprob_contract(logits, *, rank, world_size, role=LogprobRole.TRAIN):
    if world_size != 4 or type(rank) is not int or not 0 <= rank < 4:
        raise ValueError("Full Qwen logprob execution requires TP4")
    if logits.ndim != 2 or logits.shape[1] != 37984:
        raise ValueError("Each Qwen TP4 rank must own 37984 vocabulary columns")
    dtype = {torch.bfloat16: LogprobDType.BF16, torch.float32: LogprobDType.FP32}.get(logits.dtype)
    if dtype is None:
        raise ValueError("Qwen LM head produces BF16 logits, optionally promoted by VIME")
    return LogprobContract(
        role=role,
        dtype=dtype,
        mask=MaskSpec(num_tokens=logits.shape[0], active_mask=(True,) * logits.shape[0]),
        sharding=ShardingSpec(
            tp_rank=rank,
            tp_world_size=4,
            vocab_shard_bounds=tuple((index * 37984, (index + 1) * 37984) for index in range(4)),
            real_vocab_size=151936,
            padded_vocab_size=151936,
            cp_rank=0,
            cp_world_size=1,
        ),
        reduction=ReductionSpec(),
    )


def calculate_log_probs_and_entropy(
    logits,
    tokens,
    tp_group,
    with_entropy=False,
    chunk_size=-1,
    log_prob_keep_mask=None,
    with_entropy_grad=True,
):
    """VIME's numerical callback; its policy loss and response alignment stay native."""
    if log_prob_keep_mask is not None:
        raise ValueError("Fixed Qwen eager profile does not support top-p replay")
    if not dist.is_initialized() or tp_group is None:
        raise ValueError("Shared Qwen logprob requires an explicit initialized TP4 group")
    if logits.device.type != "cuda" or tokens.device != logits.device:
        raise ValueError("Logits and target IDs must share a CUDA device")
    if tokens.shape != (logits.shape[0],):
        raise ValueError("Logprob target IDs must match the token rows")
    contract = qwen_logprob_contract(
        logits, rank=dist.get_rank(tp_group), world_size=dist.get_world_size(tp_group)
    )
    # Keep one fixed numerical path even when VIME requests chunking or metrics
    # only. Its chunk_size is a memory hint; this explicit provider owns tiling.
    selected, _lse, entropy = shared_logprob(
        logits, tokens, contract=contract, tp_group=tp_group, with_entropy=True
    )
    if not with_entropy_grad:
        entropy = entropy.detach()
    return selected.unsqueeze(-1), entropy if with_entropy else None


def install_vime_logprob_provider():
    """Install explicitly in each worker, without modifying the shared VIME checkout."""
    from megatron.core import mpu
    from vime.backends.megatron_utils import loss

    topology = (mpu.get_context_parallel_world_size(), mpu.get_tensor_model_parallel_world_size())
    if topology != (1, 4):
        raise ValueError("The Qwen native policy adapter supports TP4/CP1 only")
    original = loss.calculate_log_probs_and_entropy
    if original is calculate_log_probs_and_entropy:
        return
    if original.__module__ != "vime.utils.ppo_utils":
        raise ValueError("Refusing to replace an unknown VIME logprob provider")
    loss.calculate_log_probs_and_entropy = calculate_log_probs_and_entropy
