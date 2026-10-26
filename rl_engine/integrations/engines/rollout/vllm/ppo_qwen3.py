# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Exact per-rank Qwen3 weight-install verification for the PPO acceptance runner."""
from __future__ import annotations

import hashlib

import torch


def _source(weights, name):
    try:
        return weights[name]
    except KeyError:
        raise ValueError(f"no checkpoint tensor maps to vLLM parameter {name}") from None


def expected_shard(name, weights, *, rank, tp, shape):
    if ".qkv_proj." in name:
        return torch.cat(
            [
                _source(weights, name.replace("qkv_proj", n)).chunk(tp, dim=0)[rank]
                for n in ("q_proj", "k_proj", "v_proj")
            ],
            dim=0,
        )
    if ".gate_up_proj." in name:
        return torch.cat(
            [
                _source(weights, name.replace("gate_up_proj", n)).chunk(tp, dim=0)[rank]
                for n in ("gate_proj", "up_proj")
            ],
            dim=0,
        )
    value = _source(weights, name)
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        padded = shape[0] * tp
        if padded < value.shape[0]:
            raise ValueError("vLLM vocabulary partition is smaller than the checkpoint")
        return torch.nn.functional.pad(value, (0, 0, 0, padded - value.shape[0])).chunk(tp, dim=0)[
            rank
        ]
    if ".o_proj." in name or ".down_proj." in name:
        return value.chunk(tp, dim=1)[rank]
    return value


def verify_acknowledgments(acks, *, version, tp):
    if len(acks) != tp or {a["rank"] for a in acks} != set(range(tp)):
        raise ValueError("missing or duplicate vLLM rank acknowledgment")
    if any(a["version"] != version or a["tp"] != tp or a["verified_tensors"] <= 0 for a in acks):
        raise ValueError("stale or incomplete vLLM weight acknowledgment")


def install_weights(worker, filename, version):
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    if version != getattr(worker, "ppo_actor_version", -1) + 1:
        raise ValueError("worker rejected stale/out-of-order weight installation")
    rank, tp = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    packet = torch.load(filename, map_location="cpu", weights_only=True)
    if packet["actor_version"] != version:
        raise ValueError("weight file actor version is stale")
    weights = packet["weights"]
    model = worker.model_runner.model
    loaded = model.load_weights(weights.items())
    parameters = dict(model.named_parameters())
    if loaded != set(parameters):
        raise ValueError("vLLM did not load every Qwen3 parameter")
    digest = verify_parameters(model, weights, rank=rank, tp=tp)
    torch.cuda.synchronize()
    worker.ppo_actor_version = version
    return dict(
        rank=rank,
        tp=tp,
        version=version,
        verified_tensors=len(parameters),
        installed_sha256=digest,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )


def verify_parameters(model, weights, *, rank, tp):
    parameters = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name, parameter in parameters.items():
        expected = expected_shard(name, weights, rank=rank, tp=tp, shape=parameter.shape)
        actual = parameter.detach().cpu()
        if (
            actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not torch.equal(actual, expected)
        ):
            raise ValueError(f"incorrect vLLM rank {rank} weight shard: {name}")
        digest.update(name.encode())
        digest.update(actual.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
