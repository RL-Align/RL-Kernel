# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Exact per-rank Qwen3 weight-install verification for the PPO acceptance runner."""
from __future__ import annotations

import hashlib

import torch


def _source(weights, name):
    """Return checkpoint tensor ``name``, raising ``ValueError`` when it is missing."""
    try:
        return weights[name]
    except KeyError:
        raise ValueError(f"no checkpoint tensor maps to vLLM parameter {name}") from None


def expected_shard(name, weights, *, rank, tp, shape):
    """Return the checkpoint slice that TP rank ``rank`` should hold for vLLM parameter ``name``.

    Fused QKV and gate/up concatenate each source projection's row shard; the embedding and LM
    head are zero-padded to ``shape[0] * tp`` rows before row sharding; ``o_proj`` and
    ``down_proj`` are split by columns; any other parameter is returned whole. Raises
    ``ValueError`` for a missing source tensor or a vocabulary partition smaller than the
    checkpoint.
    """
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
    """Check that every TP rank acknowledged installing actor weights ``version``.

    Raises ``ValueError`` unless ``acks`` has exactly one entry per rank in ``range(tp)``, each
    reporting this ``version`` and ``tp`` and a positive ``verified_tensors`` count.
    """
    if len(acks) != tp or {a["rank"] for a in acks} != set(range(tp)):
        raise ValueError("missing or duplicate vLLM rank acknowledgment")
    if any(a["version"] != version or a["tp"] != tp or a["verified_tensors"] <= 0 for a in acks):
        raise ValueError("stale or incomplete vLLM weight acknowledgment")


def install_weights(worker, filename, version):
    """Install actor weights ``version`` from ``filename`` into a vLLM worker and verify them.

    ``version`` must be one past ``worker.ppo_actor_version`` (the first is 0) and equal the
    file's ``actor_version``; the loaded names must equal the model's parameter names and each
    parameter must match its expected shard exactly, else ``ValueError``. On success sets
    ``worker.ppo_actor_version = version`` and returns this rank's acknowledgment dict (rank,
    tp, version, verified_tensors, installed_sha256, peak_allocated_bytes).
    """
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
    """Compare every parameter exactly with its expected TP shard; return a SHA-256 hex digest.

    The digest covers each parameter name and its raw bytes in ``named_parameters`` order.
    Raises ``ValueError`` on the first dtype, shape or value mismatch, or when
    ``expected_shard`` cannot build a parameter's shard.
    """
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
