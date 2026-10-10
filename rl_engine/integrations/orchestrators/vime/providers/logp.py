# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Runtime selected-logprob provider for Vime's Megatron backend.

The adapter intentionally accepts and returns structural objects: RL-Kernel
never imports Vime.  Vime remains responsible for constructing locally owned
CP token rows and response masks; this provider owns only the TP-vocabulary
reduction.  CP rank and layout are recorded and validated as row ownership
metadata, never passed to the numerical merge.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.contracts.operators.logprob import (
    LogprobContract,
    LogprobDType,
    LogprobRole,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.reference.logprob.vocab_parallel_logp import BACKEND_ID, DEFAULT_NUM_VOCAB_TILES
from rl_engine.runtime.registry import kernel_registry


class SelectedLogprobProviderUnavailable(RuntimeError):
    """Request Vime's native provider fallback in ``auto`` mode.

    Vime recognizes the marker instead of importing this class, which keeps
    the dependency direction from Vime to RL-Kernel at runtime only.
    """

    selected_logprob_provider_unavailable = True


@dataclass(frozen=True)
class ProviderResult:
    """Structural result understood by the Vime provider boundary."""

    selected_logprobs: torch.Tensor
    entropy: torch.Tensor | None
    backend_id: str
    contract_id: str
    provenance: Mapping[str, Any]


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SelectedLogprobProviderUnavailable(
            f"{name} must be a positive integer; got {value!r}"
        )
    return value


def _metadata(request: Any) -> Mapping[str, Any]:
    value = getattr(request, "metadata", None)
    if not isinstance(value, Mapping):
        raise SelectedLogprobProviderUnavailable(
            "request.metadata must provide vocab-parallel metadata"
        )
    return value


def _request_tensor(request: Any, name: str) -> torch.Tensor:
    value = getattr(request, name, None)
    if not isinstance(value, torch.Tensor):
        raise SelectedLogprobProviderUnavailable(f"request.{name} must be a torch.Tensor")
    return value


def _tp_coordinates(tp_group: Any) -> tuple[int, int]:
    if tp_group is not None and hasattr(tp_group, "rank") and hasattr(tp_group, "size"):
        return int(tp_group.rank()), int(tp_group.size())

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=tp_group), dist.get_world_size(group=tp_group)
    return 0, 1


def _tile_count(metadata: Mapping[str, Any], padded_vocab_size: int) -> int:
    configured = metadata.get("num_vocab_tiles", os.getenv("RL_KERNEL_LOGPROB_NUM_VOCAB_TILES"))
    if configured is None or configured == "":
        configured = DEFAULT_NUM_VOCAB_TILES
    try:
        tiles = int(configured)
    except (TypeError, ValueError) as exc:
        raise SelectedLogprobProviderUnavailable(
            f"num_vocab_tiles must be an integer; got {configured!r}"
        ) from exc
    if tiles <= 0 or padded_vocab_size % tiles:
        raise SelectedLogprobProviderUnavailable(
            f"num_vocab_tiles={tiles} must divide padded_vocab_size={padded_vocab_size}"
        )
    return tiles


def _contract_for_request(request: Any) -> tuple[LogprobContract, int]:
    logits = _request_tensor(request, "logits")
    targets = _request_tensor(request, "target_ids")
    metadata = _metadata(request)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise SelectedLogprobProviderUnavailable(
            "request must contain local [T, V] logits and aligned [T] targets"
        )
    if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise SelectedLogprobProviderUnavailable(f"unsupported logit dtype {logits.dtype}")
    if targets.device != logits.device:
        raise SelectedLogprobProviderUnavailable("target_ids must share the local logits device")

    cp = getattr(request, "context_parallel", None)
    cp_world_size = _as_positive_int(getattr(cp, "world_size", None), "context_parallel.world_size")
    cp_rank = getattr(cp, "rank", None)
    if (
        isinstance(cp_rank, bool)
        or not isinstance(cp_rank, int)
        or not 0 <= cp_rank < cp_world_size
    ):
        raise SelectedLogprobProviderUnavailable(
            f"context_parallel.rank={cp_rank!r} is invalid for CP={cp_world_size}"
        )
    if getattr(cp, "layout", None) not in (
        {"single"} if cp_world_size == 1 else {"zigzag", "allgather"}
    ):
        raise SelectedLogprobProviderUnavailable(
            "context_parallel layout does not describe local CP token ownership"
        )

    tp_rank, tp_world_size = _tp_coordinates(getattr(request, "tensor_parallel_group", None))
    declared_tp_rank = metadata.get("tp_rank")
    declared_tp_world_size = metadata.get("tp_world_size")
    if declared_tp_rank is not None and declared_tp_rank != tp_rank:
        raise SelectedLogprobProviderUnavailable(
            f"metadata tp_rank={declared_tp_rank} disagrees with TP group rank={tp_rank}"
        )
    if declared_tp_world_size is not None and declared_tp_world_size != tp_world_size:
        raise SelectedLogprobProviderUnavailable(
            f"metadata tp_world_size={declared_tp_world_size} disagrees with "
            f"TP group size={tp_world_size}"
        )

    real_vocab_size = _as_positive_int(metadata.get("real_vocab_size"), "real_vocab_size")
    padded_vocab_size = _as_positive_int(metadata.get("padded_vocab_size"), "padded_vocab_size")
    if logits.shape[1] * tp_world_size != padded_vocab_size:
        raise SelectedLogprobProviderUnavailable(
            "local vocab width and TP group do not cover padded_vocab_size exactly: "
            f"{logits.shape[1]} * {tp_world_size} != "
            f"{padded_vocab_size}"
        )
    if real_vocab_size > padded_vocab_size:
        raise SelectedLogprobProviderUnavailable(
            "real_vocab_size must not exceed padded_vocab_size"
        )

    bounds = tuple(
        (rank * logits.shape[1], (rank + 1) * logits.shape[1]) for rank in range(tp_world_size)
    )
    active_mask = (True,) * logits.shape[0]
    contract = LogprobContract(
        role=LogprobRole.TRAIN,
        dtype={
            torch.bfloat16: LogprobDType.BF16,
            torch.float16: LogprobDType.FP16,
            torch.float32: LogprobDType.FP32,
        }[logits.dtype],
        mask=MaskSpec(num_tokens=logits.shape[0], active_mask=active_mask),
        sharding=ShardingSpec(
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            vocab_shard_bounds=bounds,
            real_vocab_size=real_vocab_size,
            padded_vocab_size=padded_vocab_size,
            cp_rank=cp_rank,
            cp_world_size=cp_world_size,
        ),
        reduction=ReductionSpec(),
    )
    return contract, _tile_count(metadata, padded_vocab_size)


def provider(request: Any) -> ProviderResult:
    """Compute Vime selected logprobs on the explicit WS2 TP/CP contract.

    Top-p replay is deliberately unavailable until it has a separately
    validated fixed-order mask contract.  In Vime ``auto`` mode this signals
    native execution; in ``strict`` mode it fails instead of changing sampled
    distribution semantics.
    """

    if getattr(request, "log_prob_keep_mask", None) is not None:
        raise SelectedLogprobProviderUnavailable(
            "RL-Kernel WS2 logprob does not yet materialize Vime top-p replay masks"
        )

    contract, num_vocab_tiles = _contract_for_request(request)
    dispatch = kernel_registry.get_logprob_op(contract, requested_backend=BACKEND_ID)
    if dispatch.provenance["actual_backend"] != BACKEND_ID or dispatch.provenance["fallback"]:
        raise RuntimeError("explicit WS2 backend dispatch changed during materialization")
    if getattr(request, "with_entropy", False):
        selected_logp, _lse, entropy = dispatch.op.apply_with_entropy(
            request.logits,
            request.target_ids,
            contract=contract,
            tp_group=getattr(request, "tensor_parallel_group", None),
            num_vocab_tiles=num_vocab_tiles,
            with_entropy_grad=bool(getattr(request, "with_entropy_grad", False)),
        )
    else:
        selected_logp, _lse = dispatch.op(
            request.logits,
            request.target_ids,
            contract=contract,
            tp_group=getattr(request, "tensor_parallel_group", None),
            num_vocab_tiles=num_vocab_tiles,
        )
        entropy = None
    provenance = dict(dispatch.provenance)
    provenance["request"] = {
        "logits_shape": list(request.logits.shape),
        "logits_dtype": str(request.logits.dtype).replace("torch.", ""),
        "target_shape": list(request.target_ids.shape),
        "target_dtype": str(request.target_ids.dtype).replace("torch.", ""),
        "real_vocab_size": contract.sharding.real_vocab_size,
        "padded_vocab_size": contract.sharding.padded_vocab_size,
        "tp_rank": contract.sharding.tp_rank,
        "tp_world_size": contract.sharding.tp_world_size,
        "cp_rank": contract.sharding.cp_rank,
        "cp_world_size": contract.sharding.cp_world_size,
    }
    provenance["execution"] = {
        "role": "vime_training_selected_logprob",
        "strict_backend": True,
        "top_p_replay": False,
    }
    provenance["cp_row_ownership"] = {
        "cp_rank": contract.sharding.cp_rank,
        "cp_world_size": contract.sharding.cp_world_size,
        "layout": getattr(request.context_parallel, "layout"),
        "local_token_rows": int(request.logits.shape[0]),
        "cp_is_merge_axis": False,
    }
    provenance["num_vocab_tiles"] = num_vocab_tiles
    return ProviderResult(
        selected_logprobs=selected_logp.unsqueeze(-1),
        entropy=entropy,
        backend_id=dispatch.capability.backend_id,
        contract_id=contract.cross_rank_fingerprint(),
        provenance=provenance,
    )


__all__ = ["ProviderResult", "SelectedLogprobProviderUnavailable", "provider"]
