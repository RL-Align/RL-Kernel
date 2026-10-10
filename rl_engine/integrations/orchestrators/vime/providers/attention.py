# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Runtime strict-attention provider for Vime's Megatron backend.

The adapter intentionally accepts and returns structural objects: RL-Kernel
never imports Vime.  Vime remains responsible for materializing post-RoPE Q/K/V
for its locally owned rows; this provider owns only the attention core
arithmetic and the ``(out, lse)`` export.

Two boundaries are deliberate and are enforced rather than documented:

* Every launch carries exactly one logical batch row and one KV group (that KV
  head plus the Q heads that attend to it).  The AITER/CK reduction order
  depends on the shape of the launch, so both batching and TP head-sharding
  would otherwise change the bits: measured on MI300X, raw AITER differs by up
  to ``1.5625e-02`` between a batch and its rows submitted singly, and by up to
  ``7.8125e-03`` between TP degrees.  Pinning the launch shape makes the result
  of a row/group independent of how many rows or heads its caller happened to
  hold, which is what lets training and rollout compare bitwise across
  different batch sizes and TP degrees.  It costs roughly 3x forward time.
* CP merges through the transport, never here.  The strict ROCm core owns
  single-rank attention arithmetic only.  At ``CP > 1`` this provider hands the
  schedule to :class:`StrictRocmAttentionRuntime`, whose RCCL AG/RS transport
  combines the cross-rank ``(out, lse)`` in a fixed balanced rank tree, so no
  second merge order is ever defined here.  The layout must be ``allgather``;
  ``zigzag`` fails closed because the strict CP plan describes one contiguous
  block per rank.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.backends.rocm.attention.flash_attn import BACKEND_ID
from rl_engine.backends.rocm.attention.strict_runtime import StrictRocmAttentionRuntime
from rl_engine.contracts.operators.attention import (
    CROSS_CONFIG_BOUND_FIELDS,
    AttentionContract,
    AttentionDType,
    AttentionMode,
    AttentionRole,
    ReductionSpec,
    ShardingSpec,
    SplitKVSpec,
)
from rl_engine.runtime.registry import kernel_registry


class AttentionProviderUnavailable(RuntimeError):
    """Request Vime's native attention fallback in ``auto`` mode.

    Vime recognizes the marker instead of importing this class, which keeps the
    dependency direction from Vime to RL-Kernel at runtime only.
    """

    attention_provider_unavailable = True


@dataclass(frozen=True)
class AttentionProviderResult:
    """Structural result understood by the Vime attention boundary."""

    out: torch.Tensor
    lse: torch.Tensor
    backend_id: str
    contract_id: str
    provenance: Mapping[str, Any]


_DTYPE_TO_CONTRACT = {
    torch.bfloat16: AttentionDType.BF16,
    torch.float16: AttentionDType.FP16,
}

# Decode is deliberately absent: it requires KV-cache identity metadata that
# this core does not materialize.  A decode request fails closed here rather
# than being silently served by the dense prefill core over a cache the
# provider never validated.
_MODE_BY_NAME = {
    "prefill": AttentionMode.PREFILL,
    "chunked_prefill": AttentionMode.CHUNKED_PREFILL,
}

_ROLE_BY_NAME = {
    "train": AttentionRole.TRAIN,
    "infer": AttentionRole.INFER,
}


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AttentionProviderUnavailable(f"{name} must be a positive integer; got {value!r}")
    return value


def _metadata(request: Any) -> Mapping[str, Any]:
    value = getattr(request, "metadata", None)
    if not isinstance(value, Mapping):
        raise AttentionProviderUnavailable("request.metadata must provide attention metadata")
    return value


def _request_tensor(request: Any, name: str) -> torch.Tensor:
    value = getattr(request, name, None)
    if not isinstance(value, torch.Tensor):
        raise AttentionProviderUnavailable(f"request.{name} must be a torch.Tensor")
    return value


def _tp_coordinates(tp_group: Any) -> tuple[int, int]:
    if tp_group is not None and hasattr(tp_group, "rank") and hasattr(tp_group, "size"):
        return int(tp_group.rank()), int(tp_group.size())

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=tp_group), dist.get_world_size(group=tp_group)
    return 0, 1


def _reject_unsupported_materializations(request: Any, metadata: Mapping[str, Any]) -> None:
    """Fail closed on every knob that would change the numerical definition."""

    if getattr(request, "key_padding_mask", None) is not None:
        raise AttentionProviderUnavailable(
            "strict ROCm attention materializes each unpadded logical row; "
            "pass unpadded per-row Q/K/V instead of a key padding mask"
        )
    dropout_p = metadata.get("dropout_p", 0.0)
    if dropout_p:
        raise AttentionProviderUnavailable(
            f"strict attention requires dropout_p=0.0; got {dropout_p!r}"
        )
    for unsupported in (
        "alibi_slopes",
        "attention_bias",
        "logit_soft_cap",
        "sliding_window",
        "sink_tokens",
    ):
        if metadata.get(unsupported) is not None:
            raise AttentionProviderUnavailable(
                f"strict ROCm attention does not materialize {unsupported}"
            )
    window = metadata.get("window_size")
    if window is not None and tuple(window) != (-1, -1):
        raise AttentionProviderUnavailable(
            f"strict ROCm attention requires a full causal/full window; got {window!r}"
        )


def _contract_for_request(
    request: Any,
) -> tuple[AttentionContract, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate the request and derive the explicit WS2 attention contract."""

    metadata = _metadata(request)
    _reject_unsupported_materializations(request, metadata)

    query = _request_tensor(request, "query")
    key = _request_tensor(request, "key")
    value = _request_tensor(request, "value")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise AttentionProviderUnavailable("query/key/value must be 4-D [B, H, S, D] tensors")
    if key.shape != value.shape:
        raise AttentionProviderUnavailable("key and value must share one shape")
    if query.shape[0] != key.shape[0]:
        raise AttentionProviderUnavailable("query and key must share the logical batch size")
    if query.shape[-1] != key.shape[-1]:
        raise AttentionProviderUnavailable("query and key must share head_dim")
    if query.dtype not in _DTYPE_TO_CONTRACT:
        raise AttentionProviderUnavailable(
            f"strict ROCm attention supports BF16/FP16 only; got {query.dtype}"
        )
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise AttentionProviderUnavailable("query/key/value must share one dtype")
    if not (query.device == key.device == value.device):
        raise AttentionProviderUnavailable("query/key/value must share one device")

    batch_size, local_q_heads, query_len, head_dim = query.shape
    local_kv_heads = key.shape[1]
    kv_len = key.shape[2]
    if local_q_heads % local_kv_heads:
        raise AttentionProviderUnavailable(
            f"local Q heads={local_q_heads} must be divisible by local KV heads={local_kv_heads}"
        )

    cp = getattr(request, "context_parallel", None)
    cp_world_size = _as_positive_int(getattr(cp, "world_size", None), "context_parallel.world_size")
    cp_rank = getattr(cp, "rank", None)
    if (
        isinstance(cp_rank, bool)
        or not isinstance(cp_rank, int)
        or not 0 <= cp_rank < cp_world_size
    ):
        raise AttentionProviderUnavailable(
            f"context_parallel.rank={cp_rank!r} is invalid for CP={cp_world_size}"
        )
    cp_layout = getattr(cp, "layout", None)
    if cp_layout not in ({"single"} if cp_world_size == 1 else {"zigzag", "allgather"}):
        raise AttentionProviderUnavailable(
            "context_parallel layout does not describe local CP token ownership"
        )
    if cp_world_size > 1 and cp_layout != "allgather":
        # The strict CP plan describes one contiguous block per rank. A zigzag
        # rank owns two discontiguous token runs, so accepting it here would
        # silently disagree with the block manifest the transport validates.
        raise AttentionProviderUnavailable(
            f"CP={cp_world_size} requires the 'allgather' layout; got {cp_layout!r}, "
            "whose block ownership the strict CP plan does not describe"
        )

    tp_rank, tp_world_size = _tp_coordinates(getattr(request, "tensor_parallel_group", None))
    declared_tp_rank = metadata.get("tp_rank")
    declared_tp_world_size = metadata.get("tp_world_size")
    if declared_tp_rank is not None and declared_tp_rank != tp_rank:
        raise AttentionProviderUnavailable(
            f"metadata tp_rank={declared_tp_rank} disagrees with TP group rank={tp_rank}"
        )
    if declared_tp_world_size is not None and declared_tp_world_size != tp_world_size:
        raise AttentionProviderUnavailable(
            f"metadata tp_world_size={declared_tp_world_size} disagrees with "
            f"TP group size={tp_world_size}"
        )

    global_q_heads = _as_positive_int(metadata.get("global_q_heads"), "global_q_heads")
    global_kv_heads = _as_positive_int(metadata.get("global_kv_heads"), "global_kv_heads")
    if local_q_heads * tp_world_size != global_q_heads:
        raise AttentionProviderUnavailable(
            "local Q heads and TP group do not cover global_q_heads exactly: "
            f"{local_q_heads} * {tp_world_size} != {global_q_heads}"
        )
    if local_kv_heads * tp_world_size != global_kv_heads:
        raise AttentionProviderUnavailable(
            "local KV heads and TP group do not cover global_kv_heads exactly: "
            f"{local_kv_heads} * {tp_world_size} != {global_kv_heads}"
        )

    mode_name = str(metadata.get("attention_mode", "prefill"))
    if mode_name == "decode":
        raise AttentionProviderUnavailable(
            "decode requires KV-cache identity metadata (cache_position, block table, "
            "prefix-cache key) that the strict dense core does not materialize; use the "
            "paged decode path"
        )
    if mode_name not in _MODE_BY_NAME:
        raise AttentionProviderUnavailable(f"unsupported attention_mode={mode_name!r}")
    mode = _MODE_BY_NAME[mode_name]
    role_name = str(metadata.get("role", "train"))
    if role_name not in _ROLE_BY_NAME:
        raise AttentionProviderUnavailable(f"unsupported role={role_name!r}")
    role = _ROLE_BY_NAME[role_name]

    causal = metadata.get("causal", True)
    if not isinstance(causal, bool):
        raise AttentionProviderUnavailable(f"causal must be a bool; got {causal!r}")
    if mode is AttentionMode.PREFILL and query_len != kv_len:
        raise AttentionProviderUnavailable(
            "prefill requires the query and KV lengths to describe one logical sequence; "
            f"got Sq={query_len} and Skv={kv_len}"
        )

    if query_len > kv_len:
        raise AttentionProviderUnavailable(
            f"causal attention requires Sq <= Skv; got Sq={query_len} and Skv={kv_len}"
        )

    scale = metadata.get("softmax_scale")
    resolved_scale = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)

    # Each CP rank owns one contiguous block of the logical sequence; at CP=1
    # that block is the whole sequence. The allgather layout checked above is
    # what makes the block contiguous.
    sharding = ShardingSpec(
        tp_rank=tp_rank,
        tp_world_size=tp_world_size,
        cp_rank=cp_rank,
        cp_world_size=cp_world_size,
        global_q_heads=global_q_heads,
        global_kv_heads=global_kv_heads,
        local_q_head_start=tp_rank * local_q_heads,
        local_q_heads=local_q_heads,
        local_kv_head_start=tp_rank * local_kv_heads,
        local_kv_heads=local_kv_heads,
        global_sequence_length=kv_len * cp_world_size,
        local_sequence_length=kv_len,
        global_block_indices=(cp_rank,),
        global_block_token_starts=(cp_rank * kv_len,),
        local_block_offsets=(0, kv_len),
    )
    # Causal alignment: the query block is the tail of the logical sequence, so
    # every batch entry carries the same offset between Q row 0 and KV token 0.
    causal_offsets = (kv_len - query_len,) * batch_size if causal else None

    contract = AttentionContract(
        role=role,
        mode=mode,
        dtype=_DTYPE_TO_CONTRACT[query.dtype],
        batch_size=batch_size,
        query_sequence_length=query_len if mode is not AttentionMode.PREFILL else kv_len,
        head_dim=head_dim,
        causal=causal,
        causal_offsets=causal_offsets,
        sharding=sharding,
        reduction=ReductionSpec(),
        split_kv=SplitKVSpec.disabled(),
        kv_cache=None,
        rope=None,
        export_lse=True,
    )
    key_position_ids = _key_position_ids(metadata, query.device, batch_size, kv_len)
    return contract, resolved_scale, query, key, value, key_position_ids


def _key_position_ids(
    metadata: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    kv_len: int,
) -> torch.Tensor:
    """Resolve the global KV token positions for this request.

    Position identity is part of the contract, not an implementation detail: it
    is what makes a training-side full-sequence call and a rollout-side chunk
    provably describe the same logical tokens.  Vime may declare it; otherwise
    the canonical contiguous ``[0, kv_len)`` block is used.
    """

    declared = metadata.get("key_position_ids")
    if declared is None:
        return (
            torch.arange(kv_len, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(batch_size, kv_len)
            .contiguous()
        )
    positions = declared if isinstance(declared, torch.Tensor) else torch.as_tensor(declared)
    positions = positions.to(device=device, dtype=torch.int64)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(batch_size, -1)
    if tuple(positions.shape) != (batch_size, kv_len):
        raise AttentionProviderUnavailable(
            f"key_position_ids must have shape {(batch_size, kv_len)}; "
            f"got {tuple(positions.shape)}"
        )
    if kv_len > 1 and bool((positions[:, 1:] - positions[:, :-1] != 1).any()):
        raise AttentionProviderUnavailable(
            "key_position_ids must describe one contiguous increasing token block"
        )
    return positions.contiguous()


def attention_provider(request: Any) -> AttentionProviderResult:
    """Compute Vime attention on the explicit WS2 strict ROCm contract.

    Materializes each logical batch row independently so batch composition
    cannot change the bits, then returns the stacked ``(out, lse)`` together
    with the dispatch provenance Vime records alongside the result.
    """

    contract, scale, query, key, value, key_positions = _contract_for_request(request)
    cp_world_size = contract.sharding.cp_world_size
    if cp_world_size > 1:
        cp_group = getattr(request, "context_parallel_group", None)
        if cp_group is None:
            raise AttentionProviderUnavailable(
                f"CP={cp_world_size} requires request.context_parallel_group so the strict "
                "RCCL AG/RS transport can be built; CP=1 does not need one"
            )
    else:
        cp_group = None

    dispatch = kernel_registry.get_attention_op(contract, requested_backend=BACKEND_ID)
    if dispatch.provenance["actual_backend"] != BACKEND_ID or dispatch.provenance["fallback"]:
        raise RuntimeError("explicit strict attention dispatch changed during materialization")

    query_len = query.shape[2]

    # One runtime owns the launch schedule for both CP degrees, so the
    # per-(batch row, KV group) launch loop that makes the result TP-degree
    # invariant cannot drift between the single-rank and CP paths.
    runtime = StrictRocmAttentionRuntime(process_group=cp_group, core=dispatch.op)
    runtime_result = runtime.forward_with_lse(
        query,
        key,
        value,
        contract=contract,
        causal=contract.causal,
        scale=scale,
        cp_world_size=cp_world_size,
        query_position_ids=key_positions[:, -query_len:],
        key_position_ids=key_positions,
        # At CP=1 this rank already holds the logical sequence in position
        # order, so the reorder the CP path needs would be a no-op copy.
        positions_are_sorted=cp_world_size == 1,
    )

    out = runtime_result.out
    lse = runtime_result.lse
    core_provenance = runtime_result.provenance["core"]
    launches = runtime_result.provenance["core_launch_count"]

    provenance = dict(dispatch.provenance)
    provenance["core"] = core_provenance
    provenance["request"] = {
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "dtype": str(query.dtype).replace("torch.", ""),
        "causal": contract.causal,
        "softmax_scale": scale,
        "tp_rank": contract.sharding.tp_rank,
        "tp_world_size": contract.sharding.tp_world_size,
        "cp_rank": contract.sharding.cp_rank,
        "cp_world_size": contract.sharding.cp_world_size,
    }
    provenance["execution"] = {
        "role": "vime_attention",
        "strict_backend": True,
        "launch_granularity": "one_batch_row_one_kv_group",
        "core_launches": launches,
        "batch_rows_materialized_independently": True,
        "kv_groups_materialized_independently": True,
        "attention_mode": contract.mode.value,
    }
    provenance["cp_row_ownership"] = {
        "cp_rank": contract.sharding.cp_rank,
        "cp_world_size": contract.sharding.cp_world_size,
        "layout": getattr(request.context_parallel, "layout"),
        "local_token_rows": contract.sharding.local_sequence_length,
        # The merge happens in the RCCL AG/RS transport's fixed rank tree, not
        # in this provider; the flag records that CP was an axis at all.
        "cp_is_merge_axis": cp_world_size > 1,
        "cp_merge_owner": (
            runtime_result.provenance["communication_backend"] if cp_world_size > 1 else "none"
        ),
    }
    provenance["lse_domain"] = "attention"
    # The qualified ROCm core's reduction order depends on the launch head
    # count, so raw AITER gives a head shard computed under TP=4 different bits
    # from the same shard under TP=8 at some shapes.  RL-Kernel removes the
    # dependence instead of binding the degree: every launch carries exactly one
    # KV group, which is bitwise TP-invariant at 12 of 12 measured points.
    # ``contract_id`` still encodes TP/CP so the preflight can compare the two
    # sides, but it is no longer what buys the invariance.
    provenance["cross_config_binding"] = {
        "bound_fields": list(CROSS_CONFIG_BOUND_FIELDS),
        "tp_world_size": contract.sharding.tp_world_size,
        "cp_world_size": contract.sharding.cp_world_size,
        "binding_token": "contract_id",
        "tp_degree_invariant": True,
        "invariance_mechanism": "one_kv_group_per_launch",
        "reason": (
            "AITER/CK dense MHA reduction order depends on the launch head count, so "
            "every launch is pinned to one KV group and its Q heads; the result of a "
            "head shard is then independent of the TP degree that produced it"
        ),
    }
    return AttentionProviderResult(
        out=out,
        lse=lse,
        backend_id=dispatch.capability.backend_id,
        contract_id=contract.cross_rank_fingerprint(),
        provenance=provenance,
    )


__all__ = [
    "AttentionProviderResult",
    "AttentionProviderUnavailable",
    "attention_provider",
]
