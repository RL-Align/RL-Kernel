# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Runtime ``linear_logp`` provider for Vime's Megatron backend.

The adapter accepts only structural objects. RL-Kernel never imports Vime;
Vime owns token-row construction and loss composition, while this provider
owns the numerical backend, TP reduction, and provenance.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.integrations.ablation import Implementation, operator_ablation_case
from rl_engine.kernels.logprob_contract import (
    LogprobContract,
    LogprobDType,
    LogprobRole,
    MaskSpec,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.kernels.ops.pytorch.loss.vocab_parallel_logp import (
    BACKEND_ID,
    DEFAULT_NUM_VOCAB_TILES,
)
from rl_engine.kernels.registry import kernel_registry


class LinearLogpProviderUnavailable(RuntimeError):
    """Request Vime's native provider fallback in ``auto`` mode."""

    linear_logp_provider_unavailable = True


@dataclass(frozen=True)
class LinearLogpResult:
    """Structural result understood by Vime's provider boundary."""

    logp: torch.Tensor
    entropy: torch.Tensor | None
    backend_id: str
    contract_id: str
    provenance: Mapping[str, Any]


_DEFAULT_STRICT_LINEAR_LOGP: Any | None = None


def _default_strict_linear_logp() -> Any:
    global _DEFAULT_STRICT_LINEAR_LOGP
    if _DEFAULT_STRICT_LINEAR_LOGP is None:
        from rl_engine.integrations.linear_logp import LinearLogpWrapper

        _DEFAULT_STRICT_LINEAR_LOGP = LinearLogpWrapper()
    return _DEFAULT_STRICT_LINEAR_LOGP


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LinearLogpProviderUnavailable(f"{name} must be a positive integer; got {value!r}")
    return value


def _metadata(request: Any) -> Mapping[str, Any]:
    value = getattr(request, "metadata", None)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LinearLogpProviderUnavailable("request.metadata must be a mapping")
    return value


def _request_tensor(request: Any, name: str) -> torch.Tensor:
    value = getattr(request, name, None)
    if not isinstance(value, torch.Tensor):
        raise LinearLogpProviderUnavailable(f"request.{name} must be a torch.Tensor")
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
        raise LinearLogpProviderUnavailable(
            f"num_vocab_tiles must be an integer; got {configured!r}"
        ) from exc
    if tiles <= 0 or padded_vocab_size % tiles:
        raise LinearLogpProviderUnavailable(
            f"num_vocab_tiles={tiles} must divide padded_vocab_size={padded_vocab_size}"
        )
    return tiles


def _vocab_partition(
    request: Any, logits: torch.Tensor, tp_rank: int, tp_world_size: int
) -> tuple[int, int, int, int]:
    metadata = _metadata(request)
    context = getattr(request, "context", None)
    partition = getattr(context, "vocab_partition", None)
    if partition is not None:
        local_start = getattr(partition, "local_start", None)
        local_size = getattr(partition, "local_size", None)
        real_vocab_size = getattr(partition, "real_size", None)
        padded_vocab_size = getattr(partition, "padded_size", None)
    else:
        local_start = tp_rank * logits.shape[1]
        local_size = logits.shape[1]
        real_vocab_size = metadata.get("real_vocab_size")
        padded_vocab_size = metadata.get("padded_vocab_size")

    local_start = 0 if local_start is None else local_start
    local_size = logits.shape[1] if local_size is None else local_size
    real_vocab_size = _as_positive_int(real_vocab_size, "real_vocab_size")
    padded_vocab_size = _as_positive_int(padded_vocab_size, "padded_vocab_size")
    if local_start != tp_rank * logits.shape[1] or local_size != logits.shape[1]:
        raise LinearLogpProviderUnavailable(
            "linear_logp vocabulary partition must be rank-contiguous and match local logits: "
            f"start={local_start}, local={local_size}, rank={tp_rank}, width={logits.shape[1]}"
        )
    if logits.shape[1] * tp_world_size != padded_vocab_size:
        raise LinearLogpProviderUnavailable(
            "local vocab width and TP group do not cover padded_vocab_size exactly: "
            f"{logits.shape[1]} * {tp_world_size} != {padded_vocab_size}"
        )
    if real_vocab_size > padded_vocab_size:
        raise LinearLogpProviderUnavailable("real_vocab_size must not exceed padded_vocab_size")
    return local_start, logits.shape[1], real_vocab_size, padded_vocab_size


def _contract_for_request(request: Any) -> tuple[LogprobContract, int]:
    logits = _request_tensor(request, "logits")
    targets = _request_tensor(request, "target_ids")
    metadata = _metadata(request)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise LinearLogpProviderUnavailable(
            "request must contain local [T, V] logits and aligned [T] targets"
        )
    if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise LinearLogpProviderUnavailable(f"unsupported logit dtype {logits.dtype}")
    if targets.device != logits.device:
        raise LinearLogpProviderUnavailable("target_ids must share the local logits device")

    layout = getattr(request, "token_layout", None)
    cp_world_size = _as_positive_int(getattr(layout, "world_size", None), "token_layout.world_size")
    cp_rank = getattr(layout, "rank", None)
    if (
        isinstance(cp_rank, bool)
        or not isinstance(cp_rank, int)
        or not 0 <= cp_rank < cp_world_size
    ):
        raise LinearLogpProviderUnavailable(
            f"token_layout.rank={cp_rank!r} is invalid for CP={cp_world_size}"
        )
    if getattr(layout, "layout", None) not in (
        {"single"} if cp_world_size == 1 else {"zigzag", "allgather"}
    ):
        raise LinearLogpProviderUnavailable("token_layout does not describe local token ownership")

    tp_rank, tp_world_size = _tp_coordinates(getattr(request, "tensor_parallel_group", None))
    declared_tp_rank = metadata.get("tp_rank")
    declared_tp_world_size = metadata.get("tp_world_size")
    if declared_tp_rank is not None and declared_tp_rank != tp_rank:
        raise LinearLogpProviderUnavailable(
            f"metadata tp_rank={declared_tp_rank} disagrees with TP group rank={tp_rank}"
        )
    if declared_tp_world_size is not None and declared_tp_world_size != tp_world_size:
        raise LinearLogpProviderUnavailable(
            "metadata tp_world_size="
            f"{declared_tp_world_size} disagrees with TP group size={tp_world_size}"
        )

    local_start, local_size, real_vocab_size, padded_vocab_size = _vocab_partition(
        request, logits, tp_rank, tp_world_size
    )
    bounds = tuple((rank * local_size, (rank + 1) * local_size) for rank in range(tp_world_size))
    contract = LogprobContract(
        role=LogprobRole.TRAIN,
        dtype={
            torch.bfloat16: LogprobDType.BF16,
            torch.float16: LogprobDType.FP16,
            torch.float32: LogprobDType.FP32,
        }[logits.dtype],
        mask=MaskSpec(num_tokens=logits.shape[0], active_mask=(True,) * logits.shape[0]),
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


def _is_identity_temperature(value: Any) -> bool:
    return value is None or (not isinstance(value, torch.Tensor) and float(value) == 1.0)


@torch.no_grad()
def _metric_entropy_from_strict_lse(
    local_logits: torch.Tensor,
    lse: torch.Tensor,
    *,
    vocab_start_index: int,
    real_vocab_size: int,
    tp_group: Any,
) -> torch.Tensor:
    """Compute metric-only entropy without re-running selected-logp/LSE."""

    local_real = max(
        0,
        min(local_logits.size(1), int(real_vocab_size) - int(vocab_start_index)),
    )
    centered = local_logits[:, :local_real].float()
    centered.sub_(lse.reshape(-1, 1).float())
    probabilities = centered.exp()
    centered.neg_()
    local_entropy = torch.einsum("ij,ij->i", probabilities, centered).contiguous()
    _rank, world = _tp_coordinates(tp_group)
    if world == 1:
        return local_entropy
    import torch.distributed as dist

    gathered = [torch.empty_like(local_entropy) for _ in range(world)]
    dist.all_gather(gathered, local_entropy, group=tp_group)
    entropy = gathered[0].clone()
    for partial in gathered[1:]:
        entropy.add_(partial)
    return entropy


def _provider_impl(request: Any, *, linear_logp: Any = None) -> LinearLogpResult:
    """Compute Vime log-probabilities on the explicit TP/CP contract."""

    strict = os.getenv("VIME_RL_KERNEL_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    context = getattr(request, "context", None)
    hidden = getattr(context, "hidden", None)
    projection = getattr(context, "projection", None)
    partition = getattr(context, "vocab_partition", None)
    if strict and not isinstance(hidden, torch.Tensor):
        raise RuntimeError("strict Vime linear_logp request is missing structural context")
    if strict and getattr(request, "log_prob_keep_mask", None) is not None:
        raise RuntimeError("strict Vime linear_logp does not support top-p replay in this contract")
    if linear_logp is None and strict and isinstance(hidden, torch.Tensor):
        linear_logp = _default_strict_linear_logp()
    if linear_logp is not None and isinstance(hidden, torch.Tensor):
        if projection is None or not isinstance(getattr(projection, "weight", None), torch.Tensor):
            raise RuntimeError("linear_logp context must expose projection.weight")
        if partition is None:
            raise RuntimeError("linear_logp context must expose vocab_partition")
        reuse_local_logits = bool(getattr(context, "reuse_local_logits", False))
        # Megatron's strict output-layer patch materializes the same local
        # padded-vocabulary logits that vLLM serves.  The Vime context type in
        # this revision predates the explicit ``local_logits`` field, so use
        # the request tensor when it is the matching ROCm TP shard.  This
        # avoids a second GEMM and keeps the backward graph attached to the
        # output projection.
        request_logits = getattr(request, "logits", None)
        materialized_local_logits = (
            torch.version.hip is not None
            and isinstance(request_logits, torch.Tensor)
            and request_logits.ndim == 2
            and request_logits.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and request_logits.shape
            == (
                hidden.size(0),
                projection.weight.size(0),
            )
        )
        if materialized_local_logits:
            reuse_local_logits = True
        with_entropy = bool(getattr(request, "with_entropy", False))
        with_entropy_grad = bool(getattr(request, "with_entropy_grad", False))
        fast_metric_entropy = (
            reuse_local_logits
            and with_entropy
            and not with_entropy_grad
            and _is_identity_temperature(getattr(request, "temperature", None))
        )
        strict_lse = None
        if reuse_local_logits:
            local_logits = getattr(context, "local_logits", None)
            if materialized_local_logits:
                local_logits = request_logits
            if not isinstance(local_logits, torch.Tensor):
                raise RuntimeError("strict reusable LM-head context is missing local logits")
            from_local_logits = getattr(linear_logp, "from_local_logits", None)
            if not callable(from_local_logits):
                raise RuntimeError(
                    "strict reusable LM-head logits require a from_local_logits provider"
                )
            scored = from_local_logits(
                local_logits,
                request.target_ids,
                tp_group=getattr(request, "tensor_parallel_group", None),
                vocab_start_index=int(partition.local_start),
                global_vocab_size=int(partition.padded_size),
                real_vocab_size=int(partition.real_size),
                target="training",
                temperature=getattr(request, "temperature", None),
                return_lse=fast_metric_entropy,
                diagnostics_hidden=hidden,
                diagnostics_lm_head_weight=projection.weight,
            )
            if fast_metric_entropy:
                logp, strict_lse = scored
            else:
                logp = scored
        else:
            logp = linear_logp(
                hidden,
                projection.weight,
                request.target_ids,
                getattr(projection, "bias", None),
                tp_group=getattr(request, "tensor_parallel_group", None),
                vocab_start_index=int(partition.local_start),
                global_vocab_size=int(partition.padded_size),
                real_vocab_size=int(partition.real_size),
                temperature=getattr(request, "temperature", None),
            )
        entropy = None
        entropy_provenance: dict[str, Any] = {}
        if fast_metric_entropy:
            if strict_lse is None:
                raise RuntimeError("metric-only entropy requires the strict local-logits LSE")
            entropy = _metric_entropy_from_strict_lse(
                local_logits,
                strict_lse,
                vocab_start_index=int(partition.local_start),
                real_vocab_size=int(partition.real_size),
                tp_group=getattr(request, "tensor_parallel_group", None),
            )
            entropy_provenance = {
                "backend_id": "rlkernel.strict-lse-metric-entropy.v1",
                "logits_materialized": True,
                "strict_lse_reused": True,
                "with_entropy_grad": False,
            }
        elif with_entropy:
            entropy_contract, entropy_tiles = _contract_for_request(request)
            entropy_dispatch = kernel_registry.get_logprob_op(
                entropy_contract, requested_backend=BACKEND_ID
            )
            if (
                entropy_dispatch.provenance["actual_backend"] != BACKEND_ID
                or entropy_dispatch.provenance["fallback"]
            ):
                raise RuntimeError(
                    "explicit WS2 entropy dispatch changed during strict linear_logp execution"
                )
            _, _, entropy = entropy_dispatch.op.apply_with_entropy(
                request.logits,
                request.target_ids,
                contract=entropy_contract,
                tp_group=getattr(request, "tensor_parallel_group", None),
                num_vocab_tiles=entropy_tiles,
                with_entropy_grad=with_entropy_grad,
            )
            entropy_provenance = {
                "backend_id": entropy_dispatch.capability.backend_id,
                "contract_id": entropy_contract.cross_rank_fingerprint(),
                "num_vocab_tiles": entropy_tiles,
                "logits_materialized": True,
            }
        provenance = dict(getattr(linear_logp, "provenance", {}))
        provenance["execution"] = {
            "role": "vime_training_linear_logp",
            "strict_backend": True,
            "top_p_replay": False,
            "cp_is_merge_axis": False,
            "logits_materialized": bool(
                reuse_local_logits or getattr(request, "with_entropy", False)
            ),
            "lm_head_result_reused": reuse_local_logits,
            "entropy": entropy_provenance,
        }
        provenance["request"] = {
            "hidden_shape": list(hidden.shape),
            "hidden_dtype": str(hidden.dtype).replace("torch.", ""),
            "target_shape": list(request.target_ids.shape),
            "tp_world_size": int(getattr(request, "metadata", {}).get("tp_world_size", 1)),
            "tp_rank": int(getattr(request, "metadata", {}).get("tp_rank", 0)),
            "cp_world_size": int(getattr(request, "token_layout").world_size),
            "cp_rank": int(getattr(request, "token_layout").rank),
        }
        backend_id = str(provenance.get("actual_backend", linear_logp.backend_id))
        contract_id = (
            "linear_logp:"
            f"tp={getattr(request, 'metadata', {}).get('tp_world_size', 1)}:"
            f"cp={getattr(request, 'token_layout').world_size}:"
            f"vocab={partition.padded_size}"
        )
        return LinearLogpResult(
            logp=logp.reshape(-1, 1),
            entropy=entropy,
            backend_id=backend_id,
            contract_id=contract_id,
            provenance=provenance,
        )

    if getattr(request, "log_prob_keep_mask", None) is not None:
        raise LinearLogpProviderUnavailable(
            "RL-Kernel logp does not yet materialize Vime top-p replay masks"
        )

    contract, num_vocab_tiles = _contract_for_request(request)
    dispatch = kernel_registry.get_logprob_op(contract, requested_backend=BACKEND_ID)
    if dispatch.provenance["actual_backend"] != BACKEND_ID or dispatch.provenance["fallback"]:
        raise RuntimeError("explicit WS2 backend dispatch changed during materialization")
    if getattr(request, "with_entropy", False):
        logp, _lse, entropy = dispatch.op.apply_with_entropy(
            request.logits,
            request.target_ids,
            contract=contract,
            tp_group=getattr(request, "tensor_parallel_group", None),
            num_vocab_tiles=num_vocab_tiles,
            with_entropy_grad=bool(getattr(request, "with_entropy_grad", False)),
        )
    else:
        logp, _lse = dispatch.op(
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
        "role": "vime_training_linear_logp",
        "strict_backend": True,
        "top_p_replay": False,
    }
    provenance["token_row_ownership"] = {
        "rank": contract.sharding.cp_rank,
        "world_size": contract.sharding.cp_world_size,
        "layout": request.token_layout.layout,
        "local_token_rows": int(request.logits.shape[0]),
        "is_merge_axis": False,
    }
    provenance["num_vocab_tiles"] = num_vocab_tiles
    return LinearLogpResult(
        logp=logp.unsqueeze(-1),
        entropy=entropy,
        backend_id=dispatch.capability.backend_id,
        contract_id=contract.cross_rank_fingerprint(),
        provenance=provenance,
    )


_provider_impl.backend_id = BACKEND_ID  # type: ignore[attr-defined]


def provider(request: Any) -> LinearLogpResult:
    """Route Vime training logp through the active Megatron integration."""

    case = operator_ablation_case("logp", os.getenv("RL_KERNEL_LOGP_CASE", "P/P"))
    if case.training is Implementation.PRODUCTION:
        raise RuntimeError(
            "the RL-Kernel linear_logp provider must not be configured for a "
            "production Megatron logp route; omit --linear-logp-provider so "
            "Vime executes calculate_log_probs_and_entropy directly"
        )

    from rl_engine.integrations.state import get_active_integration

    integration = get_active_integration("megatron")
    if integration is None:
        return _provider_impl(request)

    def native_unavailable(_request: Any) -> LinearLogpResult:
        raise RuntimeError(
            "the structural provider was invoked for a production Megatron logp route"
        )

    return integration.execute("logp", native_unavailable, request)


__all__ = ["LinearLogpResult", "LinearLogpProviderUnavailable", "provider"]
