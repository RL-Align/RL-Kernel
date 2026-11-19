# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared training log-probability execution, independent of orchestrator requests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
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


class LinearLogpProviderUnavailable(RuntimeError):
    """Signal that the shared scorer cannot handle these inputs."""

    linear_logp_provider_unavailable = True


@dataclass(frozen=True)
class LinearLogpResult:
    """Selected-token log-probabilities, optional entropy, and execution provenance."""

    logp: torch.Tensor
    entropy: torch.Tensor | None
    backend_id: str
    contract_id: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class TrainingLogpInputs:
    """Local token rows and vocabulary shard supplied by a training adapter.

    Orchestrators own row selection and request conversion. ``temperature``
    applies to projection from hidden states; ``local_logits_temperature`` is
    the scaling still required by supplied logits (None if already scaled).
    CP describes row ownership and is never a log-probability reduction axis.
    """

    logits: torch.Tensor | None
    target_ids: torch.Tensor
    real_vocab_size: int
    padded_vocab_size: int
    cp_rank: int
    cp_world_size: int
    cp_layout: str
    tp_group: Any = None
    tp_rank: int | None = None
    tp_world_size: int | None = None
    vocab_start_index: int | None = None
    local_vocab_size: int | None = None
    num_vocab_tiles: int | str | None = DEFAULT_NUM_VOCAB_TILES
    hidden: torch.Tensor | None = None
    weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    local_logits: torch.Tensor | None = None
    reuse_local_logits: bool = False
    temperature: float | torch.Tensor | None = None
    local_logits_temperature: float | torch.Tensor | None = None
    keep_mask: torch.Tensor | None = None
    sparse_token_ids: torch.Tensor | None = None
    with_entropy: bool = False
    with_entropy_grad: bool = False
    execution_role: str = "training_linear_logp"


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LinearLogpProviderUnavailable(f"{name} must be a positive integer; got {value!r}")
    return value


def _tp_coordinates(tp_group: Any) -> tuple[int, int]:
    if tp_group is not None and hasattr(tp_group, "rank") and hasattr(tp_group, "size"):
        return int(tp_group.rank()), int(tp_group.size())

    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=tp_group), dist.get_world_size(group=tp_group)
    return 0, 1


def _tile_count(configured: Any, padded_vocab_size: int) -> int:
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
    inputs: TrainingLogpInputs, logits: torch.Tensor, tp_rank: int, tp_world_size: int
) -> tuple[int, int, int, int]:
    local_start = inputs.vocab_start_index
    if local_start is None:
        local_start = tp_rank * logits.shape[1]
    local_size = inputs.local_vocab_size
    real_vocab_size = inputs.real_vocab_size
    padded_vocab_size = inputs.padded_vocab_size

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


def _contract_for_inputs(inputs: TrainingLogpInputs) -> tuple[LogprobContract, int]:
    logits = inputs.logits
    targets = inputs.target_ids
    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise LinearLogpProviderUnavailable("logits and target_ids must be torch.Tensor values")
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise LinearLogpProviderUnavailable(
            "request must contain local [T, V] logits and aligned [T] targets"
        )
    if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise LinearLogpProviderUnavailable(f"unsupported logit dtype {logits.dtype}")
    if targets.device != logits.device:
        raise LinearLogpProviderUnavailable("target_ids must share the local logits device")

    cp_world_size = _as_positive_int(inputs.cp_world_size, "token_layout.world_size")
    cp_rank = inputs.cp_rank
    if (
        isinstance(cp_rank, bool)
        or not isinstance(cp_rank, int)
        or not 0 <= cp_rank < cp_world_size
    ):
        raise LinearLogpProviderUnavailable(
            f"token_layout.rank={cp_rank!r} is invalid for CP={cp_world_size}"
        )
    if inputs.cp_layout not in ({"single"} if cp_world_size == 1 else {"zigzag", "allgather"}):
        raise LinearLogpProviderUnavailable("token_layout does not describe local token ownership")

    tp_rank, tp_world_size = _tp_coordinates(inputs.tp_group)
    declared_tp_rank = inputs.tp_rank
    declared_tp_world_size = inputs.tp_world_size
    if declared_tp_rank is not None and declared_tp_rank != tp_rank:
        raise LinearLogpProviderUnavailable(
            f"metadata tp_rank={declared_tp_rank} disagrees with TP group rank={tp_rank}"
        )
    if declared_tp_world_size is not None and declared_tp_world_size != tp_world_size:
        raise LinearLogpProviderUnavailable(
            "metadata tp_world_size="
            f"{declared_tp_world_size} disagrees with TP group size={tp_world_size}"
        )

    _local_start, local_size, real_vocab_size, padded_vocab_size = _vocab_partition(
        inputs, logits, tp_rank, tp_world_size
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
    return contract, _tile_count(inputs.num_vocab_tiles, padded_vocab_size)


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
    centered = local_logits[:, :local_real].float() - lse.reshape(-1, 1).float()
    probabilities = centered.exp()
    centered.neg_()
    local_entropy = torch.einsum("ij,ij->i", probabilities, centered).contiguous()
    if tp_group is None:
        return local_entropy
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


def compute_training_logp(
    inputs: TrainingLogpInputs, *, linear_logp: Any = None
) -> LinearLogpResult:
    """Score local training rows without importing an orchestrator or engine."""

    hidden = inputs.hidden
    if linear_logp is not None and isinstance(hidden, torch.Tensor):
        if not isinstance(inputs.weight, torch.Tensor):
            raise RuntimeError("linear_logp inputs must expose projection weight")
        if inputs.vocab_start_index is None:
            raise RuntimeError("linear_logp inputs must expose vocabulary shard start")
        reuse_local_logits = inputs.reuse_local_logits
        keep_mask = inputs.keep_mask
        sparse_nucleus_ids = inputs.sparse_token_ids
        sparse_metric_entropy = None
        if keep_mask is not None and not reuse_local_logits:
            raise RuntimeError("strict top-p replay requires reusable materialized local logits")
        with_entropy = inputs.with_entropy
        with_entropy_grad = inputs.with_entropy_grad
        local_logits_temperature = inputs.local_logits_temperature
        fast_metric_entropy = (
            reuse_local_logits
            and with_entropy
            and not with_entropy_grad
            and _is_identity_temperature(local_logits_temperature)
            and keep_mask is None
            and sparse_nucleus_ids is None
        )
        strict_lse = None
        if reuse_local_logits:
            local_logits = inputs.local_logits
            if not isinstance(local_logits, torch.Tensor):
                raise RuntimeError("strict reusable LM-head context is missing local logits")
            if keep_mask is not None:
                if keep_mask.shape != local_logits.shape:
                    raise RuntimeError("strict top-p replay mask must match local logits")
                local_logits = local_logits.masked_fill(~keep_mask, float("-inf"))
            from_local_logits = getattr(linear_logp, "from_local_logits", None)
            if not callable(from_local_logits):
                raise RuntimeError(
                    "strict reusable LM-head logits require a from_local_logits provider"
                )
            if sparse_nucleus_ids is not None:
                scored = linear_logp.from_local_logits_sparse_nucleus(
                    local_logits,
                    inputs.target_ids,
                    sparse_nucleus_ids,
                    tp_group=inputs.tp_group,
                    vocab_start_index=int(inputs.vocab_start_index),
                    global_vocab_size=int(inputs.padded_vocab_size),
                    real_vocab_size=int(inputs.real_vocab_size),
                    target="training",
                    temperature=local_logits_temperature,
                    return_entropy=with_entropy and not with_entropy_grad,
                )
                if with_entropy and not with_entropy_grad:
                    scored, sparse_metric_entropy = scored
            else:
                scored = from_local_logits(
                    local_logits,
                    inputs.target_ids,
                    tp_group=inputs.tp_group,
                    vocab_start_index=int(inputs.vocab_start_index),
                    global_vocab_size=int(inputs.padded_vocab_size),
                    real_vocab_size=int(inputs.real_vocab_size),
                    target="training",
                    temperature=local_logits_temperature,
                    return_lse=fast_metric_entropy,
                    diagnostics_hidden=hidden,
                    diagnostics_lm_head_weight=inputs.weight,
                )
            if fast_metric_entropy:
                logp, strict_lse = scored
            else:
                logp = scored
        else:
            logp = linear_logp(
                hidden,
                inputs.weight,
                inputs.target_ids,
                inputs.bias,
                tp_group=inputs.tp_group,
                vocab_start_index=int(inputs.vocab_start_index),
                global_vocab_size=int(inputs.padded_vocab_size),
                real_vocab_size=int(inputs.real_vocab_size),
                temperature=inputs.temperature,
            )
        entropy = None
        entropy_provenance: dict[str, Any] = {}
        if fast_metric_entropy:
            if strict_lse is None or not isinstance(local_logits, torch.Tensor):
                raise RuntimeError("metric-only entropy requires the strict local-logits LSE")
            entropy = _metric_entropy_from_strict_lse(
                local_logits,
                strict_lse,
                vocab_start_index=int(inputs.vocab_start_index),
                real_vocab_size=int(inputs.real_vocab_size),
                tp_group=inputs.tp_group,
            )
            entropy_provenance = {
                "backend_id": "rlkernel.strict-lse-metric-entropy.v1",
                "logits_materialized": True,
                "strict_lse_reused": True,
                "with_entropy_grad": False,
            }
        elif sparse_metric_entropy is not None:
            entropy = sparse_metric_entropy
            entropy_provenance = {
                "backend_id": "rlkernel.sparse-nucleus-metric-entropy.v1",
                "sparse_nucleus_reused": True,
                "with_entropy_grad": False,
            }
        elif with_entropy:
            if not isinstance(inputs.logits, torch.Tensor):
                raise LinearLogpProviderUnavailable("entropy requires materialized logits")
            entropy_inputs = inputs
            if local_logits_temperature is not None and not _is_identity_temperature(
                local_logits_temperature
            ):
                divisor = local_logits_temperature
                if isinstance(divisor, torch.Tensor):
                    from rl_engine.integrations.common.linear_logp import LinearLogpWrapper

                    temperatures = LinearLogpWrapper._temperature_tensor(
                        divisor, rows=inputs.logits.size(0), device=inputs.logits.device
                    )
                    assert temperatures is not None
                    divisor = temperatures.unsqueeze(1)
                entropy_inputs = replace(inputs, logits=inputs.logits.float() / divisor)
            entropy_contract, entropy_tiles = _contract_for_inputs(entropy_inputs)
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
                entropy_inputs.logits,
                inputs.target_ids,
                contract=entropy_contract,
                tp_group=inputs.tp_group,
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
            "role": inputs.execution_role,
            "strict_backend": True,
            "top_p_replay": keep_mask is not None or sparse_nucleus_ids is not None,
            "cp_is_merge_axis": False,
            "logits_materialized": bool(reuse_local_logits or inputs.with_entropy),
            "lm_head_result_reused": reuse_local_logits,
            "entropy": entropy_provenance,
        }
        provenance["request"] = {
            "hidden_shape": list(hidden.shape),
            "hidden_dtype": str(hidden.dtype).replace("torch.", ""),
            "target_shape": list(inputs.target_ids.shape),
            "tp_world_size": int(inputs.tp_world_size if inputs.tp_world_size is not None else 1),
            "tp_rank": int(inputs.tp_rank if inputs.tp_rank is not None else 0),
            "cp_world_size": int(inputs.cp_world_size),
            "cp_rank": int(inputs.cp_rank),
        }
        backend_id = str(provenance.get("actual_backend", linear_logp.backend_id))
        contract_id = (
            "linear_logp:"
            f"tp={inputs.tp_world_size if inputs.tp_world_size is not None else 1}:"
            f"cp={inputs.cp_world_size}:"
            f"vocab={inputs.padded_vocab_size}"
        )
        return LinearLogpResult(
            logp=logp.reshape(-1, 1),
            entropy=entropy,
            backend_id=backend_id,
            contract_id=contract_id,
            provenance=provenance,
        )

    if inputs.keep_mask is not None:
        raise LinearLogpProviderUnavailable(
            "RL-Kernel logp does not yet materialize top-p replay masks"
        )

    contract, num_vocab_tiles = _contract_for_inputs(inputs)
    assert isinstance(inputs.logits, torch.Tensor)  # Validated by the contract builder.
    dispatch = kernel_registry.get_logprob_op(contract, requested_backend=BACKEND_ID)
    if dispatch.provenance["actual_backend"] != BACKEND_ID or dispatch.provenance["fallback"]:
        raise RuntimeError("explicit WS2 backend dispatch changed during materialization")
    if inputs.with_entropy:
        logp, _lse, entropy = dispatch.op.apply_with_entropy(
            inputs.logits,
            inputs.target_ids,
            contract=contract,
            tp_group=inputs.tp_group,
            num_vocab_tiles=num_vocab_tiles,
            with_entropy_grad=inputs.with_entropy_grad,
        )
    else:
        logp, _lse = dispatch.op(
            inputs.logits,
            inputs.target_ids,
            contract=contract,
            tp_group=inputs.tp_group,
            num_vocab_tiles=num_vocab_tiles,
        )
        entropy = None
    provenance = dict(dispatch.provenance)
    provenance["request"] = {
        "logits_shape": list(inputs.logits.shape),
        "logits_dtype": str(inputs.logits.dtype).replace("torch.", ""),
        "target_shape": list(inputs.target_ids.shape),
        "target_dtype": str(inputs.target_ids.dtype).replace("torch.", ""),
        "real_vocab_size": contract.sharding.real_vocab_size,
        "padded_vocab_size": contract.sharding.padded_vocab_size,
        "tp_rank": contract.sharding.tp_rank,
        "tp_world_size": contract.sharding.tp_world_size,
        "cp_rank": contract.sharding.cp_rank,
        "cp_world_size": contract.sharding.cp_world_size,
    }
    provenance["execution"] = {
        "role": inputs.execution_role,
        "strict_backend": True,
        "top_p_replay": False,
    }
    provenance["token_row_ownership"] = {
        "rank": contract.sharding.cp_rank,
        "world_size": contract.sharding.cp_world_size,
        "layout": inputs.cp_layout,
        "local_token_rows": int(inputs.logits.shape[0]),
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


compute_training_logp.backend_id = BACKEND_ID  # type: ignore[attr-defined]


__all__ = [
    "LinearLogpProviderUnavailable",
    "LinearLogpResult",
    "TrainingLogpInputs",
    "compute_training_logp",
]
