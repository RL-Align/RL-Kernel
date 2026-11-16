# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Runtime ``linear_logp`` provider for Vime's Megatron backend.

The adapter accepts only structural objects. RL-Kernel never imports Vime;
Vime owns token-row construction and loss composition, while this provider
converts requests for the shared training scorer.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, cast

import torch

from rl_engine.integrations.common.training_logp import (
    LinearLogpProviderUnavailable,
    LinearLogpResult,
    TrainingLogpInputs,
    compute_training_logp,
)
from rl_engine.reference.logprob.vocab_parallel_logp import BACKEND_ID, DEFAULT_NUM_VOCAB_TILES
from rl_engine.runtime.plan import Implementation, operator_ablation_case

_DEFAULT_STRICT_LINEAR_LOGP: Any | None = None


def _default_strict_linear_logp() -> Any:
    global _DEFAULT_STRICT_LINEAR_LOGP
    if _DEFAULT_STRICT_LINEAR_LOGP is None:
        from rl_engine.integrations.common.linear_logp import LinearLogpWrapper

        _DEFAULT_STRICT_LINEAR_LOGP = LinearLogpWrapper()
    return _DEFAULT_STRICT_LINEAR_LOGP


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


def _local_logits_temperature(request: Any) -> Any:
    """Return scaling still required by the Vime local-logits contract.

    Older Vime revisions scale ``request.logits`` before provider dispatch and
    advertise that fact with ``logits_are_temperature_scaled``.  Current Vime
    revisions dispatch unscaled logits and leave the marker absent.  Only the
    reused-local-logits path consumes this distinction; recomputation from
    hidden states always starts from unscaled values.
    """

    if _metadata(request).get("logits_are_temperature_scaled") is True:
        return None
    return getattr(request, "temperature", None)


def to_training_logp_inputs(request: Any) -> TrainingLogpInputs:
    """Translate Vime's structural request into shared numerical inputs."""

    metadata = _metadata(request)
    logits = getattr(request, "logits", None)
    targets = _request_tensor(request, "target_ids")
    context = getattr(request, "context", None)
    hidden = getattr(context, "hidden", None)
    projection = getattr(context, "projection", None)
    partition = getattr(context, "vocab_partition", None)
    layout = getattr(request, "token_layout", None)
    weight = getattr(projection, "weight", None)
    keep_mask = getattr(request, "log_prob_keep_mask", None)
    reuse_local_logits = bool(getattr(context, "reuse_local_logits", False))
    local_logits = getattr(context, "local_logits", None)
    # Older Vime contexts have no local_logits field. Reuse the request tensor
    # when it describes the materialized shard produced by the output patch.
    materialized_local_logits = (
        isinstance(hidden, torch.Tensor)
        and isinstance(weight, torch.Tensor)
        and isinstance(logits, torch.Tensor)
        and logits.ndim == 2
        and logits.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and logits.shape == (hidden.size(0), weight.size(0))
        and (
            torch.version.hip is not None
            or keep_mask is not None
            or metadata.get("complete_sampling_support") is True
        )
    )
    if materialized_local_logits:
        reuse_local_logits = True
        local_logits = logits

    return TrainingLogpInputs(
        logits=logits,
        target_ids=targets,
        real_vocab_size=cast(
            int,
            (
                getattr(partition, "real_size", None)
                if partition is not None
                else metadata.get("real_vocab_size")
            ),
        ),
        padded_vocab_size=cast(
            int,
            (
                getattr(partition, "padded_size", None)
                if partition is not None
                else metadata.get("padded_vocab_size")
            ),
        ),
        vocab_start_index=(
            getattr(partition, "local_start", None) or 0 if partition is not None else None
        ),
        local_vocab_size=getattr(partition, "local_size", None),
        cp_rank=cast(int, getattr(layout, "rank", None)),
        cp_world_size=cast(int, getattr(layout, "world_size", None)),
        cp_layout=cast(str, getattr(layout, "layout", None)),
        tp_group=getattr(request, "tensor_parallel_group", None),
        tp_rank=metadata.get("tp_rank"),
        tp_world_size=metadata.get("tp_world_size"),
        num_vocab_tiles=metadata.get(
            "num_vocab_tiles",
            os.getenv("RL_KERNEL_LOGPROB_NUM_VOCAB_TILES", DEFAULT_NUM_VOCAB_TILES),
        ),
        hidden=hidden,
        weight=weight,
        bias=getattr(projection, "bias", None),
        local_logits=local_logits,
        reuse_local_logits=reuse_local_logits,
        temperature=getattr(request, "temperature", None),
        local_logits_temperature=_local_logits_temperature(request),
        keep_mask=keep_mask,
        sparse_token_ids=metadata.get("top_p_sparse_token_ids"),
        with_entropy=bool(getattr(request, "with_entropy", False)),
        with_entropy_grad=bool(getattr(request, "with_entropy_grad", False)),
        execution_role="vime_training_linear_logp",
    )


def _require_strict_context(request: Any) -> bool:
    strict = os.getenv("VIME_RL_KERNEL_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict and not isinstance(
        getattr(getattr(request, "context", None), "hidden", None), torch.Tensor
    ):
        raise RuntimeError("strict Vime linear_logp request is missing structural context")
    return strict


def _provider_impl(request: Any, *, linear_logp: Any = None) -> LinearLogpResult:
    """Compatibility entry point for callers passing a Vime request directly."""

    if _require_strict_context(request) and linear_logp is None:
        linear_logp = _default_strict_linear_logp()
    return compute_training_logp(to_training_logp_inputs(request), linear_logp=linear_logp)


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

    from rl_engine.integrations.common.state import get_active_integration

    integration = get_active_integration("megatron")
    if integration is None:
        return _provider_impl(request)

    def native_unavailable(_request: Any) -> LinearLogpResult:
        raise RuntimeError(
            "the structural provider was invoked for a production Megatron logp route"
        )

    _require_strict_context(request)
    return integration.execute("logp", native_unavailable, to_training_logp_inputs(request))


__all__ = [
    "LinearLogpResult",
    "LinearLogpProviderUnavailable",
    "provider",
    "to_training_logp_inputs",
]
