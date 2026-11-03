# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P1: recorded training provider over the existing Megatron logp route.

A recorded M10 input is wrapped in VIME's ``LinearLogpRequest`` and sent once
through the existing ``MegatronLogpOperator`` → VIME provider → strict
``LinearLogpWrapper`` route under a read-only :class:`TrainingLogpObserver`;
the frozen snapshot is validated by the same gate the live provider uses. No
LM-head GEMM or log-softmax is reimplemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol

import torch

from rl_engine.alignment.cross_config.adapters.megatron_training import (
    PreUpdateWeightLock,
    T05ContractError,
)
from rl_engine.alignment.cross_config.adapters.megatron_training_gate import (
    RowLayout,
    TrainingCallContext,
    TrainingM10Evidence,
    validate_training_call,
)
from rl_engine.integrations.megatron_training_readback import TrainingLogpObserver

M10_PROVIDER_INPUT_SCHEMA_VERSION = "rlkernel.p7.t05.m10_provider_input.v1-draft"


@dataclass(frozen=True)
class M10ProviderInput:
    """Shared M10 input contract (training side), plus the identity T05 must bind."""

    final_decoder_rows: torch.Tensor  # BF16 [N, H]
    lm_head_weight_local: torch.Tensor  # BF16 [V_local, H]
    target_token_ids: torch.Tensor  # INT64 [N]
    active_token_mask: torch.Tensor  # BOOL [N]
    vocab_start: int
    real_vocab_size: int
    padded_vocab_size: int
    token_layout_world_size: int
    token_layout_rank: int
    token_layout: str
    row_layout: RowLayout
    checkpoint_id: str
    weight_id: str
    weight_lock: PreUpdateWeightLock
    lm_head_lora: Optional[bool]
    sampling_record: Optional[Mapping[str, Mapping[str, Any]]]
    temperature: Any = 1.0
    tensor_parallel_group: Any = None
    local_logits: Optional[torch.Tensor] = None  # recorded logits enable the reuse path
    metadata: Mapping[str, Any] = field(default_factory=dict)
    chunk_size: int = 1024
    schema_version: str = M10_PROVIDER_INPUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != M10_PROVIDER_INPUT_SCHEMA_VERSION:
            raise T05ContractError(
                "SCHEMA_MISMATCH", "m10_input.schema_version", self.schema_version
            )
        rows = self.final_decoder_rows
        checks = (
            ("final_decoder_rows", rows, torch.bfloat16, 2),
            ("lm_head_weight_local", self.lm_head_weight_local, torch.bfloat16, 2),
            ("target_token_ids", self.target_token_ids, torch.int64, 1),
            ("active_token_mask", self.active_token_mask, torch.bool, 1),
        )
        for name, tensor, dtype, ndim in checks:
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype or tensor.ndim != ndim:
                raise T05ContractError(
                    "SCHEMA_MISMATCH", name, f"expected {dtype} with {ndim} dims"
                )
        n = rows.size(0)
        if (
            self.target_token_ids.numel() != n
            or self.active_token_mask.numel() != n
            or self.lm_head_weight_local.size(1) != rows.size(1)
        ):
            raise T05ContractError("SCHEMA_MISMATCH", "m10_input", "row or hidden widths differ")
        if self.local_logits is not None and tuple(self.local_logits.shape) != (
            n,
            self.lm_head_weight_local.size(0),
        ):
            raise T05ContractError("SCHEMA_MISMATCH", "local_logits", "expected [N, V_local]")

    @property
    def vocab_local_size(self) -> int:
        return int(self.lm_head_weight_local.size(0))


class TrainingProvider(Protocol):
    """Recorded and live training providers share this ABI and the same gate."""

    def score(self, m10_input: M10ProviderInput) -> TrainingM10Evidence: ...


def vime_request_factory(m10_input: M10ProviderInput, logits: torch.Tensor) -> Any:
    """Build VIME's own ``LinearLogpRequest`` (its validation runs on construction)."""

    try:
        from vime.backends.megatron_utils import linear_logp_provider as vime
    except ImportError as exc:
        raise T05ContractError(
            "UNSUPPORTED_CAPABILITY", "vime", f"VIME request types are unavailable: {exc}"
        ) from exc
    return vime.LinearLogpRequest(
        logits=logits,
        target_ids=m10_input.target_token_ids,
        tensor_parallel_group=m10_input.tensor_parallel_group,
        token_layout=vime.TokenLayout(
            world_size=m10_input.token_layout_world_size,
            rank=m10_input.token_layout_rank,
            layout=m10_input.token_layout,
        ),
        with_entropy=False,
        with_entropy_grad=False,
        chunk_size=m10_input.chunk_size,
        context=vime.LinearLogpContext(
            hidden=m10_input.final_decoder_rows,
            projection=vime.LinearProjection(weight=m10_input.lm_head_weight_local),
            vocab_partition=vime.VocabPartition(
                local_start=m10_input.vocab_start,
                local_size=m10_input.vocab_local_size,
                real_size=m10_input.real_vocab_size,
                padded_size=m10_input.padded_vocab_size,
            ),
        ),
        temperature=m10_input.temperature,
        metadata=dict(m10_input.metadata),
    )


def default_operator() -> Any:
    """The operator ``install_megatron_integration`` installs for the strict logp route."""

    from rl_engine.integrations.framework_operators import MegatronLogpOperator
    from rl_engine.integrations.linear_logp import LinearLogpWrapper
    from rl_engine.integrations.vime.linear_logp_provider import _provider_impl

    return MegatronLogpOperator(_provider_impl, linear_logp=LinearLogpWrapper())


class RecordedTrainingProvider:
    """Score a recorded M10 input once through the existing strict training route."""

    def __init__(
        self,
        *,
        operator_factory: Callable[[], Any] = default_operator,
        request_factory: Callable[[M10ProviderInput, torch.Tensor], Any] = vime_request_factory,
        digest_tensors: bool = True,
    ) -> None:
        self._operator_factory = operator_factory
        self._request_factory = request_factory
        self._digest = digest_tensors

    def score(self, m10_input: M10ProviderInput) -> TrainingM10Evidence:
        rows = m10_input.final_decoder_rows
        recorded_logits = m10_input.local_logits
        placeholder = recorded_logits is None
        if recorded_logits is None:
            # Zero-stride placeholder: VIME requires [N, V_local] logits, the recompute
            # path never reads them, and no LM-head GEMM is run here to produce them.
            logits = torch.zeros((), dtype=torch.bfloat16, device=rows.device).expand(
                rows.size(0), m10_input.vocab_local_size
            )
        else:
            logits = recorded_logits
        request = self._request_factory(m10_input, logits)
        operator = self._operator_factory()
        observer = TrainingLogpObserver(max_snapshots=1, digest_tensors=self._digest)
        observer.install(type(operator))
        try:
            operator(request)
        finally:
            observer.uninstall()
        snapshots = observer.snapshots
        if len(snapshots) != 1:
            raise T05ContractError(
                "MISSING_PROVENANCE", "snapshot", f"expected 1 call, observed {len(snapshots)}"
            )
        evidence = validate_training_call(
            snapshots[0],
            TrainingCallContext(
                checkpoint_id=m10_input.checkpoint_id,
                weight_id=m10_input.weight_id,
                weight_lock=m10_input.weight_lock,
                row_layout=m10_input.row_layout,
                active_token_mask=m10_input.active_token_mask,
                lm_head_lora=m10_input.lm_head_lora,
                sampling_record=m10_input.sampling_record,
            ),
        )
        if placeholder and evidence.path != "recompute":
            raise T05ContractError(
                "MISSING_PROVENANCE",
                "logits",
                f"placeholder logits were consumed by the {evidence.path} path",
            )
        return evidence


__all__ = [
    "M10_PROVIDER_INPUT_SCHEMA_VERSION",
    "M10ProviderInput",
    "RecordedTrainingProvider",
    "TrainingProvider",
    "default_operator",
    "vime_request_factory",
]
