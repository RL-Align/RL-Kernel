# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict expert-parallel transport for Nemotron-3-Nano (RFC #434 ``ep_all_to_all``).

The caller routes tokens, runs its experts on ``x_rows`` and combines the unweighted
``(token, slot)`` rows that come back. Weighted combine and expert kernels live outside
this module. Every rank must make the same calls in the same order, including ranks
with no tokens. This is the forward exchange only and runs under ``no_grad``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.distributed.ep.cuda_transport import SUPPORTED_EP_SIZES, CudaNcclAllToAll
from rl_engine.distributed.ep.exchange import EpExchange, EpRoute
from rl_engine.distributed.ep.plan import EP_MAPPING_SCHEMA, EpIdentity, EpPlacement, EpPlanError

HIDDEN_SIZE = 2688
TOP_K = 6
NUM_EXPERTS = 128
MAX_TOKENS = 65536
_OP_CALL = 21


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise EpPlanError(message)


@dataclass(frozen=True)
class EpInvocation:
    """Names one exchange across ranks; a training-wide id cannot tell two layers apart."""

    step: int
    microbatch: int
    layer: int
    invocation: int = 0

    def __post_init__(self) -> None:
        _require(
            all(type(v) is int and v >= 0 for v in self.key),
            "step, microbatch, layer and invocation must be non-negative integers",
        )

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.step, self.microbatch, self.layer, self.invocation)

    @property
    def run_id(self) -> str:
        return "nemotron-ep.v1/s{}/m{}/l{}/i{}".format(*self.key)


class NemotronEpExchange:
    """Construction is local: make it on every rank before the first exchange."""

    def __init__(
        self,
        placement: EpPlacement,
        *,
        model_version: str,
        placement_version: str,
        group=None,
        transport=None,
        strict: bool = True,
    ) -> None:
        _require(
            placement.expert_owner.numel() == NUM_EXPERTS,
            f"placement must cover {NUM_EXPERTS} experts",
        )
        _require(
            placement.num_ranks in SUPPORTED_EP_SIZES,
            f"EP size {placement.num_ranks} is outside {SUPPORTED_EP_SIZES}",
        )
        self.exchange = EpExchange(
            group,
            placement,
            strict=strict,
            transport=CudaNcclAllToAll(group) if transport is None else transport,
        )
        self.model_version = model_version
        self.placement_version = placement_version
        self._batch: tuple[int, int] | None = None
        self._seen: set[tuple[int, int, int, int]] = set()

    def _gate(self, x: torch.Tensor, ids: torch.Tensor, invocation: EpInvocation) -> None:
        _require(
            torch._C._current_graph_task_id() == -1,
            "exchange inside backward is unsupported, including reentrant checkpoint",
        )
        device = self.exchange.transport.device
        _require(x.dtype == torch.bfloat16, f"x must be BF16, got {x.dtype}")
        _require(
            x.ndim == 2 and x.shape[1] == HIDDEN_SIZE,
            f"x must have shape [T, {HIDDEN_SIZE}], got {tuple(x.shape)}",
        )
        _require(x.shape[0] <= MAX_TOKENS, f"at most {MAX_TOKENS} tokens per rank")
        _require(ids.dtype == torch.int64, f"ids must be int64, got {ids.dtype}")
        _require(
            ids.shape == (x.shape[0], TOP_K),
            f"ids must have shape [{x.shape[0]}, {TOP_K}], got {tuple(ids.shape)}",
        )
        _require(
            x.device == device and ids.device == device,
            "inputs must be on the EP device",
        )
        _require(
            bool(((ids >= 0) & (ids < NUM_EXPERTS)).all()),
            f"expert id outside [0, {NUM_EXPERTS})",
        )
        _require(
            bool((ids[:, 1:] > ids[:, :-1]).all()),
            "expert ids must be unique and ascending per token",
        )
        _require(
            not torch.is_grad_enabled(),
            "the forward exchange has no backward; run it under no_grad",
        )
        batch = invocation.key[:2]
        seen = self._seen if batch == self._batch else set()
        _require(invocation.key not in seen, f"invocation {invocation.key} already used")
        self._batch, self._seen = batch, seen | {invocation.key}

    def dispatch(
        self, x: torch.Tensor, ids: torch.Tensor, invocation: EpInvocation
    ) -> NemotronEpCall:
        """``x`` BF16 ``[T, H]`` and router ``ids`` int64 ``[T, 6]`` to rows on the owners."""
        exchange, error, tag = self.exchange, None, 0
        identity: Any = None
        training = torch.is_grad_enabled()
        try:
            self._gate(x, ids, invocation)
            identity = EpIdentity(invocation.run_id, self.model_version, self.placement_version)
            tag = identity.run_tag(exchange.placement, TOP_K)
        except EpPlanError as exc:
            error = exc
        # Training and strict both change which collectives follow, so all ranks agree on them.
        mode = int(training) | int(exchange.strict) << 1
        exchange._agree(_OP_CALL, run_tag=tag, width=mode, error=error)

        route = exchange.build_route(ids, identity)
        rows = exchange.dispatch_tokens(route, x.detach())
        return NemotronEpCall(self, route, rows)


@dataclass(eq=False)
class NemotronEpCall:
    owner: NemotronEpExchange
    route: EpRoute
    x_rows: torch.Tensor  # BF16 [R, H], (local expert, source, token, slot)
    rows: torch.Tensor | None = None

    @property
    def expert_offsets(self) -> torch.Tensor:
        """Exclusive prefix counts of ``x_rows`` per local expert."""
        return self.route.recv.expert_offsets

    @property
    def local_experts(self) -> torch.Tensor:
        """Global expert ids owned by this rank, indexed by local expert id."""
        exchange = self.owner.exchange
        return exchange.placement.local_expert_ids(exchange.transport.rank)

    def return_outputs(self, y_rows: torch.Tensor) -> torch.Tensor:
        """Unweighted BF16 ``[T * 6, H]`` in ``(token, slot)`` order, on the source rank."""
        exchange = self.owner.exchange
        with exchange.guard():
            _require(self.rows is None, "outputs already returned")
            _require(
                y_rows.dtype == torch.bfloat16,
                f"y_rows must be BF16, got {y_rows.dtype}",
            )
            _require(
                y_rows.shape == self.x_rows.shape,
                f"y_rows must match x_rows {tuple(self.x_rows.shape)}, got {tuple(y_rows.shape)}",
            )
        self.rows = exchange.return_outputs(self.route, y_rows.detach()).rows
        return self.rows

    def provenance(self) -> dict[str, Any]:
        """What actually ran, so a claim can be read back from the run."""
        exchange = self.owner.exchange
        transport = exchange.transport
        return {
            "ep_size": transport.size,
            "rank": transport.rank,
            "backend": getattr(transport, "backend_id", type(transport).__name__),
            "payload_dtype": "bf16",
            "placement_fingerprint": exchange.placement.fingerprint,
            "run_id": self.route.send.identity.run_id,
            "run_tag": self.route.send.run_tag,
            "strict": exchange.strict,
            "fallback": False,
            "mapping_schema": EP_MAPPING_SCHEMA,
        }


__all__ = [
    "HIDDEN_SIZE",
    "MAX_TOKENS",
    "NUM_EXPERTS",
    "TOP_K",
    "EpInvocation",
    "NemotronEpCall",
    "NemotronEpExchange",
]
