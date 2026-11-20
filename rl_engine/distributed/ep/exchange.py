# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Forward expert dispatch and return for DSV4 expert parallelism.

Transport moves bytes only: every rank's slot travels as its own row and no
floating-point merge happens here. Each operation starts with a fixed-size
signature gather, so a rank that failed or diverged makes all ranks raise instead
of leaving peers blocked in an all-to-all. Callers see rows in expert order on
the way in and hand them back in that order.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist

from rl_engine.distributed.ep.plan import (
    WIRE_COLUMNS,
    EpCombineMapping,
    EpIdentity,
    EpPlacement,
    EpPlanError,
    EpRecvPlan,
    EpSendPlan,
    build_recv_plan,
    build_send_plan,
)

_OP_ROUTE, _OP_ROUTE_DONE, _OP_DISPATCH, _OP_RETURN, _OP_RETURN_DONE = range(1, 6)
_PAYLOAD_DTYPES = {torch.bfloat16: 1, torch.float16: 2, torch.float32: 3}
_SIGNATURE_NAMES = "status, op, seq, tag, dtype, width"


class EpExchangeError(RuntimeError):
    """Raised on every rank when an exchange fails; the exchange stays poisoned."""


class TorchAllToAll:
    """``all_to_all_single`` on a process group; the only validated backend is RCCL."""

    def __init__(self, group) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            raise EpExchangeError("process group required")
        if "nccl" not in str(dist.get_backend(group)).lower() or not torch.version.hip:
            raise EpExchangeError("expert exchange is only validated on ROCm RCCL")
        self.group = group
        self.rank = dist.get_rank(group)
        self.size = dist.get_world_size(group)
        self.device = torch.device("cuda", torch.cuda.current_device())

    def gather_signature(self, signature: torch.Tensor) -> torch.Tensor:
        output = signature.new_empty((self.size, signature.numel()))
        dist.all_gather_into_tensor(output, signature, group=self.group)
        return output

    def all_to_all(
        self,
        output: torch.Tensor,
        send: torch.Tensor,
        recv_counts: list[int],
        send_counts: list[int],
    ) -> None:
        dist.all_to_all_single(output, send, recv_counts, send_counts, group=self.group)


@dataclass(eq=False)
class EpRoute:
    """One forward invocation's plans; closed after its outputs are returned."""

    owner: EpExchange
    send: EpSendPlan
    recv: EpRecvPlan
    send_counts: list[int]
    recv_counts: list[int]
    returned: bool = False


class EpExchange:
    def __init__(
        self,
        group,
        placement: EpPlacement,
        *,
        strict=False,
        transport=None,
    ) -> None:
        self.transport = TorchAllToAll(group) if transport is None else transport
        if placement.num_ranks != self.transport.size:
            raise EpExchangeError("placement size differs from the EP group")
        self.placement = placement
        self.strict = strict
        self._sequence = 0
        self._poisoned = False

    # ------------------------------------------------------------ handshake
    def _agree(
        self,
        op: int,
        *,
        run_tag: int = 0,
        dtype: int = 0,
        width: int = 0,
        error: Exception | None = None,
    ) -> None:
        if self._poisoned:
            raise EpExchangeError("exchange is poisoned by an earlier failure") from error
        self._sequence += 1
        signature = torch.tensor(
            [error is not None, op, self._sequence, run_tag, dtype, width],
            dtype=torch.int64,
            device=self.transport.device,
        )
        try:
            peers = self.transport.gather_signature(signature)
        except Exception as exc:
            self._poisoned = True
            raise EpExchangeError("signature exchange failed") from exc
        failed = torch.nonzero(peers[:, 0]).reshape(-1).tolist()
        if failed:
            self._poisoned = True
            cause = f"; local cause: {error}" if error is not None else ""
            raise EpExchangeError(f"ranks {failed} failed{cause}") from error
        if not bool((peers == peers[0]).all()):
            self._poisoned = True
            raise EpExchangeError(f"ranks disagree on [{_SIGNATURE_NAMES}]: {peers.tolist()}")

    def _allocate(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=self.transport.device)

    def _move(
        self, output: torch.Tensor, send: torch.Tensor, recv: list[int], sent: list[int]
    ) -> torch.Tensor:
        try:
            self.transport.all_to_all(output, send, recv, sent)
        except Exception as exc:
            self._poisoned = True
            raise EpExchangeError("all-to-all failed") from exc
        return output

    def _payload_signature(self, route: EpRoute, rows: torch.Tensor) -> dict[str, int]:
        if route.owner is not self:
            raise EpPlanError("route belongs to another exchange")
        if route.returned:
            raise EpPlanError("route already returned; stale plans cannot be reused")
        if rows.dtype not in _PAYLOAD_DTYPES:
            raise EpPlanError(f"unsupported payload dtype {rows.dtype}")
        if rows.device != self.transport.device:
            raise EpPlanError("payload is not on the exchange device")
        if rows.requires_grad:
            raise EpPlanError("payload requires grad; all-to-all would drop it silently")
        width = 1
        for extent in rows.shape[1:]:
            width *= extent
        return {
            "run_tag": route.send.run_tag,
            "dtype": _PAYLOAD_DTYPES[rows.dtype],
            "width": width,
        }

    @contextlib.contextmanager
    def guard(self) -> Iterator[None]:
        """Report an exception raised between operations, such as in the expert kernel."""
        try:
            yield
        except Exception as exc:
            if not self._poisoned:
                with contextlib.suppress(EpExchangeError):
                    self._agree(0, error=exc)
            raise

    # ------------------------------------------------------------ operations
    def build_route(
        self,
        topk_experts: torch.Tensor,
        identity: EpIdentity,
        valid: torch.Tensor | None = None,
    ) -> EpRoute:
        send, error = None, None
        try:
            send = build_send_plan(
                topk_experts, self.placement, identity, self.transport.rank, valid=valid
            )
        except EpPlanError as exc:
            error = exc
        tag = 0 if send is None else send.run_tag
        self._agree(_OP_ROUTE, run_tag=tag, error=error)

        recv_counts = torch.empty_like(send.counts)
        self._move(recv_counts, send.counts, [1] * send.num_ranks, [1] * send.num_ranks)
        send_list, recv_list = send.counts.tolist(), recv_counts.tolist()

        # Allocation can fail on one rank, so it happens before peers enter the collective.
        metadata = None
        try:
            metadata = self._allocate((sum(recv_list), WIRE_COLUMNS), torch.int64)
        except RuntimeError as exc:
            error = exc
        self._agree(_OP_ROUTE, run_tag=tag, error=error)
        self._move(metadata, send.wire_metadata(), recv_list, send_list)

        recv = None
        try:
            recv = build_recv_plan(
                metadata, recv_counts, self.placement, identity, self.transport.rank, send.top_k
            )
        except (EpPlanError, RuntimeError) as exc:
            error = exc
        self._agree(_OP_ROUTE_DONE, run_tag=tag, error=error)
        return EpRoute(self, send, recv, send_list, recv_list)

    def _dispatch(self, route: EpRoute, rows, output, signature, error) -> torch.Tensor:
        self._agree(_OP_DISPATCH, **signature, error=error)
        self._move(output, rows, route.recv_counts, route.send_counts)
        with self.guard():
            return route.recv.to_expert_order(output)

    def dispatch_tokens(self, route: EpRoute, x: torch.Tensor) -> torch.Tensor:
        """``[T, ...]`` per-token rows to expert order on the owning ranks, byte-exact."""
        rows, output, signature, error = None, None, {}, None
        try:
            signature = self._payload_signature(route, x)
            rows = route.send.pack_token_rows(x)
            output = self._allocate((route.recv.num_rows, *rows.shape[1:]), rows.dtype)
        except (EpPlanError, RuntimeError) as exc:
            error = exc
        return self._dispatch(route, rows, output, signature, error)

    def dispatch_slots(self, route: EpRoute, values: torch.Tensor) -> torch.Tensor:
        """``[T, top_k, ...]`` per-slot rows to expert order on the owning ranks, byte-exact."""
        rows, output, signature, error = None, None, {}, None
        try:
            rows = route.send.pack_slot_values(values)
            signature = self._payload_signature(route, rows)
            output = self._allocate((route.recv.num_rows, *rows.shape[1:]), rows.dtype)
        except (EpPlanError, RuntimeError) as exc:
            error = exc
        return self._dispatch(route, rows, output, signature, error)

    def return_outputs(self, route: EpRoute, outputs: torch.Tensor) -> EpCombineMapping:
        """Expert-order outputs back to their sources, one row per slot, no merging."""
        rows, home, signature, error = None, None, {}, None
        try:
            signature = self._payload_signature(route, outputs)
            rows = route.recv.to_arrival_order(outputs)
            home = self._allocate((route.send.num_rows, *rows.shape[1:]), rows.dtype)
        except (EpPlanError, RuntimeError) as exc:
            error = exc
        self._agree(_OP_RETURN, **signature, error=error)

        self._move(home, rows, route.send_counts, route.recv_counts)
        route.returned = True

        if self.strict:
            echoed = torch.empty(
                (route.send.num_rows, WIRE_COLUMNS), dtype=torch.int64, device=home.device
            )
            metadata = _arrival_metadata(route.recv)
            self._move(echoed, metadata, route.send_counts, route.recv_counts)
            try:
                route.send.verify_echo(echoed)
            except EpPlanError as exc:
                error = exc
            self._agree(_OP_RETURN_DONE, run_tag=route.send.run_tag, error=error)
        with self.guard():
            return route.send.unpack_return(home)


def _arrival_metadata(recv: EpRecvPlan) -> torch.Tensor:
    return torch.stack(
        (
            torch.full_like(recv.tokens, recv.run_tag),
            recv.sources,
            recv.tokens,
            recv.slots,
            recv.experts,
        ),
        dim=1,
    )


__all__ = ["EpExchange", "EpExchangeError", "EpRoute", "TorchAllToAll"]
