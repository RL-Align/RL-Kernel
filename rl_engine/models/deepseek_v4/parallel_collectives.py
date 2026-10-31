# SPDX-License-Identifier: Apache-2.0
"""Opt-in DSV4 TP/SP/DP fixed-order collective execution."""

import torch
import torch.distributed as dist

from rl_engine.contracts.operators.dsv4_collectives import (
    Dsv4CollectiveContract,
    require,
)
from rl_engine.distributed.transports.rccl import (
    RCCLDeterministicCollective,
    TorchDistributedDeterministicCollective,
)


class Dsv4ParallelCollectives:
    """Execute a sealed schedule using the repository's fixed-tree backends.

    The caller creates all process groups in a common global order. Construction
    and each phase are collective over ``world_group``; an exception during a
    distributed phase poisons this instance, so it cannot be resumed silently.
    CPU/Gloo is an explicit executable reference, not a production backend.
    """

    def __init__(
        self,
        contract: Dsv4CollectiveContract,
        *,
        world_group: dist.ProcessGroup,
        groups: dict[str, dist.ProcessGroup],
        allow_cpu_reference: bool = False,
    ) -> None:
        require(isinstance(contract, Dsv4CollectiveContract), "sealed contract required")
        require(dist.is_available() and dist.is_initialized(), "process group required")
        self.contract = contract
        self.world_group = world_group
        local_error = None
        rocm = False
        try:
            require(set(groups) == {"tp", "sp", "dp"}, "TP/SP/DP groups required")
            require(
                tuple(dist.get_process_group_ranks(world_group)) == contract.global_ranks,
                "world group differs from sealed ranks",
            )
            rank = dist.get_rank()
            require(rank in contract.global_ranks, "rank outside sealed group")
            for dimension, group in groups.items():
                expected = next(g for g in getattr(contract, f"{dimension}_groups") if rank in g)
                require(
                    tuple(dist.get_process_group_ranks(group)) == expected,
                    f"{dimension} rank partition differs from sealed plan",
                )
            backend = str(dist.get_backend(world_group)).lower()
            rocm = "nccl" in backend and bool(getattr(torch.version, "hip", None))
            cpu = "gloo" in backend and allow_cpu_reference
            require(rocm or cpu, "unvalidated DSV4 collective transport")
            for dimension, group in groups.items():
                require(
                    str(dist.get_backend(group)).lower() == backend,
                    f"{dimension} backend differs from world group",
                )
            self.device = (
                torch.device("cuda", torch.cuda.current_device()) if rocm else torch.device("cpu")
            )
        except Exception as exc:
            local_error = exc
            self.device = torch.device("cpu")
        self._agree(("construct", contract.fingerprint), local_error is None)
        if local_error is not None:
            raise local_error
        cls = RCCLDeterministicCollective if rocm else TorchDistributedDeterministicCollective
        self._collectives = {}
        try:
            for dimension in ("tp", "sp", "dp"):
                self._collectives[dimension] = cls(
                    group=groups[dimension],
                    device=self.device,
                    max_size_bytes=contract.max_size_bytes,
                )
        except Exception:
            self.close()
            raise
        self._active = False
        self._poisoned = False
        self._closed = False
        self._next = 0

    def _agree(self, signature: tuple, valid: bool = True) -> None:
        peers = [None] * dist.get_world_size(self.world_group)
        dist.all_gather_object(peers, (signature, valid), group=self.world_group)
        require(
            all(peer[0] == signature for peer in peers),
            f"DSV4 rank schedule/version mismatch: {peers}",
        )
        require(all(peer[1] for peer in peers), "invalid DSV4 phase state or layout on a rank")

    def _usable(self) -> None:
        require(not self._poisoned, "collective instance is closed or poisoned")
        if self.device.type == "cuda":
            require(
                not torch.cuda.is_current_stream_capturing(), "DSV4 Graph capture is not validated"
            )

    def begin_step(self, step_id: int, *, model_version: str) -> None:
        self._usable()
        try:
            valid = (
                not self._active
                and type(step_id) is int
                and step_id >= 0
                and model_version == self.contract.model_version
            )
            self._agree(("begin", step_id, model_version, self.contract.fingerprint), valid)
        except Exception:
            self._poisoned = True
            raise
        self._active = True
        self._next = 0

    def execute(self, phase_name: str, value: torch.Tensor) -> torch.Tensor:
        self._usable()
        try:
            layout = (
                (tuple(value.shape), str(value.dtype), str(value.device), value.is_contiguous())
                if isinstance(value, torch.Tensor)
                else None
            )
            phase = (
                self.contract.phases[self._next] if self._next < len(self.contract.phases) else None
            )
            valid = (
                self._active
                and phase is not None
                and phase_name == phase.name
                and isinstance(value, torch.Tensor)
                and tuple(value.shape) == phase.shape
                and value.dtype == phase.dtype
                and value.device == self.device
                and value.is_contiguous()
            )
            self._agree(("phase", self._next, phase_name, layout), valid)
            if value.numel() == 0:
                # The world-wide phase handshake already established an empty input
                # on every rank, so no backend transport is necessary.
                group_size = self._collectives[phase.group].world_size
                rows = value.size(0)
                if phase.operation == "all_gather":
                    rows *= group_size
                elif phase.operation == "reduce_scatter":
                    rows //= group_size
                result = value.new_empty((rows, *value.shape[1:]))
            else:
                result = getattr(self._collectives[phase.group], phase.operation)(value)
            if phase.denominator is not None:
                result = result / phase.denominator
        except Exception:
            self._poisoned = True
            raise
        self._next += 1
        return result

    def end_step(self) -> None:
        self._usable()
        try:
            self._agree(
                ("end", self._next, self.contract.fingerprint),
                self._active and self._next == len(self.contract.phases),
            )
        except Exception:
            self._poisoned = True
            raise
        self._active = False

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._poisoned = True
        self._closed = True
        for collective in getattr(self, "_collectives", {}).values():
            collective.close()

    def __enter__(self):
        self._usable()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
