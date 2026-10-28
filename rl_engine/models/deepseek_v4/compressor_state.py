# SPDX-License-Identifier: Apache-2.0
"""Host-checked request leases around existing compressor/state GPU operators."""

from dataclasses import dataclass, field
import hashlib
import json
import math
from threading import RLock

import torch

from rl_engine.contracts.operators.compressor_state import CompressorStateContract
from rl_engine.contracts.operators.kv_compressor_c4 import require, source_hash


@dataclass(frozen=True)
class RequestHandle:
    request_id: int
    generation: int
    version: str
    writer_rank: int
    slot: int
    _arena: object = field(repr=False, compare=True)


@dataclass(frozen=True)
class CommitLease:
    handle: RequestHandle
    start: int
    stop: int
    first_group: int
    end_group: int


@dataclass(frozen=True)
class CommitReceipt:
    request_id: int
    generation: int
    version: str
    first_group: int
    end_group: int
    tokens: int
    arithmetic_id: str
    implementation_fingerprint: str


@dataclass
class _Request:
    handle: RequestHandle
    tokens: int = 0
    commits: int = 0


class CompressorStateCache:
    """One staging lease at a time; completed commits are synchronous.

    This conservative ownership policy avoids early buffer reuse across host
    threads and GPU streams. Communication overlap is a separate task. Input,
    parameter, state and output ownership is explicit; cache writes have no VJP.
    """

    def __init__(
        self,
        contract,
        *,
        ape,
        norm,
        cos,
        sin,
        version,
        writer_rank,
        expected_arithmetic,
        rms_eps=1e-6,
        backend=None,
        expected_implementation=None,
    ):
        require(type(contract) is CompressorStateContract, "typed state contract required")
        require(
            expected_arithmetic == contract.arithmetic_id,
            "compressor arithmetic mismatch; C128 HCA eight-wave is a different profile",
        )
        require(
            isinstance(version, str) and bool(version), "explicit model/position version required"
        )
        require(
            type(writer_rank) is int and 0 <= writer_rank < contract.world_size,
            "invalid writer rank",
        )
        require(
            type(rms_eps) in (int, float) and math.isfinite(rms_eps) and 1e-12 <= rms_eps <= 1,
            "invalid RMS epsilon",
        )
        if backend is None:
            from rl_engine.backends.rocm.attention.compressor_state import (
                AiterCompressorStateBackend,
            )

            backend = AiterCompressorStateBackend()
        self.contract, self.backend, self.rms_eps = contract, backend, rms_eps
        require(
            isinstance(ape, torch.Tensor) and ape.device.type == backend.device_type,
            "parameter/backend device mismatch",
        )
        self.device = ape.device
        if hasattr(backend, "device"):
            require(self.device == backend.device, "wrong provider device index")
        self._eager()
        self._lock, self._identity = RLock(), object()
        self._version, self.writer_rank = version, writer_rank
        self._requests, self._slots = {}, {}
        self._generation, self._pending, self._poisoned = 0, None, False
        self._graph = None
        c = contract
        with torch.inference_mode(False), torch.no_grad():
            self._ape, self._norm, self._cos, self._sin = self._parameter_copies(
                ape, norm, cos, sin
            )
            self._state = torch.zeros(
                (c.max_requests, c.state_rows, 2 * c.width), device=self.device, dtype=torch.float32
            )
            self._read_kv = torch.empty(
                (c.max_requests, c.state_rows, c.width), device=self.device, dtype=torch.float32
            )
            self._read_score = torch.empty_like(self._read_kv)
            self._cache = torch.zeros(
                (c.max_requests * c.pages_per_request, 64, 512),
                device=self.device,
                dtype=torch.bfloat16,
            )
            self._input = torch.zeros(
                (c.max_tokens_per_step, 2 * c.width), device=self.device, dtype=torch.bfloat16
            )
            self._positions = torch.zeros(
                c.max_tokens_per_step, device=self.device, dtype=torch.int64
            )
            self._write_slots = torch.full_like(self._positions, -1)
            self._plan = torch.zeros((c.plan_rows, 4), device=self.device, dtype=torch.int32)
            self._plan[:, 2] = -1
            self._state_slots = torch.arange(c.max_requests, device=self.device, dtype=torch.int32)
            self._blocks = torch.arange(
                c.max_requests * c.pages_per_request, device=self.device, dtype=torch.int32
            ).reshape(c.max_requests, -1)
        identity = {
            "contract": c.fingerprint,
            "arithmetic": c.arithmetic_id,
            "eps": rms_eps,
            "provider": getattr(backend, "provenance", {"test_backend": type(backend).__name__}),
            "model_source": source_hash(__file__),
        }
        self.implementation_fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        if expected_implementation is not None:
            require(
                expected_implementation == self.implementation_fingerprint,
                "state implementation fingerprint mismatch",
            )
        if c.graph:
            require(
                self.device.type == "cuda" and backend.supports_graph,
                "Graph is unsupported by this backend",
            )
            self._capture()

    def _eager(self):
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                require(
                    not torch.cuda.is_current_stream_capturing(),
                    "host lifecycle methods cannot run inside an external Graph capture",
                )

    def _ensure(self):
        self._eager()
        require(
            not self._poisoned,
            "state arena is poisoned; discard it rather than retrying a partial write",
        )

    def _parameter_copies(self, ape, norm, cos, sin):
        c = self.contract
        result = []
        for tensor, shape, dtype, name in (
            (ape, (c.ratio, c.width), torch.float32, "APE"),
            (norm, (512,), torch.float32, "RMS"),
            (cos, (c.max_sequence_tokens, 32), torch.bfloat16, "cos"),
            (sin, (c.max_sequence_tokens, 32), torch.bfloat16, "sin"),
        ):
            require(
                isinstance(tensor, torch.Tensor)
                and tuple(tensor.shape) == shape
                and tensor.dtype == dtype
                and tensor.device == self.device
                and tensor.is_contiguous(),
                f"invalid {name} shape/dtype/device/layout",
            )
            snapshot = tensor.detach().clone()
            limit = 1 if name in ("cos", "sin") else 2.0**20
            require(
                bool((torch.isfinite(snapshot) & (snapshot.abs() <= limit)).all()),
                f"{name} outside finite admitted domain",
            )
            result.append(snapshot)
        return tuple(result)

    def _complete_device(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            event.synchronize()

    def _capture(self):
        # Warmup/capture only PAD plans and invalid state writers. Native
        # kernels compile at full static capacity without touching real state.
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                self.backend.execute(self)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph), torch.no_grad():
            self.backend.execute(self)
        self._complete_device()
        require(
            bool((self._state == 0).all()) and bool((self._cache == 0).all()),
            "provider warmup/capture wrote real state",
        )

    def _request(self, handle):
        require(
            isinstance(handle, RequestHandle) and handle._arena is self._identity,
            "foreign request handle",
        )
        entry = self._requests.get(handle.request_id)
        require(
            entry is not None and entry.handle is handle and handle.version == self._version,
            "stale request generation or model version",
        )
        return entry

    def _clear_slot(self, slot):
        try:
            c = self.contract
            page = slot * c.pages_per_request
            self._state[slot].zero_()
            self._cache[page : page + c.pages_per_request].zero_()
            self._complete_device()
        except BaseException:
            self._poisoned = True
            raise

    def create_request(self, request_id, *, version, writer_rank):
        with self._lock, torch.inference_mode(False), torch.no_grad():
            self._ensure()
            require(
                type(request_id) is int and request_id >= 0,
                "request_id must be a nonnegative integer",
            )
            require(version == self._version, "model/position version mismatch")
            require(
                type(writer_rank) is int and 0 <= writer_rank < self.contract.world_size,
                "invalid request writer rank",
            )
            require(request_id not in self._requests, "request already active")
            available = [i for i in range(self.contract.max_requests) if i not in self._slots]
            require(bool(available), "request capacity exhausted")
            slot = available[0]
            self._generation += 1
            h = RequestHandle(
                request_id, self._generation, version, writer_rank, slot, self._identity
            )
            self._clear_slot(slot)
            self._requests[request_id] = _Request(h)
            self._slots[slot] = request_id
            return h

    def prepare(self, handle, values, *, start_position):
        with self._lock, torch.inference_mode(False), torch.no_grad():
            self._ensure()
            entry = self._request(handle)
            require(handle.writer_rank == self.writer_rank, "only the declared writer may commit")
            require(self._pending is None, "another staging lease is outstanding")
            require(
                type(start_position) is int and start_position == entry.tokens,
                "noncontiguous start or already committed token range",
            )
            c = self.contract
            require(
                isinstance(values, torch.Tensor)
                and values.dtype == torch.bfloat16
                and values.device == self.device
                and values.ndim == 2
                and values.shape[1] == 2 * c.width
                and values.is_contiguous()
                and not values.requires_grad,
                "cache inputs must be detached contiguous BF16 rows",
            )
            count = values.shape[0]
            require(count <= c.max_tokens_per_step, "step input capacity exceeded")
            stop = start_position + count
            require(stop <= c.max_sequence_tokens, "sequence capacity exceeded")
            # Own data before inspecting content; original aliases can be reused
            # after prepare returns. Concurrent writes during copying are forbidden.
            copy = values.detach().clone()
            require(
                bool((torch.isfinite(copy.float()) & (copy.float().abs() <= 2.0**30)).all()),
                "input outside finite FP32 accumulation domain",
            )
            plan = [
                [
                    p - start_position,
                    handle.slot,
                    p,
                    max(0, c.state_rows - (p - start_position + 1)),
                ]
                for p in range(start_position, stop)
                if (p + 1) % c.ratio == 0
            ]
            self._input.zero_()
            self._input[:count].copy_(copy)
            self._positions.zero_()
            self._write_slots.fill_(-1)
            if count:
                positions = torch.arange(
                    start_position, stop, dtype=torch.int64, device=self.device
                )
                self._positions[:count].copy_(positions)
                first_writer = max(0, count - c.state_rows)
                self._write_slots[first_writer:count].copy_(
                    handle.slot * c.state_rows + positions[first_writer:].remainder(c.state_rows)
                )
            self._plan.zero_()
            self._plan[:, 2] = -1
            if plan:
                self._plan[: len(plan)].copy_(
                    torch.tensor(plan, dtype=torch.int32, device=self.device)
                )
            # A different caller thread/stream may own commit(). Finish the
            # staging writes before handing off this exclusive lease.
            self._complete_device()
            lease = CommitLease(
                handle, start_position, stop, start_position // c.ratio, stop // c.ratio
            )
            self._pending = lease
            return lease

    def cancel(self, lease):
        with self._lock:
            self._ensure()
            require(
                isinstance(lease, CommitLease) and lease is self._pending,
                "lease is stale, foreign or already consumed",
            )
            self._pending = None

    def commit(self, lease):
        with self._lock, torch.inference_mode(False), torch.no_grad():
            self._ensure()
            require(
                isinstance(lease, CommitLease) and lease is self._pending,
                "lease is stale, foreign or already consumed",
            )
            entry = self._request(lease.handle)
            require(entry.tokens == lease.start, "token cursor changed after preparation")
            try:
                if lease.stop != lease.start:
                    if self._graph is None:
                        self.backend.execute(self)
                    else:
                        self._graph.replay()
                    self._complete_device()
            except BaseException:
                self._poisoned = True
                raise
            # Publish the cursor only after the ordered device writes complete.
            entry.tokens = lease.stop
            entry.commits += 1
            self._pending = None
            return CommitReceipt(
                lease.handle.request_id,
                lease.handle.generation,
                lease.handle.version,
                lease.first_group,
                lease.end_group,
                lease.stop,
                self.contract.arithmetic_id,
                self.implementation_fingerprint,
            )

    def append(self, handle, values, *, start_position):
        # Keep preparation and commit atomic relative to other host callers.
        with self._lock:
            return self.commit(self.prepare(handle, values, start_position=start_position))

    def snapshot(self, handle):
        with self._lock, torch.no_grad():
            self._ensure()
            entry = self._request(handle)
            c = self.contract
            page = handle.slot * c.pages_per_request
            rows = (
                self._cache[page : page + c.pages_per_request]
                .flatten(0, 1)[: entry.tokens // c.ratio]
                .clone()
            )
            state = self._state[handle.slot].clone()
            self._complete_device()
            return {
                "rows": rows,
                "partial_ring": state,
                "tokens": entry.tokens,
                "incomplete_tokens": entry.tokens % c.ratio,
                "commits": entry.commits,
                "generation": handle.generation,
                "version": handle.version,
                "arithmetic_id": c.arithmetic_id,
            }

    def release(self, handle):
        with self._lock, torch.no_grad():
            self._ensure()
            self._request(handle)
            require(
                self._pending is None or self._pending.handle is not handle,
                "cancel the outstanding lease before releasing its request",
            )
            self._clear_slot(handle.slot)
            del self._requests[handle.request_id]
            del self._slots[handle.slot]

    def replace_model(self, *, version, ape, norm, cos, sin):
        with self._lock, torch.inference_mode(False), torch.no_grad():
            self._ensure()
            require(
                not self._requests and self._pending is None,
                "release all requests before model replacement",
            )
            require(
                isinstance(version, str) and bool(version) and version != self._version,
                "new model/position version required",
            )
            copies = self._parameter_copies(ape, norm, cos, sin)
            try:
                for target, value in zip(
                    (self._ape, self._norm, self._cos, self._sin), copies, strict=True
                ):
                    target.copy_(value)
                self._complete_device()
            except BaseException:
                self._poisoned = True
                raise
            self._version = version
