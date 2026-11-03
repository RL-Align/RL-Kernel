# SPDX-License-Identifier: Apache-2.0
"""Independent, request-local quantized ICV with caller-owned global K materialization."""

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType

import torch

from rl_engine.contracts.operators.indexer_icv import (
    IndexerICVContract,
    IndexerICVError,
    source_sha256,
)


def require(ok, message):
    if not ok:
        raise IndexerICVError(message)


def eager(device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            require(not torch.cuda.is_current_stream_capturing(), "Graph capture is not validated")


def stamp(t):
    return (t.data_ptr(), t._version, t.dtype, t.device, tuple(t.shape), tuple(t.stride()))


def tracked(t, *, copy=False):
    require(not t.requires_grad, "item 16 is quantized forward; backward belongs to item 26")
    if copy or t.is_inference():
        with torch.inference_mode(False):
            return t.detach().clone().contiguous()
    return t.detach()


@dataclass(frozen=True)
class QuantizedICVRows:
    """Read-only by contract, not by PyTorch storage enforcement.

    Borrowing requires the caller to forbid *all* writes until consumers finish;
    version stamps only detect writes sharing PyTorch's version counter.
    Snapshot mode isolates the original caller storage, including untracked
    aliases. Neither mode authorizes writes to these bound tensor fields.
    """

    data: torch.Tensor = field(repr=False)
    scales: torch.Tensor = field(repr=False)
    role: str
    version: str
    contract: IndexerICVContract
    ownership: str
    _stamps: tuple = field(repr=False)

    @classmethod
    def bind(cls, data, scales, *, role, version, layout, contract, ownership="borrowed"):
        require(isinstance(contract, IndexerICVContract), "typed ICV contract required")
        require(ownership in ("borrowed", "snapshot"), "ownership must be borrowed or snapshot")
        require(role in ("q", "k"), "role must be q or k")
        require(isinstance(version, str) and bool(version), "Q/K model-scale version required")
        require(layout == contract.layout, "unsupported/swizzled scale or nibble layout")
        require(
            isinstance(data, torch.Tensor) and isinstance(scales, torch.Tensor),
            "independent Q/K scales are required; cannot recover them from fused head weights",
        )
        eager(data.device)
        rank = 3 if role == "q" else 2
        require(data.ndim == rank and scales.ndim == rank, "operand rank mismatch")
        require(
            data.is_contiguous() and scales.is_contiguous(),
            "row-major contiguous operands required",
        )
        require(data.device == scales.device, "data and scales must share a device")
        if role == "q":
            require(data.shape[1] == contract.num_heads, "query head count mismatch")
        # Copy before validating content: admission must examine the same owned
        # bytes later consumed by the backend, not the original caller storage.
        # The caller must still order producer writes before binding/copying.
        if data.is_inference() or scales.is_inference():
            ownership = "snapshot"
        data = tracked(data, copy=ownership == "snapshot")
        scales = tracked(scales, copy=ownership == "snapshot")
        leading = tuple(data.shape[:-1])
        if contract.precision == "mxfp4":
            require(
                data.dtype == torch.uint8 and data.shape[-1] == contract.padded_dim // 2,
                "MXFP4 requires uint8 packed E2M1 with group-32 padded dimension",
            )
            require(
                scales.dtype == torch.uint8
                and tuple(scales.shape) == (*leading, contract.padded_dim // 32),
                "MXFP4 scales must be unshuffled E8M0 bytes, one per 32 values",
            )
            # Existing aiter upcast uses exponent bitcasts. Exclude exponent-zero
            # special handling, BF16 subnormals and overflow rather than silently
            # turning tiny values into zero or clamping large values.
            packed = data.reshape(*leading, contract.padded_dim // 32, 16)
            # E2M1 has a sign bit and three magnitude bits per nibble:
            # 0x0 and 0x8 are both zero. Check BOTH nibbles, ignoring only signs.
            zero_groups = (((packed & 0x07) == 0) & (((packed >> 4) & 0x07) == 0)).all(-1)
            admitted = ((scales >= 2) & (scales <= 252)) | (zero_groups & (scales < 255))
            require(
                bool(admitted.all()),
                "MXFP4 exact-decode requires E8M0 [2,252], or finite-scale all-zero groups",
            )
        else:
            dtype = getattr(torch, "float8_" + contract.fp8_encoding)
            require(
                data.dtype == dtype and data.shape[-1] == contract.head_dim,
                "FP8 dtype/encoding or logical dimension mismatch",
            )
            require(
                scales.dtype == torch.float32 and tuple(scales.shape) == (*leading, 1),
                "FP8 requires a separate FP32 scale for each Q head / K row",
            )
            require(bool(torch.isfinite(data.float()).all()), "non-finite FP8 payload")
            require(
                bool((torch.isfinite(scales) & (scales >= 2.0**-60) & (scales <= 2.0**30)).all()),
                "FP8 scales must be finite positive powers of two within 2**[-60,30]",
            )
            require(
                bool((torch.frexp(scales)[0] == 0.5).all()),
                "FP8 scales must be FP32 power-of-two multipliers, not E8M0 exponent bytes",
            )
        return cls(data, scales, role, version, contract, ownership, (stamp(data), stamp(scales)))

    @property
    def device(self):
        return self.data.device

    def validate(self):
        require(
            (stamp(self.data), stamp(self.scales)) == self._stamps,
            "quantized data/scales changed; rebind the new version",
        )


def bind_ids(ids, rows, device):
    require(
        isinstance(ids, torch.Tensor)
        and ids.dtype == torch.int64
        and tuple(ids.shape) == (rows, 2)
        and ids.device == device,
        "logical IDs must be int64 [rows,2] on the operand device",
    )
    snapshot = tracked(ids, copy=True)
    host = tuple(tuple(row) for row in snapshot.cpu().tolist())
    require(
        all(row == (-1, -1) or (row[0] >= 0 and row[1] >= 0) for row in host),
        "IDs must be (request,global_row) or (-1,-1) padding",
    )
    active = [row for row in host if row != (-1, -1)]
    require(len(set(active)) == len(active), "duplicate logical row ID")
    return host


@dataclass(frozen=True)
class ICVQueryRequest:
    request_id: int
    positions: tuple[int, ...]
    physical_rows: tuple[int, ...]
    selector: torch.Tensor = field(repr=False)
    selector_stamp: tuple = field(repr=False)


@dataclass(frozen=True)
class BoundICVQueries:
    operand: QuantizedICVRows
    requests: tuple[ICVQueryRequest, ...]

    @classmethod
    def bind(cls, operand, query_ids):
        require(operand.role == "q", "expected a Q operand")
        operand.validate()
        host = bind_ids(query_ids, operand.data.shape[0], operand.device)
        groups = {}
        for row, (request, position) in enumerate(host):
            if request >= 0:
                groups.setdefault(request, []).append((row, position))
        requests = []
        for request, entries in sorted(groups.items()):
            physical, positions = zip(*entries, strict=True)
            selector = tracked(
                torch.tensor(physical, device=operand.device, dtype=torch.int64), copy=True
            )
            requests.append(
                ICVQueryRequest(request, positions, physical, selector, stamp(selector))
            )
        return cls(operand, tuple(requests))

    def validate(self):
        self.operand.validate()
        for request in self.requests:
            require(stamp(request.selector) == request.selector_stamp, "query selector was mutated")


@dataclass(frozen=True)
class ICVKeyRequest:
    request_id: int
    rows: int
    values: torch.Tensor = field(repr=False)
    scales: torch.Tensor | None = field(repr=False)
    _stamps: tuple = field(repr=False)

    def validate(self):
        require(
            (stamp(self.values), None if self.scales is None else stamp(self.scales))
            == self._stamps,
            "prepared key workspace was mutated",
        )


@dataclass(frozen=True)
class PreparedICVKeys:
    operand: QuantizedICVRows
    requests: object = field(repr=False)
    cp_size: int
    implementation: str

    def validate(self):
        self.operand.validate()
        for entry in self.requests.values():
            entry.validate()


@dataclass(frozen=True)
class ICVTile:
    request_id: int
    query_positions: tuple[int, ...]
    query_physical_rows: tuple[int, ...]
    key_start: int
    values: torch.Tensor
    contract_fingerprint: str
    implementation_fingerprint: str

    @property
    def key_end(self):
        return self.key_start + self.values.shape[-1]


class IndexerICV:
    def __init__(self, contract, *, backend=None, expected_implementation=None):
        require(isinstance(contract, IndexerICVContract), "typed ICV contract required")
        if backend is None:
            from rl_engine.backends.rocm.attention.indexer_icv import AiterIndexerICVBackend

            backend = AiterIndexerICVBackend()
        require(backend.backend_id == contract.backend_id, "backend identity mismatch")
        self.contract, self.backend = contract, backend
        self.contract_fingerprint = contract.fingerprint
        identity = {
            "contract": contract.fingerprint,
            "backend": getattr(backend, "provenance", {"test_backend": type(backend).__name__}),
            "model_source": source_sha256(__file__, label=__name__),
            "plan": backend.plan(contract),
        }
        self.implementation_fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode(),
        ).hexdigest()
        if expected_implementation is not None:
            require(
                expected_implementation == self.implementation_fingerprint,
                "train/rollout implementation fingerprint mismatch",
            )

    def _check_operand(self, operand):
        require(operand.contract == self.contract, "operand contract mismatch")
        require(operand.device.type == self.backend.device_type, "operand/backend device mismatch")
        if hasattr(self.backend, "device"):
            require(operand.device == self.backend.device, "bind a backend on this device")
        eager(operand.device)
        operand.validate()

    def prepare_keys(self, operand, key_ids, *, request_key_counts, key_owners=None, cp_size=1):
        """Consume COMPLETE keys after native transport, never rank-local K alone.

        request_key_counts is the scheduler's global compressed-row count per
        request. CP owners must accompany a CP>1 materialization. No collective
        is implemented here, and no claim of live distributed CP is implied.
        """
        self._check_operand(operand)
        require(operand.role == "k", "expected a K operand")
        require(type(cp_size) is int and cp_size > 0, "invalid CP size")
        counts = dict(request_key_counts)
        require(
            all(type(r) is int and r >= 0 and type(n) is int and n >= 0 for r, n in counts.items()),
            "invalid global request key counts",
        )
        host = bind_ids(key_ids, operand.data.shape[0], operand.device)
        if key_owners is None:
            require(cp_size == 1, "CP ownership metadata is required for CP>1")
            owners = (0,) * len(host)
        else:
            require(
                isinstance(key_owners, torch.Tensor)
                and key_owners.dtype == torch.int64
                and tuple(key_owners.shape) == (len(host),)
                and key_owners.device == operand.device,
                "key owners must be int64 [key_rows] on the operand device",
            )
            owners = tuple(key_owners.cpu().tolist())
        groups = {r: [] for r in counts}
        for row, ((request, logical), owner) in enumerate(zip(host, owners, strict=True)):
            if request < 0:
                continue
            require(0 <= owner < cp_size, "key owner outside CP group")
            require(request in counts, "key belongs to an undeclared request")
            groups[request].append((logical, row))
        for request, entries in groups.items():
            entries.sort()
            require(
                len(entries) == counts[request]
                and all(logical == i for i, (logical, _) in enumerate(entries)),
                "incomplete/incorrect global K: native CP transport must precede ICV",
            )
        bytes_per_row = (
            self.contract.head_dim * 2
            if self.contract.precision == "mxfp4"
            else self.contract.head_dim + 4
        )
        require(
            sum(counts.values()) * bytes_per_row <= self.contract.max_prepared_key_bytes,
            "prepared K exceeds the contract byte budget",
        )
        prepared = {}
        # Do not decode irrelevant padded rows; select each request's canonical K.
        for request, entries in sorted(groups.items()):
            selector = torch.tensor(
                [row for _, row in entries], dtype=torch.int64, device=operand.device
            )
            data = operand.data.index_select(0, selector)
            scales = operand.scales.index_select(0, selector)
            values, scales = self.backend.prepare(data, scales, self.contract)
            values = tracked(values)
            scales = None if scales is None else tracked(scales)
            prepared[request] = ICVKeyRequest(
                request,
                len(entries),
                values,
                scales,
                (stamp(values), None if scales is None else stamp(scales)),
            )
        return PreparedICVKeys(
            operand,
            MappingProxyType(prepared),
            cp_size,
            self.implementation_fingerprint,
        )

    def _check_call(self, queries, keys):
        self._check_operand(queries.operand)
        self._check_operand(keys.operand)
        queries.validate()
        keys.validate()
        require(keys.implementation == self.implementation_fingerprint, "prepared backend mismatch")
        require(keys.operand.version == queries.operand.version, "Q/K model-scale version mismatch")
        require(queries.operand.device == keys.operand.device, "Q/K device mismatch")
        require(
            all(r.request_id in keys.requests for r in queries.requests),
            "query has no declared global K",
        )

    def _tiles(self, queries, keys, *, full_key_span=False):
        c = self.contract
        for request in queries.requests:
            key = keys.requests[request.request_id]
            for begin in range(0, len(request.positions), c.query_tile_rows):
                queries.validate()
                keys.validate()
                end = begin + c.query_tile_rows
                selector = request.selector[begin:end]
                q, scale = self.backend.prepare(
                    queries.operand.data.index_select(0, selector),
                    queries.operand.scales.index_select(0, selector),
                    c,
                )
                width = max(key.rows, 1) if full_key_span else c.key_tile_rows
                for start in range(0, key.rows, width):
                    queries.operand.validate()
                    keys.validate()
                    stop = start + width
                    out = self.backend.dot(
                        q,
                        key.values[start:stop],
                        scale,
                        None if key.scales is None else key.scales[start:stop],
                        c,
                    )
                    require(
                        out.dtype == torch.float32
                        and out.device == queries.operand.device
                        and tuple(out.shape)
                        == (
                            len(request.positions[begin:end]),
                            c.num_heads,
                            min(stop, key.rows) - start,
                        )
                        and not out.requires_grad,
                        "ICV backend must produce raw FP32 [query,head,K] on the input device",
                    )
                    yield ICVTile(
                        request.request_id,
                        request.positions[begin:end],
                        request.physical_rows[begin:end],
                        start,
                        out,
                        self.contract_fingerprint,
                        self.implementation_fingerprint,
                    )

    def iter_tiles(self, queries, keys):
        """Bounded raw ICV stream. The caller owns tile retention and postprocessing."""
        self._check_call(queries, keys)
        return self._tiles(queries, keys)

    def materialize(self, queries, keys):
        """Debug/small-request convenience. Large requests must use iter_tiles."""
        self._check_call(queries, keys)
        c = self.contract
        size = sum(
            len(r.positions) * c.num_heads * keys.requests[r.request_id].rows * 4
            for r in queries.requests
        )
        require(
            size <= c.max_output_bytes, "raw ICV exceeds materialization budget; use iter_tiles"
        )
        outputs = {}
        for request in queries.requests:
            n = keys.requests[request.request_id].rows
            # A fitting request can return its one upstream GEMM output directly.
            # Longer requests still bound query decode workspace by query_tile_rows.
            outputs[request.request_id] = (
                None
                if (n and len(request.positions) <= c.query_tile_rows)
                else torch.empty(
                    (len(request.positions), c.num_heads, n),
                    dtype=torch.float32,
                    device=queries.operand.device,
                )
            )
        offsets = {r.request_id: 0 for r in queries.requests}
        for tile in self._tiles(queries, keys, full_key_span=True):
            row = offsets[tile.request_id]
            if outputs[tile.request_id] is None:
                outputs[tile.request_id] = tile.values
            else:
                outputs[tile.request_id][row : row + len(tile.query_positions)] = tile.values
            offsets[tile.request_id] += len(tile.query_positions)
        return tuple(
            ICVTile(
                r.request_id,
                r.positions,
                r.physical_rows,
                0,
                outputs[r.request_id],
                self.contract_fingerprint,
                self.implementation_fingerprint,
            )
            for r in queries.requests
        )
