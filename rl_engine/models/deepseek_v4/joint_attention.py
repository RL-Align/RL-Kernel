# SPDX-License-Identifier: Apache-2.0
"""Joint forward records and owned graph buffers. Autograd is separate item 25."""

from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import threading

import torch

from rl_engine.contracts.operators.joint_attention import JointAttentionContract, require


def stamp(tensor):
    require(not tensor.is_inference(), "versioned ordinary tensors are required")
    return (
        tensor.data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def eager(device):
    if device.type == "cuda":
        with torch.cuda.device(device):
            require(
                not torch.cuda.is_current_stream_capturing(),
                "use the plan's owned capture/replay path",
            )


def owned(tensor):
    with torch.inference_mode(False), torch.no_grad():
        return tensor.detach().clone().contiguous()


@dataclass(frozen=True)
class JointMetadata:
    query_keys: torch.Tensor = field(repr=False)
    kv_keys: torch.Tensor = field(repr=False)
    permutation: torch.Tensor = field(repr=False)
    indices: torch.Tensor = field(repr=False)
    stamps: tuple = field(repr=False)
    version: str
    contract_fingerprint: str

    @classmethod
    def bind(cls, contract, query_keys, kv_keys, compressed_indices, *, version):
        """KV keys: [request, kind(0=compressed,1=window), global_end_position].

        compressed_indices address the caller's original KV row order. We own
        a canonical KV order and append the *complete* recent window ourselves.
        All host readback/causal/owner validation occurs at this admission point.
        """
        require(isinstance(contract, JointAttentionContract), "missing joint contract")
        require(isinstance(version, str) and bool(version), "model/cache version required")
        require(
            query_keys.ndim == 2 and query_keys.shape[1] == 2 and query_keys.dtype == torch.int64,
            "query keys must be int64 [M,2]",
        )
        require(
            kv_keys.ndim == 2 and kv_keys.shape[1] == 3 and kv_keys.dtype == torch.int64,
            "KV keys must be int64 [N,3]",
        )
        require(
            query_keys.shape[0] <= contract.max_queries and kv_keys.shape[0] <= contract.max_kv,
            "metadata capacity exceeded",
        )
        require(
            compressed_indices.shape == (query_keys.shape[0], contract.compressed_topk)
            and compressed_indices.dtype in (torch.int32, torch.int64),
            "compressed Top-K shape/dtype mismatch",
        )
        device = query_keys.device
        require(kv_keys.device == compressed_indices.device == device, "metadata device mismatch")
        eager(device)
        qh, kh, topk = (
            value.detach().cpu().tolist() for value in (query_keys, kv_keys, compressed_indices)
        )
        require(
            all((r >= 0 and p >= 0) or (r == -1 and p == -1) for r, p in qh),
            "invalid query identity/padding",
        )
        active = [tuple(row) for row in qh if row[0] >= 0]
        require(len(active) == len(set(active)), "duplicate query identity")
        require(
            all(r >= 0 and kind in (0, 1) and pos >= 0 for r, kind, pos in kh),
            "invalid logical KV identity",
        )
        require(len(kh) == len(set(map(tuple, kh))), "duplicate logical KV identity")
        require(
            all(kind != 0 or (pos + 1) % contract.compression_ratio == 0 for _, kind, pos in kh),
            "compressed row is not a complete global group",
        )
        order = sorted(range(len(kh)), key=lambda i: kh[i])
        inverse = {old: new for new, old in enumerate(order)}
        window = {(r, p): i for i, (r, kind, p) in enumerate(kh) if kind == 1}
        selected = []
        for (request, position), chosen in zip(qh, topk, strict=True):
            require(all(i == -1 or 0 <= i < len(kh) for i in chosen), "Top-K index out of bounds")
            valid = [i for i in chosen if i >= 0]
            require(len(valid) == len(set(valid)), "duplicate compressed selection")
            if request < 0:
                require(not valid, "padding query cannot select KV")
                result = []
            else:
                require(
                    all(
                        kh[i][0] == request and kh[i][1] == 0 and kh[i][2] <= position
                        for i in valid
                    ),
                    "cross-request, future or noncompressed Top-K row",
                )
                valid.sort(key=lambda i: kh[i])
                recent = range(max(0, position - contract.window_size + 1), position + 1)
                require(all((request, p) in window for p in recent), "recent window is incomplete")
                result = [inverse[i] for i in valid] + [inverse[window[request, p]] for p in recent]
            selected.append(result + [-1] * (contract.selected_capacity - len(result)))
        with torch.inference_mode(False):
            qkeys = owned(query_keys)
            permutation = torch.tensor(order, dtype=torch.int64, device=device)
            keys = owned(kv_keys.index_select(0, permutation))
            indices = torch.tensor(selected, dtype=torch.int32, device=device).reshape(
                query_keys.shape[0], contract.selected_capacity
            )
        tensors = (qkeys, keys, permutation, indices)
        return cls(*tensors, tuple(stamp(t) for t in tensors), version, contract.fingerprint)

    def validate(self, contract, m, n, device, version):
        require(
            self.contract_fingerprint == contract.fingerprint and self.version == version,
            "contract/model/cache version mismatch",
        )
        require(
            self.query_keys.shape == (m, 2)
            and self.kv_keys.shape == (n, 3)
            and self.indices.shape == (m, contract.selected_capacity),
            "logical input geometry mismatch",
        )
        require(
            all(
                t.device == device
                for t in (self.query_keys, self.kv_keys, self.permutation, self.indices)
            ),
            "logical input device mismatch",
        )
        require(
            tuple(stamp(t) for t in (self.query_keys, self.kv_keys, self.permutation, self.indices))
            == self.stamps,
            "bound joint metadata was mutated",
        )


@dataclass(frozen=True)
class JointForwardRecord:
    output: torch.Tensor
    lse: torch.Tensor
    q: torch.Tensor = field(repr=False)
    kv: torch.Tensor = field(repr=False)
    sink: torch.Tensor = field(repr=False)
    indices: torch.Tensor = field(repr=False)
    metadata: JointMetadata
    implementation: str
    stamps: tuple = field(repr=False)

    def validate_saved(self):
        tensors = (self.output, self.lse, self.q, self.kv, self.sink, self.indices)
        require(tuple(stamp(t) for t in tensors) == self.stamps, "saved forward record was mutated")
        require(
            tuple(
                stamp(t)
                for t in (
                    self.metadata.query_keys,
                    self.metadata.kv_keys,
                    self.metadata.permutation,
                    self.metadata.indices,
                )
            )
            == self.metadata.stamps,
            "saved metadata was mutated",
        )


class JointAttentionPlan:
    """Owned, fixed-capacity buffers; forward records never alias replay storage.

    forward_record is deliberately not an autograd Function. It saves the exact
    kernel inputs/O/LSE for item 25, without claiming a working backward here.
    """

    def __init__(self, contract, *, backend=None, expected_implementation=None):
        require(isinstance(contract, JointAttentionContract), "missing joint contract")
        if backend is None:
            from rl_engine.backends.rocm.attention.joint_attention import AiterJointAttentionBackend

            backend = AiterJointAttentionBackend()
        require(backend.profile == contract.profile, "provider arithmetic mismatch")
        self.contract, self.backend = contract, backend
        self.device = backend.device
        eager(self.device)
        self._lock, self._event, self._graph = threading.RLock(), None, None
        self._poisoned = False
        c = contract
        with torch.inference_mode(False):
            self._q = torch.zeros(
                c.max_queries, c.num_heads, c.head_dim, dtype=torch.bfloat16, device=self.device
            )
            self._kv = torch.zeros(c.max_kv, c.head_dim, dtype=torch.bfloat16, device=self.device)
            self._sink = torch.zeros(c.num_heads, dtype=torch.float32, device=self.device)
            self._indices = torch.full(
                (c.max_queries, c.selected_capacity), -1, dtype=torch.int32, device=self.device
            )
            self._active = torch.zeros(c.max_queries, dtype=torch.bool, device=self.device)
            self._out = torch.empty_like(self._q)
            self._lse = torch.empty(
                (c.max_queries, c.num_heads), dtype=torch.float32, device=self.device
            )
        identity = {
            "contract": c.fingerprint,
            "backend": backend.provenance,
            "bridge_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self.implementation = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        require(
            expected_implementation is None or expected_implementation == self.implementation,
            "train/rollout arithmetic handshake mismatch",
        )
        if self.device.type == "cuda":
            self._event = torch.cuda.Event()
            self._event.record(torch.cuda.current_stream(self.device))

    def _wait(self):
        if self._event is not None:
            torch.cuda.current_stream(self.device).wait_event(self._event)

    def _record(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self._event = event

    @contextmanager
    def _transaction(self):
        device_context = (
            torch.cuda.device(self.device) if self.device.type == "cuda" else nullcontext()
        )
        with self._lock, device_context:
            require(not self._poisoned, "failed joint workspace is poisoned; create a new plan")
            try:
                self._wait()
                yield
                self._record()
            except BaseException:
                # A failed mutable launch cannot be safely retried using the
                # same graph/workspace, even if an old result still exists.
                self._poisoned = True
                raise

    def _launch(self):
        self.backend.forward_into(
            self._q, self._kv, self._sink, self._indices, self._out, self._lse, self.contract
        )
        self._out.masked_fill_(~self._active[:, None, None], 0)
        self._lse.masked_fill_(~self._active[:, None], float("-inf"))

    def capture(self):
        require(self.contract.graph and self.device.type == "cuda", "Graph profile is disabled")
        eager(self.device)
        with self._transaction(), torch.no_grad():
            for _ in range(3):
                self._launch()
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._launch()
            self._graph = graph

    @torch.no_grad()
    def forward_record(self, q, kv, sink, *, metadata, version, mode, use_graph=False):
        require(mode in ("train", "recompute", "prefill", "decode"), "unknown runtime mode")
        require(isinstance(metadata, JointMetadata), "bound logical metadata required")
        c = self.contract
        require(
            q.ndim == 3
            and q.shape[1:] == (c.num_heads, c.head_dim)
            and q.shape[0] <= c.max_queries
            and q.dtype == torch.bfloat16
            and q.device == self.device
            and q.is_contiguous(),
            "Q ABI mismatch",
        )
        require(
            kv.ndim == 2
            and kv.shape[1] == c.head_dim
            and kv.shape[0] <= c.max_kv
            and kv.dtype == torch.bfloat16
            and kv.device == self.device
            and kv.is_contiguous(),
            "KV ABI mismatch",
        )
        require(
            sink.shape == (c.num_heads,)
            and sink.dtype == torch.float32
            and sink.device == self.device
            and sink.is_contiguous(),
            "sink ABI mismatch",
        )
        eager(self.device)
        metadata.validate(c, q.shape[0], kv.shape[0], self.device, version)
        # These are admission checks, not operations inside the captured graph.
        require(bool(torch.isfinite(q).all() & torch.isfinite(kv).all()), "Q/KV must be finite")
        require(
            bool((torch.isfinite(sink) | torch.isneginf(sink)).all()),
            "sink may be finite or negative infinity, not NaN/+infinity",
        )
        require(not use_graph or self._graph is not None, "capture this plan before Graph replay")
        # A record owns snapshots, including in an enclosing inference_mode().
        with self._transaction(), torch.inference_mode(False), torch.no_grad():
            self._q.zero_()
            self._kv.zero_()
            self._indices.fill_(-1)
            self._active.zero_()
            m, n = q.shape[0], kv.shape[0]
            self._q[:m].copy_(q)
            self._kv[:n].copy_(kv.index_select(0, metadata.permutation))
            self._sink.copy_(sink)
            self._indices[:m].copy_(metadata.indices)
            self._active[:m].copy_(metadata.query_keys[:, 0] >= 0)
            if use_graph:
                self._graph.replay()
            else:
                self._launch()
            tensors = tuple(
                owned(t)
                for t in (
                    self._out[:m],
                    self._lse[:m],
                    self._q[:m],
                    self._kv[:n],
                    self._sink,
                    self._indices[:m],
                )
            )
            # The event is after every returned clone. A later stream must wait
            # before reusing graph storage; no global synchronize on each replay.
        return JointForwardRecord(
            *tensors, metadata, self.implementation, tuple(stamp(t) for t in tensors)
        )


@torch.no_grad()
def gather_packed_joint_kv(cache, slots, *, block_size, encoding, backend, max_bytes=1 << 30):
    """Read selected physical rows using vLLM's real DSv4 FP8 cache decoder.

    Slots must match the metadata's original KV row order. Caller owns cache
    writer/reader synchronization; this creates a detached BF16 snapshot.
    Native decoding visits only referenced pages, not the entire cache pool.
    """
    require(block_size in (64, 128) and encoding in ("ocp", "fnuz"), "unknown packed cache ABI")
    require(type(max_bytes) is int and max_bytes > 0, "packed gather budget must be positive")
    require(
        cache.ndim == 2
        and cache.dtype == torch.uint8
        and cache.is_contiguous()
        and cache.shape[1] >= block_size * 584
        and cache.stride(0) % 16 == 0
        and cache.device == backend.device,
        "cache must be contiguous uint8 [pages, block_bytes]",
    )
    require(
        slots.ndim == 1 and slots.dtype == torch.int64 and slots.device == cache.device,
        "physical slots must be int64 [rows] on the cache device",
    )
    eager(cache.device)
    require(
        bool(((slots >= 0) & (slots < cache.shape[0] * block_size)).all()),
        "physical slot outside allocated cache",
    )
    if slots.numel() == 0:
        return torch.empty((0, 512), dtype=torch.bfloat16, device=cache.device)
    pages, inverse = torch.unique(slots // block_size, sorted=True, return_inverse=True)
    needed = pages.numel() * block_size * 512 * 2 + slots.numel() * 512 * 2
    require(needed <= max_bytes, "packed gather workspace budget exceeded")
    out = torch.empty((pages.numel(), block_size, 512), dtype=torch.bfloat16, device=cache.device)
    lengths = torch.full((pages.numel(),), block_size, dtype=torch.int32, device=cache.device)
    table = pages.int().view(-1, 1).contiguous()
    with torch.cuda.device(cache.device):
        backend.gather_packed_pages(cache, table, block_size, out, lengths, encoding == "fnuz")
    return out.flatten(0, 1).index_select(0, inverse * block_size + slots % block_size)
