# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Caller-allocated recent-128 pages with upstream vLLM numerical operators.

Bind scheduler metadata ONCE per step (outside a graph), then share that immutable
snapshot across layers. No allocator, RoPE, quantizer, gather or sparse-index
kernel is implemented here. Physical pages belong to the serving framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rl_engine.contracts.operators.recent128 import (
    CacheRequest,
    CacheVersion,
    Recent128Contract,
    Recent128ContractError,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Recent128ContractError(message)


def _eager_only(device: torch.device) -> None:
    if device.type == "cuda":
        with torch.cuda.device(device):
            _require(
                not torch.cuda.is_current_stream_capturing(),
                "stateful recent128 Graph capture/replay is not validated",
            )


def _snapshot(tensor: torch.Tensor, dtype: torch.dtype, rank: int, name: str):
    _require(isinstance(tensor, torch.Tensor), f"{name} must be a tensor")
    _require(tensor.dtype == dtype and tensor.ndim == rank, f"{name}: wrong dtype/rank")
    _eager_only(tensor.device)
    # Also works when the framework created inference tensors without _version.
    with torch.inference_mode(False):
        return tensor.detach().clone().contiguous()


def _stamp(tensor: torch.Tensor):
    return (tensor.data_ptr(), tensor._version, tuple(tensor.shape), tuple(tensor.stride()))


@dataclass(frozen=True)
class Recent128Batch:
    contract: Recent128Contract
    requests: tuple[CacheRequest, ...]
    version: CacheVersion
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    gather_lens: torch.Tensor
    previous_lengths: tuple[int, ...]
    final_lengths: tuple[int, ...]
    gather_starts: tuple[int, ...]
    num_actual_tokens: int
    max_physical_page: int
    _stamps: tuple = field(repr=False)

    @classmethod
    def from_metadata(
        cls,
        metadata: Any,
        *,
        requests: tuple[CacheRequest, ...],
        version: CacheVersion,
        contract: Recent128Contract,
    ) -> Recent128Batch:
        """Bind the existing vLLM SWA metadata ABI, without importing vLLM.

        This is an eager scheduler boundary: host readback validates ownership,
        exact write slots and chunk retention once, not inside the layer kernels.
        Suffix padding is supported; holes inside a real request are rejected.
        The scheduler must not reclaim/reassign the bound pages until gather is
        finished on the consumer stream. Shared writable/prefix pages require a
        framework COW adapter and are deliberately rejected in this version.
        """
        _require(isinstance(contract, Recent128Contract), "missing cache contract")
        _require(isinstance(version, CacheVersion), "missing cache version barrier")
        requests = tuple(requests)
        _require(all(isinstance(r, CacheRequest) for r in requests), "invalid request handles")
        _require(len(set(requests)) == len(requests), "duplicate request incarnation")
        _require(
            len({r.request_id for r in requests}) == len(requests),
            "multiple incarnations of the same request in one batch",
        )
        _require(metadata.block_size == contract.block_size, "metadata page size mismatch")
        bt = _snapshot(metadata.block_table, torch.int32, 2, "block_table")
        slots = _snapshot(metadata.slot_mapping, torch.int64, 1, "slot_mapping")
        seq = _snapshot(metadata.seq_lens, torch.int32, 1, "seq_lens")
        starts = _snapshot(metadata.query_start_loc, torch.int32, 1, "query_start_loc")
        _require(
            all(t.device == bt.device for t in (slots, seq, starts)),
            "metadata tensors must be on the same device",
        )
        n = len(requests)
        _require(
            n * contract.workspace_width <= 2**31 - 1,
            "flattened workspace indices exceed int32",
        )
        _require(bt.shape[0] == n and seq.numel() == n, "request/metadata row count mismatch")
        _require(starts.numel() == n + 1, "query_start_loc must contain requests+1 entries")
        table, slot_values = bt.cpu().tolist(), slots.cpu().tolist()
        lengths, offsets = seq.cpu().tolist(), starts.cpu().tolist()
        _require(offsets[0] == 0, "query_start_loc must start at zero")
        _require(
            all(a <= b for a, b in zip(offsets, offsets[1:], strict=False)),
            "query_start_loc must be monotonic",
        )
        actual = offsets[-1]
        _require(actual <= len(slot_values), "query extent exceeds slot mapping")
        _require(all(s == -1 for s in slot_values[actual:]), "padding slots must be -1")
        previous, gather, gather_starts = [], [], []
        page_owners: dict[int, tuple[int, int]] = {}
        for row, (end, a, b) in enumerate(zip(lengths, offsets, offsets[1:], strict=False)):
            count = b - a
            prefix = end - count
            _require(prefix >= 0, "sequence length is shorter than its query")
            _require(count <= contract.max_query_tokens, "query exceeds fixed contract capacity")
            previous.append(prefix)
            g = count + min(prefix, 127) if count else 0
            gather.append(g)
            gather_starts.append(end - g)
            if not count:
                continue
            first_page = (end - g) // contract.block_size
            last_page = (end - 1) // contract.block_size
            _require(last_page < len(table[row]), "block table does not cover the query")
            # Check only live logical pages: old pages may legitimately be recycled.
            for logical in range(first_page, last_page + 1):
                physical = table[row][logical]
                _require(physical >= 0, "missing live page (prefix+chunk must be retained)")
                owner = (row, logical)
                _require(
                    physical not in page_owners or page_owners[physical] == owner,
                    "live page alias: unsafe ring overwrite or shared page without COW",
                )
                page_owners[physical] = owner
            for index in range(a, b):
                position = prefix + index - a
                physical = table[row][position // contract.block_size]
                expected = physical * contract.block_size + position % contract.block_size
                _require(slot_values[index] == expected, "slot does not match logical position")
        with torch.inference_mode(False):
            gather_tensor = torch.tensor(gather, device=bt.device, dtype=torch.int32)
        tensors = (bt, slots, seq, starts, gather_tensor)
        return cls(
            contract,
            requests,
            version,
            *tensors,
            tuple(previous),
            tuple(lengths),
            tuple(gather_starts),
            actual,
            max(page_owners, default=-1),
            tuple(_stamp(t) for t in tensors),
        )

    @property
    def device(self) -> torch.device:
        return self.slot_mapping.device

    @property
    def num_tokens(self) -> int:
        return self.slot_mapping.numel()

    def validate_snapshot(self) -> None:
        tensors = (
            self.block_table,
            self.slot_mapping,
            self.seq_lens,
            self.query_start_loc,
            self.gather_lens,
        )
        _require(tuple(_stamp(t) for t in tensors) == self._stamps, "bound metadata was mutated")


@dataclass
class Recent128Workspace:
    """Caller-reusable buffers; valid indices alone define the visible values."""

    kv: torch.Tensor
    indices: torch.Tensor
    lengths: torch.Tensor
    empty_topk: torch.Tensor

    @classmethod
    def allocate(cls, batch: Recent128Batch) -> Recent128Workspace:
        device, c = batch.device, batch.contract
        _eager_only(device)
        return cls(
            torch.empty(
                (len(batch.requests), c.workspace_width, 512),
                dtype=torch.bfloat16,
                device=device,
            ),
            torch.empty((batch.num_tokens, 128), dtype=torch.int32, device=device),
            torch.empty(batch.num_tokens, dtype=torch.int32, device=device),
            torch.empty((batch.num_tokens, 0), dtype=torch.int32, device=device),
        )

    def validate(self, batch: Recent128Batch, cache: torch.Tensor) -> None:
        expected = (
            (self.kv, (len(batch.requests), batch.contract.workspace_width, 512), torch.bfloat16),
            (self.indices, (batch.num_tokens, 128), torch.int32),
            (self.lengths, (batch.num_tokens,), torch.int32),
            (self.empty_topk, (batch.num_tokens, 0), torch.int32),
        )
        storages = set()
        protected = {
            t.untyped_storage().data_ptr()
            for t in (
                cache,
                batch.block_table,
                batch.slot_mapping,
                batch.seq_lens,
                batch.query_start_loc,
                batch.gather_lens,
            )
            if t.numel()
        }
        for tensor, shape, dtype in expected:
            _require(
                tuple(tensor.shape) == shape
                and tensor.dtype == dtype
                and tensor.device == batch.device
                and tensor.is_contiguous()
                and not tensor.requires_grad,
                "workspace shape/dtype/device/stride mismatch",
            )
            if tensor.numel():
                address = tensor.untyped_storage().data_ptr()
                _require(
                    address not in protected and address not in storages,
                    "workspace must not alias cache, metadata or another output",
                )
                storages.add(address)


@dataclass(frozen=True)
class Recent128ReadView:
    """Gathered rows + upstream workspace-relative indices, NOT physical slot IDs."""

    workspace: Recent128Workspace
    batch: Recent128Batch

    @property
    def kv(self):
        return self.workspace.kv

    @property
    def indices(self):
        return self.workspace.indices

    @property
    def lengths(self):
        return self.workspace.lengths


class Recent128Cache:
    """One layer's versioned cache. The framework, not this class, owns its pages."""

    def __init__(
        self,
        cache: torch.Tensor,
        contract: Recent128Contract,
        *,
        layer_id: str,
        backend=None,
    ) -> None:
        _require(isinstance(layer_id, str) and bool(layer_id), "layer_id must be explicit")
        _require(
            cache.dtype == torch.uint8
            and cache.ndim == 2
            and cache.is_contiguous()
            and cache.shape[1] == contract.page_bytes,
            "cache must be contiguous uint8 [pages, aligned DS-MLA page bytes]",
        )
        _eager_only(cache.device)
        if backend is None:
            from rl_engine.backends.rocm.attention.recent128_cache import VllmRecent128Backend

            backend = VllmRecent128Backend()
            _require(cache.device.type == "cuda", "production cache must be on ROCm GPU")
        _require(backend.backend_id == contract.backend_id, "backend does not match contract")
        self.cache, self.contract, self.layer_id, self.backend = cache, contract, layer_id, backend
        self._states: dict[CacheRequest, tuple[int, CacheVersion, bool]] = {}
        self._generations: dict[str, int] = {}

    def register(
        self,
        request: CacheRequest,
        version: CacheVersion,
        *,
        cached_length: int = 0,
    ) -> None:
        """Explicitly adopt a framework-owned cache; default is a new empty request."""
        _require(isinstance(request, CacheRequest), "invalid request incarnation")
        _require(isinstance(version, CacheVersion), "missing cache version")
        _require(type(cached_length) is int and cached_length >= 0, "invalid cached length")
        _require(
            not any(r.request_id == request.request_id for r in self._states),
            "request already registered; release before reusing its ID",
        )
        _require(
            request.generation > self._generations.get(request.request_id, -1),
            "request ID reuse requires a newer generation",
        )
        self._generations[request.request_id] = request.generation
        self._states[request] = (cached_length, version, False)

    def release(self, request: CacheRequest) -> None:
        _require(request in self._states, "unknown request")
        del self._states[request]

    def _prepare(self, batch, workspace, *, replay: bool):
        _eager_only(self.cache.device)
        _require(batch.contract == self.contract, "batch/cache contract mismatch")
        _require(batch.device == self.cache.device, "batch/cache device mismatch")
        batch.validate_snapshot()
        _require(batch.max_physical_page < self.cache.shape[0], "physical page out of range")
        for request, previous, final in zip(
            batch.requests,
            batch.previous_lengths,
            batch.final_lengths,
            strict=False,
        ):
            _require(request in self._states, "request must be explicitly registered")
            length, version, poisoned = self._states[request]
            _require(not poisoned, "cache step failed; release and rebuild this request")
            _require(version == batch.version, "weight/LoRA/scale version barrier mismatch")
            _require(
                length == (final if replay else previous),
                "stale/duplicate step or inconsistent logical length",
            )
        workspace = workspace if workspace is not None else Recent128Workspace.allocate(batch)
        workspace.validate(batch, self.cache)
        return workspace

    def update(
        self,
        kv: torch.Tensor,
        batch: Recent128Batch,
        *,
        workspace: Recent128Workspace | None = None,
    ) -> Recent128ReadView:
        """Write post-RoPE rows and gather all old-prefix+chunk rows before reuse.

        The upstream writer sees suffix -1 slots and ignores extra input padding.
        No host tensor reads or synchronization occur here. Ordering/reclamation
        follows the caller's current stream, exactly as for the native kernels.
        """
        workspace = self._prepare(batch, workspace, replay=False)
        _require(
            kv.dtype == torch.bfloat16
            and kv.ndim == 2
            and kv.shape[1] == 512
            and kv.shape[0] >= batch.num_tokens
            and kv.is_contiguous()
            and kv.device == self.cache.device
            and not kv.requires_grad,
            "inference writer requires contiguous post-RoPE BF16 [tokens,512] without grad",
        )
        if kv.numel():
            _require(
                kv.untyped_storage().data_ptr()
                not in {
                    self.cache.untyped_storage().data_ptr(),
                    workspace.kv.untyped_storage().data_ptr(),
                },
                "input KV must not alias mutable cache/workspace",
            )
        try:
            self.backend.write(kv, self.cache, batch, self.contract)
            self.backend.gather(self.cache, batch, workspace, self.contract)
            self.backend.indices(batch, workspace, self.contract)
        except Exception:
            for request in batch.requests:
                length, version, _ = self._states[request]
                self._states[request] = (length, version, True)
            raise
        for request, final in zip(batch.requests, batch.final_lengths, strict=False):
            self._states[request] = (final, batch.version, False)
        return Recent128ReadView(workspace, batch)

    def replay(
        self,
        batch: Recent128Batch,
        *,
        workspace: Recent128Workspace | None = None,
    ) -> Recent128ReadView:
        """Read-only diagnostic replay. Never increments length or rewrites cache."""
        workspace = self._prepare(batch, workspace, replay=True)
        self.backend.gather(self.cache, batch, workspace, self.contract)
        self.backend.indices(batch, workspace, self.contract)
        return Recent128ReadView(workspace, batch)
