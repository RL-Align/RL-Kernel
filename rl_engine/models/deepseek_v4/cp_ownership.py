# SPDX-License-Identifier: Apache-2.0
"""Contiguous DSV4 CP ownership and deterministic KV/dKV exchange."""

from dataclasses import dataclass
import hashlib
import json

import torch
import torch.distributed as dist

from rl_engine.distributed.transports.rccl import (
    RCCLDeterministicCollective,
    TorchDistributedDeterministicCollective,
)


class CpOwnershipError(RuntimeError):
    pass


def _require(ok, message):
    if not ok:
        raise CpOwnershipError(message)


@dataclass(frozen=True)
class Dsv4CpPlan:
    """Fixed contiguous CP shards of one request's global token positions."""

    request_id: int
    total_tokens: int
    ranks: tuple[int, ...]
    compression_ratio: int
    max_rows_per_rank: int
    model_version: str

    def __post_init__(self):
        _require(
            type(self.request_id) is int and self.request_id >= 0, "nonnegative request ID required"
        )
        _require(
            type(self.total_tokens) is int and self.total_tokens > 0,
            "positive global sequence required",
        )
        _require(
            isinstance(self.ranks, tuple)
            and len(self.ranks) in (1, 2, 4, 8)
            and all(type(r) is int and r >= 0 for r in self.ranks)
            and tuple(sorted(set(self.ranks))) == self.ranks,
            "supported sorted CP ranks required",
        )
        _require(
            type(self.compression_ratio) is int and self.compression_ratio > 0,
            "positive compression ratio required",
        )
        _require(
            type(self.max_rows_per_rank) is int and self.max_rows_per_rank > 0,
            "positive fixed KV row capacity required",
        )
        _require(
            isinstance(self.model_version, str) and bool(self.model_version),
            "model version required",
        )

    @property
    def fingerprint(self):
        payload = (
            self.request_id,
            self.total_tokens,
            self.ranks,
            self.compression_ratio,
            self.max_rows_per_rank,
            self.model_version,
        )
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    def bounds(self, rank):
        _require(rank in self.ranks, "rank outside CP plan")
        index = self.ranks.index(rank)
        width, extra = divmod(self.total_tokens, len(self.ranks))
        start = index * width + min(index, extra)
        return start, start + width + (index < extra)

    def owner(self, key):
        _require(len(key) == 3, "KV key must be [request, kind, global position]")
        request, kind, position = key
        _require(
            request == self.request_id
            and kind in (0, 1)
            and type(position) is int
            and 0 <= position < self.total_tokens,
            "invalid global KV key",
        )
        if kind == 0:
            _require(
                (position + 1) % self.compression_ratio == 0,
                "compressed row must end a complete global group",
            )
        return next(
            rank for rank in self.ranks if self.bounds(rank)[0] <= position < self.bounds(rank)[1]
        )

    def query_keys(self, rank, *, device):
        start, end = self.bounds(rank)
        return torch.tensor(
            [(self.request_id, pos) for pos in range(start, end)], dtype=torch.int64, device=device
        ).reshape(-1, 2)


class Dsv4CpKvExchange:
    """Rank-ordered padded KV gather and fixed-tree dKV return to row owners.

    The caller computes complete C128 rows before exchange, supplies keys in
    local order, and selects sparse/window rows from the returned global order.
    The values and keys are never interpreted from local CP offsets.
    """

    def __init__(
        self, plan: Dsv4CpPlan, *, group, allow_cpu_reference=False, max_size_bytes=64 * 1024 * 1024
    ):
        _require(isinstance(plan, Dsv4CpPlan), "CP plan required")
        _require(dist.is_available() and dist.is_initialized(), "process group required")
        _require(
            tuple(dist.get_process_group_ranks(group)) == plan.ranks, "CP group differs from plan"
        )
        backend = str(dist.get_backend(group)).lower()
        rocm = "nccl" in backend and bool(torch.version.hip)
        _require(rocm or ("gloo" in backend and allow_cpu_reference), "unvalidated CP transport")
        self.device = (
            torch.device("cuda", torch.cuda.current_device()) if rocm else torch.device("cpu")
        )
        self.plan, self.group = plan, group
        self.rank = dist.get_rank()
        self.local_index = plan.ranks.index(self.rank)
        self._phase = "gather"
        self._dtype = None
        self._local_keys = None
        self._global_keys = None
        self._agree(("construct", plan.fingerprint), True)
        cls = RCCLDeterministicCollective if rocm else TorchDistributedDeterministicCollective
        self.collective = cls(group=group, device=self.device, max_size_bytes=max_size_bytes)

    def _agree(self, signature, valid):
        peers = [None] * len(self.plan.ranks)
        dist.all_gather_object(peers, (signature, valid), group=self.group)
        _require(all(peer[0] == signature for peer in peers), f"CP version/phase mismatch: {peers}")
        _require(all(peer[1] for peer in peers), "invalid CP ownership or layout on a rank")

    def gather_kv(self, keys, values, *, model_version):
        """Return `(global_keys, global_values)` sorted by logical KV key."""
        try:
            valid = (
                self._phase == "gather"
                and model_version == self.plan.model_version
                and isinstance(keys, torch.Tensor)
                and isinstance(values, torch.Tensor)
                and keys.ndim == 2
                and keys.shape[1] == 3
                and keys.dtype == torch.int64
                and values.ndim == 2
                and values.shape[0] == keys.shape[0]
                and values.dtype in (torch.bfloat16, torch.float16, torch.float32)
                and keys.device == values.device == self.device
                and keys.is_contiguous()
                and values.is_contiguous()
                and keys.shape[0] <= self.plan.max_rows_per_rank
            )
            local_keys = tuple(tuple(map(int, row)) for row in keys.tolist()) if valid else ()
            valid = valid and len(set(local_keys)) == len(local_keys)
            if valid:
                try:
                    valid = all(self.plan.owner(key) == self.rank for key in local_keys)
                except CpOwnershipError:
                    valid = False
            self._agree(
                (
                    "gather",
                    model_version,
                    str(values.dtype) if valid else None,
                    values.shape[1] if valid else None,
                ),
                valid,
            )
            capacity = self.plan.max_rows_per_rank
            packed_keys = torch.full((capacity, 3), -1, device=self.device, dtype=torch.int64)
            packed_values = values.new_zeros((capacity, values.shape[1]))
            packed_keys[: len(local_keys)].copy_(keys)
            packed_values[: len(local_keys)].copy_(values)
            all_keys = self.collective.all_gather(packed_keys).reshape(
                len(self.plan.ranks), capacity, 3
            )
            all_values = self.collective.all_gather(packed_values).reshape(
                len(self.plan.ranks), capacity, values.shape[1]
            )
            raw_keys = all_keys.cpu().tolist()
            entries = []
            for index, rank_keys in enumerate(raw_keys):
                seen_padding = False
                for row, key in enumerate(rank_keys):
                    if key == [-1, -1, -1]:
                        seen_padding = True
                    else:
                        _require(
                            not seen_padding
                            and self.plan.owner(tuple(key)) == self.plan.ranks[index],
                            "stale or misplaced compressed/window row",
                        )
                        entries.append((tuple(key), index, row))
            _require(
                len({key for key, _, _ in entries}) == len(entries), "duplicate global KV identity"
            )
            entries.sort(key=lambda item: item[0])
            positions = torch.tensor(
                [index * capacity + row for _, index, row in entries],
                device=self.device,
                dtype=torch.long,
            )
            self._global_keys = tuple(key for key, _, _ in entries)
            self._local_keys = local_keys
            self._dtype = values.dtype
            self._hidden_size = values.shape[1]
            result_keys = all_keys.reshape(-1, 3).index_select(0, positions)
            result_values = all_values.reshape(-1, values.shape[1]).index_select(0, positions)
        except Exception:
            self._phase = "poisoned"
            raise
        self._phase = "return"
        return result_keys, result_values

    def return_dkv(self, global_dkv, *, model_version):
        """Reduce globally in fixed rank order, then extract local owned rows."""
        try:
            valid = (
                self._phase == "return"
                and model_version == self.plan.model_version
                and isinstance(global_dkv, torch.Tensor)
                and tuple(global_dkv.shape) == (len(self._global_keys), self._hidden_size)
                and global_dkv.dtype == self._dtype
                and global_dkv.device == self.device
                and global_dkv.is_contiguous()
            )
            self._agree(
                (
                    "return",
                    model_version,
                    str(global_dkv.dtype) if isinstance(global_dkv, torch.Tensor) else None,
                ),
                valid,
            )
            reduced = (
                self.collective.all_reduce(global_dkv) if global_dkv.numel() else global_dkv.clone()
            )
            lookup = {key: index for index, key in enumerate(self._global_keys)}
            indices = torch.tensor(
                [lookup[key] for key in self._local_keys], device=self.device, dtype=torch.long
            )
            result = reduced.index_select(0, indices)
        except Exception:
            self._phase = "poisoned"
            raise
        self._phase = "used"
        return result

    def close(self):
        self._phase = "poisoned"
        self.collective.close()
