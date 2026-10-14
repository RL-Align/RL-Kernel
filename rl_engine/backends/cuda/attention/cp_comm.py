# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CP/TP Attention communication and a P2P NCCL reference.

PR7 evaluates fused attention backends for the #235 target
``Qwen3-8B, TP=2, CP=2, BF16``. The self-owned CUDA communication operators are
AG/RS and compute-communication decoupled. This module adapts the deterministic
collectives from PR311/PR312 to the Attention partial-state contract.

The strict production path uses one arithmetic graph at every CP size:

```text
owner-local Q/K/V and position IDs
  -> deterministic AG(Q/K/V/positions)
  -> shared no-Split-K CUDA Attention core on full logical Q/K/V
  -> deterministic RS(Out, LSE)
  -> owner-local query result
```

The older partial-state path remains available as a reference interface:

```text
owner-local deterministic fallback or TE attention over rank-owned KV blocks
  -> AttentionCPPartialState(out, lse, global_block_index, tp/cp rank metadata)
  -> custom CUDA AG communication operator
  -> sort by global_block_index
  -> PR3 FP32 online-softmax merge
  -> custom CUDA RS communication operator
```
The CUDA backend uses the self-owned deterministic CUDA collectives when PR311 /
PR312 are present.  It keeps the same manifest and FP32 merge contract as the
P2P NCCL reference, and fails closed when those compiled operators are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

import torch

CPCommunicationBackend = Literal[
    "cuda_ag_rs",
    "rccl_ag_rs",
    "p2p_nccl_reference",
    "local_debug",
]
CPCommunicationStatus = Literal["interface_only", "implemented"]


class AttentionCPCommunicationUnavailable(RuntimeError):
    """Raised when a requested CP communication backend is not implemented."""


@dataclass(frozen=True)
class AttentionParallelSpec:
    """TP/CP identity carried by PR7 attention backend reports."""

    tp_world_size: int = 2
    tp_rank: int = 0
    cp_world_size: int = 2
    cp_rank: int = 0

    def validate(self) -> None:
        _positive_int(self.tp_world_size, "tp_world_size")
        _positive_int(self.cp_world_size, "cp_world_size")
        _rank_in_world(self.tp_rank, self.tp_world_size, "tp_rank")
        _rank_in_world(self.cp_rank, self.cp_world_size, "cp_rank")

    def provenance(self) -> dict[str, int]:
        self.validate()
        return {
            "tp_world_size": int(self.tp_world_size),
            "tp_rank": int(self.tp_rank),
            "cp_world_size": int(self.cp_world_size),
            "cp_rank": int(self.cp_rank),
        }


@dataclass(frozen=True)
class AttentionCPBlockMetadata:
    """Logical identity for one attention partial state."""

    global_block_index: int
    kv_block_start: int
    kv_block_end: int
    owner_cp_rank: int
    owner_tp_rank: int

    def validate(self, parallel: AttentionParallelSpec) -> None:
        parallel.validate()
        if (
            isinstance(self.global_block_index, bool)
            or not isinstance(self.global_block_index, int)
            or self.global_block_index < 0
        ):
            raise ValueError("global_block_index must be non-negative")
        if (
            isinstance(self.kv_block_start, bool)
            or isinstance(self.kv_block_end, bool)
            or not isinstance(self.kv_block_start, int)
            or not isinstance(self.kv_block_end, int)
            or self.kv_block_start < 0
            or self.kv_block_end <= self.kv_block_start
        ):
            raise ValueError("KV block bounds must satisfy 0 <= start < end")
        _rank_in_world(self.owner_cp_rank, parallel.cp_world_size, "owner_cp_rank")
        _rank_in_world(self.owner_tp_rank, parallel.tp_world_size, "owner_tp_rank")

    def provenance(self) -> dict[str, int]:
        return {
            "global_block_index": int(self.global_block_index),
            "kv_block_start": int(self.kv_block_start),
            "kv_block_end": int(self.kv_block_end),
            "owner_cp_rank": int(self.owner_cp_rank),
            "owner_tp_rank": int(self.owner_tp_rank),
        }


@dataclass(frozen=True)
class AttentionCPPartialState:
    """One local or received ``(out, lse)`` state before CP merge."""

    out: torch.Tensor
    lse: torch.Tensor
    block: AttentionCPBlockMetadata

    def validate(self, parallel: AttentionParallelSpec) -> None:
        self.block.validate(parallel)
        if self.out.ndim != 4:
            raise ValueError("partial out must have shape [B, Hq, Sq, D]")
        if self.lse.ndim != 3:
            raise ValueError("partial lse must have shape [B, Hq, Sq]")
        if self.out.shape[:3] != self.lse.shape:
            raise ValueError("partial out and lse must share [B, Hq, Sq]")
        if self.out.device != self.lse.device:
            raise ValueError("partial out and lse must be on the same device")
        if self.lse.dtype != torch.float32:
            raise ValueError("partial lse must be attention-domain FP32")
        if self.out.dtype != torch.float32:
            raise ValueError("partial out must remain FP32 until the final write")


@dataclass(frozen=True)
class AttentionCPMergedState:
    """Merged attention state before the CUDA RS communication operator."""

    out: torch.Tensor
    lse: torch.Tensor

    def validate(self) -> None:
        if self.out.ndim != 4:
            raise ValueError("merged out must have shape [B, Hq, Sq, D]")
        if self.lse.ndim != 3:
            raise ValueError("merged lse must have shape [B, Hq, Sq]")
        if self.out.shape[:3] != self.lse.shape:
            raise ValueError("merged out and lse must share [B, Hq, Sq]")
        if self.out.device != self.lse.device:
            raise ValueError("merged out and lse must be on the same device")
        if self.lse.dtype != torch.float32:
            raise ValueError("merged lse must be attention-domain FP32")
        if self.out.dtype != torch.float32:
            raise ValueError("merged out must remain FP32 until the final write")


@dataclass(frozen=True)
class AttentionCPOutputShard:
    """Final strict-core output shard after RS."""

    out: torch.Tensor
    lse: torch.Tensor

    def validate(self) -> None:
        if self.out.ndim != 4 or self.lse.ndim != 3:
            raise ValueError("strict output must have out [B,Hq,Sq,D] and lse [B,Hq,Sq]")
        if self.out.shape[:3] != self.lse.shape:
            raise ValueError("strict output and lse must share [B,Hq,Sq]")
        if self.out.device != self.lse.device:
            raise ValueError("strict output and lse must be on the same device")
        if self.out.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict Attention output must be FP16 or BF16")
        if self.lse.dtype is not torch.float32:
            raise ValueError("strict Attention LSE must be FP32")


@dataclass(frozen=True)
class AttentionCPCommunicationPlan:
    """Requested AG/RS communication contract for CP attention partial states."""

    parallel: AttentionParallelSpec
    backend: CPCommunicationBackend = "cuda_ag_rs"
    status: CPCommunicationStatus = "interface_only"
    pattern: str = "ag_rs"
    compute_communication: str = "decoupled"
    merge_order: str = "global_block_index"
    accum_dtype: torch.dtype = torch.float32
    return_lse: bool = True
    expected_blocks: tuple[AttentionCPBlockMetadata, ...] = ()
    expected_kv_token_range: tuple[int, int] | None = None
    query_token_ranges: tuple[tuple[int, int], ...] = ()
    merge_root_cp_rank: int = 0

    def validate(self) -> None:
        self.parallel.validate()
        if self.backend not in {
            "cuda_ag_rs",
            "rccl_ag_rs",
            "p2p_nccl_reference",
            "local_debug",
        }:
            raise ValueError(f"unsupported CP communication backend: {self.backend}")
        if self.status not in {"interface_only", "implemented"}:
            raise ValueError(f"unsupported CP communication status: {self.status}")
        if self.pattern != "ag_rs":
            raise ValueError("PR7 CP communication must use the self-owned AG/RS interface")
        if self.compute_communication != "decoupled":
            raise ValueError("PR7 CP communication must keep compute and communication decoupled")
        if self.merge_order != "global_block_index":
            raise ValueError("PR7 CP communication must preserve global_block_index merge order")
        if self.accum_dtype is not torch.float32:
            raise ValueError("PR7 CP merge accumulation must be FP32")
        if not self.return_lse:
            raise ValueError("PR7 CP communication requires LSE-carrying partial states")
        _rank_in_world(
            self.merge_root_cp_rank,
            self.parallel.cp_world_size,
            "merge_root_cp_rank",
        )
        _validate_expected_block_manifest(self)
        _validate_query_token_ranges(self)
        if self.backend == "p2p_nccl_reference":
            if self.status != "implemented":
                raise ValueError("P2P NCCL reference plans must use status='implemented'")
            if not self.expected_blocks or self.expected_kv_token_range is None:
                raise ValueError("P2P NCCL reference requires a complete expected block manifest")
            if not self.query_token_ranges:
                raise ValueError("P2P NCCL reference requires one query range per CP rank")

    def provenance(self) -> dict[str, object]:
        self.validate()
        return {
            "cp_comm_backend": self.backend,
            "cp_comm_status": self.status,
            "cp_comm_pattern": self.pattern,
            "cp_comm_compute_communication": self.compute_communication,
            "cp_comm_merge_order": self.merge_order,
            "cp_comm_accum_dtype": "fp32",
            "cp_comm_return_lse": self.return_lse,
            "cp_comm_contract": "partial_out_lse_global_block_index",
            "cp_comm_strict_contract": "ag_qkv_positions_shared_core_rs_out_lse",
            "cp_comm_strict_kv_communication": "all_gather",
            "cp_comm_strict_position_communication": "all_gather",
            "cp_comm_strict_backward": "rs_out_backward_ag_then_ag_qkv_backward_rs",
            "cp_comm_runtime": (
                "rccl"
                if self.backend == "rccl_ag_rs"
                else "nccl" if self.backend in {"cuda_ag_rs", "p2p_nccl_reference"} else "local"
            ),
            # The ROCm path intentionally transports tensors and performs the
            # arithmetic in the deterministic core; only the CUDA IPC path
            # owns a numeric collective reduction kernel.
            "cp_comm_attention_numeric_reduction": self.backend == "cuda_ag_rs",
            "cp_comm_expected_kv_token_range": (
                None if self.expected_kv_token_range is None else list(self.expected_kv_token_range)
            ),
            "cp_comm_expected_blocks": [block.provenance() for block in self.expected_blocks],
            "cp_comm_query_token_ranges": [list(bounds) for bounds in self.query_token_ranges],
            "cp_comm_merge_root_cp_rank": int(self.merge_root_cp_rank),
            **self.parallel.provenance(),
        }


class AttentionCPCommunication(Protocol):
    """Protocol implemented by custom CUDA AG/RS communication operators."""

    def all_gather_query(
        self,
        local_q: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> torch.Tensor:
        """Gather the query sequence shards in logical CP-rank order."""

    def all_gather_kv(
        self,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather owner-local K/V in logical CP-rank order."""

    def all_gather_position_ids(
        self,
        local_query_positions: torch.Tensor,
        local_key_positions: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the position IDs paired with owner-local Q/K."""

    def all_gather_partial_states(
        self,
        local_states: tuple[AttentionCPPartialState, ...],
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[AttentionCPPartialState, ...]:
        """Run the custom CUDA AG operator and return gathered partial states."""

    def reduce_scatter_merged_state(
        self,
        merged_state: AttentionCPMergedState,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPMergedState:
        """Run the custom CUDA RS operator and return this rank's output shard."""

    def reduce_scatter_strict_result(
        self,
        out: torch.Tensor,
        lse: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPOutputShard:
        """RS a shared-core result without changing its output dtype."""


class _AllGatherSequence(torch.autograd.Function):
    """Autograd bridge for the self-owned rank-ordered sequence AllGather."""

    @staticmethod
    def forward(ctx, local: torch.Tensor, collective: Any, sequence_dim: int) -> torch.Tensor:
        ctx.collective = collective
        ctx.sequence_dim = int(sequence_dim)
        packed = local.movedim(ctx.sequence_dim, 0).contiguous()
        gathered = collective.all_gather(packed)
        return gathered.movedim(0, ctx.sequence_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_global: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        packed = grad_global.movedim(ctx.sequence_dim, 0).contiguous()
        grad_local = ctx.collective.reduce_scatter(packed)
        return grad_local.movedim(0, ctx.sequence_dim).contiguous(), None, None


class _AllGatherKVSequence(torch.autograd.Function):
    """Gather equal-shape K/V shards through one deterministic collective."""

    @staticmethod
    def forward(
        ctx,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        collective: Any,
        sequence_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.collective = collective
        ctx.sequence_dim = int(sequence_dim)
        packed = torch.stack(
            (
                local_k.movedim(ctx.sequence_dim, 0),
                local_v.movedim(ctx.sequence_dim, 0),
            ),
            dim=1,
        )
        gathered = collective.all_gather(packed)
        return (
            gathered[:, 0].movedim(0, ctx.sequence_dim).contiguous(),
            gathered[:, 1].movedim(0, ctx.sequence_dim).contiguous(),
        )

    @staticmethod
    def backward(
        ctx,
        grad_global_k: torch.Tensor,
        grad_global_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        packed = torch.stack(
            (
                grad_global_k.movedim(ctx.sequence_dim, 0),
                grad_global_v.movedim(ctx.sequence_dim, 0),
            ),
            dim=1,
        )
        local = ctx.collective.reduce_scatter(packed)
        return (
            local[:, 0].movedim(0, ctx.sequence_dim).contiguous(),
            local[:, 1].movedim(0, ctx.sequence_dim).contiguous(),
            None,
            None,
        )


class _AllGatherSequenceFirst(torch.autograd.Function):
    """Gather a local Attention tensor and keep the sequence-leading result."""

    @staticmethod
    def forward(ctx, local: torch.Tensor, collective: Any, sequence_dim: int) -> torch.Tensor:
        ctx.collective = collective
        ctx.sequence_dim = int(sequence_dim)
        packed = local.movedim(ctx.sequence_dim, 0).contiguous()
        return collective.all_gather(packed)

    @staticmethod
    def backward(ctx, grad_global: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        grad_local = ctx.collective.reduce_scatter(grad_global.contiguous())
        return grad_local.movedim(0, ctx.sequence_dim).contiguous(), None, None


class _AllGatherKVSequenceFirst(torch.autograd.Function):
    """Gather K/V together and retain sequence-leading views for direct reorder."""

    @staticmethod
    def forward(
        ctx,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        collective: Any,
        sequence_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.collective = collective
        ctx.sequence_dim = int(sequence_dim)
        packed = torch.stack(
            (
                local_k.movedim(ctx.sequence_dim, 0),
                local_v.movedim(ctx.sequence_dim, 0),
            ),
            dim=1,
        )
        gathered = collective.all_gather(packed)
        return gathered[:, 0], gathered[:, 1]

    @staticmethod
    def backward(
        ctx,
        grad_global_k: torch.Tensor,
        grad_global_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        packed = torch.stack((grad_global_k, grad_global_v), dim=1)
        local = ctx.collective.reduce_scatter(packed)
        return (
            local[:, 0].movedim(0, ctx.sequence_dim).contiguous(),
            local[:, 1].movedim(0, ctx.sequence_dim).contiguous(),
            None,
            None,
        )


class _AllGatherQKVSequenceFirst(torch.autograd.Function):
    """Gather Q/K/V through one collective and expose sequence-leading views."""

    @staticmethod
    def forward(
        ctx,
        local_q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        collective: Any,
        sequence_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.collective = collective
        ctx.sequence_dim = int(sequence_dim)
        ctx.q_heads = int(local_q.size(1))
        ctx.kv_heads = int(local_k.size(1))
        packed = torch.cat(
            (
                local_q.movedim(ctx.sequence_dim, 0),
                local_k.movedim(ctx.sequence_dim, 0),
                local_v.movedim(ctx.sequence_dim, 0),
            ),
            dim=2,
        )
        gathered = collective.all_gather(packed)
        k_start = ctx.q_heads
        v_start = k_start + ctx.kv_heads
        return (
            gathered[:, :, :k_start],
            gathered[:, :, k_start:v_start],
            gathered[:, :, v_start:],
        )

    @staticmethod
    def backward(
        ctx,
        grad_global_q: torch.Tensor,
        grad_global_k: torch.Tensor,
        grad_global_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        packed = torch.cat(
            (grad_global_q, grad_global_k, grad_global_v),
            dim=2,
        )
        local = ctx.collective.reduce_scatter(packed)
        k_start = ctx.q_heads
        v_start = k_start + ctx.kv_heads
        return (
            local[:, :, :k_start].movedim(0, ctx.sequence_dim).contiguous(),
            local[:, :, k_start:v_start].movedim(0, ctx.sequence_dim).contiguous(),
            local[:, :, v_start:].movedim(0, ctx.sequence_dim).contiguous(),
            None,
            None,
        )


class _RootReduceScatterSequence(torch.autograd.Function):
    """Scatter one authoritative full result and gather its gradient back to the root."""

    @staticmethod
    def forward(
        ctx,
        full: torch.Tensor,
        collective: Any,
        sequence_dim: int,
        rank: int,
        root: int,
    ) -> torch.Tensor:
        ctx.collective = collective
        ctx.sequence_dim = int(sequence_dim)
        ctx.rank = int(rank)
        ctx.root = int(root)
        ctx.world_size = int(getattr(collective, "world_size", 1))
        packed = full.movedim(ctx.sequence_dim, 0).contiguous()
        ctx.full_shape = tuple(packed.shape)
        if ctx.rank != ctx.root:
            packed = torch.zeros_like(packed)
        # The ROCm transport adapter exposes an explicit root-owned scatter;
        # CUDA's IPC collective keeps the historical reduce_scatter entrypoint.
        scatter = getattr(collective, "scatter", None)
        local = scatter(packed) if callable(scatter) else collective.reduce_scatter(packed)
        return local.movedim(0, ctx.sequence_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_local: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        packed = grad_local.movedim(ctx.sequence_dim, 0).contiguous()
        # The forward scatter has one authoritative full input on root. Its
        # backward is the dual gather of every rank's local output gradient;
        # non-root full inputs were zeroed in forward and receive no gradient.
        grad_full = ctx.collective.all_gather(packed).movedim(0, ctx.sequence_dim).contiguous()
        if ctx.rank != ctx.root:
            grad_full.zero_()
        return grad_full, None, None, None, None


class _RootReduceScatterSequenceFirst(torch.autograd.Function):
    """RS an already sequence-leading full tensor without another layout copy."""

    @staticmethod
    def forward(
        ctx,
        full: torch.Tensor,
        collective: Any,
        output_sequence_dim: int,
        rank: int,
        root: int,
    ) -> torch.Tensor:
        ctx.collective = collective
        ctx.output_sequence_dim = int(output_sequence_dim)
        ctx.rank = int(rank)
        ctx.root = int(root)
        packed = full if ctx.rank == ctx.root else torch.zeros_like(full)
        local = collective.reduce_scatter(packed)
        return local.movedim(0, ctx.output_sequence_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_local: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        packed = grad_local.movedim(ctx.output_sequence_dim, 0).contiguous()
        grad_full = ctx.collective.all_gather(packed)
        if ctx.rank != ctx.root:
            grad_full.zero_()
        return grad_full, None, None, None, None


class CUDAAGRSAttentionCPCommunication:
    """Deterministic CUDA AG/RS adapter backed by PR311/PR312."""

    backend_id = "cuda_ag_rs"
    collective_label = "self-owned CUDA AG/RS"
    supports_autograd = True

    def __init__(self, *, process_group: Any = None, collective: Any = None) -> None:
        self._process_group = process_group
        self._collective = collective

    def _get_collective(self, plan: AttentionCPCommunicationPlan):
        if self._collective is not None:
            return self._collective
        try:
            from rl_engine.distributed.algorithms.collectives import collective_for_group
        except ImportError as exc:
            raise AttentionCPCommunicationUnavailable(
                f"{self.collective_label} requires PR311/PR312 DeterministicCollective"
            ) from exc
        try:
            dist = self._dist()
            group = self._process_group if self._process_group is not None else dist.group.WORLD
            self._collective = collective_for_group(
                group=group,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            if self._collective is None:
                raise RuntimeError("the CP process group is unavailable")
        except (RuntimeError, ValueError, TypeError) as exc:
            raise AttentionCPCommunicationUnavailable(
                f"{self.collective_label} is unavailable: {exc}"
            ) from exc
        if self._collective.world_size != plan.parallel.cp_world_size:
            raise AttentionCPCommunicationUnavailable(
                "self-owned collective world size does not match CP communication plan"
            )
        return self._collective

    def all_gather_query(
        self,
        local_q: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> torch.Tensor:
        """Gather Q with the self-owned CUDA AG operator.

        The collective works on the leading dimension, so Q is temporarily
        laid out as ``[S_local, B, H, D]``.  The returned tensor is restored to
        the Attention layout ``[B, H, S_global, D]`` in CP-rank order.
        """

        self._validate_cuda_plan(plan)
        _validate_query_shard(local_q, plan)
        ranges = plan.query_token_ranges
        if len({end - start for start, end in ranges}) != 1:
            raise AttentionCPCommunicationUnavailable(
                "self-owned CUDA AG requires equal query shard lengths"
            )
        collective = self._get_collective(plan)
        return self._all_gather_sequence_tensor(local_q, collective, sequence_dim=2)

    def all_gather_kv(
        self,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_cuda_plan(plan)
        _validate_local_kv_shard(local_k, local_v, plan)
        _require_equal_kv_owner_widths(plan, "self-owned CUDA AG")
        collective = self._get_collective(plan)
        return _AllGatherKVSequence.apply(local_k, local_v, collective, 2)

    def all_gather_query_sequence_first(
        self,
        local_q: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> torch.Tensor:
        """Gather Q without copying the sequence-leading collective result back."""

        self._validate_cuda_plan(plan)
        _validate_query_shard(local_q, plan)
        ranges = plan.query_token_ranges
        if len({end - start for start, end in ranges}) != 1:
            raise AttentionCPCommunicationUnavailable(
                "self-owned CUDA AG requires equal query shard lengths"
            )
        return _AllGatherSequenceFirst.apply(local_q, self._get_collective(plan), 2)

    def all_gather_kv_sequence_first(
        self,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V once and expose sequence-leading views to strict FA4."""

        self._validate_cuda_plan(plan)
        _validate_local_kv_shard(local_k, local_v, plan)
        _require_equal_kv_owner_widths(plan, "self-owned CUDA AG")
        return _AllGatherKVSequenceFirst.apply(
            local_k,
            local_v,
            self._get_collective(plan),
            2,
        )

    def all_gather_qkv_sequence_first(
        self,
        local_q: torch.Tensor,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather equal-length Q/K/V shards with one collective invocation."""

        self._validate_cuda_plan(plan)
        _validate_query_shard(local_q, plan)
        _validate_local_kv_shard(local_k, local_v, plan)
        _require_equal_kv_owner_widths(plan, "self-owned CUDA AG")
        ranges = plan.query_token_ranges
        if len({end - start for start, end in ranges}) != 1:
            raise AttentionCPCommunicationUnavailable(
                "self-owned CUDA AG requires equal query shard lengths"
            )
        if (
            local_q.size(0) != local_k.size(0)
            or local_q.size(2) != local_k.size(2)
            or local_q.size(3) != local_k.size(3)
        ):
            raise AttentionCPCommunicationUnavailable(
                "fused Q/K/V gather requires matching batch, sequence, and head dimensions"
            )
        return _AllGatherQKVSequenceFirst.apply(
            local_q,
            local_k,
            local_v,
            self._get_collective(plan),
            2,
        )

    def all_gather_position_ids(
        self,
        local_query_positions: torch.Tensor,
        local_key_positions: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_cuda_plan(plan)
        _validate_local_position_ids(local_query_positions, local_key_positions, plan)
        _require_equal_kv_owner_widths(plan, "self-owned CUDA AG")
        collective = self._get_collective(plan)
        if local_query_positions is local_key_positions:
            positions = self._all_gather_sequence_tensor(
                local_query_positions, collective, sequence_dim=1
            )
            return positions, positions
        query_positions = self._all_gather_sequence_tensor(
            local_query_positions, collective, sequence_dim=1
        )
        key_positions = self._all_gather_sequence_tensor(
            local_key_positions, collective, sequence_dim=1
        )
        return query_positions, key_positions

    @staticmethod
    def _all_gather_sequence_tensor(
        local: torch.Tensor,
        collective: Any,
        *,
        sequence_dim: int,
    ) -> torch.Tensor:
        return _AllGatherSequence.apply(local, collective, sequence_dim)

    def all_gather_partial_states(
        self,
        local_states: tuple[AttentionCPPartialState, ...],
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[AttentionCPPartialState, ...]:
        self._validate_cuda_plan(plan)
        _validate_local_partial_states(local_states, plan)
        ordered = tuple(sorted(local_states, key=lambda state: state.block.global_block_index))
        block_count = len(ordered)
        counts = [None] * plan.parallel.cp_world_size
        self._dist().all_gather_object(counts, block_count, group=self._process_group)
        if any(count != block_count for count in counts):
            raise AttentionCPCommunicationUnavailable(
                "self-owned CUDA AG requires equal block counts on all CP ranks"
            )
        collective = self._get_collective(plan)
        packed_out = torch.stack([state.out for state in ordered], dim=0).contiguous()
        packed_lse = torch.stack([state.lse for state in ordered], dim=0).contiguous()
        gathered_out = collective.all_gather(packed_out)
        gathered_lse = collective.all_gather(packed_lse)
        received: list[AttentionCPPartialState] = []
        blocks_by_rank = [
            _expected_blocks_for_cp_rank(plan, cp_rank)
            for cp_rank in range(plan.parallel.cp_world_size)
        ]
        for cp_rank, blocks in enumerate(blocks_by_rank):
            for block_index, block in enumerate(blocks):
                row = cp_rank * block_count + block_index
                received.append(
                    AttentionCPPartialState(
                        out=gathered_out[row],
                        lse=gathered_lse[row],
                        block=block,
                    )
                )
        return sort_attention_cp_partial_states(tuple(received), plan=plan)

    def reduce_scatter_merged_state(
        self,
        merged_state: AttentionCPMergedState,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPMergedState:
        self._validate_cuda_plan(plan)
        merged_state.validate()
        ranges = plan.query_token_ranges
        if not ranges or len({end - start for start, end in ranges}) != 1:
            raise AttentionCPCommunicationUnavailable(
                "self-owned CUDA RS currently requires equal contiguous query ranges"
            )
        collective = self._get_collective(plan)
        rank = plan.parallel.cp_rank
        root = plan.merge_root_cp_rank
        out_local = _RootReduceScatterSequence.apply(merged_state.out, collective, 2, rank, root)
        lse_local = _RootReduceScatterSequence.apply(merged_state.lse, collective, 2, rank, root)
        result = AttentionCPMergedState(out=out_local, lse=lse_local)
        result.validate()
        return result

    def reduce_scatter_strict_result(
        self,
        out: torch.Tensor,
        lse: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPOutputShard:
        self._validate_cuda_plan(plan)
        _validate_strict_full_result(out, lse, plan)
        ranges = plan.query_token_ranges
        if len({end - start for start, end in ranges}) != 1:
            raise AttentionCPCommunicationUnavailable(
                "self-owned CUDA RS requires equal contiguous query ranges"
            )
        collective = self._get_collective(plan)
        result = AttentionCPOutputShard(
            out=_RootReduceScatterSequence.apply(
                out,
                collective,
                2,
                plan.parallel.cp_rank,
                plan.merge_root_cp_rank,
            ),
            lse=_RootReduceScatterSequence.apply(
                lse,
                collective,
                2,
                plan.parallel.cp_rank,
                plan.merge_root_cp_rank,
            ),
        )
        result.validate()
        return result

    def reduce_scatter_strict_result_sequence_first(
        self,
        out: torch.Tensor,
        lse: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPOutputShard:
        """RS strict outputs already laid out as [S, B, H, ...]."""

        self._validate_cuda_plan(plan)
        ranges = plan.query_token_ranges
        if len({end - start for start, end in ranges}) != 1:
            raise AttentionCPCommunicationUnavailable(
                "self-owned CUDA RS requires equal contiguous query ranges"
            )
        full_tokens = ranges[-1][1]
        if (
            out.ndim != 4
            or lse.ndim != 3
            or out.size(0) != full_tokens
            or lse.size(0) != full_tokens
            or out.shape[1:3] != lse.shape[1:3]
        ):
            raise ValueError("sequence-first strict outputs must use [S,B,H,D] and [S,B,H]")
        if not out.is_cuda or not lse.is_cuda or not out.is_contiguous() or not lse.is_contiguous():
            raise ValueError("sequence-first strict outputs must be contiguous CUDA tensors")
        collective = self._get_collective(plan)
        rank = plan.parallel.cp_rank
        root = plan.merge_root_cp_rank
        result = AttentionCPOutputShard(
            out=_RootReduceScatterSequenceFirst.apply(
                out,
                collective,
                2,
                rank,
                root,
            ),
            lse=_RootReduceScatterSequenceFirst.apply(
                lse,
                collective,
                2,
                rank,
                root,
            ),
        )
        result.validate()
        return result

    def _dist(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise AttentionCPCommunicationUnavailable(
                "self-owned CUDA AG/RS requires initialized torch.distributed"
            )
        return dist

    def _validate_cuda_plan(self, plan: AttentionCPCommunicationPlan) -> None:
        plan.validate()
        if plan.backend != "cuda_ag_rs" or plan.status != "implemented":
            raise AttentionCPCommunicationUnavailable(
                "self-owned CUDA AG/RS requires an implemented cuda_ag_rs plan"
            )
        if not torch.cuda.is_available():
            raise AttentionCPCommunicationUnavailable("self-owned CUDA AG/RS requires CUDA")


class _RCCLRankOrderedTransport:
    """RCCL transport with rank-ordered all-gather and root-owned scatter.

    The scatter half deliberately performs no floating-point reduction. The
    strict Attention arithmetic remains entirely in the deterministic core.
    """

    def __init__(self, *, process_group: Any, root: int) -> None:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise AttentionCPCommunicationUnavailable(
                "self-owned RCCL AG/RS requires initialized torch.distributed"
            )
        backend = str(dist.get_backend(process_group)).lower()
        if "nccl" not in backend or torch.version.hip is None:
            raise AttentionCPCommunicationUnavailable(
                "self-owned RCCL AG/RS requires the PyTorch NCCL API on ROCm"
            )
        self.group = process_group
        self.rank = int(dist.get_rank(process_group))
        self.world_size = int(dist.get_world_size(process_group))
        self.root = int(root)
        if self.root < 0 or self.root >= self.world_size:
            raise AttentionCPCommunicationUnavailable("RCCL scatter root is outside the group")

    def all_gather(self, local: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        if not local.is_cuda or not local.is_contiguous():
            raise AttentionCPCommunicationUnavailable(
                "RCCL AllGather requires a contiguous ROCm tensor"
            )
        shape = (self.world_size * local.size(0), *local.shape[1:])
        gathered = torch.empty(shape, dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(gathered, local, group=self.group)
        return gathered

    def scatter(self, full: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        if not full.is_cuda or not full.is_contiguous():
            raise AttentionCPCommunicationUnavailable(
                "RCCL ReduceScatter transport requires a contiguous ROCm tensor"
            )
        if full.size(0) % self.world_size:
            raise AttentionCPCommunicationUnavailable(
                "RCCL scatter leading dimension must divide the CP world size"
            )
        # Leading-dimension chunks are already contiguous. Avoid materializing
        # copies for every rank; non-root ranks do not need a scatter list.
        local_shape = (full.size(0) // self.world_size, *full.shape[1:])
        local = torch.empty(local_shape, dtype=full.dtype, device=full.device)
        if self.world_size == 1:
            local.copy_(full)
            return local
        global_root = self.root
        if self.group is not None:
            get_global_rank = getattr(dist, "get_global_rank", None)
            if callable(get_global_rank):
                global_root = int(get_global_rank(self.group, self.root))
            else:
                global_root = int(dist.get_process_group_ranks(self.group)[self.root])
        scatter_list = list(full.chunk(self.world_size, dim=0)) if self.rank == self.root else None
        dist.scatter(local, scatter_list=scatter_list, src=global_root, group=self.group)
        return local

    def reduce_scatter(self, full: torch.Tensor) -> torch.Tensor:
        """Deterministic rank-order sum followed by local scatter.

        RCCL is used for point-to-point transport. The floating-point sum is
        performed locally in source-rank order, so collective reduction order
        is not delegated to RCCL.
        """
        if not full.is_cuda or not full.is_contiguous():
            raise AttentionCPCommunicationUnavailable(
                "RCCL ReduceScatter requires a contiguous ROCm tensor"
            )
        if full.size(0) % self.world_size:
            raise AttentionCPCommunicationUnavailable(
                "RCCL ReduceScatter leading dimension must divide the CP world size"
            )
        chunks = tuple(chunk.contiguous() for chunk in full.chunk(self.world_size, dim=0))
        gathered = self.all_gather(full)
        local = gathered[
            self.rank * chunks[self.rank].size(0) : (self.rank + 1) * chunks[self.rank].size(0)
        ].clone()
        chunk_rows = chunks[self.rank].size(0)
        # Start with source rank 0, then add the remaining source ranks in
        # ascending order. This avoids counting source 0 twice.
        for source in range(1, self.world_size):
            source_full = gathered[source * full.size(0) : (source + 1) * full.size(0)]
            local.add_(source_full[self.rank * chunk_rows : (self.rank + 1) * chunk_rows])
        return local


class RCCLAGRSAttentionCPCommunication(CUDAAGRSAttentionCPCommunication):
    """ROCm AG/RS adapter using RCCL only as rank-ordered tensor transport."""

    backend_id = "rccl_ag_rs"
    collective_label = "self-owned RCCL AG/RS"
    supports_autograd = True
    transport_only = True
    supports_async_overlap = False
    supports_compute_communication_fusion = False

    def _dist(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise AttentionCPCommunicationUnavailable(
                "self-owned RCCL AG/RS requires initialized torch.distributed"
            )
        return dist

    def _validate_cuda_plan(self, plan: AttentionCPCommunicationPlan) -> None:
        plan.validate()
        if plan.backend != "rccl_ag_rs" or plan.status != "implemented":
            raise AttentionCPCommunicationUnavailable(
                "self-owned RCCL AG/RS requires an implemented rccl_ag_rs plan"
            )
        if torch.version.hip is None or not torch.cuda.is_available():
            raise AttentionCPCommunicationUnavailable(
                "self-owned RCCL AG/RS requires an available ROCm device"
            )


class P2PNCCLAttentionCPCommunication:
    """Correctness-first P2P NCCL implementation of the CP protocol.

    The block manifest is authoritative.  Only ``out`` and ``lse`` tensors are
    transported, in deterministic peer/block order; received metadata is
    reconstructed from the manifest and then validated as a complete set.
    ``reduce_scatter_merged_state`` uses a designated root and explicit P2P
    sends so its numerical behavior is easy to compare with a future CUDA RS.
    """

    backend_id = "p2p_nccl_reference"
    supports_autograd = False

    def __init__(
        self,
        *,
        process_group: Any = None,
        dist_module: Any = None,
        validate_cuda_tensors: bool = True,
    ) -> None:
        if dist_module is None:
            import torch.distributed as dist

            dist_module = dist
        self._dist = dist_module
        self._group = process_group
        self._validate_cuda_tensors = validate_cuda_tensors

    def all_gather_query(
        self,
        local_q: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> torch.Tensor:
        """Reference query AG implemented with explicit NCCL P2P traffic."""

        self._validate_runtime(plan)
        _validate_query_shard(local_q, plan)
        self._require_cuda(local_q)
        ranges = plan.query_token_ranges
        if len({end - start for start, end in ranges}) != 1:
            raise AttentionCPCommunicationUnavailable(
                "P2P NCCL query AG currently requires equal query shard lengths"
            )
        received: dict[int, torch.Tensor] = {plan.parallel.cp_rank: local_q.contiguous()}
        operations: list[Any] = []
        for peer_cp_rank in range(plan.parallel.cp_world_size):
            if peer_cp_rank == plan.parallel.cp_rank:
                continue
            peer = self._global_peer(peer_cp_rank)
            remote = torch.empty_like(local_q)
            received[peer_cp_rank] = remote
            operations.extend(
                (
                    self._dist.P2POp(self._dist.irecv, remote, peer, group=self._group),
                    self._dist.P2POp(
                        self._dist.isend,
                        local_q.contiguous(),
                        peer,
                        group=self._group,
                    ),
                )
            )
        self._run_operations(operations)
        return torch.cat(
            [received[rank] for rank in range(plan.parallel.cp_world_size)],
            dim=2,
        )

    def all_gather_kv(
        self,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_runtime(plan)
        _validate_local_kv_shard(local_k, local_v, plan)
        self._require_cuda(local_k, local_v)
        ranges = _kv_owner_ranges(plan)
        local_rank = plan.parallel.cp_rank
        gathered_k: dict[int, torch.Tensor] = {local_rank: local_k.contiguous()}
        gathered_v: dict[int, torch.Tensor] = {local_rank: local_v.contiguous()}
        operations: list[Any] = []
        for peer_rank, (start, end) in enumerate(ranges):
            if peer_rank == local_rank:
                continue
            peer = self._global_peer(peer_rank)
            shape = (*local_k.shape[:2], end - start, local_k.size(3))
            peer_k = torch.empty(shape, dtype=local_k.dtype, device=local_k.device)
            peer_v = torch.empty(shape, dtype=local_v.dtype, device=local_v.device)
            gathered_k[peer_rank] = peer_k
            gathered_v[peer_rank] = peer_v
            operations.extend(
                (
                    self._dist.P2POp(self._dist.irecv, peer_k, peer, group=self._group),
                    self._dist.P2POp(self._dist.irecv, peer_v, peer, group=self._group),
                    self._dist.P2POp(
                        self._dist.isend, local_k.contiguous(), peer, group=self._group
                    ),
                    self._dist.P2POp(
                        self._dist.isend, local_v.contiguous(), peer, group=self._group
                    ),
                )
            )
        self._run_operations(operations)
        return (
            torch.cat([gathered_k[rank] for rank in range(len(ranges))], dim=2),
            torch.cat([gathered_v[rank] for rank in range(len(ranges))], dim=2),
        )

    def all_gather_position_ids(
        self,
        local_query_positions: torch.Tensor,
        local_key_positions: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_runtime(plan)
        _validate_local_position_ids(local_query_positions, local_key_positions, plan)
        self._require_cuda(local_query_positions, local_key_positions)
        query_ranges = plan.query_token_ranges
        key_ranges = _kv_owner_ranges(plan)
        local_rank = plan.parallel.cp_rank
        gathered_q = {local_rank: local_query_positions.contiguous()}
        gathered_k = {local_rank: local_key_positions.contiguous()}
        operations: list[Any] = []
        for peer_rank, ((q_start, q_end), (k_start, k_end)) in enumerate(
            zip(query_ranges, key_ranges, strict=True)
        ):
            if peer_rank == local_rank:
                continue
            peer = self._global_peer(peer_rank)
            peer_q = torch.empty(
                (local_query_positions.size(0), q_end - q_start),
                dtype=local_query_positions.dtype,
                device=local_query_positions.device,
            )
            peer_k = torch.empty(
                (local_key_positions.size(0), k_end - k_start),
                dtype=local_key_positions.dtype,
                device=local_key_positions.device,
            )
            gathered_q[peer_rank] = peer_q
            gathered_k[peer_rank] = peer_k
            operations.extend(
                (
                    self._dist.P2POp(self._dist.irecv, peer_q, peer, group=self._group),
                    self._dist.P2POp(self._dist.irecv, peer_k, peer, group=self._group),
                    self._dist.P2POp(
                        self._dist.isend,
                        local_query_positions.contiguous(),
                        peer,
                        group=self._group,
                    ),
                    self._dist.P2POp(
                        self._dist.isend,
                        local_key_positions.contiguous(),
                        peer,
                        group=self._group,
                    ),
                )
            )
        self._run_operations(operations)
        return (
            torch.cat([gathered_q[rank] for rank in range(len(query_ranges))], dim=1),
            torch.cat([gathered_k[rank] for rank in range(len(key_ranges))], dim=1),
        )

    def all_gather_partial_states(
        self,
        local_states: tuple[AttentionCPPartialState, ...],
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[AttentionCPPartialState, ...]:
        self._validate_runtime(plan)
        _validate_local_partial_states(local_states, plan)
        self._require_cuda(local_states[0].out, local_states[0].lse)
        ordered_local_states = tuple(
            sorted(local_states, key=lambda state: state.block.global_block_index)
        )
        template = ordered_local_states[0]
        if template.out.size(2) != plan.query_token_ranges[-1][1]:
            raise ValueError(
                "local partial states must cover the complete query range before gather"
            )
        received: list[AttentionCPPartialState] = []
        operations: list[Any] = []
        receive_tensors: list[tuple[AttentionCPBlockMetadata, torch.Tensor, torch.Tensor]] = []

        for peer_cp_rank in range(plan.parallel.cp_world_size):
            if peer_cp_rank == plan.parallel.cp_rank:
                continue
            peer = self._global_peer(peer_cp_rank)
            peer_blocks = _expected_blocks_for_cp_rank(plan, peer_cp_rank)
            for block in peer_blocks:
                out = torch.empty_like(template.out)
                lse = torch.empty_like(template.lse)
                operations.extend(
                    (
                        self._dist.P2POp(
                            self._dist.irecv,
                            out,
                            peer,
                            group=self._group,
                        ),
                        self._dist.P2POp(
                            self._dist.irecv,
                            lse,
                            peer,
                            group=self._group,
                        ),
                    )
                )
                receive_tensors.append((block, out, lse))
            for state in ordered_local_states:
                operations.extend(
                    (
                        self._dist.P2POp(
                            self._dist.isend,
                            state.out.contiguous(),
                            peer,
                            group=self._group,
                        ),
                        self._dist.P2POp(
                            self._dist.isend,
                            state.lse.contiguous(),
                            peer,
                            group=self._group,
                        ),
                    )
                )

        self._run_operations(operations)
        for block, out, lse in receive_tensors:
            received.append(AttentionCPPartialState(out=out, lse=lse, block=block))
        return sort_attention_cp_partial_states(
            (*ordered_local_states, *received),
            plan=plan,
        )

    def reduce_scatter_merged_state(
        self,
        merged_state: AttentionCPMergedState,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPMergedState:
        self._validate_runtime(plan)
        merged_state.validate()
        self._require_cuda(merged_state.out, merged_state.lse)
        ranges = plan.query_token_ranges
        full_query_tokens = ranges[-1][1]
        if merged_state.out.size(2) != full_query_tokens:
            raise ValueError("merged state query length does not match query_token_ranges coverage")

        rank = plan.parallel.cp_rank
        root = plan.merge_root_cp_rank
        local_start, local_end = ranges[rank]
        if rank == root:
            operations: list[Any] = []
            for peer_cp_rank, (start, end) in enumerate(ranges):
                if peer_cp_rank == root or start == end:
                    continue
                peer = self._global_peer(peer_cp_rank)
                operations.extend(
                    (
                        self._dist.P2POp(
                            self._dist.isend,
                            merged_state.out[:, :, start:end, :].contiguous(),
                            peer,
                            group=self._group,
                        ),
                        self._dist.P2POp(
                            self._dist.isend,
                            merged_state.lse[:, :, start:end].contiguous(),
                            peer,
                            group=self._group,
                        ),
                    )
                )
            self._run_operations(operations)
            result = AttentionCPMergedState(
                out=merged_state.out[:, :, local_start:local_end, :].contiguous(),
                lse=merged_state.lse[:, :, local_start:local_end].contiguous(),
            )
        else:
            local_query_tokens = local_end - local_start
            if local_query_tokens == 0:
                result = AttentionCPMergedState(
                    out=merged_state.out[:, :, 0:0, :].contiguous(),
                    lse=merged_state.lse[:, :, 0:0].contiguous(),
                )
                result.validate()
                return result
            out = torch.empty(
                (
                    *merged_state.out.shape[:2],
                    local_query_tokens,
                    merged_state.out.size(3),
                ),
                dtype=merged_state.out.dtype,
                device=merged_state.out.device,
            )
            lse = torch.empty(
                (*merged_state.lse.shape[:2], local_query_tokens),
                dtype=merged_state.lse.dtype,
                device=merged_state.lse.device,
            )
            peer = self._global_peer(root)
            self._run_operations(
                [
                    self._dist.P2POp(
                        self._dist.irecv,
                        out,
                        peer,
                        group=self._group,
                    ),
                    self._dist.P2POp(
                        self._dist.irecv,
                        lse,
                        peer,
                        group=self._group,
                    ),
                ]
            )
            result = AttentionCPMergedState(out=out, lse=lse)
        result.validate()
        return result

    def reduce_scatter_strict_result(
        self,
        out: torch.Tensor,
        lse: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> AttentionCPOutputShard:
        self._validate_runtime(plan)
        _validate_strict_full_result(out, lse, plan)
        self._require_cuda(out, lse)
        rank = plan.parallel.cp_rank
        root = plan.merge_root_cp_rank
        local_start, local_end = plan.query_token_ranges[rank]
        if rank == root:
            operations: list[Any] = []
            for peer_rank, (start, end) in enumerate(plan.query_token_ranges):
                if peer_rank == root or start == end:
                    continue
                peer = self._global_peer(peer_rank)
                operations.extend(
                    (
                        self._dist.P2POp(
                            self._dist.isend,
                            out[:, :, start:end, :].contiguous(),
                            peer,
                            group=self._group,
                        ),
                        self._dist.P2POp(
                            self._dist.isend,
                            lse[:, :, start:end].contiguous(),
                            peer,
                            group=self._group,
                        ),
                    )
                )
            self._run_operations(operations)
            result = AttentionCPOutputShard(
                out=out[:, :, local_start:local_end, :].contiguous(),
                lse=lse[:, :, local_start:local_end].contiguous(),
            )
        else:
            out_local = torch.empty(
                (*out.shape[:2], local_end - local_start, out.size(3)),
                dtype=out.dtype,
                device=out.device,
            )
            lse_local = torch.empty(
                (*lse.shape[:2], local_end - local_start),
                dtype=lse.dtype,
                device=lse.device,
            )
            peer = self._global_peer(root)
            self._run_operations(
                (
                    self._dist.P2POp(self._dist.irecv, out_local, peer, group=self._group),
                    self._dist.P2POp(self._dist.irecv, lse_local, peer, group=self._group),
                )
            )
            result = AttentionCPOutputShard(out=out_local, lse=lse_local)
        result.validate()
        return result

    def _validate_runtime(self, plan: AttentionCPCommunicationPlan) -> None:
        plan.validate()
        if plan.backend != "p2p_nccl_reference" or plan.status != "implemented":
            raise AttentionCPCommunicationUnavailable(
                "P2P NCCL communication requires an implemented p2p_nccl_reference plan"
            )
        dist = self._dist
        if not dist.is_available() or not dist.is_initialized():
            raise AttentionCPCommunicationUnavailable(
                "P2P NCCL communication requires initialized torch.distributed"
            )
        backend = str(dist.get_backend(self._group)).lower()
        if "nccl" not in backend:
            raise AttentionCPCommunicationUnavailable(
                f"P2P NCCL communication requires the NCCL backend; got {backend}"
            )
        world_size = int(dist.get_world_size(self._group))
        rank = int(dist.get_rank(self._group))
        if world_size != plan.parallel.cp_world_size:
            raise AttentionCPCommunicationUnavailable(
                "process-group world size does not match cp_world_size"
            )
        if rank != plan.parallel.cp_rank:
            raise AttentionCPCommunicationUnavailable(
                "process-group rank does not match the communication plan cp_rank"
            )
        local_blocks = _expected_blocks_for_cp_rank(plan, plan.parallel.cp_rank)
        if not local_blocks:
            raise AttentionCPCommunicationUnavailable(
                "P2P NCCL communication requires every CP rank to own at least one block"
            )

    def _global_peer(self, peer_cp_rank: int) -> int:
        get_global_rank = getattr(self._dist, "get_global_rank", None)
        if self._group is not None and callable(get_global_rank):
            return int(get_global_rank(self._group, peer_cp_rank))
        return peer_cp_rank

    def _run_operations(self, operations: Sequence[Any]) -> None:
        if not operations:
            return
        requests = self._dist.batch_isend_irecv(list(operations))
        for request in requests:
            request.wait()

    def _require_cuda(self, *tensors: torch.Tensor) -> None:
        if self._validate_cuda_tensors and any(tensor.device.type != "cuda" for tensor in tensors):
            raise AttentionCPCommunicationUnavailable(
                "P2P NCCL communication requires CUDA tensors"
            )


def sort_attention_cp_partial_states(
    states: tuple[AttentionCPPartialState, ...],
    *,
    plan: AttentionCPCommunicationPlan,
) -> tuple[AttentionCPPartialState, ...]:
    """Validate and sort partial states by ``global_block_index``."""

    plan.validate()
    if not states:
        raise ValueError("at least one CP attention partial state is required")
    for state in states:
        state.validate(plan.parallel)
    ordered = tuple(sorted(states, key=lambda state: state.block.global_block_index))
    indices = [state.block.global_block_index for state in ordered]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate global_block_index values are not allowed")
    if not plan.expected_blocks and indices != list(range(len(ordered))):
        raise ValueError(
            "partial states without a manifest must cover global_block_index "
            "values [0, block_count)"
        )
    if not plan.expected_blocks and ordered[0].block.kv_block_start != 0:
        raise ValueError("partial states without a manifest must start at KV token 0")
    if plan.expected_kv_token_range is not None:
        expected_start, expected_end = plan.expected_kv_token_range
        if (
            ordered[0].block.kv_block_start != expected_start
            or ordered[-1].block.kv_block_end != expected_end
        ):
            raise ValueError("partial states do not cover the declared expected KV token range")
    _validate_partial_state_set(ordered, plan)
    return ordered


def _validate_expected_block_manifest(plan: AttentionCPCommunicationPlan) -> None:
    blocks = plan.expected_blocks
    if not blocks:
        if plan.expected_kv_token_range is not None:
            raise ValueError("expected_kv_token_range requires expected_blocks")
        return
    for block in blocks:
        block.validate(plan.parallel)
        if block.owner_tp_rank != plan.parallel.tp_rank:
            raise ValueError("expected block owner_tp_rank must match the plan TP shard")
    ordered = tuple(sorted(blocks, key=lambda block: block.global_block_index))
    indices = tuple(block.global_block_index for block in ordered)
    if len(set(indices)) != len(indices):
        raise ValueError("expected block manifest contains duplicate global_block_index values")
    if len(set(blocks)) != len(blocks):
        raise ValueError("expected block manifest contains duplicate metadata")
    owners = {block.owner_cp_rank for block in blocks}
    if owners != set(range(plan.parallel.cp_world_size)):
        raise ValueError("expected block manifest must assign work to every CP rank")
    expected_range = plan.expected_kv_token_range
    if expected_range is None:
        raise ValueError("expected block manifest requires expected_kv_token_range")
    try:
        start, end = expected_range
    except (TypeError, ValueError) as exc:
        raise ValueError("expected KV range must contain exactly (start, end)") from exc
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ValueError("expected KV range must satisfy 0 <= start < end")
    if ordered[0].kv_block_start != start or ordered[-1].kv_block_end != end:
        raise ValueError("expected block manifest does not cover the declared KV range")
    previous_end = start
    for block in ordered:
        if block.kv_block_start != previous_end:
            raise ValueError("expected block manifest must be gap-free and non-overlapping")
        previous_end = block.kv_block_end


def _validate_query_token_ranges(plan: AttentionCPCommunicationPlan) -> None:
    ranges = plan.query_token_ranges
    if not ranges:
        return
    if len(ranges) != plan.parallel.cp_world_size:
        raise ValueError("query_token_ranges must contain one range per CP rank")
    previous_end = 0
    for bounds in ranges:
        try:
            start, end = bounds
        except (TypeError, ValueError) as exc:
            raise ValueError("query token ranges must contain (start, end) pairs") from exc
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError("query token ranges must contain (start, end) pairs")
        if start != previous_end or end < start:
            raise ValueError("query token ranges must be non-negative, contiguous, and start at 0")
        previous_end = end
    if previous_end == 0:
        raise ValueError("query token ranges must cover at least one query token")


def _validate_query_shard(
    local_q: torch.Tensor,
    plan: AttentionCPCommunicationPlan,
) -> None:
    if local_q.ndim != 4:
        raise ValueError("local Q must have shape [B, Hq, Sq_local, D]")
    if not plan.query_token_ranges:
        raise ValueError("query AG requires query_token_ranges")
    start, end = plan.query_token_ranges[plan.parallel.cp_rank]
    if local_q.size(2) != end - start:
        raise ValueError("local Q sequence length does not match its query_token_range")


def _kv_owner_ranges(
    plan: AttentionCPCommunicationPlan,
) -> tuple[tuple[int, int], ...]:
    """Return one gap-free logical KV range for each CP owner."""

    ranges: list[tuple[int, int]] = []
    for owner_rank in range(plan.parallel.cp_world_size):
        blocks = _expected_blocks_for_cp_rank(plan, owner_rank)
        if not blocks:
            raise ValueError(f"CP block manifest has no blocks for owner rank {owner_rank}")
        ordered = sorted(blocks, key=lambda block: block.kv_block_start)
        start = ordered[0].kv_block_start
        cursor = start
        for block in ordered:
            if block.kv_block_start != cursor:
                raise ValueError("CP owner KV blocks must form a contiguous range")
            cursor = block.kv_block_end
        ranges.append((start, cursor))
    expected = plan.expected_kv_token_range
    if expected is None or ranges[0][0] != expected[0] or ranges[-1][1] != expected[1]:
        raise ValueError("CP owner ranges do not cover expected_kv_token_range")
    for left, right in zip(ranges, ranges[1:], strict=False):
        if left[1] != right[0]:
            raise ValueError("CP owner KV ranges must be gap-free and rank ordered")
    return tuple(ranges)


def _require_equal_kv_owner_widths(
    plan: AttentionCPCommunicationPlan,
    backend: str,
) -> None:
    widths = {end - start for start, end in _kv_owner_ranges(plan)}
    if len(widths) != 1:
        raise AttentionCPCommunicationUnavailable(f"{backend} requires equal KV shard lengths")


def _validate_local_kv_shard(
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    plan: AttentionCPCommunicationPlan,
) -> None:
    if local_k.ndim != 4 or local_v.ndim != 4:
        raise ValueError("local K/V must have shape [B,Hkv,Skv_local,D]")
    if local_k.shape != local_v.shape:
        raise ValueError("local K/V must have matching shapes")
    if local_k.dtype != local_v.dtype or local_k.device != local_v.device:
        raise ValueError("local K/V must have matching dtype and device")
    start, end = _kv_owner_ranges(plan)[plan.parallel.cp_rank]
    if local_k.size(2) != end - start:
        raise ValueError("local K/V width does not match the CP owner range")


def _validate_local_position_ids(
    local_query_positions: torch.Tensor,
    local_key_positions: torch.Tensor,
    plan: AttentionCPCommunicationPlan,
) -> None:
    integer_dtypes = (torch.int32, torch.int64)
    if local_query_positions.ndim != 2 or local_key_positions.ndim != 2:
        raise ValueError("local Q/K position IDs must have shape [B,S_local]")
    if local_query_positions.dtype not in integer_dtypes or (
        local_key_positions.dtype not in integer_dtypes
    ):
        raise ValueError("local Q/K position IDs must contain integers")
    if local_query_positions.device != local_key_positions.device:
        raise ValueError("local Q/K position IDs must be on the same device")
    query_start, query_end = plan.query_token_ranges[plan.parallel.cp_rank]
    key_start, key_end = _kv_owner_ranges(plan)[plan.parallel.cp_rank]
    if local_query_positions.size(1) != query_end - query_start:
        raise ValueError("local query position width does not match query ownership")
    if local_key_positions.size(1) != key_end - key_start:
        raise ValueError("local key position width does not match KV ownership")
    if local_query_positions.size(0) != local_key_positions.size(0):
        raise ValueError("local Q/K position IDs must share batch size")


def _validate_strict_full_result(
    out: torch.Tensor,
    lse: torch.Tensor,
    plan: AttentionCPCommunicationPlan,
) -> None:
    result = AttentionCPOutputShard(out=out, lse=lse)
    result.validate()
    total_queries = plan.query_token_ranges[-1][1]
    if out.size(2) != total_queries:
        raise ValueError("strict full result does not cover all query_token_ranges")


def _validate_local_partial_states(
    states: tuple[AttentionCPPartialState, ...],
    plan: AttentionCPCommunicationPlan,
) -> None:
    expected = _expected_blocks_for_cp_rank(plan, plan.parallel.cp_rank)
    if not states:
        raise ValueError("each CP rank must provide at least one local partial state")
    for state in states:
        state.validate(plan.parallel)
        if state.block.owner_cp_rank != plan.parallel.cp_rank:
            raise ValueError("local partial state has the wrong CP owner")
        if state.block.owner_tp_rank != plan.parallel.tp_rank:
            raise ValueError("local partial state has the wrong TP owner")
    actual = tuple(
        state.block for state in sorted(states, key=lambda item: item.block.global_block_index)
    )
    if actual != expected:
        raise ValueError("local partial states do not exactly match the rank manifest")
    _validate_common_state_shapes(states)


def _expected_blocks_for_cp_rank(
    plan: AttentionCPCommunicationPlan,
    cp_rank: int,
) -> tuple[AttentionCPBlockMetadata, ...]:
    return tuple(
        block
        for block in sorted(
            plan.expected_blocks,
            key=lambda item: item.global_block_index,
        )
        if block.owner_cp_rank == cp_rank
    )


def _validate_partial_state_set(
    states: tuple[AttentionCPPartialState, ...],
    plan: AttentionCPCommunicationPlan,
) -> None:
    _validate_common_state_shapes(states)
    previous_end = states[0].block.kv_block_start
    for state in states:
        if state.block.owner_tp_rank != plan.parallel.tp_rank:
            raise ValueError("partial state has the wrong TP owner for this CP group")
        if state.block.kv_block_start != previous_end:
            raise ValueError("partial state KV ranges must be gap-free and non-overlapping")
        previous_end = state.block.kv_block_end
    if plan.expected_blocks:
        actual = tuple(state.block for state in states)
        expected = tuple(sorted(plan.expected_blocks, key=lambda block: block.global_block_index))
        if actual != expected:
            raise ValueError(
                "gathered partial states do not exactly match the complete block manifest"
            )


def _validate_common_state_shapes(states: Sequence[AttentionCPPartialState]) -> None:
    first = states[0]
    for state in states[1:]:
        if state.out.shape != first.out.shape or state.lse.shape != first.lse.shape:
            raise ValueError("all CP partial states must have matching out/lse shapes")
        if state.out.dtype != first.out.dtype or state.lse.dtype != first.lse.dtype:
            raise ValueError("all CP partial states must have matching out/lse dtypes")
        if state.out.device != first.out.device or state.lse.device != first.lse.device:
            raise ValueError("all CP partial states must be on the same device")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _rank_in_world(rank: int, world_size: int, name: str) -> None:
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank >= world_size:
        raise ValueError(f"{name} must be in [0, world_size)")


__all__ = [
    "AttentionCPBlockMetadata",
    "AttentionCPCommunication",
    "AttentionCPCommunicationPlan",
    "AttentionCPCommunicationUnavailable",
    "AttentionCPMergedState",
    "AttentionCPOutputShard",
    "AttentionCPPartialState",
    "AttentionParallelSpec",
    "CPCommunicationBackend",
    "CPCommunicationStatus",
    "CUDAAGRSAttentionCPCommunication",
    "RCCLAGRSAttentionCPCommunication",
    "P2PNCCLAttentionCPCommunication",
    "sort_attention_cp_partial_states",
]
