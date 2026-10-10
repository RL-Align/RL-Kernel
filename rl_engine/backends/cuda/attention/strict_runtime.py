# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Framework-neutral strict CUDA Attention runtime.

The runtime composes the production FA4 core and the self-owned CUDA AG/RS
transport. Framework integrations provide only local layout metadata and
logical position IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.backends.cuda.attention.cp_comm import (
    AttentionCPBlockMetadata,
    AttentionCPCommunicationPlan,
    AttentionParallelSpec,
    CUDAAGRSAttentionCPCommunication,
)
from rl_engine.backends.cuda.attention.flash_attn import StrictFlashAttention4Core
from rl_engine.backends.cuda.attention.permutation import permute_sequence
from rl_engine.contracts.operators.attention import (
    STRICT_ATTENTION_FA4_SCHEDULE_ID,
    STRICT_ATTENTION_PRODUCTION_CORE_ID,
    AttentionContract,
)


@dataclass(frozen=True)
class StrictCUDAAttentionResult:
    out: torch.Tensor
    lse: torch.Tensor
    provenance: dict[str, Any]


class StrictCUDAAttentionRuntime:
    """Run one FA4 arithmetic identity at CP=1 or through CUDA AG/RS."""

    backend_id = "rlkernel.cuda.attention.fa4_ag_rs.v1"
    core_id = STRICT_ATTENTION_PRODUCTION_CORE_ID
    strict_schedule = STRICT_ATTENTION_FA4_SCHEDULE_ID

    def __init__(
        self,
        *,
        process_group: Any = None,
        core: Any | None = None,
        communication: Any | None = None,
    ) -> None:
        self._core = StrictFlashAttention4Core() if core is None else core
        self._communication = (
            CUDAAGRSAttentionCPCommunication(process_group=process_group)
            if communication is None
            else communication
        )
        if getattr(self._core, "core_id", None) != self.core_id:
            raise RuntimeError("strict CUDA Attention runtime requires the FA4 production core")
        if getattr(self._core, "strict_schedule", None) != self.strict_schedule:
            raise RuntimeError("strict CUDA Attention runtime requires the FA4 fixed schedule")
        self.communication_executed = False
        self._position_layout_cache: dict[tuple[Any, ...], tuple[torch.Tensor, ...]] = {}
        self._validated_global_position_layouts: dict[tuple[Any, ...], None] = {}

    def _position_layout(
        self,
        query_position_ids: torch.Tensor,
        key_position_ids: torch.Tensor,
        plan: AttentionCPCommunicationPlan,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            query_position_ids.data_ptr(),
            int(query_position_ids._version),
            key_position_ids.data_ptr(),
            int(key_position_ids._version),
            tuple(query_position_ids.shape),
            tuple(key_position_ids.shape),
            plan.parallel.cp_rank,
            plan.parallel.cp_world_size,
        )
        cached = self._position_layout_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        global_q_positions, global_k_positions = self._communication.all_gather_position_ids(
            query_position_ids,
            key_position_ids,
            plan,
        )
        q_sort = torch.argsort(global_q_positions, dim=1)
        if global_q_positions is global_k_positions:
            k_sort = q_sort
            q_positions_sorted = torch.gather(global_q_positions, 1, q_sort)
            k_positions_sorted = q_positions_sorted
        else:
            k_sort = torch.argsort(global_k_positions, dim=1)
            q_positions_sorted = torch.gather(global_q_positions, 1, q_sort)
            k_positions_sorted = torch.gather(global_k_positions, 1, k_sort)
        inverse_q_sort = torch.argsort(q_sort, dim=1)
        value = (
            q_positions_sorted,
            k_positions_sorted,
            q_sort,
            k_sort,
            inverse_q_sort,
        )
        if len(self._position_layout_cache) >= 128:
            self._position_layout_cache.pop(next(iter(self._position_layout_cache)))
        self._position_layout_cache[key] = value
        return value

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        contract: AttentionContract,
        causal: bool,
        scale: float | None,
        cp_world_size: int,
        query_position_ids: torch.Tensor,
        key_position_ids: torch.Tensor,
        positions_are_sorted: bool = False,
    ) -> StrictCUDAAttentionResult:
        self._require_nvidia_cuda(q)
        if cp_world_size != contract.sharding.cp_world_size:
            raise RuntimeError("runtime CP world size does not match AttentionContract")
        self._validate_local_positions(q, k, query_position_ids, key_position_ids)

        if cp_world_size == 1:
            global_q, global_k, global_v = q, k, v
            global_q_positions = query_position_ids
            global_k_positions = key_position_ids
            communication_backend = "none"
            self.communication_executed = False
            gathered_tensors_are_sequence_first = False
        else:
            plan = self._communication_plan(contract, q.size(2), k.size(2))
            gather_query_sequence_first = getattr(
                self._communication,
                "all_gather_query_sequence_first",
                None,
            )
            gather_kv_sequence_first = getattr(
                self._communication,
                "all_gather_kv_sequence_first",
                None,
            )
            gather_qkv_sequence_first = getattr(
                self._communication,
                "all_gather_qkv_sequence_first",
                None,
            )
            if callable(gather_qkv_sequence_first):
                global_q, global_k, global_v = gather_qkv_sequence_first(
                    q,
                    k,
                    v,
                    plan,
                )
                gathered_tensors_are_sequence_first = True
            elif callable(gather_query_sequence_first) and callable(gather_kv_sequence_first):
                global_q = gather_query_sequence_first(q, plan)
                global_k, global_v = gather_kv_sequence_first(k, v, plan)
                gathered_tensors_are_sequence_first = True
            else:
                global_q = self._communication.all_gather_query(q, plan)
                global_k, global_v = self._communication.all_gather_kv(k, v, plan)
                gathered_tensors_are_sequence_first = False
            (
                q_positions_sorted,
                k_positions_sorted,
                q_sort,
                k_sort,
                inverse_q_sort,
            ) = self._position_layout(query_position_ids, key_position_ids, plan)
            communication_backend = "cuda_ag_rs"
            self.communication_executed = True

        if positions_are_sorted:
            if cp_world_size != 1:
                raise RuntimeError("pre-sorted Attention positions are supported only at CP=1")
            q_sorted, k_sorted, v_sorted = global_q, global_k, global_v
            q_positions_sorted, k_positions_sorted = global_q_positions, global_k_positions
            q_sort = None
            sorted_tensors_are_bshd = False
        else:
            if cp_world_size > 1:
                # Gather directly into FA4's [B, S, H, D] layout.  Gathering
                # into [B, H, S, D] and transposing afterward materializes
                # every global Q/K/V tensor twice on every Attention call.
                gather_for_fa4 = (
                    self._gather_sequence_first_bshd
                    if gathered_tensors_are_sequence_first
                    else self._gather_sequence_bshd
                )
                q_sorted = gather_for_fa4(global_q, q_sort)
                k_sorted = gather_for_fa4(global_k, k_sort)
                sorted_tensors_are_bshd = True
            else:
                q_sorted, q_positions_sorted, q_sort = self._sort_by_position(
                    global_q, global_q_positions
                )
                k_sorted, k_positions_sorted, k_sort = self._sort_by_position(
                    global_k, global_k_positions
                )
                sorted_tensors_are_bshd = False
            v_sorted = (
                gather_for_fa4(global_v, k_sort)
                if sorted_tensors_are_bshd
                else self._gather_sequence(global_v, k_sort)
            )
            self._validate_global_positions_cached(
                q_positions_sorted,
                k_positions_sorted,
                causal,
            )

        # FA4 consumes [B, S, H, D]. Materialize that layout once for every
        # logical sequence instead of once per causal prefix.
        if sorted_tensors_are_bshd:
            q_fa, k_fa, v_fa = q_sorted, k_sorted, v_sorted
        else:
            q_fa = q_sorted.transpose(1, 2).contiguous()
            k_fa = k_sorted.transpose(1, 2).contiguous()
            v_fa = v_sorted.transpose(1, 2).contiguous()

        # FA4's full causal schedule produces the same output, LSE, and dQ as
        # launching one single-query prefix at a time. Keeping all query rows
        # in one launch removes O(sequence_length) Python and CUDA dispatches;
        # deterministic=True still pins the backward implementation.
        result = self._core.forward_bshd_with_lse(
            q_fa,
            k_fa,
            v_fa,
            causal=causal,
            scale=scale,
            query_position_ids=q_positions_sorted,
            key_position_ids=k_positions_sorted,
            output_dtype=q.dtype,
        )
        backend = (
            result.provenance.get("actual_backend")
            or result.provenance.get("attention_backend")
            or getattr(self._core, "backend_id", None)
        )

        if cp_world_size > 1:
            if q_sort is None:
                raise RuntimeError("CP Attention requires a framework position reorder")
            reduce_sequence_first = getattr(
                self._communication,
                "reduce_scatter_strict_result_sequence_first",
                None,
            )
            if callable(reduce_sequence_first):
                out_rank_packed = self._gather_bshd_sequence_first(
                    result.out,
                    inverse_q_sort,
                )
                lse_rank_packed = self._gather_bhs_sequence_first(
                    result.lse,
                    inverse_q_sort,
                )
                shard = reduce_sequence_first(
                    out_rank_packed,
                    lse_rank_packed,
                    plan,
                )
            else:
                out_sorted = result.out.transpose(1, 2).contiguous()
                lse_sorted = result.lse
                out_rank_packed = self._gather_sequence(out_sorted, inverse_q_sort)
                lse_rank_packed = self._gather_sequence(lse_sorted, inverse_q_sort)
                shard = self._communication.reduce_scatter_strict_result(
                    out_rank_packed,
                    lse_rank_packed,
                    plan,
                )
            out, lse = shard.out, shard.lse
        else:
            out = result.out.transpose(1, 2).contiguous()
            lse = result.lse

        return StrictCUDAAttentionResult(
            out=out,
            lse=lse,
            provenance={
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "actual_backend": self.backend_id,
                "communication_backend": communication_backend,
                "communication_executed": self.communication_executed,
                "native_attention_arithmetic": True,
                "production_ready": True,
                "fallback": False,
                "fallback_reason": None,
                "reference_only": False,
                "split_kv": "disabled",
                "framework_position_reorder": True,
                "query_schedule": "full_sequence_causal_single_launch",
                "backward_schedule": "fa4_deterministic_full_sequence",
                "core_row_count": q_fa.size(0) * q_fa.size(1),
                "core_launch_count": 1,
                "core_batch_size": q_fa.size(0),
                "core_query_length": q_fa.size(1),
                "core_actual_backends": [] if backend is None else [str(backend)],
            },
        )

    def forward_paged_with_lse(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        scale: float | None,
        out: torch.Tensor | None = None,
    ) -> StrictCUDAAttentionResult:
        """Run strict inference Attention without materializing paged KV rows."""

        self._require_nvidia_cuda(q)
        if out is not None and (
            out.shape != q.shape or out.dtype != q.dtype or out.device != q.device
        ):
            raise ValueError("paged Attention out must match q shape, dtype, and device")
        paged_out = None
        if out is not None:
            paged_out = torch.empty(
                q.transpose(1, 2).shape,
                dtype=q.dtype,
                device=q.device,
            )
        result = self._core.forward_paged_bshd_with_lse(
            q.transpose(1, 2).contiguous(),
            k_cache,
            v_cache,
            page_table=page_table,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            scale=scale,
            output_dtype=q.dtype,
            out=paged_out,
        )
        self.communication_executed = False
        result_out = result.out.transpose(1, 2).contiguous()
        if out is not None:
            out.copy_(result_out)
            result_out = out
        return StrictCUDAAttentionResult(
            out=result_out,
            lse=result.lse,
            provenance={
                **result.provenance,
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "actual_backend": self.backend_id,
                "communication_backend": "none",
                "communication_executed": False,
                "native_attention_arithmetic": True,
                "production_ready": True,
                "fallback": False,
                "fallback_reason": None,
                "reference_only": False,
                "query_schedule": "paged_single_query_batch",
                "core_row_count": q.size(0),
                "core_launch_count": 1,
                "core_batch_size": q.size(0),
                "core_actual_backends": [
                    result.provenance.get("attention_backend", "flash_attention_4.cute.paged")
                ],
            },
        )

    @staticmethod
    def _require_nvidia_cuda(tensor: torch.Tensor) -> None:
        if tensor.device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("strict Attention R/R requires NVIDIA CUDA tensors")

    @staticmethod
    def _communication_plan(
        contract: AttentionContract,
        local_q_tokens: int,
        local_kv_tokens: int,
    ) -> AttentionCPCommunicationPlan:
        sharding = contract.sharding
        parallel = AttentionParallelSpec(
            tp_world_size=sharding.tp_world_size,
            tp_rank=sharding.tp_rank,
            cp_world_size=sharding.cp_world_size,
            cp_rank=sharding.cp_rank,
        )
        query_ranges = tuple(
            (rank * local_q_tokens, (rank + 1) * local_q_tokens)
            for rank in range(sharding.cp_world_size)
        )
        blocks = tuple(
            AttentionCPBlockMetadata(
                global_block_index=rank,
                kv_block_start=rank * local_kv_tokens,
                kv_block_end=(rank + 1) * local_kv_tokens,
                owner_cp_rank=rank,
                owner_tp_rank=sharding.tp_rank,
            )
            for rank in range(sharding.cp_world_size)
        )
        return AttentionCPCommunicationPlan(
            parallel=parallel,
            backend="cuda_ag_rs",
            status="implemented",
            expected_blocks=blocks,
            expected_kv_token_range=(0, local_kv_tokens * sharding.cp_world_size),
            query_token_ranges=query_ranges,
        )

    @staticmethod
    def _validate_local_positions(
        q: torch.Tensor,
        k: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> None:
        if query_positions.shape != (q.size(0), q.size(2)):
            raise RuntimeError("query_position_ids must describe every local query token")
        if key_positions.shape != (k.size(0), k.size(2)):
            raise RuntimeError("key_position_ids must describe every local KV token")
        if query_positions.device != q.device or key_positions.device != k.device:
            raise RuntimeError("Attention position IDs must be on the Q/K CUDA device")
        if query_positions.dtype not in (torch.int32, torch.int64) or (
            key_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise RuntimeError("Attention position IDs must contain integers")

    @staticmethod
    def _sort_by_position(
        tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        order = torch.argsort(positions, dim=1)
        return (
            StrictCUDAAttentionRuntime._gather_sequence(tensor, order),
            torch.gather(positions, 1, order),
            order,
        )

    @staticmethod
    def _gather_sequence(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        if tensor.ndim not in (3, 4):
            raise RuntimeError("strict Attention sequence reorder expects a 3-D or 4-D tensor")
        return permute_sequence(tensor, order, sequence_dim=2, batch_dim=0).contiguous()

    @staticmethod
    def _gather_sequence_bshd(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise RuntimeError("strict Attention BSHD reorder expects a 4-D tensor")
        return permute_sequence(
            tensor.transpose(1, 2), order, sequence_dim=1, batch_dim=0
        ).contiguous()

    @staticmethod
    def _gather_sequence_first_bshd(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise RuntimeError("strict Attention sequence-first BSHD reorder expects a 4-D tensor")
        return permute_sequence(
            tensor.permute(1, 0, 2, 3), order, sequence_dim=1, batch_dim=0
        ).contiguous()

    @staticmethod
    def _gather_bshd_sequence_first(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4:
            raise RuntimeError("strict Attention output reorder expects a BSHD tensor")
        return permute_sequence(
            tensor.permute(1, 0, 2, 3), order, sequence_dim=0, batch_dim=1
        ).contiguous()

    @staticmethod
    def _gather_bhs_sequence_first(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 3:
            raise RuntimeError("strict Attention LSE reorder expects a BHS tensor")
        return permute_sequence(
            tensor.permute(2, 0, 1), order, sequence_dim=0, batch_dim=1
        ).contiguous()

    @staticmethod
    def _validate_global_positions(
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        causal: bool,
    ) -> None:
        for positions, name in (
            (query_positions, "query"),
            (key_positions, "key"),
        ):
            if positions.size(1) > 1:
                torch._assert_async(
                    torch.all(positions[:, 1:] > positions[:, :-1]),
                    f"global {name} positions must be unique and increasing",
                )
        if causal:
            torch._assert_async(
                torch.all(query_positions == key_positions[:, -query_positions.size(1) :]),
                "causal Attention queries must be the trailing global KV positions",
            )

    def _validate_global_positions_cached(
        self,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        causal: bool,
    ) -> None:
        """Validate each immutable sorted position layout only once.

        The position-layout cache owns these tensors and reuses their exact
        storage across layers and recomputation. Re-launching three device
        assertions on every attention call adds work without adding coverage.
        Tensor versions keep the cache fail-safe if a layout is ever mutated.
        """

        key = (
            query_positions.data_ptr(),
            int(query_positions._version),
            tuple(query_positions.shape),
            key_positions.data_ptr(),
            int(key_positions._version),
            tuple(key_positions.shape),
            bool(causal),
        )
        if key in self._validated_global_position_layouts:
            return
        self._validate_global_positions(query_positions, key_positions, causal)
        if len(self._validated_global_position_layouts) >= 128:
            self._validated_global_position_layouts.pop(
                next(iter(self._validated_global_position_layouts))
            )
        self._validated_global_position_layouts[key] = None


__all__ = ["StrictCUDAAttentionResult", "StrictCUDAAttentionRuntime"]
