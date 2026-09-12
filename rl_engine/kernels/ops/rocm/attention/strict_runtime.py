# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Framework-neutral strict ROCm Attention runtime.

Composes the production AITER/CK core with the self-owned RCCL AG/RS
transport, mirroring :class:`StrictCUDAAttentionRuntime` so both platforms
present one runtime shape to framework integrations. Before this existed the
CP schedule had no home on ROCm: the core is single-rank arithmetic, the Vime
provider fails closed at ``CP > 1``, and the only working AG/core/RS sequence
lived in the benchmark script.

Two things differ from the CUDA runtime and both are load-bearing:

* The core is launched once per ``(batch row, KV group)`` rather than once per
  sequence. AITER/CK's reduction order depends on how many heads shared the
  launch, so a head shard computed under TP=N is otherwise not bit-identical
  to the same shard under a different TP degree. The CUDA FA4 core has no such
  dependence and runs one launch per sequence.
* RCCL moves tensors but never reduces them. The cross-rank ``(out, lse)``
  combine order comes from the fixed balanced rank tree in the shared
  ``RCCLDeterministicCollective``, not from RCCL's own algorithm selection.
  That is the collective the CUDA runtime also resolves through
  ``collective_for_group``, so both platforms run one reduction order from
  one implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
    AttentionContract,
)
from rl_engine.kernels.ops.cuda.attention.cp_comm import (
    AttentionCPBlockMetadata,
    AttentionCPCommunicationPlan,
    AttentionCPCommunicationUnavailable,
    AttentionParallelSpec,
    CUDAAGRSAttentionCPCommunication,
)
from rl_engine.kernels.ops.cuda.attention.strict_runtime import StrictCUDAAttentionRuntime
from rl_engine.kernels.ops.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore
from rl_engine.kernels.ops.rocm.attention.paged_gather import fused_paged_kv_gather_bhsd

# The sequence reorder and the position validation are platform-neutral tensor
# bookkeeping. They are bound from the CUDA runtime rather than reimplemented
# so the two runtimes cannot drift into two different global orderings.
_gather_sequence = StrictCUDAAttentionRuntime._gather_sequence
_validate_local_positions = StrictCUDAAttentionRuntime._validate_local_positions
_validate_global_positions = StrictCUDAAttentionRuntime._validate_global_positions


class RCCLAGRSAttentionCPCommunication(CUDAAGRSAttentionCPCommunication):
    """ROCm adapter over the shared deterministic fixed-tree collective."""

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
        if plan.backend != "rccl_ag_rs" or plan.status != "implemented":
            raise AttentionCPCommunicationUnavailable(
                "self-owned RCCL AG/RS requires an implemented rccl_ag_rs plan"
            )
        replace(plan, backend="cuda_ag_rs").validate()
        if torch.version.hip is None or not torch.cuda.is_available():
            raise AttentionCPCommunicationUnavailable(
                "self-owned RCCL AG/RS requires an available ROCm device"
            )


@dataclass(frozen=True)
class StrictRocmAttentionResult:
    out: torch.Tensor
    lse: torch.Tensor
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _PageBoundsEpoch:
    """Opaque proof scope issued by one strict ROCm runtime."""

    owner: object


@dataclass(frozen=True)
class _PageBoundsValidation:
    """Keep validated metadata alive so object/address reuse cannot spoof a hit."""

    epoch: _PageBoundsEpoch
    page_table: torch.Tensor
    seqused_k: torch.Tensor
    signature: tuple[Any, ...]


class StrictRocmAttentionRuntime:
    """Run one AITER/CK arithmetic identity at CP=1 or through RCCL AG/RS."""

    backend_id = "rlkernel.rocm.attention.aiter_ck_ag_rs.v1"
    core_id = STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID
    strict_schedule = STRICT_ATTENTION_ROCM_SCHEDULE_ID
    communication_backend_id = "rccl_ag_rs"
    # Communication and compute are deliberately decoupled; see
    # ``AttentionCPCommunicationPlan.validate``.
    supports_async_overlap = False
    supports_compute_communication_fusion = False

    def __init__(
        self,
        *,
        process_group: Any = None,
        core: Any | None = None,
        communication: Any | None = None,
    ) -> None:
        self._core = StrictRocmAiterCKAttentionCore() if core is None else core
        self._communication = (
            RCCLAGRSAttentionCPCommunication(process_group=process_group)
            if communication is None
            else communication
        )
        if getattr(self._core, "core_id", None) != self.core_id:
            raise RuntimeError(
                "strict ROCm Attention runtime requires the AITER/CK production core"
            )
        if getattr(self._core, "strict_schedule", None) != self.strict_schedule:
            raise RuntimeError("strict ROCm Attention runtime requires the AITER/CK fixed schedule")
        self.communication_executed = False
        self._page_bounds_epoch_owner = object()
        self._page_bounds_validation: _PageBoundsValidation | None = None
        self._causal_prefill_position_cache: tuple[torch.device, int, torch.Tensor] | None = None
        self._position_plan_cache: dict[
            tuple[int, int, int, bool],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}

    def _position_plan(
        self,
        query_position_ids: torch.Tensor,
        key_position_ids: torch.Tensor,
        *,
        plan: AttentionCPCommunicationPlan | None,
        cp_world_size: int,
        causal: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Cache immutable CP position transport and reorder indices across layers."""

        key = (id(query_position_ids), id(key_position_ids), cp_world_size, causal)
        cached = self._position_plan_cache.get(key)
        if cached is not None and cached[0] is query_position_ids and cached[1] is key_position_ids:
            return cached[2:]

        if cp_world_size == 1:
            global_query_positions = query_position_ids
            global_key_positions = key_position_ids
        else:
            if plan is None:
                raise RuntimeError("CP Attention position plan requires a communication plan")
            global_query_positions, global_key_positions = (
                self._communication.all_gather_position_ids(
                    query_position_ids,
                    key_position_ids,
                    plan,
                )
            )
        query_sort = torch.argsort(global_query_positions, dim=1)
        if global_key_positions is global_query_positions:
            key_sort = query_sort
        else:
            key_sort = torch.argsort(global_key_positions, dim=1)
        query_positions_sorted = torch.gather(global_query_positions, 1, query_sort)
        key_positions_sorted = torch.gather(global_key_positions, 1, key_sort)
        _validate_global_positions(
            query_positions_sorted,
            key_positions_sorted,
            causal,
        )
        inverse_query_sort = torch.argsort(query_sort, dim=1)
        value = (
            query_position_ids,
            key_position_ids,
            query_positions_sorted,
            key_positions_sorted,
            query_sort,
            key_sort,
            inverse_query_sort,
        )
        if len(self._position_plan_cache) >= 16:
            self._position_plan_cache.pop(next(iter(self._position_plan_cache)))
        self._position_plan_cache[key] = value
        return value[2:]

    def new_page_bounds_epoch(self) -> object:
        """Issue a proof scope while its page table and lengths remain immutable.

        vLLM creates fresh materialized metadata for the next model forward,
        which must receive a fresh epoch as well.
        """

        return _PageBoundsEpoch(self._page_bounds_epoch_owner)

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
    ) -> StrictRocmAttentionResult:
        self._require_rocm(q)
        if cp_world_size != contract.sharding.cp_world_size:
            raise RuntimeError("runtime CP world size does not match AttentionContract")
        _validate_local_positions(q, k, query_position_ids, key_position_ids)

        plan = None
        if cp_world_size == 1:
            global_q, global_k, global_v = q, k, v
            communication_backend = "none"
            self.communication_executed = False
        else:
            plan = self._communication_plan(contract, q.size(2), k.size(2))
            global_q = self._communication.all_gather_query(q, plan)
            global_k, global_v = self._communication.all_gather_kv(k, v, plan)
            communication_backend = self.communication_backend_id
            self.communication_executed = True

        if positions_are_sorted:
            if cp_world_size != 1:
                raise RuntimeError("pre-sorted Attention positions are supported only at CP=1")
            q_sorted, k_sorted, v_sorted = global_q, global_k, global_v
            q_positions_sorted, k_positions_sorted = query_position_ids, key_position_ids
            q_sort = None
            inverse_q_sort = None
        else:
            (
                q_positions_sorted,
                k_positions_sorted,
                q_sort,
                k_sort,
                inverse_q_sort,
            ) = self._position_plan(
                query_position_ids,
                key_position_ids,
                plan=plan,
                cp_world_size=cp_world_size,
                causal=causal,
            )
            q_sorted = _gather_sequence(global_q, q_sort)
            k_sorted = _gather_sequence(global_k, k_sort)
            v_sorted = _gather_sequence(global_v, k_sort)

        paged_schedule = bool(getattr(self._core, "supports_paged_schedule", False))
        if paged_schedule:
            core_result = self._core.forward_with_lse(
                q_sorted,
                k_sorted,
                v_sorted,
                causal=causal,
                scale=scale,
                key_padding_mask=None,
                query_position_ids=q_positions_sorted,
                key_position_ids=k_positions_sorted,
                output_dtype=q.dtype,
            )
            out_sorted = core_result.out
            lse_sorted = core_result.lse
            core_provenance = dict(core_result.provenance)
            launches = 1
        else:
            out_sorted, lse_sorted, core_provenance, launches = self._run_core(
                q_sorted,
                k_sorted,
                v_sorted,
                causal=causal,
                scale=scale,
                query_position_ids=q_positions_sorted,
                key_position_ids=k_positions_sorted,
                output_dtype=q.dtype,
            )

        if cp_world_size > 1:
            if q_sort is None or inverse_q_sort is None:
                raise RuntimeError("CP Attention requires a framework position reorder")
            out_rank_packed = _gather_sequence(out_sorted, inverse_q_sort)
            lse_rank_packed = _gather_sequence(lse_sorted, inverse_q_sort)
            shard = self._communication.reduce_scatter_strict_result(
                out_rank_packed,
                lse_rank_packed,
                plan,
            )
            out, lse = shard.out, shard.lse
        else:
            out, lse = out_sorted, lse_sorted

        backend = (
            core_provenance.get("attention_backend")
            or core_provenance.get("actual_backend")
            or getattr(self._core, "backend_id", None)
        )
        return StrictRocmAttentionResult(
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
                "query_schedule": (
                    "one_local_gqa_batch_paged_ck"
                    if paged_schedule
                    else "one_batch_row_one_kv_group"
                ),
                "backward_schedule": "aiter_ck_deterministic_per_kv_group",
                "launch_granularity": (
                    "one_local_gqa_batch" if paged_schedule else "one_batch_row_one_kv_group"
                ),
                "tp_degree_invariant": True,
                "invariance_mechanism": "one_kv_group_per_launch",
                "core_row_count": q_sorted.size(0) * q_sorted.size(2),
                "core_launch_count": launches,
                "core_batch_size": q_sorted.size(0),
                "core_query_length": q_sorted.size(2),
                "core_actual_backends": [] if backend is None else [str(backend)],
                "core": core_provenance,
            },
        )

    def forward_paged_varlen_with_lse(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        kv_indptr: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        scale: float | None,
        out: torch.Tensor | None = None,
        return_lse: bool = False,
        page_table_validated: bool = False,
    ) -> StrictRocmAttentionResult:
        """Run packed prefill/decode directly from vLLM's paged KV cache."""

        self._require_rocm(q)
        direct = getattr(self._core, "forward_paged_varlen_with_lse", None)
        if not callable(direct) or not bool(getattr(self._core, "supports_paged_schedule", False)):
            raise RuntimeError("strict ROCm direct paged-varlen CK is unavailable")
        if q.ndim != 3:
            raise ValueError("packed paged Q must use [tokens, heads, head_dim]")
        if not page_table_validated:
            bounds_ok = torch.all((page_table >= 0) & (page_table < k_cache.size(0)))
            torch._assert_async(bounds_ok, "page_table entries are outside the KV cache")
            valid_lengths = torch.all((seqused_k > 0) & (seqused_k <= max_seqlen_k))
            torch._assert_async(
                valid_lengths,
                "seqused_k entries must be positive and within max_seqlen_k",
            )
        core_result = direct(
            q,
            k_cache,
            v_cache,
            page_table=page_table,
            seqused_k=seqused_k,
            cu_seqlens_q=cu_seqlens_q,
            kv_indptr=kv_indptr,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            scale=scale,
            out=out,
            return_lse=return_lse,
        )
        core_provenance = dict(core_result.provenance)
        backend = (
            core_provenance.get("attention_backend")
            or core_provenance.get("actual_backend")
            or getattr(self._core, "backend_id", None)
        )
        self.communication_executed = False
        return StrictRocmAttentionResult(
            out=core_result.out,
            lse=core_result.lse,
            provenance={
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
                "split_kv": "disabled",
                "query_schedule": "paged_varlen_batch",
                "paged_execution": "direct_vllm_pages_to_aiter_batch_prefill_ck",
                "paged_kernel": self._paged_kernel_id(),
                "dense_kv_materialized": False,
                "lse_returned": bool(return_lse),
                "launch_granularity": "one_local_gqa_batch",
                "tp_degree_invariant": True,
                "invariance_mechanism": "matched_train_and_rollout_paged_ck_schedule",
                "core_row_count": q.size(0),
                "core_launch_count": 1,
                "core_batch_size": page_table.size(0),
                "core_query_length": max_seqlen_q,
                "core_actual_backends": [] if backend is None else [str(backend)],
                "core": core_provenance,
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
        cached_lengths: Sequence[int] | None = None,
        page_bounds_epoch: object | None = None,
        return_lse: bool = True,
        page_table_validated: bool = False,
        cu_seqlens_q: torch.Tensor | None = None,
        kv_indptr: torch.Tensor | None = None,
    ) -> StrictRocmAttentionResult:
        """Run strict decode Attention over a paged KV cache.

        AITER exposes no paged entry point that this contract can use. Every
        ``paged_attention_*`` kernel partitions KV and reduces the partials
        (``partition_size``, ``exp_sums``/``max_logits``/``tmp_out``), so the
        partition count moves with the cached length; AITER's
        ``flash_attn_varlen_func`` takes a ``block_table`` but has no
        ``num_splits`` knob to pin, unlike CUDA's FA4. Either way the strict
        contract could not prove Split-KV disabled.

        So the pages are gathered into logical KV order and handed to the same
        dense core the prefill path uses, at the same one-launch-per
        ``(batch row, KV group)`` granularity. The arithmetic is then identical
        to a CP=1 prefill over the same logical sequence, which is what makes
        decode replay comparable against it. The cost is materializing the
        cached KV; a native paged kernel would avoid that, and can replace this
        once AITER can pin its split count.
        """

        self._require_rocm(q)
        self._validate_paged_inputs(
            q,
            k_cache,
            v_cache,
            page_table=page_table,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
        )
        if out is not None:
            if out.shape != q.shape:
                raise ValueError("paged Attention out must have the same shape as q")
            if out.dtype != q.dtype or out.device != q.device:
                raise ValueError("paged Attention out must match the Q dtype and device")
            if not out.is_contiguous():
                raise ValueError("paged Attention out must be contiguous")
        direct_core_out = (
            out is not None
            and not torch.is_grad_enabled()
            and not any(tensor.requires_grad for tensor in (q, k_cache, v_cache, out))
            and q.size(2) == 1
            and callable(getattr(self._core, "forward_decode_with_lse_into", None))
            and self._storage_is_disjoint(
                out,
                q,
                k_cache,
                v_cache,
                page_table,
                seqused_k,
            )
        )

        direct_paged = callable(getattr(self._core, "forward_paged_with_lse", None)) and bool(
            getattr(self._core, "supports_paged_schedule", False)
        )
        if direct_paged:
            if not page_table_validated:
                valid_lengths = torch.all((seqused_k > 0) & (seqused_k <= max_seqlen_k))
                torch._assert_async(
                    valid_lengths,
                    "seqused_k entries must be positive and within max_seqlen_k",
                )
                bounds_ok = torch.all((page_table >= 0) & (page_table < k_cache.size(0)))
                torch._assert_async(bounds_ok, "page_table entries are outside the KV cache")
            if cu_seqlens_q is None:
                cu_seqlens_q = torch.arange(q.size(0) + 1, dtype=torch.int32, device=q.device)
            if kv_indptr is None:
                kv_indptr = torch.arange(
                    q.size(0) + 1, dtype=torch.int32, device=q.device
                ) * page_table.size(1)
            core_result = self._core.forward_paged_with_lse(
                q,
                k_cache,
                v_cache,
                page_table=page_table,
                seqused_k=seqused_k,
                cu_seqlens_q=cu_seqlens_q,
                kv_indptr=kv_indptr,
                max_seqlen_k=max_seqlen_k,
                causal=False,
                scale=scale,
                out=out,
                return_lse=return_lse,
            )
            core_provenance = dict(core_result.provenance)
            backend = (
                core_provenance.get("attention_backend")
                or core_provenance.get("actual_backend")
                or getattr(self._core, "backend_id", None)
            )
            self.communication_executed = False
            return StrictRocmAttentionResult(
                out=core_result.out,
                lse=core_result.lse,
                provenance={
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
                    "split_kv": "disabled",
                    "query_schedule": "paged_single_query_batch",
                    "paged_execution": "direct_vllm_pages_to_aiter_batch_prefill_ck",
                    "paged_kernel": self._paged_kernel_id(),
                    "dense_kv_materialized": False,
                    "lse_returned": bool(return_lse),
                    "launch_granularity": "one_local_gqa_batch",
                    "tp_degree_invariant": True,
                    "invariance_mechanism": "matched_train_and_rollout_paged_ck_schedule",
                    "core_row_count": q.size(0) * q.size(2),
                    "core_launch_count": 1,
                    "core_batch_size": q.size(0),
                    "core_query_length": q.size(2),
                    "core_actual_backends": [] if backend is None else [str(backend)],
                    "core": core_provenance,
                },
            )

        if cached_lengths is None:
            cached_lengths = tuple(int(value) for value in seqused_k.tolist())
        else:
            cached_lengths = tuple(int(value) for value in cached_lengths)
        if len(cached_lengths) != q.size(0):
            raise ValueError("cached_lengths must carry one length per query")
        for row, cached_length in enumerate(cached_lengths):
            if cached_length <= 0 or cached_length > max_seqlen_k:
                raise ValueError(
                    "seqused_k entries must be positive and within max_seqlen_k; "
                    f"row {row} requested {cached_length}"
                )
        causal_prefill = (
            q.size(0) > 1
            and q.size(2) == 1
            and not torch.is_grad_enabled()
            and not any(tensor.requires_grad for tensor in (q, k_cache, v_cache))
            and cached_lengths == tuple(range(1, q.size(0) + 1))
        )
        use_fused_gather = False

        resolved_epoch: _PageBoundsEpoch | None
        if page_bounds_epoch is None:
            resolved_epoch = None
        elif (
            not isinstance(page_bounds_epoch, _PageBoundsEpoch)
            or page_bounds_epoch.owner is not self._page_bounds_epoch_owner
        ):
            raise ValueError("page_bounds_epoch was not issued by this ROCm runtime")
        else:
            resolved_epoch = page_bounds_epoch
        bounds_signature = (
            None
            if resolved_epoch is None
            else self._page_bounds_signature(
                page_table,
                seqused_k,
                cached_lengths=cached_lengths,
                cache_pages=k_cache.size(0),
                page_size=k_cache.size(1),
                max_seqlen_k=max_seqlen_k,
            )
        )
        cached_validation = self._page_bounds_validation
        bounds_reused = bool(
            resolved_epoch is not None
            and cached_validation is not None
            and cached_validation.epoch is resolved_epoch
            and cached_validation.page_table is page_table
            and cached_validation.seqused_k is seqused_k
            and cached_validation.signature == bounds_signature
        )

        if causal_prefill:
            required_pages = (cached_lengths[-1] + k_cache.size(1) - 1) // k_cache.size(1)
            if not (page_table_validated or bounds_reused):
                page_rows_equal = torch.all(
                    page_table[:, :required_pages] == page_table[-1:, :required_pages]
                )
                if page_table.is_cuda:
                    torch._assert_async(
                        page_rows_equal,
                        "causal prefill rows must share one logical page table",
                    )
                elif not bool(page_rows_equal.item()):
                    raise ValueError("causal prefill rows must share one logical page table")
            k_row, v_row = self._gather_paged_row(
                k_cache,
                v_cache,
                page_table[-1],
                cached_lengths[-1],
                validate_bounds=not (page_table_validated or bounds_reused),
            )
            q_sequence = q.squeeze(2).permute(1, 0, 2).unsqueeze(0)
            positions = self._causal_prefill_positions(q.size(0), q.device)
            sequence_out, sequence_lse, core_provenance, launches = self._run_core(
                q_sequence,
                k_row,
                v_row,
                causal=True,
                scale=scale,
                query_position_ids=positions,
                key_position_ids=positions,
                output_dtype=q.dtype,
                collect_lse=return_lse,
            )
            reordered_out = sequence_out.permute(2, 1, 0, 3)
            if out is None:
                result_out = reordered_out.contiguous()
            else:
                out.copy_(reordered_out)
                result_out = out
            result_lse = (
                sequence_lse.permute(2, 1, 0).contiguous()
                if return_lse
                else torch.empty((0,), dtype=torch.float32, device=q.device)
            )
        else:
            row_outs: list[torch.Tensor] = []
            row_lses: list[torch.Tensor] = []
            core_provenance = None
            launches = 0
            for row, cached_length in enumerate(cached_lengths):
                k_row, v_row = self._gather_paged_row(
                    k_cache,
                    v_cache,
                    page_table[row],
                    cached_length,
                    validate_bounds=not (page_table_validated or bounds_reused),
                )
                # Decode attends over the whole cached prefix, so neither the mask
                # nor the AITER call consumes position IDs. Avoid allocating two
                # dead tensors for every row of every decoder layer.
                row_out, row_lse, row_provenance, row_launches = self._run_core(
                    q[row : row + 1],
                    k_row,
                    v_row,
                    causal=False,
                    scale=scale,
                    query_position_ids=None,
                    key_position_ids=None,
                    output_dtype=q.dtype,
                    out=None if out is None else out[row : row + 1],
                    direct_core_out=direct_core_out,
                    collect_lse=return_lse,
                )
                if out is None:
                    row_outs.append(row_out)
                if return_lse:
                    row_lses.append(row_lse)
                launches += row_launches
                if core_provenance is None:
                    core_provenance = row_provenance

            if core_provenance is None:
                raise RuntimeError("strict ROCm paged Attention executed no core launch")

            result_out = out if out is not None else torch.cat(row_outs, dim=0)
            result_lse = (
                row_lses[0]
                if return_lse and direct_core_out and len(row_lses) == 1
                else (
                    torch.cat(row_lses, dim=0)
                    if return_lse
                    else torch.empty((0,), dtype=torch.float32, device=q.device)
                )
            )
        self.communication_executed = False
        if resolved_epoch is not None and not bounds_reused:
            assert bounds_signature is not None
            self._page_bounds_validation = _PageBoundsValidation(
                epoch=resolved_epoch,
                page_table=page_table,
                seqused_k=seqused_k,
                signature=bounds_signature,
            )

        backend = (
            core_provenance.get("attention_backend")
            or core_provenance.get("actual_backend")
            or getattr(self._core, "backend_id", None)
        )
        return StrictRocmAttentionResult(
            out=result_out,
            lse=result_lse,
            provenance={
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
                "split_kv": "disabled",
                "query_schedule": (
                    "paged_causal_prefill_batch" if causal_prefill else "paged_single_query_batch"
                ),
                # The dense core runs; the pages are gathered first. Recorded so
                # a reader never mistakes this for a native paged kernel.
                "paged_execution": (
                    "fused_paged_kv_gather_to_aiter_ck_bshd"
                    if use_fused_gather
                    else "logical_kv_gather_then_dense_core"
                ),
                "paged_kernel": ("triton_fused_kv_gather_bhsd" if use_fused_gather else "none"),
                "lse_returned": bool(return_lse),
                "launch_granularity": "one_batch_row_one_kv_group",
                "tp_degree_invariant": True,
                "invariance_mechanism": "one_kv_group_per_launch",
                "core_row_count": q.size(0) * q.size(2),
                "core_launch_count": launches,
                "core_batch_size": 1 if causal_prefill else q.size(0),
                "core_query_length": q.size(0) if causal_prefill else q.size(2),
                "core_actual_backends": [] if backend is None else [str(backend)],
                "core_output_staging": (
                    "runtime_causal_prefill"
                    if causal_prefill
                    else ("aiter_direct_caller_group" if direct_core_out else "runtime_group_cat")
                ),
                "causal_prefill_collapsed": causal_prefill,
                "page_bounds_validation_reused": bounds_reused,
                "core": core_provenance,
            },
        )

    def _causal_prefill_positions(
        self,
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        cached = self._causal_prefill_position_cache
        if cached is not None and cached[0] == device and cached[1] == token_count:
            return cached[2]
        positions = torch.arange(
            token_count,
            dtype=torch.int64,
            device=device,
        ).unsqueeze(0)
        self._causal_prefill_position_cache = (device, token_count, positions)
        return positions

    @staticmethod
    def _gather_paged_row(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_row: torch.Tensor,
        cached_length: int,
        *,
        validate_bounds: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize one row's cached KV in logical order as ``[1, H, S, D]``.

        Physical page order never reaches the core: the pages are read through
        the page table, so the launch sees the same logical sequence a prefill
        over the same tokens would have seen.
        """

        page_size = k_cache.size(1)
        page_count = (cached_length + page_size - 1) // page_size
        if page_count > page_row.numel():
            raise ValueError("page_table row is shorter than the cached length requires")
        pages = page_row[:page_count]
        if validate_bounds:
            bounds_ok = torch.all((pages >= 0) & (pages < k_cache.size(0)))
            if pages.is_cuda:
                # Keep malformed metadata fail-closed without synchronizing the host
                # twice per row. vLLM page tables are already int32, which
                # index_select accepts directly on ROCm.
                torch._assert_async(bounds_ok, "page_table entries are outside the KV cache")
            elif not bool(bounds_ok.item()):
                raise ValueError("page_table entries are outside the KV cache")

        use_head_major_gather = (
            not torch.is_grad_enabled()
            and not k_cache.requires_grad
            and not v_cache.requires_grad
            # With either singleton dimension, the legacy transpose is already
            # contiguous and does not pay the second materialization copy.
            and k_cache.size(2) > 1
            and cached_length > 1
        )

        def _gather(cache: torch.Tensor) -> torch.Tensor:
            if use_head_major_gather:
                # Index the non-contiguous [H, pages, page, D] view directly.
                # index_select writes head-major storage, leaving only views
                # before the runtime slices one contiguous KV group at a time.
                selected = cache.permute(2, 0, 1, 3).index_select(1, pages)
                return selected.flatten(1, 2)[:, :cached_length].unsqueeze(0)
            selected = cache.index_select(0, pages)
            flat = selected.reshape(page_count * page_size, cache.size(2), cache.size(3))
            return flat[:cached_length].permute(1, 0, 2).unsqueeze(0).contiguous()

        return _gather(k_cache), _gather(v_cache)

    def _gather_paged_rows_fused_bhsd(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_rows: torch.Tensor,
        page_count: int,
        *,
        validate_bounds: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pages = page_rows[:, :page_count]
        flat_pages = pages.reshape(-1)
        if validate_bounds:
            bounds_ok = torch.all((flat_pages >= 0) & (flat_pages < k_cache.size(0)))
            torch._assert_async(bounds_ok, "page_table entries are outside the KV cache")
        rows = pages.size(0)
        shape = (
            rows,
            k_cache.size(2),
            page_count * k_cache.size(1),
            k_cache.size(3),
        )
        key = (
            k_cache.device.type,
            k_cache.device.index,
            k_cache.dtype,
            *shape,
        )
        buffers = self._paged_bhsd_workspaces.get(key)
        if buffers is None:
            if len(self._paged_bhsd_workspaces) >= 128:
                self._paged_bhsd_workspaces.pop(next(iter(self._paged_bhsd_workspaces)))
            buffers = (
                torch.empty(shape, dtype=k_cache.dtype, device=k_cache.device),
                torch.empty(shape, dtype=v_cache.dtype, device=v_cache.device),
            )
            self._paged_bhsd_workspaces[key] = buffers
        return fused_paged_kv_gather_bhsd(
            k_cache,
            v_cache,
            pages,
            page_count,
            k_out=buffers[0],
            v_out=buffers[1],
        )

    @staticmethod
    def _gather_paged_rows_by_page_count(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_rows: torch.Tensor,
        page_count: int,
        *,
        validate_bounds: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gather one page-count group with shared index-selects.

        if page_count <= 0 or page_rows.ndim != 2:
            raise ValueError("page_rows must be 2-D with a positive page count")
        if page_rows.size(1) < page_count:
            raise ValueError("page_table rows are shorter than the requested page count")
        pages = page_rows[:, :page_count]
        flat_pages = pages.reshape(-1)
        if validate_bounds:
            bounds_ok = torch.all((flat_pages >= 0) & (flat_pages < k_cache.size(0)))
            if flat_pages.is_cuda:
                torch._assert_async(bounds_ok, "page_table entries are outside the KV cache")
            elif not bool(bounds_ok.item()):
                raise ValueError("page_table entries are outside the KV cache")

        rows = pages.size(0)
        page_size = k_cache.size(1)

        def _gather(cache: torch.Tensor) -> torch.Tensor:
            selected = cache.index_select(0, flat_pages)
            flat = selected.reshape(
                rows,
                page_count * page_size,
                cache.size(2),
                cache.size(3),
            )
            return flat.permute(0, 2, 1, 3).contiguous()

        return _gather(k_cache), _gather(v_cache)

    @staticmethod
    def _page_bounds_signature(
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        *,
        cached_lengths: Sequence[int],
        cache_pages: int,
        page_size: int,
        max_seqlen_k: int,
    ) -> tuple[Any, ...]:
        """Fingerprint metadata that can affect which physical pages are read."""

        def tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
            try:
                version: int | None = tensor._version
            except RuntimeError:
                # Inference tensors have no version counter. Their lifetime is
                # still fenced by the adapter-issued materialization epoch.
                version = None
            return (
                id(tensor),
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
                tensor.device,
                version,
            )

        stream = torch.cuda.current_stream(page_table.device) if page_table.is_cuda else None
        return (
            tensor_signature(page_table),
            tensor_signature(seqused_k),
            tuple(cached_lengths),
            int(cache_pages),
            int(page_size),
            int(max_seqlen_k),
            stream,
        )

    @staticmethod
    def _validate_paged_inputs(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
    ) -> None:
        if q.ndim != 4:
            raise ValueError("paged q must use [B, H, S, D]")
        if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
            raise ValueError("paged k/v must use [pages, page_size, H, D]")
        if q.size(1) % k_cache.size(2) != 0 or q.size(3) != k_cache.size(3):
            raise ValueError("paged q/k head counts or head dimensions are incompatible")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("strict paged Attention supports FP16/BF16 only")
        if k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
            raise ValueError("paged q/k/v must share one dtype")
        if not (q.device == k_cache.device == v_cache.device):
            raise ValueError("paged q/k/v must be on one ROCm device")
        if page_table.ndim != 2 or page_table.size(0) != q.size(0):
            raise ValueError("page_table must be 2-D with one row per query")
        if page_table.dtype not in (torch.int32, torch.int64):
            raise ValueError("page_table must be an integer tensor")
        if seqused_k.shape != (q.size(0),):
            raise ValueError("seqused_k must carry one cached length per query")
        if seqused_k.dtype not in (torch.int32, torch.int64):
            raise ValueError("seqused_k must be an integer tensor")
        if page_table.device != q.device or seqused_k.device != q.device:
            raise ValueError("paged Attention metadata must be on the Q device")
        if max_seqlen_k <= 0 or max_seqlen_k > page_table.size(1) * k_cache.size(1):
            raise ValueError("max_seqlen_k exceeds the page table capacity")

    def _run_paged_core_bshd(
        self,
        q: torch.Tensor,
        k_bhsd: torch.Tensor,
        v_bhsd: torch.Tensor,
        *,
        scale: float | None,
        output_dtype: torch.dtype,
        out: torch.Tensor | None,
        collect_lse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], int]:
        """Launch strict CK from fused group-major KV without layout copies."""

        if output_dtype != q.dtype:
            raise ValueError("strict paged Attention output dtype must match Q")
        local_kv_heads = k_bhsd.size(1)
        if local_kv_heads <= 0 or q.size(1) % local_kv_heads:
            raise RuntimeError("paged Q heads must be divisible by local KV heads")
        group_size = q.size(1) // local_kv_heads
        group_outs: list[torch.Tensor] = []
        group_lses: list[torch.Tensor] = []
        core_provenance: dict[str, Any] | None = None
        direct = getattr(self._core, "forward_bshd_with_lse")
        for group in range(local_kv_heads):
            q_lo, q_hi = group * group_size, (group + 1) * group_size
            q_bshd = q[:, q_lo:q_hi].transpose(1, 2)
            k_group_bshd = k_bhsd[:, group : group + 1].transpose(1, 2)
            v_group_bshd = v_bhsd[:, group : group + 1].transpose(1, 2)
            if not q_bshd.is_contiguous():
                q_bshd = q_bshd.contiguous()
            if not k_group_bshd.is_contiguous() or not v_group_bshd.is_contiguous():
                raise RuntimeError("fused paged gather did not produce group-contiguous KV")
            group_out = None if out is None else out[:, q_lo:q_hi]
            result = direct(
                q_bshd,
                k_group_bshd,
                v_group_bshd,
                causal=False,
                scale=scale,
                out=group_out,
            )
            if out is None:
                group_outs.append(result.out)
            if collect_lse:
                group_lses.append(result.lse)
            if core_provenance is None:
                core_provenance = dict(result.provenance)
        if core_provenance is None:
            raise RuntimeError("strict ROCm paged Attention executed no CK launch")
        return (
            out if out is not None else torch.cat(group_outs, dim=1),
            (
                torch.cat(group_lses, dim=1)
                if collect_lse
                else torch.empty((0,), dtype=torch.float32, device=q.device)
            ),
            core_provenance,
            local_kv_heads,
        )

    def _run_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: float | None,
        query_position_ids: torch.Tensor | None,
        key_position_ids: torch.Tensor | None,
        output_dtype: torch.dtype,
        out: torch.Tensor | None = None,
        direct_core_out: bool = False,
        collect_lse: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], int]:
        """Launch the core once per ``(batch row, KV group)`` and concatenate.

        Every launch therefore sees exactly one KV group and its Q heads, so
        the result does not depend on the TP degree that produced the shard.
        """

        local_kv_heads = k.size(1)
        if local_kv_heads <= 0 or q.size(1) % local_kv_heads:
            raise RuntimeError(
                f"local Q heads={q.size(1)} must be divisible by local KV heads={local_kv_heads}"
            )
        group_size = q.size(1) // local_kv_heads
        direct_output = q
        grouped_decode = False
        if direct_core_out:
            if out is None or causal or q.size(2) != 1:
                raise RuntimeError("direct ROCm core output is only valid for paged decode")
            direct_output = out
            if torch.is_grad_enabled() or any(
                tensor.requires_grad for tensor in (q, k, v, direct_output)
            ):
                raise RuntimeError("direct ROCm core output requires disabled gradient mode")
            if not callable(getattr(self._core, "forward_decode_with_lse_into", None)):
                raise RuntimeError("strict ROCm core has no direct decode output entry point")
            grouped_decode = local_kv_heads > 1 and callable(
                getattr(self._core, "forward_grouped_decode_with_lse_into", None)
            )

        if grouped_decode:
            batch_size, _query_heads, query_length, head_dim = q.shape
            key_length = k.size(2)
            grouped_q = q.reshape(
                batch_size,
                local_kv_heads,
                group_size,
                query_length,
                head_dim,
            ).reshape(batch_size * local_kv_heads, group_size, query_length, head_dim)
            grouped_k = k.reshape(batch_size * local_kv_heads, 1, key_length, head_dim)
            grouped_v = v.reshape(batch_size * local_kv_heads, 1, key_length, head_dim)
            grouped_out = direct_output.reshape(
                batch_size,
                local_kv_heads,
                group_size,
                query_length,
                head_dim,
            ).reshape(batch_size * local_kv_heads, group_size, query_length, head_dim)
            result = self._core.forward_grouped_decode_with_lse_into(
                grouped_q,
                grouped_k,
                grouped_v,
                out=grouped_out,
                scale=scale,
                output_dtype=output_dtype,
            )
            if result.out.data_ptr() != grouped_out.data_ptr():
                raise RuntimeError("strict ROCm grouped core did not write to its output view")
            grouped_lse = (
                result.lse.reshape(
                    batch_size,
                    local_kv_heads,
                    group_size,
                    query_length,
                ).reshape(
                    batch_size,
                    local_kv_heads * group_size,
                    query_length,
                )
                if collect_lse
                else torch.empty((0,), dtype=torch.float32, device=q.device)
            )
            return (
                direct_output,
                grouped_lse,
                dict(result.provenance),
                1,
            )

        row_outs: list[torch.Tensor] = []
        row_lses: list[torch.Tensor] = []
        core_provenance: dict[str, Any] | None = None
        launches = 0
        for row in range(q.size(0)):
            row_query_positions = (
                None if query_position_ids is None else query_position_ids[row : row + 1]
            )
            row_key_positions = (
                None if key_position_ids is None else key_position_ids[row : row + 1]
            )
            group_outs: list[torch.Tensor] = []
            group_lses: list[torch.Tensor] = []
            for group in range(local_kv_heads):
                q_lo, q_hi = group * group_size, (group + 1) * group_size
                group_out = None
                if direct_core_out:
                    group_out = direct_output[row : row + 1, q_lo:q_hi]
                    result = self._core.forward_decode_with_lse_into(
                        q[row : row + 1, q_lo:q_hi],
                        k[row : row + 1, group : group + 1],
                        v[row : row + 1, group : group + 1],
                        out=group_out,
                        scale=scale,
                        output_dtype=output_dtype,
                    )
                else:
                    result = self._core.forward_with_lse(
                        q[row : row + 1, q_lo:q_hi],
                        k[row : row + 1, group : group + 1],
                        v[row : row + 1, group : group + 1],
                        causal=causal,
                        scale=scale,
                        key_padding_mask=None,
                        query_position_ids=row_query_positions if causal else None,
                        key_position_ids=row_key_positions if causal else None,
                        output_dtype=output_dtype,
                    )
                if group_out is None:
                    group_outs.append(result.out)
                elif result.out.data_ptr() != group_out.data_ptr():
                    raise RuntimeError("strict ROCm core did not write to its output slice")
                if collect_lse:
                    group_lses.append(result.lse)
                launches += 1
                if core_provenance is None:
                    core_provenance = dict(result.provenance)
            if out is None:
                row_outs.append(torch.cat(group_outs, dim=1))
            elif not direct_core_out:
                torch.cat(group_outs, dim=1, out=out[row : row + 1])
            if collect_lse:
                row_lses.append(
                    group_lses[0]
                    if direct_core_out and len(group_lses) == 1
                    else torch.cat(group_lses, dim=1)
                )

        if core_provenance is None:
            raise RuntimeError("strict ROCm Attention runtime executed no core launch")
        return (
            out if out is not None else torch.cat(row_outs, dim=0),
            (
                row_lses[0]
                if collect_lse and direct_core_out and len(row_lses) == 1
                else (
                    torch.cat(row_lses, dim=0)
                    if collect_lse
                    else torch.empty((0,), dtype=torch.float32, device=q.device)
                )
            ),
            core_provenance,
            launches,
        )

    def _paged_kernel_id(self) -> str:
        return str(getattr(self._core, "paged_kernel_id", "aiter_mha_batch_prefill_non_split_ck"))

    @staticmethod
    def _storage_is_disjoint(output: torch.Tensor, *inputs: torch.Tensor) -> bool:
        output_storage = output.untyped_storage().data_ptr()
        return all(tensor.untyped_storage().data_ptr() != output_storage for tensor in inputs)

    @staticmethod
    def _require_rocm(tensor: torch.Tensor) -> None:
        if tensor.device.type != "cuda" or torch.version.hip is None:
            raise RuntimeError("strict ROCm Attention requires ROCm GPU tensors")

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
            backend="rccl_ag_rs",
            status="implemented",
            expected_blocks=blocks,
            expected_kv_token_range=(0, local_kv_tokens * sharding.cp_world_size),
            query_token_ranges=query_ranges,
        )


__all__ = ["StrictRocmAttentionResult", "StrictRocmAttentionRuntime"]
