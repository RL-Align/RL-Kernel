# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""FlashInfer paged-attention candidate for WS2 PR7.

This module is intentionally opt-in.  It adapts RL-Kernel's
``[B, H, S, D]`` attention tensors and PR6-style paged-KV metadata to
FlashInfer's paged attention wrappers, while recording the three PR7 contract
choices that affect rollout/training alignment:

* Qwen3-exact RoPE fused into attention through ``ROPE_LLAMA``;
* split-KV policy, with auto split rejected when batch invariance is required;
* LSE export and provenance for downstream drift reports.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from rl_engine.backends.cuda.attention.cp_comm import (
    AttentionCPCommunication,
    AttentionCPCommunicationPlan,
    AttentionCPMergedState,
    AttentionCPPartialState,
    AttentionParallelSpec,
)
from rl_engine.backends.cuda.attention.deterministic_attn import DeterministicAttentionCoreResult
from rl_engine.backends.cuda.attention.flash_attn import StrictFlashAttention4Core
from rl_engine.contracts.operators.attention import (
    STRICT_ATTENTION_CORE_ID,
    STRICT_ATTENTION_FA4_SCHEDULE_ID,
    STRICT_ATTENTION_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
    STRICT_ATTENTION_SCHEDULE_ID,
    AttentionContractError,
    SplitKVExecutionPlan,
    SplitKVMode,
    SplitKVRuntimeCoordinate,
    SplitKVRuntimePlanEntry,
    SplitKVRuntimePlanSet,
    SplitKVSpec,
    validate_split_kv_alignment,
)
from rl_engine.reference.attention.cp_attention import (
    AttentionPartialState,
    AttentionRingSchedule,
    DeterministicCPAttentionReferenceOp,
    build_reference_split_kv_runtime_plan_set,
    merge_attention_partial_states,
)
from rl_engine.reference.rope.rope import NativeRoPEOp

RoPEState = Literal["pre_rope", "post_rope"]
FlashInferAttentionMode = Literal["prefill", "decode"]
_FLASHINFER_MODULE = "flashinfer"


class FlashInferUnavailable(RuntimeError):
    """Raised when FlashInfer cannot be imported or lacks required symbols."""


@dataclass(frozen=True)
class FlashInferRoPEFusionConfig:
    """Qwen3 RoPE settings used when FlashInfer performs RoPE inside attention."""

    pos_encoding_mode: str = "ROPE_LLAMA"
    rope_theta: float = 1_000_000.0
    rope_scale: float = 1.0
    rotary_dim: int | None = None
    q_rope_state: RoPEState = "pre_rope"
    k_cache_rope_state: RoPEState = "pre_rope"

    def validate(self, head_dim: int) -> None:
        if self.pos_encoding_mode != "ROPE_LLAMA":
            raise ValueError("PR7 RoPE fusion requires FlashInfer pos_encoding_mode='ROPE_LLAMA'")
        if float(self.rope_theta) != 1_000_000.0:
            raise ValueError("Qwen3-8B RoPE fusion requires rope_theta=1_000_000.0")
        if float(self.rope_scale) != 1.0:
            raise ValueError("Qwen3-8B RoPE fusion requires rope_scale=1.0")
        rotary_dim = head_dim if self.rotary_dim is None else int(self.rotary_dim)
        if rotary_dim != head_dim:
            raise ValueError("FlashInfer PR7 candidate supports full-head Qwen3 RoPE only")
        if self.q_rope_state != "pre_rope" or self.k_cache_rope_state != "pre_rope":
            raise ValueError(
                "FlashInfer ROPE_LLAMA attention fusion expects pre-RoPE Q and pre-RoPE K cache; "
                "post-RoPE tensors would be rotated twice"
            )

    def provenance(self, head_dim: int) -> dict[str, Any]:
        rotary_dim = head_dim if self.rotary_dim is None else int(self.rotary_dim)
        return {
            "rope_fusion": True,
            "rope_fusion_boundary": "flashinfer_attention_kernel",
            "pos_encoding_mode": self.pos_encoding_mode,
            "rope_backend": "flashinfer",
            "rope_theta": float(self.rope_theta),
            "rope_scale": float(self.rope_scale),
            "rotary_dim": rotary_dim,
            "rope_layout": "qwen3_rotate_half_non_interleaved",
            "q_rope_state": self.q_rope_state,
            "k_cache_rope_state": self.k_cache_rope_state,
        }


FlashInferSplitKVPolicy = SplitKVSpec


@dataclass(frozen=True)
class FlashInferPagedAttentionConfig:
    """Runtime knobs for the opt-in FlashInfer paged attention candidate."""

    mode: FlashInferAttentionMode = "prefill"
    causal: bool = True
    kv_layout: str = "NHD"
    softmax_scale: float | None = None
    return_lse: bool = True
    require_batch_invariant: bool = True
    workspace_size_bytes: int = 128 * 1024 * 1024
    rope: FlashInferRoPEFusionConfig = field(default_factory=FlashInferRoPEFusionConfig)
    split_kv: SplitKVSpec = field(default_factory=SplitKVSpec.disabled)
    cp_comm_plan: AttentionCPCommunicationPlan = field(
        default_factory=lambda: AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        )
    )
    require_cp_comm: bool = False
    require_verified_arithmetic: bool = True
    cp_communication: AttentionCPCommunication | None = None
    strict_mode: bool = False
    deterministic_core: Any | None = None
    strict_rope_op: Any | None = None

    def validate(self, *, head_dim: int, query_len: int) -> None:
        if self.mode not in {"prefill", "decode"}:
            raise ValueError("mode must be 'prefill' or 'decode'")
        if self.mode == "decode" and query_len != 1:
            raise ValueError("BatchDecodeWithPagedKVCacheWrapper requires Sq == 1")
        if self.kv_layout != "NHD":
            raise ValueError("PR7 FlashInfer adapter currently supports kv_layout='NHD' only")
        if not self.return_lse:
            raise ValueError("PR7 requires attention-domain LSE export")
        if self.workspace_size_bytes <= 0:
            raise ValueError("workspace_size_bytes must be positive")
        self.rope.validate(head_dim)
        if not isinstance(self.split_kv, SplitKVSpec):
            raise ValueError("split_kv must be a SplitKVSpec")
        if self.require_batch_invariant and self.split_kv.mode is SplitKVMode.AUTO:
            raise ValueError(
                "FlashInfer auto split-KV is not a batch-invariant candidate; "
                "use disabled split-KV or a fixed split size"
            )
        self.cp_comm_plan.validate()
        if self.require_cp_comm:
            if self.cp_comm_plan.status != "implemented":
                raise ValueError(
                    "require_cp_comm=True needs an implemented CP communication plan; "
                    "interface-only plans cannot produce owner-local partial states"
                )
        elif self.cp_comm_plan.status != "interface_only":
            raise ValueError("implemented CP communication plans require require_cp_comm=True")
        if not isinstance(self.require_verified_arithmetic, bool):
            raise ValueError("require_verified_arithmetic must be a bool")
        if not isinstance(self.strict_mode, bool):
            raise ValueError("strict_mode must be a bool")
        if self.strict_mode:
            if not self.require_batch_invariant:
                raise ValueError("strict Attention requires batch invariance")
            if self.split_kv.mode is not SplitKVMode.DISABLED:
                raise ValueError("strict Attention requires Split-KV to be disabled")
            if self.deterministic_core is not None:
                _validate_strict_core(self.deterministic_core)
            if self.require_cp_comm:
                if self.cp_communication is None:
                    raise ValueError("strict CP Attention requires a communication adapter")
                for method_name in (
                    "all_gather_query",
                    "all_gather_kv",
                    "all_gather_position_ids",
                    "reduce_scatter_strict_result",
                ):
                    if not callable(getattr(self.cp_communication, method_name, None)):
                        raise ValueError(
                            "strict CP communication adapter must implement " f"{method_name}"
                        )


@dataclass(frozen=True)
class FlashInferPagedKVPlan:
    """FlashInfer paged-KV tensors derived from PR6-style metadata."""

    qo_indptr: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    kv_seq_lens: torch.Tensor
    seq_lens_q: torch.Tensor
    page_size: int
    physical_page_count_per_batch: int
    logical_block_counts: tuple[int, ...]

    def provenance(self) -> dict[str, Any]:
        return {
            "page_size": self.page_size,
            "physical_page_count_per_batch": self.physical_page_count_per_batch,
            "logical_block_counts": list(self.logical_block_counts),
            "qo_indptr": self.qo_indptr.detach().cpu().tolist(),
            "paged_kv_indptr": self.paged_kv_indptr.detach().cpu().tolist(),
            "paged_kv_indices": self.paged_kv_indices.detach().cpu().tolist(),
            "paged_kv_last_page_len": self.paged_kv_last_page_len.detach().cpu().tolist(),
            "kv_seq_lens": self.kv_seq_lens.detach().cpu().tolist(),
            "seq_lens_q": self.seq_lens_q.detach().cpu().tolist(),
        }


@dataclass(frozen=True)
class FlashInferAttentionResult:
    """Output of the FlashInfer PR7 candidate."""

    out: torch.Tensor
    lse: torch.Tensor
    provenance: dict[str, Any]


def build_flashinfer_paged_kv_plan(
    metadata: Any,
    *,
    batch_size: int,
    query_len: int,
    cache_capacity: int,
    device: torch.device,
) -> FlashInferPagedKVPlan:
    """Convert PR6-style paged metadata to FlashInfer page table tensors."""

    page_size = _positive_int(int(metadata.page_size), "page_size")
    if cache_capacity % page_size != 0:
        raise ValueError("physical KV cache capacity must be divisible by page_size")
    physical_page_count = cache_capacity // page_size
    if metadata.kv_seq_lens.shape != (batch_size,):
        raise ValueError("kv_seq_lens must have shape [B]")
    if metadata.block_table.ndim != 2 or metadata.block_table.size(0) != batch_size:
        raise ValueError("block_table must have shape [B, max_blocks]")

    qo_indptr = [0]
    paged_kv_indptr = [0]
    paged_kv_indices: list[int] = []
    paged_kv_last_page_len: list[int] = []
    kv_seq_lens: list[int] = []
    seq_lens_q: list[int] = []
    logical_block_counts: list[int] = []
    for batch_index in range(batch_size):
        seq_len = _positive_int(int(metadata.kv_seq_lens[batch_index].item()), "kv_seq_len")
        if seq_len > cache_capacity:
            raise ValueError("kv_seq_len must not exceed cache capacity")
        block_count = (seq_len + page_size - 1) // page_size
        if block_count > metadata.block_table.size(1):
            raise ValueError("block_table does not contain enough logical KV blocks")
        kv_seq_lens.append(seq_len)
        seq_lens_q.append(query_len)
        logical_block_counts.append(block_count)
        qo_indptr.append(qo_indptr[-1] + query_len)
        paged_kv_indptr.append(paged_kv_indptr[-1] + block_count)
        last_len = ((seq_len - 1) % page_size) + 1
        paged_kv_last_page_len.append(last_len)
        for logical_block in range(block_count):
            local_page = int(metadata.block_table[batch_index, logical_block].item())
            if local_page < 0 or local_page >= physical_page_count:
                raise ValueError("block_table contains an out-of-range physical page")
            paged_kv_indices.append(batch_index * physical_page_count + local_page)
        active_pages = metadata.block_table[batch_index, :block_count]
        if torch.unique(active_pages).numel() != block_count:
            raise ValueError("active block_table entries must not contain duplicate pages")
        if bool((metadata.block_table[batch_index, block_count:] != -1).any()):
            raise ValueError("unused block_table entries must be -1")
        _validate_metadata_logical_positions(
            metadata,
            batch_index=batch_index,
            seq_len=seq_len,
            page_size=page_size,
            block_count=block_count,
            device=device,
        )

    return FlashInferPagedKVPlan(
        qo_indptr=torch.tensor(qo_indptr, device=device, dtype=torch.int32),
        paged_kv_indptr=torch.tensor(paged_kv_indptr, device=device, dtype=torch.int32),
        paged_kv_indices=torch.tensor(paged_kv_indices, device=device, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor(
            paged_kv_last_page_len,
            device=device,
            dtype=torch.int32,
        ),
        kv_seq_lens=torch.tensor(kv_seq_lens, device=device, dtype=torch.int32),
        seq_lens_q=torch.tensor(seq_lens_q, device=device, dtype=torch.int32),
        page_size=page_size,
        physical_page_count_per_batch=physical_page_count,
        logical_block_counts=tuple(logical_block_counts),
    )


def materialize_flashinfer_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten ``[B, Hkv, P*page, D]`` caches to FlashInfer NHD pages."""

    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have matching shape")
    if k_cache.ndim != 4:
        raise ValueError("k_cache and v_cache must have shape [B, Hkv, cache_capacity, D]")
    batch, heads, cache_capacity, head_dim = k_cache.shape
    if cache_capacity % page_size != 0:
        raise ValueError("cache capacity must be divisible by page_size")
    page_count = cache_capacity // page_size
    k_pages = (
        k_cache.contiguous()
        .reshape(batch, heads, page_count, page_size, head_dim)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch * page_count, page_size, heads, head_dim)
        .contiguous()
    )
    v_pages = (
        v_cache.contiguous()
        .reshape(batch, heads, page_count, page_size, head_dim)
        .permute(0, 2, 3, 1, 4)
        .reshape(batch * page_count, page_size, heads, head_dim)
        .contiguous()
    )
    return k_pages, v_pages


class _NativeFlashInferRuntimeAdapter:
    """Expose strict provenance from FlashInfer's materialized FA2 plan.

    Upstream FlashInfer does not provide the RL-Kernel provenance callbacks.
    Its FA2 scheduler does, however, materialize the request/tile schedule and
    the token chunk size into the wrapper's caller-owned workspace. Read that
    schedule after plan() so strict acceptance describes the kernel plan that
    will actually run instead of merely echoing requested knobs.
    """

    def __init__(self, wrapper: Any, cfg: FlashInferPagedAttentionConfig) -> None:
        self._wrapper = wrapper
        self._cfg = cfg
        self._plan_kwargs: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapper, name)

    def plan(self, *args: Any, **kwargs: Any) -> Any:
        result = self._wrapper.plan(*args, **kwargs)
        self._plan_kwargs = dict(kwargs)
        self._validate_materialized_plan()
        return result

    def get_actual_split_kv_plan(self) -> list[dict[str, Any]]:
        seq_lens, page_size, fixed_split_pages = self._runtime_split_layout()
        result = []
        for seq_len in seq_lens:
            if fixed_split_pages is None:
                boundaries = [(0, (seq_len + page_size - 1) // page_size)]
                mode = SplitKVMode.DISABLED.value
            else:
                page_count = (seq_len + page_size - 1) // page_size
                boundaries = [
                    (start, min(start + fixed_split_pages, page_count))
                    for start in range(0, page_count, fixed_split_pages)
                ]
                mode = SplitKVMode.FIXED.value
            result.append(
                {
                    "mode": mode,
                    "split_size": fixed_split_pages,
                    "split_size_unit": "pages",
                    "boundary_unit": "pages",
                    "boundaries": boundaries,
                    "fallback": False,
                    "fallback_reason": None,
                }
            )
        return result

    def get_actual_split_kv_plan_set(self) -> dict[str, Any]:
        seq_lens, page_size, fixed_split_pages = self._runtime_split_layout()
        parallel = self._cfg.cp_comm_plan.parallel
        entries = []
        for batch_index, total in enumerate(seq_lens):
            owner_ranges = _balanced_token_ranges(total, parallel.cp_world_size)
            for tp_rank in range(parallel.tp_world_size):
                for cp_rank in range(parallel.cp_world_size):
                    for owner_cp_rank, (owner_start, owner_end) in enumerate(owner_ranges):
                        if owner_start % page_size or owner_end % page_size:
                            raise FlashInferUnavailable(
                                "strict FlashInfer CP owner ranges must align to KV pages"
                            )
                        owner_pages = (owner_end - owner_start) // page_size
                        if fixed_split_pages is None:
                            boundaries = [(0, owner_pages)]
                            mode = SplitKVMode.DISABLED.value
                        else:
                            boundaries = [
                                (start, min(start + fixed_split_pages, owner_pages))
                                for start in range(0, owner_pages, fixed_split_pages)
                            ]
                            mode = SplitKVMode.FIXED.value
                        entries.append(
                            {
                                "batch_index": batch_index,
                                "tp_rank": tp_rank,
                                "cp_rank": cp_rank,
                                "owner_cp_rank": owner_cp_rank,
                                "expected_kv_range": [owner_start, owner_end],
                                "mode": mode,
                                "split_size": fixed_split_pages,
                                "split_size_unit": "pages",
                                "boundary_unit": "pages",
                                "boundaries": boundaries,
                                "merge_order": "global_block_index",
                                "accum_dtype": "fp32",
                                "downcast_at": "final_write",
                                "fallback": False,
                                "fallback_reason": None,
                            }
                        )
        return {
            "batch_size": len(seq_lens),
            "tp_world_size": parallel.tp_world_size,
            "cp_world_size": parallel.cp_world_size,
            "total_kv_tokens": list(seq_lens),
            "entries": entries,
        }

    def get_attention_arithmetic_provenance(self) -> dict[str, str]:
        self._validate_materialized_plan()
        return {
            "accum_dtype": "fp32",
            "downcast_at": "final_write",
            "lse_dtype": "fp32",
            "backend_lse_log_base": "2",
            "export_lse_log_base": "e",
            "source": "flashinfer_fa2_materialized_plan",
        }

    @staticmethod
    def normalize_lse(lse: torch.Tensor) -> torch.Tensor:
        """Convert FlashInfer FA2's log2 LSE to the natural-log contract."""

        return lse * math.log(2.0)

    def _runtime_split_layout(self) -> tuple[tuple[int, ...], int, int | None]:
        self._validate_materialized_plan()
        assert self._plan_kwargs is not None
        seq_lens_raw = self._plan_kwargs.get("seq_lens")
        if not isinstance(seq_lens_raw, torch.Tensor):
            raise FlashInferUnavailable("native FlashInfer provenance requires explicit seq_lens")
        seq_lens = tuple(int(value) for value in seq_lens_raw.detach().cpu().tolist())
        page_size = int(self._plan_kwargs["page_size"])
        disabled = bool(self._plan_kwargs.get("disable_split_kv", False))
        fixed_split_pages_raw = self._plan_kwargs.get("fixed_split_size")
        fixed_split_pages = (
            None if disabled or fixed_split_pages_raw is None else int(fixed_split_pages_raw)
        )
        return seq_lens, page_size, fixed_split_pages

    def _validate_materialized_plan(self) -> None:
        if self._plan_kwargs is None:
            raise FlashInferUnavailable("FlashInfer plan() has not materialized a runtime plan")
        if getattr(self._wrapper, "_backend", None) != "fa2":
            raise FlashInferUnavailable(
                "strict native FlashInfer provenance currently requires the FA2 backend"
            )
        plan_info = getattr(self._wrapper, "_plan_info", None)
        if plan_info is None or not hasattr(plan_info, "__getitem__") or len(plan_info) != 15:
            raise FlashInferUnavailable(
                "native FlashInfer FA2 did not expose the expected PrefillPlanInfo"
            )
        seq_lens_raw = self._plan_kwargs.get("seq_lens")
        if not isinstance(seq_lens_raw, torch.Tensor):
            raise FlashInferUnavailable("native FlashInfer provenance requires explicit seq_lens")
        seq_lens = tuple(int(value) for value in seq_lens_raw.detach().cpu().tolist())
        page_size = int(self._plan_kwargs["page_size"])
        disabled = bool(self._plan_kwargs.get("disable_split_kv", False))
        fixed_split_pages_raw = self._plan_kwargs.get("fixed_split_size")
        fixed_split_pages = (
            None if disabled or fixed_split_pages_raw is None else int(fixed_split_pages_raw)
        )
        runtime_split = bool(plan_info[14])
        if disabled and runtime_split:
            raise FlashInferUnavailable(
                "FlashInfer materialized split-KV despite disable_split_kv=True"
            )
        if fixed_split_pages is not None:
            expected_chunk_tokens = fixed_split_pages * page_size
            actual_chunk_tokens = self._workspace_i32(int(plan_info[9]), 1)[0]
            if actual_chunk_tokens != expected_chunk_tokens:
                raise FlashInferUnavailable(
                    "FlashInfer materialized KV chunk differs from fixed_split_size"
                )
            expected_runtime_split = any(seq_len > expected_chunk_tokens for seq_len in seq_lens)
            if runtime_split != expected_runtime_split:
                raise FlashInferUnavailable(
                    "FlashInfer materialized split flag differs from fixed Split-KV plan"
                )
        padded_batch_size = int(plan_info[0])
        request_indices = self._workspace_i32(int(plan_info[4]), padded_batch_size)
        kv_tile_indices = self._workspace_i32(int(plan_info[6]), padded_batch_size)
        actual_tiles = set(zip(request_indices, kv_tile_indices, strict=True))
        expected_tiles = {
            (batch_index, tile_index)
            for batch_index, seq_len in enumerate(seq_lens)
            for tile_index in range(
                1
                if fixed_split_pages is None
                else (
                    (seq_len + fixed_split_pages * page_size - 1) // (fixed_split_pages * page_size)
                )
            )
        }
        if actual_tiles != expected_tiles:
            raise FlashInferUnavailable(
                "FlashInfer materialized request/KV-tile schedule differs from the strict plan"
            )

    def _workspace_i32(self, byte_offset: int, count: int) -> tuple[int, ...]:
        workspace = getattr(self._wrapper, "_pin_memory_int_workspace_buffer", None)
        if not isinstance(workspace, torch.Tensor) or workspace.device.type != "cpu":
            raise FlashInferUnavailable(
                "native FlashInfer did not expose its materialized host plan workspace"
            )
        byte_count = count * torch.tensor([], dtype=torch.int32).element_size()
        values = workspace.narrow(0, byte_offset, byte_count).view(torch.int32)
        return tuple(int(value) for value in values.tolist())


def _balanced_token_ranges(total: int, parts: int) -> tuple[tuple[int, int], ...]:
    base, extra = divmod(total, parts)
    ranges = []
    start = 0
    for index in range(parts):
        end = start + base + (1 if index < extra else 0)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


class FlashInferQwen3PagedAttentionOp:
    """Opt-in FlashInfer paged attention backend candidate for #235 PR7."""

    op_class = "attention"

    def __init__(self, *, flashinfer_module: Any | None = None) -> None:
        self._flashinfer_module = flashinfer_module

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Any,
        *,
        config: FlashInferPagedAttentionConfig | None = None,
    ) -> FlashInferAttentionResult:
        return self.forward(q, k_cache, v_cache, metadata, config=config)

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Any,
        *,
        config: FlashInferPagedAttentionConfig | None = None,
    ) -> FlashInferAttentionResult:
        """Run FlashInfer paged attention and return RL-Kernel shaped tensors.

        Args:
            q: pre-RoPE query tensor, ``[B, Hq, Sq, D]``.
            k_cache: pre-RoPE paged key cache, ``[B, Hkv, cache_capacity, D]``.
            v_cache: paged value cache, ``[B, Hkv, cache_capacity, D]``.
            metadata: PR6 ``DecodeKVCacheMetadata``-compatible object.
            config: PR7 FlashInfer backend knobs.
        """

        _validate_qkv_cache(q, k_cache, v_cache)
        cfg = FlashInferPagedAttentionConfig() if config is None else config
        batch_size, q_heads, query_len, head_dim = q.shape
        kv_heads = k_cache.size(1)
        cfg.validate(head_dim=head_dim, query_len=query_len)
        if cfg.require_cp_comm and cfg.strict_mode:
            return self._run_strict_cp(q, k_cache, v_cache, metadata, cfg)
        if self._flashinfer_module is None and q.device.type != "cuda" and not cfg.require_cp_comm:
            raise FlashInferUnavailable("FlashInfer PR7 candidate requires CUDA tensors")

        plan = build_flashinfer_paged_kv_plan(
            metadata,
            batch_size=batch_size,
            query_len=query_len,
            cache_capacity=k_cache.size(2),
            device=q.device,
        )
        _validate_flashinfer_rope_metadata(metadata, cfg, q)
        _validate_flashinfer_prefix_cache(q, k_cache, v_cache, metadata, cfg)
        if cfg.require_cp_comm:
            return self._forward_deterministic_cp_fallback(
                q,
                k_cache,
                v_cache,
                metadata,
                cfg,
                plan,
            )
        if cfg.strict_mode:
            return self._run_strict_core(q, k_cache, v_cache, metadata, cfg, plan)
        q_flat = q.transpose(1, 2).reshape(batch_size * query_len, q_heads, head_dim).contiguous()
        k_pages, v_pages = materialize_flashinfer_paged_kv_cache(
            k_cache,
            v_cache,
            page_size=plan.page_size,
        )
        wrapper = self._make_wrapper(cfg, q)
        applied_plan_kwargs = self._plan_wrapper(
            wrapper,
            cfg,
            plan,
            q_dtype=q.dtype,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            query_len=query_len,
        )
        actual_split_plans = self._actual_split_kv_plans(wrapper, cfg, plan)
        actual_split_plan_set = self._actual_split_kv_plan_set(
            wrapper,
            cfg,
            plan,
        )
        arithmetic = self._actual_arithmetic_semantics(wrapper, cfg)
        out_flat, lse_flat = self._run_wrapper(wrapper, q_flat, (k_pages, v_pages), cfg)
        self._validate_runtime_outputs(
            out_flat,
            lse_flat,
            q,
            require_fp32_output=cfg.require_cp_comm,
        )
        out = _restore_out(out_flat, batch_size=batch_size, query_len=query_len)
        lse = _restore_lse(
            lse_flat,
            batch_size=batch_size,
            query_len=query_len,
            q_heads=q_heads,
        )
        provenance = {
            "attention_backend": "flashinfer",
            "requested_backend": "flashinfer_qwen3_rope_paged_attention",
            "actual_backend": f"flashinfer_batch_{cfg.mode}_paged_kv",
            "attention_mode": cfg.mode,
            "materialization": "flashinfer_rope_llama_paged_kv",
            "kv_layout": cfg.kv_layout,
            "causal": cfg.causal,
            "softmax_scale": cfg.softmax_scale,
            "lse_domain": "attention",
            "lse_exported": True,
            **arithmetic,
            "fallback": False,
            "fallback_reason": None,
            "paged_kv_policy": "flashinfer_page_table",
        }
        provenance.update(cfg.rope.provenance(head_dim))
        provenance.update(
            _split_kv_provenance(
                cfg.split_kv,
                actual_split_plans,
                applied_plan_kwargs=applied_plan_kwargs,
                require_batch_invariant=cfg.require_batch_invariant,
            )
        )
        provenance["actual_split_kv_plan_set"] = (
            None if actual_split_plan_set is None else actual_split_plan_set.to_dict()
        )
        provenance.update(cfg.cp_comm_plan.provenance())
        provenance["cp_comm_required"] = cfg.require_cp_comm
        provenance.update(plan.provenance())
        return FlashInferAttentionResult(
            out=out.to(dtype=q.dtype),
            lse=lse,
            provenance=provenance,
        )

    @staticmethod
    def _run_strict_core(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Any,
        cfg: FlashInferPagedAttentionConfig,
        paged_plan: FlashInferPagedKVPlan,
    ) -> FlashInferAttentionResult:
        """Use FlashInfer only for paged-KV layout, never Attention arithmetic."""

        core = _resolve_strict_core(cfg)
        _validate_strict_core(core)
        rope = _resolve_strict_rope(cfg)
        logical_k, logical_v, key_positions = _materialize_strict_logical_kv(
            k_cache, v_cache, metadata, paged_plan
        )
        query_positions = getattr(metadata, "query_position_ids", None)
        if not isinstance(query_positions, torch.Tensor):
            raise FlashInferUnavailable(
                "strict Attention requires query_position_ids runtime metadata"
            )

        outputs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        row_provenance: list[dict[str, object]] = []
        for batch_index, seq_len_value in enumerate(paged_plan.kv_seq_lens.tolist()):
            seq_len = int(seq_len_value)
            q_row = q[batch_index : batch_index + 1]
            k_row = logical_k[batch_index : batch_index + 1, :, :seq_len, :]
            v_row = logical_v[batch_index : batch_index + 1, :, :seq_len, :]
            q_pos = query_positions[batch_index : batch_index + 1]
            k_pos = key_positions[batch_index : batch_index + 1, :seq_len]
            q_ready = _apply_strict_rope(rope, q_row, q_pos, cfg.rope.rope_theta)
            k_ready = _apply_strict_rope(rope, k_row, k_pos, cfg.rope.rope_theta)
            result = core.forward_with_lse(
                q_ready,
                k_ready,
                v_row,
                causal=cfg.causal,
                scale=cfg.softmax_scale,
                query_position_ids=q_pos,
                key_position_ids=k_pos,
                output_dtype=q.dtype,
            )
            _validate_strict_core_result(result, core)
            outputs.append(result.out)
            lses.append(result.lse)
            row_provenance.append(dict(result.provenance))

        provenance = _strict_attention_provenance(
            cfg,
            core_provenance=row_provenance[0],
            materialization="flashinfer_paged_kv_layout_shared_core",
            cp_required=False,
            rope=rope,
        )
        provenance["strict_core_row_plans"] = [item["split_kv"] for item in row_provenance]
        provenance.update(cfg.cp_comm_plan.provenance())
        provenance.update(paged_plan.provenance())
        return FlashInferAttentionResult(
            out=torch.cat(outputs, dim=0),
            lse=torch.cat(lses, dim=0),
            provenance=provenance,
        )

    @staticmethod
    def _run_strict_cp(
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        metadata: Any,
        cfg: FlashInferPagedAttentionConfig,
    ) -> FlashInferAttentionResult:
        """AG full Q/K/V, execute the shared core, then RS final Out/LSE."""

        plan = cfg.cp_comm_plan
        communication = cfg.cp_communication
        assert communication is not None
        if any(tensor.requires_grad for tensor in (q_local, k_local, v_local)) and not getattr(
            communication, "supports_autograd", False
        ):
            raise FlashInferUnavailable(
                "strict training requires an autograd-capable self-owned CUDA AG/RS "
                "or ROCm RCCL AG/RS backend"
            )
        query_start, query_end = plan.query_token_ranges[plan.parallel.cp_rank]
        key_start, key_end = _cp_owner_ranges(plan)[plan.parallel.cp_rank]
        if q_local.size(2) != query_end - query_start:
            raise ValueError("strict CP Q must contain only the owner-local query range")
        if k_local.size(2) != key_end - key_start:
            raise ValueError("strict CP K/V must contain only the owner-local KV range")
        if getattr(metadata, "q_rope_state", None) != "pre_rope" or (
            getattr(metadata, "k_cache_rope_state", None) != "pre_rope"
        ):
            raise ValueError("strict CP Attention requires pre-RoPE Q and K")
        local_query_positions = _strict_local_query_positions(
            metadata, q_local, (query_start, query_end)
        )
        local_key_positions = _strict_local_key_positions(metadata, k_local, (key_start, key_end))

        global_q = communication.all_gather_query(q_local, plan)
        global_k, global_v = communication.all_gather_kv(k_local, v_local, plan)
        global_q_positions, global_k_positions = communication.all_gather_position_ids(
            local_query_positions,
            local_key_positions,
            plan,
        )
        expected_q_tokens = plan.query_token_ranges[-1][1]
        expected_kv_range = plan.expected_kv_token_range
        if expected_kv_range is None:
            raise FlashInferUnavailable("strict CP Attention requires an expected KV range")
        expected_k_tokens = expected_kv_range[1] - expected_kv_range[0]
        ring_schedule = AttentionRingSchedule.build(
            expected_k_tokens,
            cp_world_size=plan.parallel.cp_world_size,
            kv_chunk_size=None,
        )
        if global_q.size(2) != expected_q_tokens:
            raise FlashInferUnavailable("strict AG(Q) returned the wrong global width")
        if global_k.size(2) != expected_k_tokens or global_v.shape != global_k.shape:
            raise FlashInferUnavailable("strict AG(K/V) returned the wrong global shape")

        rope = _resolve_strict_rope(cfg)
        q_ready = _apply_strict_rope(rope, global_q, global_q_positions, cfg.rope.rope_theta)
        k_ready = _apply_strict_rope(rope, global_k, global_k_positions, cfg.rope.rope_theta)
        core = _resolve_strict_core(cfg)
        _validate_strict_core(core)

        outputs: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        row_provenance: list[dict[str, object]] = []
        for batch_index in range(global_q.size(0)):
            result = core.forward_with_lse(
                q_ready[batch_index : batch_index + 1],
                k_ready[batch_index : batch_index + 1],
                global_v[batch_index : batch_index + 1],
                causal=cfg.causal,
                scale=cfg.softmax_scale,
                query_position_ids=global_q_positions[batch_index : batch_index + 1],
                key_position_ids=global_k_positions[batch_index : batch_index + 1],
                output_dtype=q_local.dtype,
            )
            _validate_strict_core_result(result, core)
            outputs.append(result.out)
            lses.append(result.lse)
            row_provenance.append(dict(result.provenance))
        full_out = torch.cat(outputs, dim=0)
        full_lse = torch.cat(lses, dim=0)
        local_result = communication.reduce_scatter_strict_result(full_out, full_lse, plan)

        provenance = _strict_attention_provenance(
            cfg,
            core_provenance=row_provenance[0],
            materialization="ag_qkv_positions_shared_core_rs",
            cp_required=True,
            rope=rope,
        )
        provenance.update(plan.provenance())
        provenance.update(
            {
                "strict_core_row_plans": [item["split_kv"] for item in row_provenance],
                "strict_full_qkv_all_gather": True,
                "strict_position_ids_all_gather": True,
                "strict_split_kv": "disabled",
                "strict_comm_autograd": bool(getattr(communication, "supports_autograd", False)),
                "strict_local_query_range": [query_start, query_end],
                "strict_local_kv_range": [key_start, key_end],
                **ring_schedule.provenance(),
                "ring_schedule_default": True,
                "ring_partial_arithmetic": False,
            }
        )
        return FlashInferAttentionResult(
            out=local_result.out,
            lse=local_result.lse,
            provenance=provenance,
        )

    @staticmethod
    def _forward_deterministic_cp_fallback(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: Any,
        cfg: FlashInferPagedAttentionConfig,
        paged_plan: FlashInferPagedKVPlan,
    ) -> FlashInferAttentionResult:
        """Execute the owner-local CP contract without relabeling full-KV output.

        FlashInfer's public paged wrapper does not expose one FP32 partial state
        per CP-owned KV block.  Until it does, strict CP execution uses the
        deterministic reference arithmetic while preserving the production
        communication boundary: AG Q, owner-local partials, ordered FP32 merge,
        and RS of the merged ``(Out, LSE)`` state.
        """

        communication = cfg.cp_communication
        if communication is None:
            raise FlashInferUnavailable(
                "strict CP fallback requires an AttentionCPCommunication implementation"
            )
        cp_plan = cfg.cp_comm_plan
        total_query_tokens = cp_plan.query_token_ranges[-1][1]
        if q.size(2) != total_query_tokens:
            raise ValueError("strict CP fallback expects the complete logical Q sequence before AG")
        kv_seq_lens = tuple(int(value) for value in paged_plan.kv_seq_lens.tolist())
        if len(set(kv_seq_lens)) != 1:
            raise ValueError(
                "strict CP fallback currently requires equal KV lengths across the batch"
            )
        total_kv_tokens = kv_seq_lens[0]
        if cp_plan.expected_kv_token_range != (0, total_kv_tokens):
            raise ValueError("CP block manifest must cover the complete logical KV token range")

        query_start, query_end = cp_plan.query_token_ranges[cp_plan.parallel.cp_rank]
        local_q = q[:, :, query_start:query_end, :].contiguous()
        try:
            gathered_q = communication.all_gather_query(local_q, cp_plan)
        except AttributeError as exc:
            raise FlashInferUnavailable(
                "strict CP communication must implement the query AllGather boundary"
            ) from exc
        if gathered_q.shape != q.shape:
            raise FlashInferUnavailable(
                "query AllGather did not reconstruct the complete logical Q tensor"
            )

        logical_k, logical_v, logical_key_positions = _materialize_logical_kv_cache(
            k_cache,
            v_cache,
            metadata,
            total_kv_tokens=total_kv_tokens,
        )
        rope = NativeRoPEOp()
        q_ready = gathered_q
        if cfg.rope.q_rope_state == "pre_rope":
            q_ready = rope.forward_fp32(
                gathered_q,
                metadata.query_position_ids,
                theta=cfg.rope.rope_theta,
            ).to(gathered_q.dtype)
        k_ready = logical_k
        if cfg.rope.k_cache_rope_state == "pre_rope":
            k_ready = rope.forward_fp32(
                logical_k,
                logical_key_positions,
                theta=cfg.rope.rope_theta,
            ).to(logical_k.dtype)

        reference = DeterministicCPAttentionReferenceOp()
        local_states = []
        for block in cp_plan.expected_blocks:
            if block.owner_cp_rank != cp_plan.parallel.cp_rank:
                continue
            partial = reference.local_partial_state(
                q_ready,
                k_ready[:, :, block.kv_block_start : block.kv_block_end, :],
                logical_v[:, :, block.kv_block_start : block.kv_block_end, :],
                q_start=0,
                k_start=block.kv_block_start,
                total_kv_len=total_kv_tokens,
                total_query_len=q_ready.size(2),
                causal=cfg.causal,
                scale=cfg.softmax_scale,
                query_position_offsets=metadata.query_position_ids[:, 0],
                key_position_offsets=logical_key_positions[:, 0],
            )
            local_states.append(
                AttentionCPPartialState(
                    out=partial.out,
                    lse=partial.lse,
                    block=block,
                )
            )
        gathered_states = communication.all_gather_partial_states(
            tuple(local_states),
            cp_plan,
        )
        merged = merge_attention_partial_states(
            [
                AttentionPartialState(
                    out=state.out,
                    lse=state.lse,
                    block_start=state.block.kv_block_start,
                    block_end=state.block.kv_block_end,
                )
                for state in gathered_states
            ]
        )
        local = communication.reduce_scatter_merged_state(
            AttentionCPMergedState(out=merged.out, lse=merged.lse),
            cp_plan,
        )

        kv_chunk_size = (
            cfg.split_kv.fixed_split_size if cfg.split_kv.mode is SplitKVMode.FIXED else None
        )
        runtime_plan_set = build_reference_split_kv_runtime_plan_set(
            kv_seq_lens,
            tp_world_size=cp_plan.parallel.tp_world_size,
            cp_world_size=cp_plan.parallel.cp_world_size,
            kv_chunk_size=kv_chunk_size,
            backend="deterministic_cp_fallback",
        )
        actual_plans = tuple(
            cfg.split_kv.resolve(length, backend="deterministic_cp_fallback")
            for length in kv_seq_lens
        )
        applied_split_knobs = (
            {"fixed_split_size": cfg.split_kv.fixed_split_size}
            if cfg.split_kv.mode is SplitKVMode.FIXED
            else {"disable_split_kv": True}
        )
        provenance = {
            "attention_backend": "deterministic_cp_fallback",
            "requested_backend": "flashinfer_qwen3_rope_paged_attention",
            "actual_backend": "rlkernel_deterministic_cp_reference",
            "attention_mode": cfg.mode,
            "materialization": "logical_paged_kv_owner_local",
            "kv_layout": cfg.kv_layout,
            "causal": cfg.causal,
            "softmax_scale": cfg.softmax_scale,
            "lse_domain": "attention",
            "lse_exported": True,
            "accum_dtype": "fp32",
            "downcast_at": "final_write",
            "lse_dtype": "fp32",
            "arithmetic_plan_source": "rlkernel_deterministic_cp_reference",
            "arithmetic_semantics_verified": True,
            "fallback": True,
            "fallback_reason": "flashinfer_owner_local_cp_partial_api_unavailable",
            "paged_kv_policy": "validated_logical_page_table",
            "cp_comm_required": True,
            "query_ag": "cp_rank_order",
            "query_range": [query_start, query_end],
        }
        provenance.update(cfg.rope.provenance(q.size(-1)))
        provenance.update(
            _split_kv_provenance(
                cfg.split_kv,
                actual_plans,
                applied_plan_kwargs=applied_split_knobs,
                require_batch_invariant=cfg.require_batch_invariant,
            )
        )
        provenance["actual_split_kv_plan_set"] = runtime_plan_set.to_dict()
        provenance.update(cp_plan.provenance())
        provenance.update(paged_plan.provenance())
        return FlashInferAttentionResult(
            out=local.out.to(dtype=q.dtype),
            lse=local.lse,
            provenance=provenance,
        )

    def _load_flashinfer(self) -> Any:
        if self._flashinfer_module is not None:
            return self._flashinfer_module
        try:
            self._flashinfer_module = importlib.import_module(_FLASHINFER_MODULE)
        except (ImportError, OSError, RuntimeError) as exc:
            raise FlashInferUnavailable(str(exc)) from exc
        return self._flashinfer_module

    def _make_wrapper(self, cfg: FlashInferPagedAttentionConfig, q: torch.Tensor) -> Any:
        module = self._load_flashinfer()
        namespace_name = "decode" if cfg.mode == "decode" else "prefill"
        class_name = (
            "BatchDecodeWithPagedKVCacheWrapper"
            if cfg.mode == "decode"
            else "BatchPrefillWithPagedKVCacheWrapper"
        )
        namespace = getattr(module, namespace_name, None)
        wrapper_cls = getattr(namespace, class_name, None) if namespace is not None else None
        if wrapper_cls is None:
            raise FlashInferUnavailable(f"flashinfer.{namespace_name}.{class_name} is unavailable")

        workspace = torch.zeros(cfg.workspace_size_bytes, dtype=torch.uint8, device=q.device)
        constructor_kwargs: dict[str, Any] = {"kv_layout": cfg.kv_layout}
        if cfg.mode == "decode":
            constructor_kwargs["use_tensor_cores"] = True
        try:
            wrapper = wrapper_cls(workspace, **constructor_kwargs)
        except TypeError:
            try:
                wrapper = wrapper_cls(
                    float_workspace_buffer=workspace,
                    **constructor_kwargs,
                )
            except TypeError:
                constructor_kwargs.pop("use_tensor_cores", None)
                try:
                    wrapper = wrapper_cls(workspace, **constructor_kwargs)
                except TypeError as exc:
                    raise FlashInferUnavailable(
                        f"could not instantiate flashinfer.{namespace_name}.{class_name}"
                    ) from exc
        if type(wrapper).__module__.startswith("flashinfer."):
            return _NativeFlashInferRuntimeAdapter(wrapper, cfg)
        return wrapper

    @staticmethod
    def _plan_wrapper(
        wrapper: Any,
        cfg: FlashInferPagedAttentionConfig,
        plan: FlashInferPagedKVPlan,
        *,
        q_dtype: torch.dtype,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        query_len: int,
    ) -> dict[str, Any]:
        plan_kwargs = {
            "num_qo_heads": q_heads,
            "num_kv_heads": kv_heads,
            "page_size": plan.page_size,
            "pos_encoding_mode": cfg.rope.pos_encoding_mode,
            "rope_scale": float(cfg.rope.rope_scale),
            "rope_theta": float(cfg.rope.rope_theta),
            "q_data_type": q_dtype,
            "kv_data_type": q_dtype,
            "o_data_type": torch.float32 if cfg.require_cp_comm else q_dtype,
            "seq_lens": plan.kv_seq_lens,
        }
        if cfg.mode == "decode":
            plan_kwargs.update(
                {
                    "indptr": plan.paged_kv_indptr,
                    "indices": plan.paged_kv_indices,
                    "last_page_len": plan.paged_kv_last_page_len,
                    "head_dim": head_dim,
                    "q_len_per_req": query_len,
                }
            )
        else:
            plan_kwargs.update(
                {
                    "qo_indptr": plan.qo_indptr,
                    "paged_kv_indptr": plan.paged_kv_indptr,
                    "paged_kv_indices": plan.paged_kv_indices,
                    "paged_kv_last_page_len": plan.paged_kv_last_page_len,
                    "head_dim_qk": head_dim,
                    "causal": cfg.causal,
                    "seq_lens_q": plan.seq_lens_q,
                }
            )
        scale = cfg.softmax_scale
        if scale is not None:
            plan_kwargs["sm_scale"] = float(scale)
        plan_kwargs.update(
            _flashinfer_split_kv_plan_kwargs(
                cfg.split_kv,
                page_size=plan.page_size,
            )
        )
        applied = _call_with_supported_kwargs(wrapper.plan, plan_kwargs, return_applied=True)
        assert isinstance(applied, dict)
        if cfg.split_kv.mode is SplitKVMode.FIXED and "fixed_split_size" not in applied:
            raise FlashInferUnavailable(
                "FlashInfer plan() did not accept required Split-KV knob 'fixed_split_size'"
            )
        if cfg.split_kv.mode is SplitKVMode.DISABLED and "disable_split_kv" not in applied:
            raise FlashInferUnavailable(
                "FlashInfer plan() did not accept required Split-KV knob 'disable_split_kv'"
            )
        return applied

    @staticmethod
    def _actual_split_kv_plans(
        wrapper: Any,
        cfg: FlashInferPagedAttentionConfig,
        plan: FlashInferPagedKVPlan,
    ) -> tuple[SplitKVExecutionPlan, ...]:
        getter = getattr(wrapper, "get_actual_split_kv_plan", None)
        if not callable(getter):
            if cfg.split_kv.mode is SplitKVMode.DISABLED:
                return tuple(
                    cfg.split_kv.resolve(int(seq_len), backend="flashinfer_disabled_verified")
                    for seq_len in plan.kv_seq_lens.tolist()
                )
            if cfg.require_batch_invariant:
                raise FlashInferUnavailable(
                    "strict fixed Split-KV consistency requires runtime actual-plan provenance; "
                    "FlashInfer wrapper has no get_actual_split_kv_plan() callback. A requested "
                    "max-splits/count knob is not proof of token boundaries"
                )
            return tuple(
                cfg.split_kv.resolve(int(seq_len), backend="flashinfer_requested_only")
                for seq_len in plan.kv_seq_lens.tolist()
            )
        raw_plans = getter()
        if not isinstance(raw_plans, (list, tuple)) or len(raw_plans) != len(plan.kv_seq_lens):
            raise FlashInferUnavailable(
                "get_actual_split_kv_plan() must return one plan per batch request"
            )
        result: list[SplitKVExecutionPlan] = []
        for batch_index, (raw, seq_len) in enumerate(
            zip(raw_plans, plan.kv_seq_lens.tolist(), strict=True)
        ):
            if not isinstance(raw, dict):
                raise FlashInferUnavailable("actual Split-KV runtime plan entries must be dicts")
            try:
                required_keys = {
                    "mode",
                    "split_size",
                    "boundaries",
                    "fallback",
                    "fallback_reason",
                }
                missing_keys = sorted(required_keys.difference(raw))
                if missing_keys:
                    raise FlashInferUnavailable(
                        "actual Split-KV runtime plan is missing required fields: "
                        + ", ".join(missing_keys)
                    )
                execution = SplitKVExecutionPlan(
                    requested_mode=cfg.split_kv.mode,
                    requested_split_size=cfg.split_kv.fixed_split_size,
                    actual_mode=raw.get("mode"),
                    actual_split_size=(
                        None
                        if raw.get("split_size") is None
                        else _flashinfer_split_size_tokens(
                            raw.get("split_size"),
                            page_size=plan.page_size,
                            unit=raw.get("split_size_unit"),
                        )
                    ),
                    boundaries=_normalize_flashinfer_split_boundaries(
                        raw.get("boundaries", ()),
                        page_size=plan.page_size,
                        seq_len=int(seq_len),
                        unit=raw.get("boundary_unit"),
                    ),
                    backend="flashinfer",
                    source="runtime_callback",
                    fallback=raw["fallback"],
                    fallback_reason=raw["fallback_reason"],
                )
            except AttentionContractError as exc:
                raise FlashInferUnavailable(
                    f"invalid actual Split-KV plan for batch {batch_index}: {exc}"
                ) from exc
            expected = cfg.split_kv.resolve(int(seq_len), backend="flashinfer_contract")
            if cfg.require_batch_invariant:
                try:
                    validate_split_kv_alignment(expected, execution)
                except AttentionContractError as exc:
                    raise FlashInferUnavailable(
                        f"FlashInfer actual Split-KV plan for batch {batch_index} "
                        f"does not match the requested strict logical plan: {exc}"
                    ) from exc
            result.append(execution)
        return tuple(result)

    @staticmethod
    def _actual_split_kv_plan_set(
        wrapper: Any,
        cfg: FlashInferPagedAttentionConfig,
        plan: FlashInferPagedKVPlan,
    ) -> SplitKVRuntimePlanSet | None:
        getter = getattr(wrapper, "get_actual_split_kv_plan_set", None)
        if not callable(getter):
            if cfg.require_batch_invariant:
                raise FlashInferUnavailable(
                    "strict Split-KV consistency requires a complete "
                    "batch/TP/CP/owner "
                    "runtime plan set; FlashInfer wrapper has no "
                    "get_actual_split_kv_plan_set() callback"
                )
            return None
        raw = getter()
        if not isinstance(raw, dict):
            raise FlashInferUnavailable("get_actual_split_kv_plan_set() must return a dict")
        try:
            entries = tuple(
                SplitKVRuntimePlanEntry(
                    coordinate=SplitKVRuntimeCoordinate(
                        batch_index=int(entry["batch_index"]),
                        tp_rank=int(entry["tp_rank"]),
                        cp_rank=int(entry["cp_rank"]),
                        owner_cp_rank=int(entry["owner_cp_rank"]),
                    ),
                    expected_kv_range=tuple(entry["expected_kv_range"]),
                    execution=SplitKVExecutionPlan(
                        requested_mode=cfg.split_kv.mode,
                        requested_split_size=cfg.split_kv.fixed_split_size,
                        actual_mode=entry["mode"],
                        actual_split_size=(
                            None
                            if entry["split_size"] is None
                            else _flashinfer_split_size_tokens(
                                entry["split_size"],
                                page_size=plan.page_size,
                                unit=entry.get("split_size_unit"),
                            )
                        ),
                        boundaries=_normalize_flashinfer_split_boundaries(
                            entry["boundaries"],
                            page_size=plan.page_size,
                            seq_len=int(
                                entry["expected_kv_range"][1] - entry["expected_kv_range"][0]
                            ),
                            unit=entry.get("boundary_unit"),
                            offset=int(entry["expected_kv_range"][0]),
                        ),
                        merge_order=entry["merge_order"],
                        acc_dtype=entry["accum_dtype"],
                        downcast_at=entry["downcast_at"],
                        backend="flashinfer",
                        source="runtime_plan_set_callback",
                        fallback=entry["fallback"],
                        fallback_reason=entry["fallback_reason"],
                    ),
                )
                for entry in raw["entries"]
            )
            plan_set = SplitKVRuntimePlanSet(
                batch_size=int(raw["batch_size"]),
                tp_world_size=int(raw["tp_world_size"]),
                cp_world_size=int(raw["cp_world_size"]),
                total_kv_tokens=tuple(int(value) for value in raw["total_kv_tokens"]),
                entries=entries,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FlashInferUnavailable(
                f"invalid FlashInfer runtime Split-KV plan set: {exc}"
            ) from exc
        parallel = cfg.cp_comm_plan.parallel
        expected_topology = (
            len(plan.kv_seq_lens),
            parallel.tp_world_size,
            parallel.cp_world_size,
            tuple(int(value) for value in plan.kv_seq_lens.tolist()),
        )
        actual_topology = (
            plan_set.batch_size,
            plan_set.tp_world_size,
            plan_set.cp_world_size,
            plan_set.total_kv_tokens,
        )
        if actual_topology != expected_topology:
            raise FlashInferUnavailable(
                "FlashInfer runtime Split-KV plan-set topology does not match the request"
            )
        if cfg.require_batch_invariant:
            for entry in plan_set.entries:
                start, end = entry.expected_kv_range
                expected_local = cfg.split_kv.resolve(
                    end - start,
                    backend="flashinfer_contract",
                )
                expected = SplitKVExecutionPlan(
                    requested_mode=expected_local.requested_mode,
                    requested_split_size=expected_local.requested_split_size,
                    actual_mode=expected_local.actual_mode,
                    actual_split_size=expected_local.actual_split_size,
                    boundaries=tuple(
                        (start + local_start, start + local_end)
                        for local_start, local_end in expected_local.boundaries
                    ),
                    backend="flashinfer_contract",
                    source="contract_exact",
                )
                try:
                    validate_split_kv_alignment(expected, entry.execution)
                except AttentionContractError as exc:
                    raise FlashInferUnavailable(
                        "FlashInfer runtime Split-KV plan-set entry does not match "
                        f"the strict owner-local plan at {entry.coordinate}: {exc}"
                    ) from exc
        return plan_set

    @staticmethod
    def _actual_arithmetic_semantics(
        wrapper: Any,
        cfg: FlashInferPagedAttentionConfig,
    ) -> dict[str, Any]:
        getter = getattr(wrapper, "get_attention_arithmetic_provenance", None)
        if not callable(getter):
            if cfg.require_verified_arithmetic:
                raise FlashInferUnavailable(
                    "strict attention consistency requires runtime arithmetic provenance; "
                    "FlashInfer wrapper has no get_attention_arithmetic_provenance() callback"
                )
            return {
                "accum_dtype": None,
                "downcast_at": None,
                "lse_dtype": None,
                "arithmetic_plan_source": "unverified_backend_internal",
                "arithmetic_semantics_verified": False,
            }
        raw = getter()
        if not isinstance(raw, dict):
            raise FlashInferUnavailable("get_attention_arithmetic_provenance() must return a dict")
        required = {
            "accum_dtype": "fp32",
            "downcast_at": "final_write",
            "lse_dtype": "fp32",
        }
        mismatches = [key for key, expected in required.items() if raw.get(key) != expected]
        source = raw.get("source")
        if not isinstance(source, str) or not source.strip():
            mismatches.append("source")
        if mismatches:
            raise FlashInferUnavailable(
                "FlashInfer runtime arithmetic semantics do not satisfy the strict "
                "attention contract: " + ", ".join(mismatches)
            )
        result = {
            **required,
            "arithmetic_plan_source": source,
            "arithmetic_semantics_verified": True,
        }
        for key in ("backend_lse_log_base", "export_lse_log_base"):
            if key in raw:
                result[key] = raw[key]
        return result

    @staticmethod
    def _run_wrapper(
        wrapper: Any,
        q_flat: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        cfg: FlashInferPagedAttentionConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(wrapper, "run_return_lse"):
            result = wrapper.run_return_lse(q_flat, paged_kv_cache)
        else:
            result = _call_with_supported_kwargs(
                wrapper.run,
                {
                    "q": q_flat,
                    "paged_kv_cache": paged_kv_cache,
                    "return_lse": cfg.return_lse,
                },
            )
        if not isinstance(result, tuple) or len(result) != 2:
            raise FlashInferUnavailable("FlashInfer PR7 candidate must return (out, lse)")
        out_flat, lse_flat = result
        normalize_lse = getattr(wrapper, "normalize_lse", None)
        if callable(normalize_lse):
            lse_flat = normalize_lse(lse_flat)
        return out_flat, lse_flat

    @staticmethod
    def _validate_runtime_outputs(
        out_flat: torch.Tensor,
        lse_flat: torch.Tensor,
        q: torch.Tensor,
        *,
        require_fp32_output: bool,
    ) -> None:
        if not isinstance(out_flat, torch.Tensor) or not isinstance(lse_flat, torch.Tensor):
            raise FlashInferUnavailable("FlashInfer output and LSE must be tensors")
        if out_flat.device != q.device or lse_flat.device != q.device:
            raise FlashInferUnavailable("FlashInfer output and LSE must remain on the query device")
        expected_out_dtype = torch.float32 if require_fp32_output else q.dtype
        if out_flat.dtype != expected_out_dtype:
            raise FlashInferUnavailable(
                "FlashInfer final output dtype does not match the requested output dtype"
            )
        if lse_flat.dtype != torch.float32:
            raise FlashInferUnavailable("FlashInfer attention-domain LSE must be FP32")


def _validate_strict_core(core: Any) -> None:
    if not callable(getattr(core, "forward_with_lse", None)):
        raise ValueError("strict Attention core must implement forward_with_lse")
    expected_schedules = {
        STRICT_ATTENTION_PRODUCTION_CORE_ID: STRICT_ATTENTION_FA4_SCHEDULE_ID,
        STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID: STRICT_ATTENTION_ROCM_SCHEDULE_ID,
        STRICT_ATTENTION_CORE_ID: STRICT_ATTENTION_SCHEDULE_ID,
    }
    core_id = getattr(core, "core_id", None)
    if core_id not in expected_schedules:
        raise ValueError("strict Attention core ID is not an exact supported identity")
    if getattr(core, "strict_schedule", None) != expected_schedules[core_id]:
        raise ValueError("strict Attention core schedule does not match its exact core identity")
    required = {
        "merge_order": "global_block_index",
        "accum_dtype": "fp32",
        "downcast_at": "final_write",
        "fallback": False,
    }
    mismatches = [
        name for name, expected in required.items() if getattr(core, name, None) != expected
    ]
    if mismatches:
        raise ValueError(
            "strict Attention core has incompatible arithmetic identity: " + ", ".join(mismatches)
        )
    if core_id in {STRICT_ATTENTION_PRODUCTION_CORE_ID, STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID}:
        production_required = {
            "native_attention_arithmetic": True,
            "num_splits": 1,
            "deterministic_backward": True,
            "production_ready": True,
            "reference_only": False,
        }
        production_mismatches = [
            name
            for name, expected in production_required.items()
            if getattr(core, name, None) != expected
        ]
        if production_mismatches:
            raise ValueError(
                "strict production core has incompatible controls: "
                + ", ".join(production_mismatches)
            )
    if (
        core_id == STRICT_ATTENTION_PRODUCTION_CORE_ID
        and getattr(core, "backend_id", None) != "flash_attention_4.cute"
    ):
        raise ValueError("strict CUDA production core must be FlashAttention-4 CuTe")
    if core_id == STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID:
        if getattr(core, "backend_id", None) != "aiter.rocm.ck_dense_mha":
            raise ValueError("strict ROCm production core must be AITER CK dense MHA")
        if getattr(core, "split_kv_control", None) != "dense_non_split_api":
            raise ValueError("strict ROCm production core must use the non-Split-K CK API")


def _resolve_strict_core(cfg: FlashInferPagedAttentionConfig) -> Any:
    if cfg.deterministic_core is not None:
        return cfg.deterministic_core
    if torch.version.hip is not None:
        from rl_engine.backends.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore

        return StrictRocmAiterCKAttentionCore(split_kv=cfg.split_kv)
    return StrictFlashAttention4Core(split_kv=cfg.split_kv)


def _validate_strict_core_result(
    result: Any,
    core: Any,
) -> None:
    if not isinstance(result, DeterministicAttentionCoreResult):
        raise FlashInferUnavailable(
            "strict Attention core must return DeterministicAttentionCoreResult"
        )
    if result.out.dtype not in (torch.float16, torch.bfloat16):
        raise FlashInferUnavailable("strict Attention core output must be FP16/BF16")
    if result.lse.dtype is not torch.float32:
        raise FlashInferUnavailable("strict Attention core LSE must be FP32")
    expected = {
        "strict_core_id": core.core_id,
        "strict_schedule": core.strict_schedule,
        "attention_backend": core.backend_id,
        "merge_order": core.merge_order,
        "accum_dtype": core.accum_dtype,
        "downcast_at": core.downcast_at,
        "fallback": False,
        "native_attention_arithmetic": core.native_attention_arithmetic,
    }
    mismatches = [name for name, value in expected.items() if result.provenance.get(name) != value]
    if mismatches:
        raise FlashInferUnavailable(
            "strict core result changed its declared arithmetic identity: " + ", ".join(mismatches)
        )
    split_plan = result.provenance.get("split_kv")
    if not isinstance(split_plan, dict) or split_plan.get("actual_split_kv_policy") != (
        SplitKVMode.DISABLED.value
    ):
        raise FlashInferUnavailable("strict core result did not prove Split-KV disabled")


def _resolve_strict_rope(cfg: FlashInferPagedAttentionConfig) -> Any:
    if cfg.strict_rope_op is not None:
        return cfg.strict_rope_op
    try:
        from rl_engine.backends.cuda.rope.rope import RocmDeterministicRoPEOp, RoPESM90Op

        if torch.version.hip is not None:
            return RocmDeterministicRoPEOp()
        return RoPESM90Op()
    except (ImportError, RuntimeError) as exc:
        raise FlashInferUnavailable(
            "strict Attention requires the RL-Kernel deterministic RoPE operator"
        ) from exc


def _apply_strict_rope(
    rope: Any,
    x: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
) -> torch.Tensor:
    """Execute one batch row at a time to preserve the strict row schedule."""

    if position_ids.shape != (x.size(0), x.size(2)):
        raise ValueError("strict RoPE position IDs must have shape [B,S]")
    rows = []
    for batch_index in range(x.size(0)):
        row = x[batch_index : batch_index + 1]
        positions = position_ids[batch_index]
        try:
            rotated = rope(row, positions, theta=float(theta))
        except TypeError:
            rotated = rope(row, positions)
        if not isinstance(rotated, torch.Tensor) or rotated.shape != row.shape:
            raise FlashInferUnavailable("strict RoPE returned an invalid tensor")
        if rotated.dtype != row.dtype or rotated.device != row.device:
            raise FlashInferUnavailable("strict RoPE changed dtype or device")
        rows.append(rotated)
    return torch.cat(rows, dim=0)


def _materialize_strict_logical_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    metadata: Any,
    plan: FlashInferPagedKVPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Restore each paged cache row to logical order without padding arithmetic."""

    max_len = int(plan.kv_seq_lens.max().item())
    logical_k = torch.zeros(
        (k_cache.size(0), k_cache.size(1), max_len, k_cache.size(3)),
        dtype=k_cache.dtype,
        device=k_cache.device,
    )
    logical_v = torch.zeros_like(logical_k)
    positions = torch.full(
        (k_cache.size(0), max_len),
        -1,
        dtype=torch.long,
        device=k_cache.device,
    )
    page_size = int(metadata.page_size)
    for batch_index, seq_len_value in enumerate(plan.kv_seq_lens.tolist()):
        seq_len = int(seq_len_value)
        block_count = (seq_len + page_size - 1) // page_size
        token_index = torch.arange(seq_len, device=k_cache.device, dtype=torch.long)
        pages = metadata.block_table[batch_index, :block_count].long()
        slots = pages[token_index // page_size] * page_size + token_index % page_size
        logical_k[batch_index, :, :seq_len, :] = k_cache[batch_index, :, slots, :]
        logical_v[batch_index, :, :seq_len, :] = v_cache[batch_index, :, slots, :]
        source_positions = getattr(metadata, "key_position_ids", None)
        if not isinstance(source_positions, torch.Tensor):
            source_positions = getattr(metadata, "global_token_positions", None)
        if not isinstance(source_positions, torch.Tensor):
            raise FlashInferUnavailable(
                "strict Attention requires key_position_ids runtime metadata"
            )
        positions[batch_index, :seq_len] = source_positions[batch_index, slots].long()
    return logical_k, logical_v, positions


def _cp_owner_ranges(
    plan: AttentionCPCommunicationPlan,
) -> tuple[tuple[int, int], ...]:
    ranges = []
    for owner_rank in range(plan.parallel.cp_world_size):
        blocks = sorted(
            (
                block
                for block in plan.expected_blocks
                if block.owner_cp_rank == owner_rank
                and block.owner_tp_rank == plan.parallel.tp_rank
            ),
            key=lambda block: block.kv_block_start,
        )
        if not blocks:
            raise ValueError(f"strict CP manifest has no blocks for owner {owner_rank}")
        start = blocks[0].kv_block_start
        cursor = start
        for block in blocks:
            if block.kv_block_start != cursor:
                raise ValueError("strict CP owner blocks must form a contiguous range")
            cursor = block.kv_block_end
        ranges.append((start, cursor))
    expected = plan.expected_kv_token_range
    if expected is None or ranges[0][0] != expected[0] or ranges[-1][1] != expected[1]:
        raise ValueError("strict CP owner ranges do not cover expected_kv_token_range")
    for left, right in zip(ranges, ranges[1:], strict=False):
        if left[1] != right[0]:
            raise ValueError("strict CP owner ranges must be gap-free and rank ordered")
    return tuple(ranges)


def _strict_local_query_positions(
    metadata: Any,
    q_local: torch.Tensor,
    query_range: tuple[int, int],
) -> torch.Tensor:
    positions = getattr(metadata, "query_position_ids", None)
    if not isinstance(positions, torch.Tensor):
        raise ValueError("strict CP requires query_position_ids")
    if positions.shape == (q_local.size(0), q_local.size(2)):
        return positions.contiguous()
    start, end = query_range
    if positions.ndim == 2 and positions.size(0) == q_local.size(0) and positions.size(1) >= end:
        return positions[:, start:end].contiguous()
    raise ValueError("strict CP query_position_ids do not match local query ownership")


def _strict_local_key_positions(
    metadata: Any,
    k_local: torch.Tensor,
    key_range: tuple[int, int],
) -> torch.Tensor:
    positions = getattr(metadata, "key_position_ids", None)
    if not isinstance(positions, torch.Tensor):
        positions = getattr(metadata, "global_token_positions", None)
    if not isinstance(positions, torch.Tensor):
        raise ValueError("strict CP requires key_position_ids")
    if positions.shape == (k_local.size(0), k_local.size(2)):
        return positions.contiguous()
    start, end = key_range
    if positions.ndim == 2 and positions.size(0) == k_local.size(0) and positions.size(1) >= end:
        return positions[:, start:end].contiguous()
    raise ValueError("strict CP key_position_ids do not match local KV ownership")


def _strict_attention_provenance(
    cfg: FlashInferPagedAttentionConfig,
    *,
    core_provenance: dict[str, object],
    materialization: str,
    cp_required: bool,
    rope: Any,
) -> dict[str, Any]:
    communication_id = getattr(cfg.cp_communication, "backend_id", "unknown")
    communication_backend = (
        "self_owned_cuda_ag_rs" if communication_id == "cuda_ag_rs" else communication_id
    )
    expected_communication = "rccl_ag_rs" if torch.version.hip is not None else "cuda_ag_rs"
    return {
        "attention_backend": core_provenance["attention_backend"],
        "requested_backend": "flashinfer_layout_adapter",
        "actual_backend": core_provenance["attention_backend"],
        "adapter_backend": "flashinfer",
        "attention_mode": cfg.mode,
        "materialization": materialization,
        "causal": cfg.causal,
        "softmax_scale": cfg.softmax_scale,
        "lse_domain": "attention",
        "lse_exported": True,
        "lse_dtype": "fp32",
        "strict_mode": True,
        "strict_core_id": core_provenance["strict_core_id"],
        "strict_schedule": core_provenance["strict_schedule"],
        "accum_dtype": core_provenance["accum_dtype"],
        "downcast_at": core_provenance["downcast_at"],
        "arithmetic_plan_source": core_provenance.get(
            "fa_api_source",
            core_provenance.get("aiter_api_source", "rlkernel_reference_core"),
        ),
        "arithmetic_semantics_verified": True,
        "native_attention_arithmetic": core_provenance["native_attention_arithmetic"],
        "fallback": False,
        "fallback_reason": None,
        "rope_backend": getattr(rope, "backend_id", "rlkernel.unknown.rope"),
        "rope_theta": float(cfg.rope.rope_theta),
        "rotary_dim": cfg.rope.rotary_dim,
        "rope_fusion": False,
        "rope_fusion_boundary": "rlkernel_rope_then_attention",
        "q_rope_state": "post_rope",
        "k_cache_rope_state": "post_rope",
        "batch_invariant_claim": "strict_runtime_verified",
        "cp_comm_required": cp_required,
        "communication_backend": communication_backend if cp_required else "none",
        "platform": core_provenance.get("platform", "cuda"),
        "num_splits": core_provenance.get("num_splits"),
        "split_kv_control": core_provenance.get("split_kv_control"),
        "deterministic_backward": core_provenance.get("deterministic_backward"),
        "fa_api_source": core_provenance.get("fa_api_source"),
        "fa_package_version": core_provenance.get("fa_package_version"),
        "aiter_api_source": core_provenance.get("aiter_api_source"),
        "aiter_source_sha256": core_provenance.get("aiter_source_sha256"),
        "reference_only": bool(core_provenance.get("reference_only", False)),
        "production_ready": bool(
            core_provenance.get("production_ready", False)
            and (not cp_required or communication_id == expected_communication)
        ),
    }


def flashinfer_qwen3_paged_attention_available() -> bool:
    """Return whether the FlashInfer paged attention wrappers are importable."""

    try:
        module = FlashInferQwen3PagedAttentionOp()._load_flashinfer()
        prefill = getattr(
            getattr(module, "prefill", None),
            "BatchPrefillWithPagedKVCacheWrapper",
            None,
        )
        decode = getattr(
            getattr(module, "decode", None),
            "BatchDecodeWithPagedKVCacheWrapper",
            None,
        )
        if not callable(prefill) or not callable(decode):
            return False
    except FlashInferUnavailable:
        return False
    return True


def _validate_metadata_logical_positions(
    metadata: Any,
    *,
    batch_index: int,
    seq_len: int,
    page_size: int,
    block_count: int,
    device: torch.device,
) -> None:
    if not hasattr(metadata, "global_token_positions"):
        return
    global_token_positions = metadata.global_token_positions
    if global_token_positions.ndim != 2 or global_token_positions.size(0) <= batch_index:
        raise ValueError("global_token_positions must have shape [B, cache_capacity]")
    physical_slots: list[int] = []
    for logical_block in range(block_count):
        local_page = int(metadata.block_table[batch_index, logical_block].item())
        token_count = min(page_size, seq_len - logical_block * page_size)
        for page_offset in range(token_count):
            physical_slots.append(local_page * page_size + page_offset)
    slot_index = torch.tensor(physical_slots, device=device, dtype=torch.long)
    actual = global_token_positions[batch_index, slot_index]
    expected = torch.arange(
        0,
        seq_len,
        device=device,
        dtype=global_token_positions.dtype,
    )
    if not torch.equal(actual, expected):
        raise ValueError(
            "block_table/global_token_positions must reconstruct logical positions "
            "as one contiguous global range"
        )
    if hasattr(metadata, "key_position_ids"):
        key_positions = metadata.key_position_ids[batch_index, slot_index]
        if not torch.equal(key_positions, expected.to(dtype=key_positions.dtype)):
            raise ValueError("key_position_ids must match cached global token positions")
    active_slot_mask = torch.zeros(
        global_token_positions.size(1),
        device=device,
        dtype=torch.bool,
    )
    active_slot_mask[slot_index] = True
    if bool((global_token_positions[batch_index, ~active_slot_mask] != -1).any()):
        raise ValueError("unused global_token_positions entries must be -1")
    if hasattr(metadata, "key_position_ids") and bool(
        (metadata.key_position_ids[batch_index, ~active_slot_mask] != -1).any()
    ):
        raise ValueError("unused key_position_ids entries must be -1")


def _validate_flashinfer_rope_metadata(
    metadata: Any,
    cfg: FlashInferPagedAttentionConfig,
    q: torch.Tensor,
) -> None:
    """Bind the configured fused-RoPE boundary to the actual cache metadata."""

    for name, expected in (
        ("q_rope_state", cfg.rope.q_rope_state),
        ("k_cache_rope_state", cfg.rope.k_cache_rope_state),
    ):
        actual = getattr(metadata, name, None)
        if actual != expected:
            raise ValueError(
                f"metadata.{name}={actual!r} does not match the FlashInfer fused-RoPE "
                f"contract {expected!r}"
            )
    cache_position = getattr(metadata, "cache_position", None)
    query_position_ids = getattr(metadata, "query_position_ids", None)
    if not isinstance(cache_position, torch.Tensor) or not isinstance(
        query_position_ids, torch.Tensor
    ):
        raise ValueError("cache_position and query_position_ids are required tensors")
    if cache_position.device != q.device or query_position_ids.device != q.device:
        raise ValueError("query position metadata must be on the query device")
    if cache_position.dtype not in {torch.int32, torch.int64, torch.long} or (
        query_position_ids.dtype not in {torch.int32, torch.int64, torch.long}
    ):
        raise ValueError("query position metadata must contain integers")
    if cache_position.shape != (q.size(0), q.size(2)):
        raise ValueError("cache_position must have shape [B, Sq]")
    if query_position_ids.shape != cache_position.shape:
        raise ValueError("query_position_ids must have shape [B, Sq]")
    if not torch.equal(cache_position, query_position_ids):
        raise ValueError("cache_position and query_position_ids must match exactly")
    kv_seq_lens = getattr(metadata, "kv_seq_lens", None)
    if not isinstance(kv_seq_lens, torch.Tensor) or kv_seq_lens.shape != (q.size(0),):
        raise ValueError("kv_seq_lens must have shape [B]")
    if bool((cache_position < 0).any()) or bool((cache_position >= kv_seq_lens[:, None]).any()):
        raise ValueError("cache_position must identify tokens present in each KV sequence")
    if q.size(2) > 1 and bool((cache_position[:, 1:] <= cache_position[:, :-1]).any()):
        raise ValueError("few-query cache_position values must be strictly increasing")
    expected_query_positions = torch.stack(
        [
            torch.arange(
                int(seq_len.item()) - q.size(2),
                int(seq_len.item()),
                device=q.device,
                dtype=cache_position.dtype,
            )
            for seq_len in kv_seq_lens
        ]
    )
    if not torch.equal(cache_position, expected_query_positions):
        raise ValueError(
            "FlashInfer implicit RoPE positions require queries to be the trailing "
            "contiguous positions of each KV sequence"
        )


def _materialize_logical_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    metadata: Any,
    *,
    total_kv_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Restore paged KV to logical order for the deterministic CP fallback."""

    page_size = int(metadata.page_size)
    block_count = (total_kv_tokens + page_size - 1) // page_size
    logical_k: list[torch.Tensor] = []
    logical_v: list[torch.Tensor] = []
    logical_positions: list[torch.Tensor] = []
    for batch_index in range(k_cache.size(0)):
        pages = metadata.block_table[batch_index, :block_count].long()
        token_index = torch.arange(total_kv_tokens, device=k_cache.device, dtype=torch.long)
        slots = pages[token_index // page_size] * page_size + token_index % page_size
        logical_k.append(k_cache[batch_index, :, slots, :])
        logical_v.append(v_cache[batch_index, :, slots, :])
        if hasattr(metadata, "key_position_ids"):
            logical_positions.append(metadata.key_position_ids[batch_index, slots].long())
        else:
            logical_positions.append(token_index)
    return (
        torch.stack(logical_k, dim=0).contiguous(),
        torch.stack(logical_v, dim=0).contiguous(),
        torch.stack(logical_positions, dim=0).contiguous(),
    )


def flashinfer_prefix_cache_fingerprint(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    metadata: Any,
    cfg: FlashInferPagedAttentionConfig,
    *,
    prefix_length: int,
) -> str:
    """Hash the logical prefix and the fused-RoPE materialization identity."""

    prefix_length = _positive_int(prefix_length, "prefix_length")
    if bool((metadata.kv_seq_lens < prefix_length).any()):
        raise ValueError("prefix_length must not exceed any kv_seq_lens entry")
    digest = hashlib.sha256()
    rotary_dim = q.size(-1) if cfg.rope.rotary_dim is None else cfg.rope.rotary_dim
    digest.update(
        (
            f"k_rope_state={metadata.k_cache_rope_state};"
            f"rope_theta={float(cfg.rope.rope_theta):.17g};"
            f"rotary_dim={rotary_dim};"
            "rope_cast_at=after_rope;"
            f"k_rope_output_dtype={k_cache.dtype}\n"
        ).encode()
    )
    digest.update(f"k_dtype={k_cache.dtype};v_dtype={v_cache.dtype}\n".encode())
    page_size = int(metadata.page_size)
    for batch_index in range(q.size(0)):
        logical_index = torch.arange(prefix_length, device=q.device, dtype=torch.long)
        pages = metadata.block_table[batch_index].long()
        slots = pages[logical_index // page_size] * page_size + logical_index % page_size
        for tensor in (
            metadata.global_token_positions[batch_index, slots],
            metadata.key_position_ids[batch_index, slots],
            k_cache[batch_index, :, slots, :],
            v_cache[batch_index, :, slots, :],
        ):
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _validate_flashinfer_prefix_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    metadata: Any,
    cfg: FlashInferPagedAttentionConfig,
) -> None:
    enabled = bool(getattr(metadata, "prefix_cache_enabled", False))
    key = getattr(metadata, "prefix_cache_key", None)
    length = getattr(metadata, "prefix_length", 0)
    fingerprint = getattr(metadata, "prefix_cache_fingerprint", None)
    if not enabled:
        if key is not None or length != 0 or fingerprint is not None:
            raise ValueError(
                "prefix cache key/fingerprint must be None and prefix_length must be 0 "
                "when prefix cache is disabled"
            )
        return
    if not isinstance(key, str) or not key:
        raise ValueError("prefix_cache_key is required when prefix cache is enabled")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("prefix_cache_fingerprint is required when prefix cache is enabled")
    actual = flashinfer_prefix_cache_fingerprint(
        q,
        k_cache,
        v_cache,
        metadata,
        cfg,
        prefix_length=length,
    )
    if actual != fingerprint:
        raise ValueError(
            "prefix_cache_fingerprint does not match logical prefix content or fused-RoPE identity"
        )


def _validate_qkv_cache(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
    if q.ndim != 4 or k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("q, k_cache, and v_cache must have shape [B, H, S, D]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have matching shape")
    if q.size(0) != k_cache.size(0) or q.size(3) != k_cache.size(3):
        raise ValueError("q and KV cache must share batch size and head_dim")
    if q.size(1) % k_cache.size(1) != 0:
        raise ValueError("Q head count must be divisible by KV head count")


def _restore_out(out_flat: torch.Tensor, *, batch_size: int, query_len: int) -> torch.Tensor:
    if out_flat.ndim != 3:
        raise FlashInferUnavailable("FlashInfer output must have shape [B*Sq, Hq, D]")
    _, q_heads, head_dim = out_flat.shape
    expected = batch_size * query_len
    if out_flat.size(0) != expected:
        raise FlashInferUnavailable(
            f"FlashInfer output first dim must be B*Sq={expected}, got {out_flat.size(0)}"
        )
    return out_flat.reshape(batch_size, query_len, q_heads, head_dim).transpose(1, 2).contiguous()


def _restore_lse(
    lse_flat: torch.Tensor,
    *,
    batch_size: int,
    query_len: int,
    q_heads: int,
) -> torch.Tensor:
    expected_tokens = batch_size * query_len
    if lse_flat.shape == (expected_tokens, q_heads):
        return lse_flat.reshape(batch_size, query_len, q_heads).transpose(1, 2).contiguous()
    if lse_flat.shape == (q_heads, expected_tokens):
        return lse_flat.transpose(0, 1).reshape(batch_size, query_len, q_heads).transpose(1, 2)
    raise FlashInferUnavailable(
        f"FlashInfer LSE must have shape [B*Sq, Hq] or [Hq, B*Sq]; got {tuple(lse_flat.shape)}"
    )


def _call_with_supported_kwargs(
    fn: Any,
    kwargs: dict[str, Any],
    *,
    return_applied: bool = False,
) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        result = fn(**kwargs)
        return dict(kwargs) if return_applied else result
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        result = fn(**kwargs)
        return dict(kwargs) if return_applied else result
    supported = {name: value for name, value in kwargs.items() if name in parameters}
    missing_required = [
        name
        for name, param in parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        and name not in supported
    ]
    if missing_required:
        raise FlashInferUnavailable(
            f"{getattr(fn, '__qualname__', fn)} missing supported arguments: "
            f"{', '.join(missing_required)}"
        )
    result = fn(**supported)
    return supported if return_applied else result


def _flashinfer_split_kv_plan_kwargs(
    spec: SplitKVSpec,
    *,
    page_size: int,
) -> dict[str, Any]:
    if spec.mode is SplitKVMode.DISABLED:
        return {"disable_split_kv": True}
    if spec.mode is SplitKVMode.FIXED:
        assert spec.fixed_split_size is not None
        if spec.fixed_split_size % page_size != 0:
            raise FlashInferUnavailable(
                "FlashInfer fixed Split-K size is expressed in pages; the WS2 token "
                "split size must be divisible by page_size"
            )
        return {
            # FlashInfer names this in pages, while the WS2 contract is tokens.
            "fixed_split_size": int(spec.fixed_split_size) // page_size,
            "disable_split_kv": False,
        }
    return {"disable_split_kv": False}


def _flashinfer_split_size_tokens(value: Any, *, page_size: int, unit: Any) -> int:
    if unit != "pages":
        raise FlashInferUnavailable(
            "FlashInfer fixed Split-K provenance must declare split_size_unit='pages'"
        )
    return _positive_int(int(value), "split_size") * page_size


def _normalize_flashinfer_split_boundaries(
    boundaries: Any,
    *,
    page_size: int,
    seq_len: int,
    unit: Any,
    offset: int = 0,
) -> tuple[tuple[int, int], ...]:
    if unit != "pages":
        raise FlashInferUnavailable(
            "FlashInfer Split-K boundaries must declare boundary_unit='pages'"
        )
    normalized = []
    for boundary in boundaries:
        start_page, end_page = boundary
        normalized.append(
            (
                offset + int(start_page) * page_size,
                offset + min(int(end_page) * page_size, seq_len),
            )
        )
    return tuple(normalized)


def _split_kv_provenance(
    spec: SplitKVSpec,
    plans: tuple[SplitKVExecutionPlan, ...],
    *,
    applied_plan_kwargs: dict[str, Any],
    require_batch_invariant: bool,
) -> dict[str, Any]:
    policy = spec.mode.value
    if spec.mode is SplitKVMode.FIXED:
        policy = f"fixed:{spec.fixed_split_size}"
    return {
        "split_kv_policy": policy,
        "requested_split_kv_policy": spec.mode.value,
        "requested_split_kv_size": spec.fixed_split_size,
        "actual_split_kv_plans": [plan.to_dict() for plan in plans],
        "backend_native_split_kv_knobs": {
            key: value
            for key, value in applied_plan_kwargs.items()
            if key in {"fixed_split_size", "disable_split_kv"}
        },
        "batch_invariant_required": bool(require_batch_invariant),
        "batch_invariant_claim": (
            "strict_runtime_verified" if require_batch_invariant else "diagnostic_only"
        ),
    }


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


__all__ = [
    "FlashInferAttentionMode",
    "FlashInferAttentionResult",
    "FlashInferPagedAttentionConfig",
    "FlashInferPagedKVPlan",
    "FlashInferQwen3PagedAttentionOp",
    "FlashInferRoPEFusionConfig",
    "FlashInferSplitKVPolicy",
    "FlashInferUnavailable",
    "build_flashinfer_paged_kv_plan",
    "flashinfer_qwen3_paged_attention_available",
    "flashinfer_prefix_cache_fingerprint",
    "materialize_flashinfer_paged_kv_cache",
]
