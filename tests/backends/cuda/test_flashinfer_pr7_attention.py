# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rl_engine.backends.cuda.attention import flashinfer_paged_attention as paged_attention_module
from rl_engine.backends.cuda.attention.cp_comm import (
    AttentionCPBlockMetadata,
    AttentionCPCommunicationPlan,
    AttentionCPCommunicationUnavailable,
    AttentionCPMergedState,
    AttentionCPOutputShard,
    AttentionCPPartialState,
    AttentionParallelSpec,
    CUDAAGRSAttentionCPCommunication,
    P2PNCCLAttentionCPCommunication,
    sort_attention_cp_partial_states,
)
from rl_engine.backends.cuda.attention.deterministic_attn import DeterministicAttentionCoreResult
from rl_engine.backends.cuda.attention.flash_attn import (
    StrictFlashAttention4Core,
    StrictFlashAttentionUnavailable,
)
from rl_engine.backends.cuda.attention.flashinfer_paged_attention import (
    FlashInferPagedAttentionConfig,
    FlashInferQwen3PagedAttentionOp,
    FlashInferRoPEFusionConfig,
    FlashInferUnavailable,
    _NativeFlashInferRuntimeAdapter,
    build_flashinfer_paged_kv_plan,
    flashinfer_prefix_cache_fingerprint,
    materialize_flashinfer_paged_kv_cache,
)
from rl_engine.contracts.operators.attention import (
    STRICT_ATTENTION_CORE_ID,
    STRICT_ATTENTION_FA4_SCHEDULE_ID,
    STRICT_ATTENTION_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_RING_SCHEDULE_ID,
    STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
    STRICT_ATTENTION_ROCM_SCHEDULE_ID,
    STRICT_ATTENTION_SCHEDULE_ID,
    SplitKVSpec,
)
from rl_engine.validation.common.attention_comparison import DecodeKVCacheMetadata


def _load_repo_script(name: str):
    path = Path(__file__).resolve().parents[3] / "tools/validation/distributed" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"rl_kernel_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load repository script {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


p2p_check_script = _load_repo_script("ws2_p2p_nccl_attention_reference_check")
check_script = _load_repo_script("ws2_pr7_flashinfer_attention_check")


class _FakeFlashInferWrapper:
    instances: list["_FakeFlashInferWrapper"] = []

    def __init__(self, workspace_buffer, *, kv_layout):
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self.plan_kwargs = None
        self.run_q = None
        self.run_cache = None
        self.instances.append(self)

    def plan(self, **kwargs):
        self.plan_kwargs = kwargs

    def run_return_lse(self, q, paged_kv_cache):
        self.run_q = q
        self.run_cache = paged_kv_cache
        out = torch.zeros(
            q.shape,
            dtype=self.plan_kwargs.get("o_data_type", q.dtype),
            device=q.device,
        )
        lse = torch.zeros(q.size(0), q.size(1), dtype=torch.float32, device=q.device)
        return out, lse

    def get_actual_split_kv_plan(self):
        seq_lens = self.plan_kwargs["seq_lens"].tolist()
        disabled = bool(self.plan_kwargs.get("disable_split_kv", False))
        split_size = self.plan_kwargs.get("fixed_split_size")
        plans = []
        for seq_len in seq_lens:
            if disabled:
                boundaries = [(0, (seq_len + 1) // 2)]
                mode = "disabled"
                actual_size = None
            else:
                assert split_size is not None
                boundaries = [
                    (start, min(start + split_size, (seq_len + 1) // 2))
                    for start in range(0, (seq_len + 1) // 2, split_size)
                ]
                mode = "fixed"
                actual_size = split_size
            plans.append(
                {
                    "mode": mode,
                    "split_size": actual_size,
                    "split_size_unit": "pages",
                    "boundary_unit": "pages",
                    "boundaries": boundaries,
                    "fallback": False,
                    "fallback_reason": None,
                }
            )
        return plans

    def get_attention_arithmetic_provenance(self):
        return {
            "accum_dtype": "fp32",
            "downcast_at": "final_write",
            "lse_dtype": "fp32",
            "source": "fake_runtime_capability",
        }

    def get_actual_split_kv_plan_set(self):
        seq_lens = self.plan_kwargs["seq_lens"].tolist()
        disabled = bool(self.plan_kwargs.get("disable_split_kv", False))
        split_size = self.plan_kwargs.get("fixed_split_size")
        entries = []
        for batch_index, total in enumerate(seq_lens):
            owner_ranges = ((0, total - 2), (total - 2, total))
            for tp_rank in range(2):
                for cp_rank in range(2):
                    for owner_cp_rank, (owner_start, owner_end) in enumerate(owner_ranges):
                        if disabled:
                            mode = "disabled"
                            actual_size = None
                            boundaries = [(0, (owner_end - owner_start + 1) // 2)]
                        else:
                            mode = "fixed"
                            actual_size = split_size
                            boundaries = [
                                (
                                    (start - owner_start) // 2,
                                    min(
                                        (start - owner_start) // 2 + split_size,
                                        (owner_end - owner_start + 1) // 2,
                                    ),
                                )
                                for start in range(owner_start, owner_end, split_size * 2)
                            ]
                        entries.append(
                            {
                                "batch_index": batch_index,
                                "tp_rank": tp_rank,
                                "cp_rank": cp_rank,
                                "owner_cp_rank": owner_cp_rank,
                                "expected_kv_range": [owner_start, owner_end],
                                "mode": mode,
                                "split_size": actual_size,
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
            "tp_world_size": 2,
            "cp_world_size": 2,
            "total_kv_tokens": seq_lens,
            "entries": entries,
        }


def test_native_flashinfer_adapter_reads_materialized_plan_and_normalizes_lse():
    class _NativeWrapper:
        def __init__(self):
            self._backend = "fa2"
            self._plan_info = (2, 2, 0, 16, 0, 16, 32, 0, 48, 60, 0, 0, 0, 0, 0)
            self._pin_memory_int_workspace_buffer = torch.zeros(64, dtype=torch.uint8)
            self._pin_memory_int_workspace_buffer[0:8].view(torch.int32).copy_(
                torch.tensor([0, 1], dtype=torch.int32)
            )
            self._pin_memory_int_workspace_buffer[32:40].view(torch.int32).zero_()

        def plan(self, **kwargs):
            return None

    adapter = _NativeFlashInferRuntimeAdapter(
        _NativeWrapper(),
        FlashInferPagedAttentionConfig(workspace_size_bytes=1024),
    )
    adapter.plan(
        seq_lens=torch.tensor([16, 16], dtype=torch.int32),
        page_size=4,
        disable_split_kv=True,
    )

    assert adapter.get_actual_split_kv_plan()[0]["boundaries"] == [(0, 4)]
    plan_set = adapter.get_actual_split_kv_plan_set()
    assert len(plan_set["entries"]) == 16
    assert plan_set["entries"][1]["expected_kv_range"] == [8, 16]
    normalized = adapter.normalize_lse(torch.ones(1))
    assert torch.allclose(normalized, torch.log(torch.tensor([2.0])))


def _fake_flashinfer():
    _FakeFlashInferWrapper.instances = []
    return types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_FakeFlashInferWrapper,
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_FakeFlashInferWrapper,
        ),
    )


def _metadata(*, batch: int = 2, query_len: int = 1) -> DecodeKVCacheMetadata:
    page_size = 2
    cache_capacity = 6
    positions = torch.arange(cache_capacity, dtype=torch.long).repeat(batch, 1)
    query_positions = torch.arange(
        cache_capacity - query_len,
        cache_capacity,
        dtype=torch.long,
    ).repeat(batch, 1)
    return DecodeKVCacheMetadata(
        cache_position=query_positions.clone(),
        kv_seq_lens=torch.full((batch,), cache_capacity, dtype=torch.long),
        block_table=torch.tensor([[0, 1, 2]] * batch, dtype=torch.long),
        global_token_positions=positions,
        query_position_ids=query_positions.clone(),
        key_position_ids=positions.clone(),
        page_size=page_size,
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
    )


def _qkv(*, batch: int = 2, query_len: int = 1):
    gen = torch.Generator().manual_seed(7)
    q = torch.randn(batch, 4, query_len, 8, generator=gen)
    k = torch.randn(batch, 2, 6, 8, generator=gen)
    v = torch.randn(batch, 2, 6, 8, generator=gen)
    return q, k, v


def _partial_state(global_block_index: int) -> AttentionCPPartialState:
    return AttentionCPPartialState(
        out=torch.full((1, 2, 1, 4), float(global_block_index)),
        lse=torch.full((1, 2, 1), float(global_block_index), dtype=torch.float32),
        block=AttentionCPBlockMetadata(
            global_block_index=global_block_index,
            kv_block_start=global_block_index * 2,
            kv_block_end=global_block_index * 2 + 2,
            owner_cp_rank=global_block_index % 2,
            owner_tp_rank=0,
        ),
    )


def _p2p_plan(*, cp_rank: int = 0) -> AttentionCPCommunicationPlan:
    return AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(
            tp_world_size=2,
            tp_rank=0,
            cp_world_size=2,
            cp_rank=cp_rank,
        ),
        backend="p2p_nccl_reference",
        status="implemented",
        expected_blocks=(
            AttentionCPBlockMetadata(0, 0, 2, 0, 0),
            AttentionCPBlockMetadata(1, 2, 4, 1, 0),
        ),
        expected_kv_token_range=(0, 4),
        query_token_ranges=((0, 1), (1, 2)),
    )


class _CompletedRequest:
    def wait(self):
        return True


class _FakeP2POp:
    def __init__(self, op, tensor, peer, *, group=None):
        self.op = op
        self.tensor = tensor
        self.peer = peer
        self.group = group


class _FakeNCCLDistributed:
    def __init__(self, *, rank: int, receive_payloads=()):
        self.rank = rank
        self.receive_payloads = list(receive_payloads)

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_initialized():
        return True

    @staticmethod
    def get_backend(group=None):
        return "nccl"

    @staticmethod
    def get_world_size(group=None):
        return 2

    def get_rank(self, group=None):
        return self.rank

    @staticmethod
    def get_global_rank(group, rank):
        return rank

    @staticmethod
    def isend(tensor, dst, group=None):
        raise AssertionError("P2POp should defer isend")

    @staticmethod
    def irecv(tensor, src, group=None):
        raise AssertionError("P2POp should defer irecv")

    P2POp = _FakeP2POp

    def batch_isend_irecv(self, operations):
        for operation in operations:
            if getattr(operation.op, "__name__", None) == "irecv":
                operation.tensor.copy_(self.receive_payloads.pop(0))
        return [_CompletedRequest() for _ in operations]


class _FakeCPCommunication:
    def all_gather_query(self, local_q, plan):
        return torch.cat([local_q] * plan.parallel.cp_world_size, dim=2)

    def all_gather_partial_states(self, local_states, plan):
        local = local_states[0]
        remote_block = next(
            block for block in plan.expected_blocks if block.owner_cp_rank != plan.parallel.cp_rank
        )
        remote = AttentionCPPartialState(
            out=torch.ones_like(local.out),
            lse=torch.ones_like(local.lse),
            block=remote_block,
        )
        return tuple(sorted((local, remote), key=lambda state: state.block.global_block_index))

    def reduce_scatter_merged_state(self, merged_state, plan):
        start, end = plan.query_token_ranges[plan.parallel.cp_rank]
        return AttentionCPMergedState(
            out=merged_state.out[:, :, start:end, :],
            lse=merged_state.lse[:, :, start:end],
        )


class _IdentityStrictRoPE:
    backend_id = "rlkernel.cuda.rope_sm90"

    def __call__(self, x, positions, *, theta=1_000_000.0):
        assert positions.ndim == 1
        assert float(theta) == 1_000_000.0
        return x


class _RecordingStrictCore:
    core_id = STRICT_ATTENTION_CORE_ID
    strict_schedule = STRICT_ATTENTION_SCHEDULE_ID
    backend_id = "test.strict_cuda_core"
    merge_order = "global_block_index"
    accum_dtype = "fp32"
    downcast_at = "final_write"
    fallback = False
    native_attention_arithmetic = False

    def __init__(self):
        self.calls = []

    def forward_with_lse(
        self,
        q,
        k,
        v,
        *,
        query_position_ids,
        key_position_ids,
        **_kwargs,
    ):
        self.calls.append(
            {
                "q": q.clone(),
                "k": k.clone(),
                "v": v.clone(),
                "query_position_ids": query_position_ids.clone(),
                "key_position_ids": key_position_ids.clone(),
            }
        )
        return DeterministicAttentionCoreResult(
            out=torch.zeros_like(q),
            lse=torch.zeros(q.shape[:3], dtype=torch.float32, device=q.device),
            provenance={
                "strict_core_id": self.core_id,
                "strict_schedule": self.strict_schedule,
                "attention_backend": self.backend_id,
                "split_kv": {
                    "actual_split_kv_policy": "disabled",
                    "actual_split_boundaries": [[0, k.size(2)]],
                },
                "merge_order": self.merge_order,
                "accum_dtype": self.accum_dtype,
                "downcast_at": self.downcast_at,
                "fallback": False,
                "fallback_reason": None,
                "native_attention_arithmetic": False,
            },
        )


class _StrictCPCommunication:
    backend_id = "p2p_nccl_reference"

    def all_gather_query(self, local_q, plan):
        return torch.cat((local_q, local_q + 1), dim=2)

    def all_gather_kv(self, local_k, local_v, plan):
        return (
            torch.cat((local_k, local_k + 2), dim=2),
            torch.cat((local_v, local_v + 3), dim=2),
        )

    def all_gather_position_ids(self, local_q_positions, local_k_positions, plan):
        return (
            torch.cat((local_q_positions, local_q_positions + 1), dim=1),
            torch.cat((local_k_positions, local_k_positions + 2), dim=1),
        )

    def reduce_scatter_strict_result(self, out, lse, plan):
        start, end = plan.query_token_ranges[plan.parallel.cp_rank]
        return AttentionCPOutputShard(
            out=out[:, :, start:end, :].contiguous(),
            lse=lse[:, :, start:end].contiguous(),
        )


class _FakeDeterministicCollective:
    world_size = 2

    @staticmethod
    def all_gather(tensor):
        return torch.cat((tensor, tensor + 10), dim=0)

    @staticmethod
    def reduce_scatter(tensor):
        return tensor.chunk(2, dim=0)[0].contiguous()


class _FakeAutogradCollective:
    world_size = 2

    def __init__(self):
        self.all_gather_calls = 0
        self.reduce_scatter_calls = 0

    def all_gather(self, tensor):
        self.all_gather_calls += 1
        return torch.cat((tensor, tensor), dim=0)

    def reduce_scatter(self, tensor):
        self.reduce_scatter_calls += 1
        first, second = tensor.chunk(2, dim=0)
        return (first + second).contiguous()


class _FakeCollectiveDist:
    @staticmethod
    def all_gather_object(outputs, value, group=None):
        outputs[:] = [value, value]


def test_flashinfer_pr7_prefill_adapter_passes_qwen3_rope_and_splitk_policy():
    q, k, v = _qkv(query_len=2)
    metadata = _metadata(query_len=2)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    result = op(
        q,
        k,
        v,
        metadata,
        config=FlashInferPagedAttentionConfig(
            mode="prefill",
            workspace_size_bytes=1024,
            split_kv=SplitKVSpec.fixed(4),
        ),
    )

    wrapper = _FakeFlashInferWrapper.instances[-1]
    assert wrapper.kv_layout == "NHD"
    assert wrapper.run_q.shape == (q.size(0) * q.size(2), q.size(1), q.size(3))
    assert result.out.shape == q.shape
    assert result.lse.shape == q.shape[:3]
    assert result.provenance["actual_backend"] == "flashinfer_batch_prefill_paged_kv"
    assert result.provenance["rope_fusion_boundary"] == "flashinfer_attention_kernel"
    assert result.provenance["pos_encoding_mode"] == "ROPE_LLAMA"
    assert result.provenance["rope_theta"] == 1_000_000.0
    assert result.provenance["rope_scale"] == 1.0
    assert result.provenance["split_kv_policy"] == "fixed:4"
    assert result.provenance["batch_invariant_claim"] == "strict_runtime_verified"
    assert result.provenance["requested_split_kv_policy"] == "fixed"
    assert result.provenance["requested_split_kv_size"] == 4
    assert result.provenance["actual_split_kv_plans"][0]["actual_split_boundaries"] == [
        [0, 4],
        [4, 6],
    ]
    assert result.provenance["tp_world_size"] == 2
    assert result.provenance["cp_world_size"] == 2
    assert result.provenance["cp_comm_backend"] == "cuda_ag_rs"
    assert result.provenance["cp_comm_status"] == "interface_only"
    assert result.provenance["cp_comm_pattern"] == "ag_rs"
    assert result.provenance["cp_comm_compute_communication"] == "decoupled"
    assert result.provenance["cp_comm_merge_order"] == "global_block_index"
    assert result.provenance["cp_comm_accum_dtype"] == "fp32"
    assert result.provenance["cp_comm_return_lse"] is True
    assert result.provenance["cp_comm_contract"] == "partial_out_lse_global_block_index"
    assert result.provenance["cp_comm_required"] is False
    assert result.provenance["accum_dtype"] == "fp32"
    assert result.provenance["downcast_at"] == "final_write"
    assert result.provenance["arithmetic_semantics_verified"] is True
    assert result.provenance["actual_split_kv_plan_set"]["coverage"] == (
        "complete_batch_tp_cp_owner_cartesian_product"
    )

    plan = wrapper.plan_kwargs
    assert plan["qo_indptr"].tolist() == [0, 2, 4]
    assert plan["paged_kv_indptr"].tolist() == [0, 3, 6]
    assert plan["paged_kv_indices"].tolist() == [0, 1, 2, 3, 4, 5]
    assert plan["paged_kv_last_page_len"].tolist() == [2, 2]
    assert plan["pos_encoding_mode"] == "ROPE_LLAMA"
    assert plan["rope_theta"] == 1_000_000.0
    assert plan["rope_scale"] == 1.0
    assert plan["q_data_type"] == q.dtype
    assert plan["kv_data_type"] == q.dtype
    assert plan["fixed_split_size"] == 2
    assert plan["disable_split_kv"] is False


def test_flashinfer_pr7_required_cp_comm_uses_explicit_deterministic_fallback():
    q, k, v = _qkv(query_len=2)
    metadata = _metadata(query_len=2)
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        backend="p2p_nccl_reference",
        status="implemented",
        expected_blocks=(
            AttentionCPBlockMetadata(0, 0, 3, 0, 0),
            AttentionCPBlockMetadata(1, 3, 6, 1, 0),
        ),
        expected_kv_token_range=(0, 6),
        query_token_ranges=((0, 1), (1, 2)),
    )
    config = FlashInferPagedAttentionConfig(
        mode="prefill",
        workspace_size_bytes=1024,
        cp_comm_plan=plan,
        require_cp_comm=True,
        cp_communication=_FakeCPCommunication(),
    )
    result = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())(
        q,
        k,
        v,
        metadata,
        config=config,
    )

    assert result.out.shape == (q.size(0), q.size(1), 1, q.size(3))
    assert result.lse.shape == (q.size(0), q.size(1), 1)
    assert result.provenance["actual_backend"] == "rlkernel_deterministic_cp_reference"
    assert result.provenance["fallback"] is True
    assert result.provenance["fallback_reason"] == (
        "flashinfer_owner_local_cp_partial_api_unavailable"
    )
    assert result.provenance["cp_comm_required"] is True
    assert result.provenance["query_ag"] == "cp_rank_order"
    assert result.provenance["actual_split_kv_plan_set"]["coverage"] == (
        "complete_batch_tp_cp_owner_cartesian_product"
    )


def test_strict_paged_path_uses_shared_core_and_never_runs_flashinfer_arithmetic():
    q, k, v = (tensor.to(torch.bfloat16) for tensor in _qkv(query_len=1))
    core = _RecordingStrictCore()
    _FakeFlashInferWrapper.instances = []

    result = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())(
        q,
        k,
        v,
        _metadata(query_len=1),
        config=FlashInferPagedAttentionConfig(
            mode="decode",
            workspace_size_bytes=1024,
            strict_mode=True,
            deterministic_core=core,
            strict_rope_op=_IdentityStrictRoPE(),
        ),
    )

    assert _FakeFlashInferWrapper.instances == []
    assert len(core.calls) == q.size(0)
    assert core.calls[0]["query_position_ids"].tolist() == [[5]]
    assert core.calls[0]["key_position_ids"].tolist() == [[0, 1, 2, 3, 4, 5]]
    assert result.provenance["strict_core_id"] == STRICT_ATTENTION_CORE_ID
    assert result.provenance["native_attention_arithmetic"] is False
    assert result.provenance["fallback"] is False
    assert result.provenance["materialization"] == ("flashinfer_paged_kv_layout_shared_core")
    assert result.out.dtype is torch.bfloat16
    assert result.lse.dtype is torch.float32


def test_strict_cp_path_gathers_qkv_and_real_position_ids_before_shared_core():
    generator = torch.Generator().manual_seed(19)
    q = torch.randn(1, 4, 1, 8, generator=generator, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 2, 8, generator=generator, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 2, 8, generator=generator, dtype=torch.bfloat16)
    core = _RecordingStrictCore()
    metadata = types.SimpleNamespace(
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
        query_position_ids=torch.tensor([[2]], dtype=torch.long),
        key_position_ids=torch.tensor([[0, 1]], dtype=torch.long),
    )

    result = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())(
        q,
        k,
        v,
        metadata,
        config=FlashInferPagedAttentionConfig(
            mode="prefill",
            workspace_size_bytes=1024,
            cp_comm_plan=_p2p_plan(),
            require_cp_comm=True,
            strict_mode=True,
            cp_communication=_StrictCPCommunication(),
            deterministic_core=core,
            strict_rope_op=_IdentityStrictRoPE(),
        ),
    )

    assert len(core.calls) == 1
    assert core.calls[0]["query_position_ids"].tolist() == [[2, 3]]
    assert core.calls[0]["key_position_ids"].tolist() == [[0, 1, 2, 3]]
    assert core.calls[0]["q"].shape[2] == 2
    assert core.calls[0]["k"].shape[2] == 4
    assert result.out.shape == q.shape
    assert result.provenance["materialization"] == ("ag_qkv_positions_shared_core_rs")
    assert result.provenance["strict_full_qkv_all_gather"] is True
    assert result.provenance["strict_position_ids_all_gather"] is True
    assert result.provenance["compute_communication"] == "decoupled"
    assert result.provenance["compute_schedule"] == ("rlkernel.attention.strict_ring_state.v1")
    assert result.provenance["communication_overlap"] == "disabled"
    assert result.provenance["ring_schedule_default"] is True
    assert result.provenance["ring_partial_arithmetic"] is False
    assert result.provenance["fallback"] is False


def test_strict_cp_training_rejects_non_autograd_p2p_reference():
    generator = torch.Generator().manual_seed(29)
    q = torch.randn(1, 4, 1, 8, generator=generator, dtype=torch.bfloat16).requires_grad_()
    k = torch.randn(1, 2, 2, 8, generator=generator, dtype=torch.bfloat16).requires_grad_()
    v = torch.randn(1, 2, 2, 8, generator=generator, dtype=torch.bfloat16).requires_grad_()
    metadata = types.SimpleNamespace(
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
        query_position_ids=torch.tensor([[2]], dtype=torch.long),
        key_position_ids=torch.tensor([[0, 1]], dtype=torch.long),
    )

    with pytest.raises(FlashInferUnavailable, match="autograd-capable self-owned CUDA AG/RS"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="prefill",
                workspace_size_bytes=1024,
                cp_comm_plan=_p2p_plan(),
                require_cp_comm=True,
                strict_mode=True,
                cp_communication=_StrictCPCommunication(),
                deterministic_core=_RecordingStrictCore(),
                strict_rope_op=_IdentityStrictRoPE(),
            ),
        )


def test_strict_mode_rejects_split_k_and_unverified_core():
    with pytest.raises(ValueError, match="Split-KV to be disabled"):
        FlashInferPagedAttentionConfig(
            strict_mode=True,
            split_kv=SplitKVSpec.fixed(2),
            deterministic_core=_RecordingStrictCore(),
        ).validate(head_dim=8, query_len=1)

    bad_core = types.SimpleNamespace(
        core_id="different",
        forward_with_lse=lambda *_args, **_kwargs: None,
    )
    with pytest.raises(ValueError, match="core ID"):
        FlashInferPagedAttentionConfig(
            strict_mode=True,
            deterministic_core=bad_core,
        ).validate(head_dim=8, query_len=1)


def test_flashinfer_pr7_decode_adapter_can_disable_splitk_for_strict_candidate():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    result = op(
        q,
        k,
        v,
        metadata,
        config=FlashInferPagedAttentionConfig(
            mode="decode",
            workspace_size_bytes=1024,
            split_kv=SplitKVSpec.disabled(),
        ),
    )

    wrapper = _FakeFlashInferWrapper.instances[-1]
    assert result.provenance["actual_backend"] == "flashinfer_batch_decode_paged_kv"
    assert result.provenance["split_kv_policy"] == "disabled"
    assert result.provenance["batch_invariant_claim"] == "strict_runtime_verified"
    assert wrapper.plan_kwargs["disable_split_kv"] is True


def test_flashinfer_pr7_rejects_auto_splitk_when_batch_invariance_is_required():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    with pytest.raises(ValueError, match="auto split-KV"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                split_kv=SplitKVSpec.auto(),
            ),
        )


def test_flashinfer_pr7_strict_fixed_mode_requires_actual_runtime_split_plan():
    class _NoRuntimePlanWrapper(_FakeFlashInferWrapper):
        get_actual_split_kv_plan = None

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=_NoRuntimePlanWrapper),
        decode=types.SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=_NoRuntimePlanWrapper),
    )

    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="actual-plan provenance"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                split_kv=SplitKVSpec.fixed(2),
            ),
        )


def test_flashinfer_pr7_disabled_plan_is_exact_when_disable_knob_is_accepted():
    class _NoRuntimePlanWrapper(_FakeFlashInferWrapper):
        get_actual_split_kv_plan = None

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=_NoRuntimePlanWrapper),
        decode=types.SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=_NoRuntimePlanWrapper),
    )
    q, k, v = _qkv(query_len=1)
    result = FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
        q,
        k,
        v,
        _metadata(query_len=1),
        config=FlashInferPagedAttentionConfig(
            mode="decode",
            workspace_size_bytes=1024,
            split_kv=SplitKVSpec.disabled(),
        ),
    )

    assert result.provenance["actual_split_kv_plans"][0]["actual_split_boundaries"] == [[0, 6]]
    assert result.provenance["actual_split_kv_plans"][0]["split_kv_backend"] == (
        "flashinfer_disabled_verified"
    )


def test_flashinfer_pr7_rejects_actual_split_plan_mismatch():
    class _MismatchedRuntimePlanWrapper(_FakeFlashInferWrapper):
        def get_actual_split_kv_plan(self):
            return [
                {"mode": "fixed", "split_size": 3, "boundaries": [(0, 3), (3, 6)]},
                {"mode": "fixed", "split_size": 3, "boundaries": [(0, 3), (3, 6)]},
            ]

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_MismatchedRuntimePlanWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_MismatchedRuntimePlanWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(
        FlashInferUnavailable,
        match="does not match|invalid actual|missing required fields",
    ):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                split_kv=SplitKVSpec.fixed(2),
            ),
        )


def test_flashinfer_pr7_strict_mode_requires_runtime_arithmetic_provenance():
    class _NoArithmeticProvenanceWrapper(_FakeFlashInferWrapper):
        get_attention_arithmetic_provenance = None

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_NoArithmeticProvenanceWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_NoArithmeticProvenanceWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="arithmetic provenance"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_strict_mode_requires_complete_runtime_plan_set():
    class _NoRuntimePlanSetWrapper(_FakeFlashInferWrapper):
        get_actual_split_kv_plan_set = None

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=_NoRuntimePlanSetWrapper),
        decode=types.SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=_NoRuntimePlanSetWrapper),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="complete batch/TP/CP/owner"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_runtime_plan_set_requires_explicit_reduction_semantics():
    class _MissingReductionSemanticsWrapper(_FakeFlashInferWrapper):
        def get_actual_split_kv_plan_set(self):
            plan_set = super().get_actual_split_kv_plan_set()
            del plan_set["entries"][0]["merge_order"]
            return plan_set

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_MissingReductionSemanticsWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_MissingReductionSemanticsWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="merge_order"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_runtime_split_plan_requires_explicit_fallback_fields():
    class _MissingFallbackFieldsWrapper(_FakeFlashInferWrapper):
        def get_actual_split_kv_plan(self):
            plans = super().get_actual_split_kv_plan()
            del plans[0]["fallback"]
            return plans

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(
            BatchPrefillWithPagedKVCacheWrapper=_MissingFallbackFieldsWrapper
        ),
        decode=types.SimpleNamespace(
            BatchDecodeWithPagedKVCacheWrapper=_MissingFallbackFieldsWrapper
        ),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="fallback"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_rejects_non_fp32_runtime_accumulation():
    class _WrongArithmeticWrapper(_FakeFlashInferWrapper):
        def get_attention_arithmetic_provenance(self):
            return {
                "accum_dtype": "bf16",
                "downcast_at": "per_split",
                "lse_dtype": "fp32",
                "source": "fake_runtime_capability",
            }

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=_WrongArithmeticWrapper),
        decode=types.SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=_WrongArithmeticWrapper),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="accum_dtype, downcast_at"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_rejects_runtime_output_dtype_boundary_mismatch():
    class _WrongOutputDTypeWrapper(_FakeFlashInferWrapper):
        def run_return_lse(self, q, paged_kv_cache):
            out, lse = super().run_return_lse(q, paged_kv_cache)
            return out.double(), lse

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=_WrongOutputDTypeWrapper),
        decode=types.SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=_WrongOutputDTypeWrapper),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="final output dtype"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_rejects_non_fp32_runtime_lse():
    class _WrongLSEDTypeWrapper(_FakeFlashInferWrapper):
        def run_return_lse(self, q, paged_kv_cache):
            out, lse = super().run_return_lse(q, paged_kv_cache)
            return out, lse.to(torch.bfloat16)

    fake = types.SimpleNamespace(
        prefill=types.SimpleNamespace(BatchPrefillWithPagedKVCacheWrapper=_WrongLSEDTypeWrapper),
        decode=types.SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=_WrongLSEDTypeWrapper),
    )
    q, k, v = _qkv(query_len=1)

    with pytest.raises(FlashInferUnavailable, match="LSE must be FP32"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=fake)(
            q,
            k,
            v,
            _metadata(query_len=1),
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_required_cp_comm_needs_implemented_plan():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    with pytest.raises(ValueError, match="implemented CP communication plan"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                require_cp_comm=True,
            ),
        )


def test_flashinfer_pr7_implemented_cp_comm_status_requires_execution():
    config = FlashInferPagedAttentionConfig(
        cp_comm_plan=AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
            status="implemented",
        )
    )

    with pytest.raises(ValueError, match="require_cp_comm"):
        config.validate(head_dim=8, query_len=1)


def test_flashinfer_pr7_rejects_post_rope_inputs_for_rope_llama_fusion():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())

    with pytest.raises(ValueError, match="rotated twice"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
                rope=FlashInferRoPEFusionConfig(q_rope_state="post_rope"),
            ),
        )


def test_flashinfer_pr7_rejects_metadata_rope_state_mismatch():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    metadata = DecodeKVCacheMetadata(**{**metadata.__dict__, "k_cache_rope_state": "post_rope"})

    with pytest.raises(ValueError, match="metadata.k_cache_rope_state"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_rejects_query_position_identity_mismatch():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    metadata = DecodeKVCacheMetadata(
        **{
            **metadata.__dict__,
            "query_position_ids": torch.zeros_like(metadata.query_position_ids),
        }
    )

    with pytest.raises(ValueError, match="must match exactly"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_rejects_nontrailing_query_positions():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    metadata = DecodeKVCacheMetadata(
        **{
            **metadata.__dict__,
            "cache_position": torch.full_like(metadata.cache_position, 4),
            "query_position_ids": torch.full_like(metadata.query_position_ids, 4),
        }
    )

    with pytest.raises(ValueError, match="trailing contiguous positions"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(
                mode="decode",
                workspace_size_bytes=1024,
            ),
        )


def test_flashinfer_pr7_prefix_cache_fingerprint_binds_rope_identity():
    q, k, v = _qkv(batch=1, query_len=1)
    metadata = _metadata(batch=1, query_len=1)
    config = FlashInferPagedAttentionConfig(mode="decode", workspace_size_bytes=1024)
    fingerprint = flashinfer_prefix_cache_fingerprint(
        q,
        k,
        v,
        metadata,
        config,
        prefix_length=4,
    )
    cached = DecodeKVCacheMetadata(
        **{
            **metadata.__dict__,
            "prefix_cache_enabled": True,
            "prefix_cache_key": "prefix-0",
            "prefix_length": 4,
            "prefix_cache_fingerprint": fingerprint,
        }
    )

    FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())(
        q,
        k,
        v,
        cached,
        config=config,
    )
    drifted = replace(
        config,
        rope=replace(config.rope, rope_theta=10_000.0),
    )
    with pytest.raises(ValueError, match="rope_theta"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())(
            q,
            k,
            v,
            cached,
            config=drifted,
        )


def test_flashinfer_pr7_rejects_stale_prefix_cache_content():
    q, k, v = _qkv(batch=1, query_len=1)
    metadata = _metadata(batch=1, query_len=1)
    config = FlashInferPagedAttentionConfig(mode="decode", workspace_size_bytes=1024)
    fingerprint = flashinfer_prefix_cache_fingerprint(
        q,
        k,
        v,
        metadata,
        config,
        prefix_length=4,
    )
    cached = DecodeKVCacheMetadata(
        **{
            **metadata.__dict__,
            "prefix_cache_enabled": True,
            "prefix_cache_key": "prefix-0",
            "prefix_length": 4,
            "prefix_cache_fingerprint": fingerprint,
        }
    )
    stale_k = k.clone()
    stale_k[0, 0, 0, 0] += 1.0

    with pytest.raises(ValueError, match="prefix_cache_fingerprint"):
        FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())(
            q,
            stale_k,
            v,
            cached,
            config=config,
        )


def test_attention_cp_partial_states_sort_by_global_block_index():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        status="implemented",
    )

    ordered = sort_attention_cp_partial_states(
        (_partial_state(2), _partial_state(0), _partial_state(1)),
        plan=plan,
    )

    assert [state.block.global_block_index for state in ordered] == [0, 1, 2]


def test_attention_cp_partial_states_reject_duplicate_global_block_index():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2)
    )

    with pytest.raises(ValueError, match="duplicate global_block_index"):
        sort_attention_cp_partial_states(
            (_partial_state(1), _partial_state(1)),
            plan=plan,
        )


def test_cuda_ag_rs_attention_cp_comm_requires_compiled_collective():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        expected_blocks=(
            AttentionCPBlockMetadata(0, 0, 2, 0, 0),
            AttentionCPBlockMetadata(1, 2, 4, 1, 0),
        ),
        expected_kv_token_range=(0, 4),
        query_token_ranges=((0, 1), (1, 2)),
        status="implemented",
    )
    communication = CUDAAGRSAttentionCPCommunication()

    with pytest.raises(
        AttentionCPCommunicationUnavailable,
        match="requires CUDA|requires initialized|unavailable|extension|DeterministicCollective",
    ):
        communication.all_gather_partial_states((_partial_state(0),), plan)

    merged = AttentionCPMergedState(
        out=torch.zeros(1, 2, 1, 4),
        lse=torch.zeros(1, 2, 1, dtype=torch.float32),
    )
    with pytest.raises(
        AttentionCPCommunicationUnavailable,
        match="requires CUDA|requires initialized|unavailable|extension|DeterministicCollective",
    ):
        communication.reduce_scatter_merged_state(merged, plan)


def test_cuda_ag_rs_attention_cp_comm_executes_injected_collective(monkeypatch):
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        backend="cuda_ag_rs",
        status="implemented",
        expected_blocks=(
            AttentionCPBlockMetadata(0, 0, 2, 0, 0),
            AttentionCPBlockMetadata(1, 2, 4, 1, 0),
        ),
        expected_kv_token_range=(0, 4),
        query_token_ranges=((0, 1), (1, 2)),
    )
    communication = CUDAAGRSAttentionCPCommunication(collective=_FakeDeterministicCollective())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(communication, "_dist", lambda: _FakeCollectiveDist())

    local_q = torch.zeros(1, 2, 1, 4)
    gathered_q = communication.all_gather_query(local_q, plan)
    assert gathered_q.shape == (1, 2, 2, 4)
    assert torch.equal(gathered_q[:, :, :1, :], local_q)
    assert torch.equal(gathered_q[:, :, 1:, :], local_q + 10)

    local_state = AttentionCPPartialState(
        out=torch.zeros(1, 2, 2, 4),
        lse=torch.zeros(1, 2, 2, dtype=torch.float32),
        block=plan.expected_blocks[0],
    )
    gathered = communication.all_gather_partial_states((local_state,), plan)
    assert [state.block.global_block_index for state in gathered] == [0, 1]
    assert torch.equal(gathered[1].out, local_state.out + 10)

    merged = AttentionCPMergedState(
        out=torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4),
        lse=torch.arange(4, dtype=torch.float32).reshape(1, 2, 2),
    )
    shard = communication.reduce_scatter_merged_state(merged, plan)
    assert torch.equal(shard.out, merged.out[:, :, :1, :])
    assert torch.equal(shard.lse, merged.lse[:, :, :1])


def test_cuda_ag_rs_sequence_collectives_preserve_attention_gradients(monkeypatch):
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        backend="cuda_ag_rs",
        status="implemented",
        expected_blocks=(
            AttentionCPBlockMetadata(0, 0, 2, 0, 0),
            AttentionCPBlockMetadata(1, 2, 4, 1, 0),
        ),
        expected_kv_token_range=(0, 4),
        query_token_ranges=((0, 1), (1, 2)),
    )
    collective = _FakeAutogradCollective()
    communication = CUDAAGRSAttentionCPCommunication(collective=collective)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    local_q = torch.zeros(1, 2, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    gathered_q = communication.all_gather_query(local_q, plan)
    gathered_q.float().sum().backward()
    assert torch.equal(local_q.grad, torch.full_like(local_q, 2))
    assert collective.reduce_scatter_calls == 1

    full_out = torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    full_lse = torch.zeros(1, 2, 2, dtype=torch.float32)
    shard = communication.reduce_scatter_strict_result(full_out, full_lse, plan)
    shard.out.float().sum().backward()
    assert torch.equal(full_out.grad, torch.ones_like(full_out))
    assert collective.all_gather_calls == 2


def test_cp_manifest_rejects_gap_wrong_owner_and_incomplete_gather():
    with pytest.raises(ValueError, match="gap-free"):
        AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
            backend="p2p_nccl_reference",
            status="implemented",
            expected_blocks=(
                AttentionCPBlockMetadata(0, 0, 2, 0, 0),
                AttentionCPBlockMetadata(1, 3, 4, 1, 0),
            ),
            expected_kv_token_range=(0, 4),
            query_token_ranges=((0, 1), (1, 2)),
        ).validate()

    with pytest.raises(ValueError, match="owner_tp_rank"):
        AttentionCPCommunicationPlan(
            parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
            backend="p2p_nccl_reference",
            status="implemented",
            expected_blocks=(
                AttentionCPBlockMetadata(0, 0, 2, 0, 1),
                AttentionCPBlockMetadata(1, 2, 4, 1, 1),
            ),
            expected_kv_token_range=(0, 4),
            query_token_ranges=((0, 1), (1, 2)),
        ).validate()

    with pytest.raises(ValueError, match="expected KV token range|complete block manifest"):
        sort_attention_cp_partial_states((_partial_state(0),), plan=_p2p_plan())


def test_cp_manifest_rejects_wrong_local_cp_owner():
    communication = P2PNCCLAttentionCPCommunication(
        dist_module=_FakeNCCLDistributed(rank=0),
        validate_cuda_tensors=False,
    )
    wrong_owner = AttentionCPPartialState(
        out=torch.zeros(1, 2, 2, 4),
        lse=torch.zeros(1, 2, 2, dtype=torch.float32),
        block=AttentionCPBlockMetadata(0, 0, 2, 1, 0),
    )

    with pytest.raises(ValueError, match="wrong CP owner"):
        communication.all_gather_partial_states((wrong_owner,), _p2p_plan())


def test_p2p_nccl_reference_query_ag_preserves_cp_rank_order():
    local_q = torch.zeros(1, 2, 1, 4)
    remote_q = torch.ones_like(local_q)
    communication = P2PNCCLAttentionCPCommunication(
        dist_module=_FakeNCCLDistributed(rank=0, receive_payloads=[remote_q]),
        validate_cuda_tensors=False,
    )

    gathered = communication.all_gather_query(local_q, _p2p_plan())

    assert gathered.shape == (1, 2, 2, 4)
    assert torch.equal(gathered[:, :, :1, :], local_q)
    assert torch.equal(gathered[:, :, 1:, :], remote_q)


def test_p2p_nccl_reference_gathers_kv_and_position_ids_in_owner_order():
    local_k = torch.zeros(1, 2, 2, 4)
    local_v = torch.ones_like(local_k)
    remote_k = torch.full_like(local_k, 7)
    remote_v = torch.full_like(local_v, 9)
    communication = P2PNCCLAttentionCPCommunication(
        dist_module=_FakeNCCLDistributed(
            rank=0,
            receive_payloads=(remote_k, remote_v),
        ),
        validate_cuda_tensors=False,
    )

    global_k, global_v = communication.all_gather_kv(local_k, local_v, _p2p_plan())

    assert torch.equal(global_k[:, :, :2], local_k)
    assert torch.equal(global_k[:, :, 2:], remote_k)
    assert torch.equal(global_v[:, :, :2], local_v)
    assert torch.equal(global_v[:, :, 2:], remote_v)

    local_q_pos = torch.tensor([[2]], dtype=torch.long)
    local_k_pos = torch.tensor([[0, 1]], dtype=torch.long)
    remote_q_pos = torch.tensor([[3]], dtype=torch.long)
    remote_k_pos = torch.tensor([[2, 3]], dtype=torch.long)
    communication = P2PNCCLAttentionCPCommunication(
        dist_module=_FakeNCCLDistributed(
            rank=0,
            receive_payloads=(remote_q_pos, remote_k_pos),
        ),
        validate_cuda_tensors=False,
    )

    global_q_pos, global_k_pos = communication.all_gather_position_ids(
        local_q_pos, local_k_pos, _p2p_plan()
    )

    assert global_q_pos.tolist() == [[2, 3]]
    assert global_k_pos.tolist() == [[0, 1, 2, 3]]


def test_cp_manifest_allows_sparse_global_block_indices_and_rejects_short_query_state():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        backend="p2p_nccl_reference",
        status="implemented",
        expected_blocks=(
            AttentionCPBlockMetadata(10, 0, 2, 0, 0),
            AttentionCPBlockMetadata(20, 2, 4, 1, 0),
        ),
        expected_kv_token_range=(0, 4),
        query_token_ranges=((0, 1), (1, 2)),
    )
    plan.validate()
    communication = P2PNCCLAttentionCPCommunication(
        dist_module=_FakeNCCLDistributed(rank=0),
        validate_cuda_tensors=False,
    )
    short_query = AttentionCPPartialState(
        out=torch.zeros(1, 2, 1, 4),
        lse=torch.zeros(1, 2, 1, dtype=torch.float32),
        block=plan.expected_blocks[0],
    )

    with pytest.raises(ValueError, match="complete query range"):
        communication.all_gather_partial_states((short_query,), plan)


def test_p2p_nccl_reference_gathers_manifest_order_and_scatters_query_range():
    remote_out = torch.full((1, 2, 2, 4), 7.0)
    remote_lse = torch.full((1, 2, 2), 3.0, dtype=torch.float32)
    distributed = _FakeNCCLDistributed(
        rank=0,
        receive_payloads=(remote_out, remote_lse),
    )
    communication = P2PNCCLAttentionCPCommunication(
        dist_module=distributed,
        validate_cuda_tensors=False,
    )
    local = AttentionCPPartialState(
        out=torch.zeros(1, 2, 2, 4),
        lse=torch.zeros(1, 2, 2, dtype=torch.float32),
        block=AttentionCPBlockMetadata(0, 0, 2, 0, 0),
    )

    gathered = communication.all_gather_partial_states((local,), _p2p_plan())

    assert [state.block.global_block_index for state in gathered] == [0, 1]
    torch.testing.assert_close(gathered[1].out, remote_out)
    torch.testing.assert_close(gathered[1].lse, remote_lse)

    merged = AttentionCPMergedState(
        out=torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4),
        lse=torch.arange(4, dtype=torch.float32).reshape(1, 2, 2),
    )
    shard = communication.reduce_scatter_merged_state(merged, _p2p_plan())
    torch.testing.assert_close(shard.out, merged.out[:, :, 0:1, :])
    torch.testing.assert_close(shard.lse, merged.lse[:, :, 0:1])


def test_p2p_nccl_reference_fails_closed_on_non_nccl_backend():
    class _FakeGlooDistributed(_FakeNCCLDistributed):
        @staticmethod
        def get_backend(group=None):
            return "gloo"

    communication = P2PNCCLAttentionCPCommunication(
        dist_module=_FakeGlooDistributed(rank=0),
        validate_cuda_tensors=False,
    )

    with pytest.raises(AttentionCPCommunicationUnavailable, match="NCCL backend"):
        communication.all_gather_partial_states((_partial_state(0),), _p2p_plan())


def test_p2p_query_ranges_allow_empty_decode_shards():
    plan = AttentionCPCommunicationPlan(
        parallel=AttentionParallelSpec(tp_world_size=2, cp_world_size=2),
        backend="p2p_nccl_reference",
        status="implemented",
        expected_blocks=(
            AttentionCPBlockMetadata(0, 0, 2, 0, 0),
            AttentionCPBlockMetadata(1, 2, 4, 1, 0),
        ),
        expected_kv_token_range=(0, 4),
        query_token_ranges=((0, 1), (1, 1)),
    )
    plan.validate()
    communication = P2PNCCLAttentionCPCommunication(
        dist_module=_FakeNCCLDistributed(rank=0),
        validate_cuda_tensors=False,
    )
    merged = AttentionCPMergedState(
        out=torch.zeros(1, 2, 1, 4),
        lse=torch.zeros(1, 2, 1, dtype=torch.float32),
    )

    local = communication.reduce_scatter_merged_state(merged, plan)

    assert local.out.shape == (1, 2, 1, 4)
    assert local.lse.shape == (1, 2, 1)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.distributed.is_available()
    or not torch.distributed.is_initialized()
    or "nccl" not in str(torch.distributed.get_backend()).lower()
    or torch.distributed.get_world_size() != 2,
    reason="requires an initialized two-rank NCCL process group",
)
def test_p2p_nccl_reference_real_process_group_smoke():
    rank = torch.distributed.get_rank()
    local = AttentionCPPartialState(
        out=torch.full((1, 2, 2, 4), float(rank), device="cuda"),
        lse=torch.full((1, 2, 2), float(rank), dtype=torch.float32, device="cuda"),
        block=AttentionCPBlockMetadata(rank, rank * 2, rank * 2 + 2, rank, 0),
    )

    gathered = P2PNCCLAttentionCPCommunication().all_gather_partial_states(
        (local,),
        _p2p_plan(cp_rank=rank),
    )

    assert [state.block.global_block_index for state in gathered] == [0, 1]


def test_flashinfer_pr7_real_backend_requires_cuda_before_importing_flashinfer():
    q, k, v = _qkv(query_len=1)
    metadata = _metadata(query_len=1)
    op = FlashInferQwen3PagedAttentionOp()

    with pytest.raises(FlashInferUnavailable, match="requires CUDA"):
        op(
            q,
            k,
            v,
            metadata,
            config=FlashInferPagedAttentionConfig(mode="decode", workspace_size_bytes=1024),
        )


def test_flashinfer_pr7_plan_and_cache_materialization_follow_logical_page_order():
    q, k, v = _qkv(batch=1, query_len=1)
    positions = torch.full((1, 6), -1, dtype=torch.long)
    positions[:, 4:6] = torch.tensor([0, 1], dtype=torch.long)
    positions[:, 0:2] = torch.tensor([2, 3], dtype=torch.long)
    positions[:, 2:4] = torch.tensor([4, 5], dtype=torch.long)
    metadata = DecodeKVCacheMetadata(
        cache_position=torch.tensor([[5]], dtype=torch.long),
        kv_seq_lens=torch.tensor([6], dtype=torch.long),
        block_table=torch.tensor([[2, 0, 1]], dtype=torch.long),
        global_token_positions=positions,
        query_position_ids=torch.tensor([[5]], dtype=torch.long),
        key_position_ids=positions.clone(),
        page_size=2,
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
    )

    plan = build_flashinfer_paged_kv_plan(
        metadata,
        batch_size=1,
        query_len=1,
        cache_capacity=k.size(2),
        device=q.device,
    )
    k_pages, v_pages = materialize_flashinfer_paged_kv_cache(k, v, page_size=2)

    assert plan.paged_kv_indices.tolist() == [2, 0, 1]
    torch.testing.assert_close(k_pages[2], k[0, :, 4:6, :].transpose(0, 1))
    torch.testing.assert_close(v_pages[0], v[0, :, 0:2, :].transpose(0, 1))


def test_flashinfer_pr7_plan_rejects_position_metadata_mismatch():
    q, k, _ = _qkv(batch=1, query_len=1)
    metadata = DecodeKVCacheMetadata(
        cache_position=torch.tensor([[5]], dtype=torch.long),
        kv_seq_lens=torch.tensor([6], dtype=torch.long),
        block_table=torch.tensor([[2, 0, 1]], dtype=torch.long),
        global_token_positions=torch.arange(6, dtype=torch.long).unsqueeze(0),
        query_position_ids=torch.tensor([[5]], dtype=torch.long),
        key_position_ids=torch.arange(6, dtype=torch.long).unsqueeze(0),
        page_size=2,
        q_rope_state="pre_rope",
        k_cache_rope_state="pre_rope",
    )

    with pytest.raises(ValueError, match="reconstruct logical positions"):
        build_flashinfer_paged_kv_plan(
            metadata,
            batch_size=1,
            query_len=1,
            cache_capacity=k.size(2),
            device=q.device,
        )


def test_pr7_check_dry_run_writes_non_eligible_report(tmp_path):
    output = tmp_path / "pr7-dry-run.json"

    assert check_script.main(["--dry-run", "--output", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "dry_run"
    assert report["passed"] is False
    assert report["acceptance_eligible"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for the real entrypoint")
def test_pr7_check_writes_not_available_report_for_missing_flashinfer(monkeypatch, tmp_path):
    class MissingFlashInfer:
        def __call__(self, *args, **kwargs):
            raise FlashInferUnavailable("No module named 'flashinfer'")

    monkeypatch.setattr(check_script, "FlashInferQwen3PagedAttentionOp", MissingFlashInfer)
    output = tmp_path / "pr7-not-available.json"

    assert (
        check_script.main(["--no-dry-run", "--device", "cuda", "--json", "--output", str(output)])
        == 1
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "not_available"
    assert report["passed"] is False
    assert report["acceptance_eligible"] is False
    assert "Attention backend unavailable" in report["errors"][0]


def test_pr7_check_acceptance_errors_require_all_drift_and_invariance_fields():
    args = check_script._parse_args([])
    report = {
        "device": "cuda:0",
        "shape": {"q_heads": 16, "kv_heads": 4, "head_dim": 128},
        "candidate_provenance": {
            "attention_mode": "decode",
            "fallback": False,
            "pos_encoding_mode": "ROPE_LLAMA",
            "rope_theta": 1_000_000.0,
            "rotary_dim": 128,
            "arithmetic_semantics_verified": True,
            "actual_split_kv_plans": [{"actual_split_boundaries": [[0, 4]]}],
            "actual_split_kv_plan_set": {
                "coverage": "complete_batch_tp_cp_owner_cartesian_product"
            },
        },
        "drift": {
            "out": {"max_abs": 0.0},
            "lse": {"max_abs": 0.0},
            "dlogp": {"max_abs": 0.0},
        },
        "batch_invariant_sweep": {"passed": False},
        "page_layout_invariant_sweep": {"passed": True},
    }

    assert check_script._acceptance_errors(report, args) == ["batch_invariant_sweep failed"]


def test_pr7_check_accepts_strict_cuda_production_core():
    args = check_script._parse_args(["--strict", "--device", "cuda"])
    report = {
        "device": "cuda:0",
        "shape": {"q_heads": 16, "kv_heads": 4, "head_dim": 128},
        "candidate_provenance": {
            "attention_mode": "decode",
            "fallback": False,
            "strict_mode": True,
            "strict_core_id": STRICT_ATTENTION_PRODUCTION_CORE_ID,
            "strict_schedule": STRICT_ATTENTION_FA4_SCHEDULE_ID,
            "actual_backend": "flash_attention_4.cute",
            "native_attention_arithmetic": True,
            "num_splits": 1,
            "deterministic_backward": True,
            "fa_api_source": "flash_attn.cute.interface",
            "reference_only": False,
            "strict_core_row_plans": [{"actual_split_kv_policy": "disabled"}],
            "rope_backend": "rlkernel.cuda.rope_sm90",
            "rope_theta": 1_000_000.0,
            "rotary_dim": 128,
            "arithmetic_semantics_verified": True,
        },
        "drift": {
            "out": {"max_abs": 0.0},
            "lse": {"max_abs": 0.0},
            "dlogp": {"max_abs": 0.0},
        },
        "batch_invariant_sweep": {"passed": True},
        "page_layout_invariant_sweep": {"passed": True},
    }

    assert check_script._acceptance_errors(report, args) == []


def test_pr7_check_accepts_strict_rocm_production_core():
    args = check_script._parse_args(["--strict", "--device", "cuda"])
    report = {
        "device": "cuda:0",
        "shape": {"q_heads": 16, "kv_heads": 4, "head_dim": 128},
        "candidate_provenance": {
            "attention_mode": "decode",
            "fallback": False,
            "strict_mode": True,
            "strict_core_id": STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
            "strict_schedule": STRICT_ATTENTION_ROCM_SCHEDULE_ID,
            "actual_backend": "aiter.rocm.ck_dense_mha",
            "platform": "rocm",
            "native_attention_arithmetic": True,
            "num_splits": 1,
            "deterministic_backward": True,
            "reference_only": False,
            "split_kv_control": "dense_non_split_api",
            "aiter_api_source": "aiter.ops.mha",
            "aiter_source_sha256": "a" * 64,
            "strict_core_row_plans": [{"actual_split_kv_policy": "disabled"}],
            "rope_backend": "rlkernel.rocm.deterministic_rope",
            "rope_theta": 1_000_000.0,
            "rotary_dim": 128,
            "arithmetic_semantics_verified": True,
        },
        "drift": {
            "out": {"max_abs": 0.0},
            "lse": {"max_abs": 0.0},
            "dlogp": {"max_abs": 0.0},
        },
        "batch_invariant_sweep": {"passed": True},
        "page_layout_invariant_sweep": {"passed": True},
    }

    assert check_script._acceptance_errors(report, args) == []


def test_pr7_check_rejects_reference_core_as_production():
    args = check_script._parse_args(["--strict", "--device", "cuda"])
    report = {
        "device": "cuda:0",
        "shape": {"q_heads": 16, "kv_heads": 4, "head_dim": 128},
        "candidate_provenance": {
            "attention_mode": "decode",
            "fallback": False,
            "strict_mode": True,
            "strict_core_id": STRICT_ATTENTION_CORE_ID,
            "strict_schedule": STRICT_ATTENTION_SCHEDULE_ID,
            "actual_backend": "rlkernel.cuda.deterministic_attention",
            "native_attention_arithmetic": False,
            "reference_only": True,
            "strict_core_row_plans": [{"actual_split_kv_policy": "disabled"}],
            "rope_backend": "rlkernel.cuda.rope_sm90",
            "rope_theta": 1_000_000.0,
            "rotary_dim": 128,
            "arithmetic_semantics_verified": True,
        },
        "drift": {
            "out": {"max_abs": 0.0},
            "lse": {"max_abs": 0.0},
            "dlogp": {"max_abs": 0.0},
        },
        "batch_invariant_sweep": {"passed": True},
        "page_layout_invariant_sweep": {"passed": True},
    }

    errors = check_script._acceptance_errors(report, args)
    assert "strict runtime did not execute the native production arithmetic" in errors
    assert "strict runtime selected the reference core" in errors


@pytest.mark.parametrize(
    ("backend", "rope_backend"),
    [
        ("flash_attention_4.cute", "rlkernel.cuda.rope_sm90"),
        ("aiter.rocm.ck_dense_mha", "rlkernel.rocm.deterministic_rope"),
    ],
)
def test_strict_report_uses_executed_platform_provenance(backend, rope_backend):
    fields = check_script._strict_execution_report_fields(
        {
            "actual_backend": backend,
            "rope_backend": rope_backend,
            "rope_fusion": False,
            "rope_fusion_boundary": "rlkernel_rope_then_attention",
            "rope_theta": 1_000_000.0,
            "rotary_dim": 128,
            "q_rope_state": "post_rope",
            "k_cache_rope_state": "post_rope",
        }
    )

    assert fields["reference_backend"] == backend
    assert backend in fields["target"]
    assert fields["rope"] == {
        "rope_backend": rope_backend,
        "rope_fusion": False,
        "rope_fusion_boundary": "rlkernel_rope_then_attention",
        "rope_theta": 1_000_000.0,
        "rotary_dim": 128,
        "q_rope_state": "post_rope",
        "k_cache_rope_state": "post_rope",
    }


def test_pr7_check_rejects_nonfinite_drift_and_wrong_tp_local_shape():
    args = check_script._parse_args([])
    report = {
        "device": "cuda:0",
        "shape": {"q_heads": 32, "kv_heads": 8, "head_dim": 128},
        "candidate_provenance": {
            "attention_mode": "decode",
            "fallback": False,
            "pos_encoding_mode": "ROPE_LLAMA",
            "rope_theta": 1_000_000.0,
            "rotary_dim": 128,
            "arithmetic_semantics_verified": True,
            "actual_split_kv_plans": [{"actual_split_boundaries": [[0, 4]]}],
            "actual_split_kv_plan_set": {
                "coverage": "complete_batch_tp_cp_owner_cartesian_product"
            },
        },
        "drift": {
            "out": {"max_abs": float("nan")},
            "lse": {"max_abs": 0.0},
            "dlogp": {"max_abs": 0.0},
        },
        "batch_invariant_sweep": {"passed": True},
        "page_layout_invariant_sweep": {"passed": True},
    }

    errors = check_script._acceptance_errors(report, args)
    assert any("finite and non-negative" in error for error in errors)
    assert any("TP-local head shard" in error for error in errors)


def test_pr7_check_rejects_nonlocal_qwen3_head_arguments():
    with pytest.raises(SystemExit):
        check_script._parse_args(["--q-heads", "32", "--kv-heads", "8"])


def test_p2p_entrypoint_validates_qwen3_tp_local_shape_before_cuda_math():
    args = p2p_check_script.parse_args(["--q-heads", "32", "--kv-heads", "8"])

    with pytest.raises(ValueError, match="TP=2 Qwen3-8B local heads"):
        p2p_check_script.run_check(
            args,
            global_rank=0,
            tp_rank=0,
            cp_rank=0,
            replica_index=0,
            cp_group=None,
            device=torch.device("cpu"),
        )


def test_strict_shared_core_entrypoint_requires_self_owned_ag_rs():
    with pytest.raises(SystemExit):
        p2p_check_script.parse_args(["--strict-shared-core"])

    args = p2p_check_script.parse_args(["--transport", "cuda_ag_rs", "--strict-shared-core"])
    assert args.strict_shared_core is True


def test_strict_shared_core_reference_executes_one_logical_row(monkeypatch):
    calls = []

    class _RowCore:
        def forward_with_lse(self, q, k, v, **kwargs):
            calls.append((q.shape[0], kwargs["query_position_ids"].clone()))
            return types.SimpleNamespace(
                out=q + k + v,
                lse=(q + k).sum(dim=-1),
            )

    monkeypatch.setattr(p2p_check_script, "_strict_attention_core", _RowCore)
    q = torch.randn(2, 1, 3, 4, requires_grad=True)
    k = torch.randn(2, 1, 3, 4, requires_grad=True)
    v = torch.randn(2, 1, 3, 4, requires_grad=True)
    positions = torch.arange(3).expand(2, -1)

    result = p2p_check_script._strict_attention_reference_rows(
        q,
        k,
        v,
        positions=positions,
        output_dtype=q.dtype,
    )

    assert [batch for batch, _positions in calls] == [1, 1]
    assert [item.tolist() for _batch, item in calls] == [[[0, 1, 2]], [[0, 1, 2]]]
    assert torch.equal(result.out, q + k + v)
    assert torch.equal(result.lse, (q + k).sum(dim=-1))
    (result.out.sum() + result.lse.sum()).backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def _strict_acceptance_provenance(**overrides):
    provenance = {
        "strict_core_id": STRICT_ATTENTION_PRODUCTION_CORE_ID,
        "strict_schedule": STRICT_ATTENTION_FA4_SCHEDULE_ID,
        "attention_backend": "flash_attention_4.cute",
        "actual_backend": "flash_attention_4.cute",
        "rope_backend": "rlkernel.cuda.rope_sm90",
        "strict_mode": True,
        "native_attention_arithmetic": True,
        "num_splits": 1,
        "deterministic_backward": True,
        "reference_only": False,
        "fa_api_source": "flash_attn.cute.interface",
        "fallback": False,
        "strict_split_kv": "disabled",
        "strict_comm_autograd": True,
        "communication_backend": "self_owned_cuda_ag_rs",
        "production_ready": True,
        "strict_full_qkv_all_gather": True,
        "strict_position_ids_all_gather": True,
        "compute_communication": "decoupled",
        "compute_schedule": STRICT_ATTENTION_RING_SCHEDULE_ID,
        "communication_overlap": "disabled",
        "ring_schedule_default": True,
        "ring_partial_arithmetic": False,
        "rope_fusion": False,
        "q_rope_state": "post_rope",
        "k_cache_rope_state": "post_rope",
    }
    provenance.update(overrides)
    return provenance


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("compute_schedule", "dynamic_ring"),
        ("communication_overlap", "enabled"),
        ("ring_schedule_default", False),
        ("ring_partial_arithmetic", True),
        ("actual_backend", "flashinfer"),
        ("rope_backend", "native_rope"),
        ("strict_comm_autograd", False),
    ],
)
def test_strict_shared_core_acceptance_rejects_provenance_drift(field, invalid):
    errors = p2p_check_script._strict_shared_core_identity_errors(
        _strict_acceptance_provenance(**{field: invalid}),
        transport="cuda_ag_rs",
        is_rocm=False,
    )

    assert any(error.startswith(f"{field}=") for error in errors)


def test_strict_shared_core_acceptance_requires_rocm_backend_and_rope():
    provenance = _strict_acceptance_provenance(
        strict_core_id=STRICT_ATTENTION_ROCM_PRODUCTION_CORE_ID,
        strict_schedule=STRICT_ATTENTION_ROCM_SCHEDULE_ID,
        attention_backend="aiter.rocm.ck_dense_mha",
        actual_backend="aiter.rocm.ck_dense_mha",
        rope_backend="rlkernel.rocm.deterministic_rope",
        communication_backend="rccl_ag_rs",
        split_kv_control="dense_non_split_api",
        aiter_api_source="aiter.ops.mha",
        aiter_source_sha256="a" * 64,
    )

    assert not p2p_check_script._strict_shared_core_identity_errors(
        provenance,
        transport="rccl_ag_rs",
        is_rocm=True,
    )


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--batch", "0"], "batch must be positive"),
        (["--atol", "inf"], "atol must be finite and non-negative"),
        (
            ["--final-write-atol", "nan"],
            "final_write_atol must be finite and non-negative",
        ),
        (["--repeats", "1"], "repeats must be at least 2"),
    ],
)
def test_p2p_entrypoint_rejects_non_acceptance_arguments(argv, message):
    args = p2p_check_script.parse_args(argv)

    with pytest.raises(ValueError, match=message):
        p2p_check_script.run_check(
            args,
            global_rank=0,
            tp_rank=0,
            cp_rank=0,
            replica_index=0,
            cp_group=None,
            device=torch.device("cpu"),
        )


def test_fa4_core_rejects_api_without_reduction_controls():
    def legacy_flash_attn(q, k, v, *, causal=False):
        return q

    with pytest.raises(StrictFlashAttentionUnavailable, match="strict controls"):
        StrictFlashAttention4Core(_op=legacy_flash_attn)


def test_fa4_core_fixes_reduction_controls_and_exports_fp32_lse(monkeypatch):
    calls = []

    def fake_fa4(
        q,
        k,
        v,
        *,
        softmax_scale=None,
        causal=False,
        num_splits=0,
        pack_gqa=None,
        deterministic=False,
        return_lse=False,
    ):
        calls.append(
            {
                "q_shape": tuple(q.shape),
                "k_shape": tuple(k.shape),
                "softmax_scale": softmax_scale,
                "causal": causal,
                "num_splits": num_splits,
                "pack_gqa": pack_gqa,
                "deterministic": deterministic,
                "return_lse": return_lse,
            }
        )
        return q.clone(), torch.zeros(
            q.size(0), q.size(2), q.size(1), dtype=torch.float32, device=q.device
        )

    core = StrictFlashAttention4Core(_op=fake_fa4, _package_version="4.test")
    monkeypatch.setattr(core, "_validate_inputs", lambda *_args: None)
    monkeypatch.setattr(core, "_validate_bshd_inputs", lambda *_args: None)
    q = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
    result = core.forward_with_lse(
        q,
        k,
        v,
        causal=True,
        scale=0.125,
        query_position_ids=torch.tensor([[1, 2]]),
        key_position_ids=torch.tensor([[0, 1, 2]]),
    )

    assert calls == [
        {
            "q_shape": (1, 2, 4, 8),
            "k_shape": (1, 3, 2, 8),
            "softmax_scale": 0.125,
            "causal": True,
            "num_splits": 1,
            "pack_gqa": True,
            "deterministic": True,
            "return_lse": True,
        }
    ]
    assert result.out.shape == q.shape
    assert result.lse.shape == q.shape[:3]
    assert result.lse.dtype is torch.float32
    assert result.provenance["strict_core_id"] == STRICT_ATTENTION_PRODUCTION_CORE_ID
    assert result.provenance["strict_schedule"] == STRICT_ATTENTION_FA4_SCHEDULE_ID
    assert result.provenance["num_splits"] == 1
    assert result.provenance["dropout_p"] == 0.0
    assert result.provenance["native_attention_arithmetic"] is True
    assert result.provenance["production_ready"] is True


def test_fa4_core_bshd_entrypoint_preserves_framework_layout(monkeypatch):
    calls = []

    def fake_fa4(
        q,
        k,
        v,
        *,
        softmax_scale=None,
        causal=False,
        num_splits=0,
        pack_gqa=None,
        deterministic=False,
        return_lse=False,
    ):
        calls.append(
            {
                "q_shape": tuple(q.shape),
                "k_shape": tuple(k.shape),
                "pack_gqa": pack_gqa,
                "num_splits": num_splits,
                "return_lse": return_lse,
            }
        )
        return q.clone(), torch.zeros(
            q.size(0), q.size(2), q.size(1), dtype=torch.float32, device=q.device
        )

    core = StrictFlashAttention4Core(_op=fake_fa4, _package_version="4.test")
    monkeypatch.setattr(core, "_validate_bshd_inputs", lambda *_args: None)
    q = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 3, 2, 8, dtype=torch.bfloat16)
    v = torch.randn(1, 3, 2, 8, dtype=torch.bfloat16)

    result = core.forward_bshd_with_lse(
        q,
        k,
        v,
        causal=False,
        query_position_ids=torch.tensor([[2, 3]]),
        key_position_ids=torch.tensor([[0, 1, 2]]),
    )

    assert calls == [
        {
            "q_shape": (1, 2, 4, 8),
            "k_shape": (1, 3, 2, 8),
            "pack_gqa": True,
            "num_splits": 1,
            "return_lse": True,
        }
    ]
    assert result.out.shape == q.shape
    assert result.lse.shape == (1, 4, 2)
    assert result.lse.dtype is torch.float32


# FA4 is the CUDA production core. On ROCm the strict default correctly resolves
# to the AITER CK core instead, so the StrictFlashAttention4Core monkeypatch below
# is never consulted and the assertions cannot hold there.
@pytest.mark.cuda_only
def test_strict_paged_default_selects_fa4_production_core(monkeypatch):
    core = _RecordingStrictCore()
    core.core_id = STRICT_ATTENTION_PRODUCTION_CORE_ID
    core.strict_schedule = STRICT_ATTENTION_FA4_SCHEDULE_ID
    core.backend_id = "flash_attention_4.cute"
    core.native_attention_arithmetic = True
    core.num_splits = 1
    core.deterministic_backward = True
    core.production_ready = True
    core.reference_only = False

    original_forward = core.forward_with_lse

    def production_forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        result.provenance.update(
            {
                "native_attention_arithmetic": True,
                "production_ready": True,
                "reference_only": False,
                "num_splits": 1,
                "deterministic_backward": True,
                "fa_api_source": "flash_attn.cute.interface",
                "fa_package_version": "4.test",
            }
        )
        return result

    core.forward_with_lse = production_forward
    monkeypatch.setattr(
        paged_attention_module,
        "_resolve_strict_core",
        lambda _config: core,
    )
    q, k, v = (tensor.to(torch.bfloat16) for tensor in _qkv(query_len=1))
    result = FlashInferQwen3PagedAttentionOp(flashinfer_module=_fake_flashinfer())(
        q,
        k,
        v,
        _metadata(query_len=1),
        config=FlashInferPagedAttentionConfig(
            mode="decode",
            workspace_size_bytes=1024,
            strict_mode=True,
            strict_rope_op=_IdentityStrictRoPE(),
        ),
    )

    assert len(core.calls) == q.size(0)
    assert result.provenance["strict_core_id"] == STRICT_ATTENTION_PRODUCTION_CORE_ID
    assert result.provenance["actual_backend"] == "flash_attention_4.cute"
    assert result.provenance["num_splits"] == 1
    assert result.provenance["native_attention_arithmetic"] is True
    assert result.provenance["reference_only"] is False
