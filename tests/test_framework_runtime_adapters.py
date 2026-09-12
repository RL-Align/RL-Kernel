# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import ast
import os
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import rl_engine.integrations.framework_operators as framework_operators
from rl_engine.integrations.ablation import (
    IntegrationPlan,
    configure_integration_environment,
    integration_plan_from_environment,
)
from rl_engine.integrations.framework_operators import (
    MegatronAttentionOperator,
    SemanticOperatorHandle,
    VllmAttentionOperator,
    VllmLogpOperator,
    _megatron_zigzag_layout,
    _packed_local_sequence_layout,
    _vllm_kv_cache_views,
)
from rl_engine.integrations.megatron_runtime import (
    _deterministic_reduce_from_tensor_model_parallel_region,
    _install_torch_dist_object_compatibility,
    _patch_strict_attention_projections,
    install_megatron_integration,
)
from rl_engine.integrations.runtime import FrameworkOperatorIntegration
from rl_engine.integrations.state import clear_active_integration
from rl_engine.integrations.vllm_runtime import (
    _patch_qwen3_strict_model,
    _patch_strict_rocm_rotary_embedding,
    _register_attention_backend,
    configure_vllm_environment,
)
from rl_engine.kernels.attention_contract import (
    STRICT_ATTENTION_FA4_SCHEDULE_ID,
    STRICT_ATTENTION_PRODUCTION_CORE_ID,
    AttentionContract,
    AttentionDType,
    AttentionMode,
    AttentionRole,
    ReductionSpec,
    ShardingSpec,
)
from rl_engine.kernels.ops.cuda.attention.strict_runtime import (
    StrictCUDAAttentionRuntime,
)


def test_torch_dist_object_compatibility_deserializes_scalar_bytes_io(monkeypatch):
    strategy_name = "megatron.core.dist_checkpointing.strategies.torch"
    strategy = ModuleType(strategy_name)
    calls = []

    def replace(state_dict, flat_mapping, rename_mapping):
        calls.append((state_dict, flat_mapping, rename_mapping))
        return state_dict

    strategy._replace_sharded_keys_with_state_dict_keys = replace
    monkeypatch.setitem(sys.modules, strategy_name, strategy)

    _install_torch_dist_object_compatibility()
    installed = strategy._replace_sharded_keys_with_state_dict_keys
    _install_torch_dist_object_compatibility()
    payload = __import__("io").BytesIO()
    torch.save([{"recipe": "checkpoint object"}], payload)

    assert strategy._replace_sharded_keys_with_state_dict_keys is installed
    assert installed({"state": payload}, "flat", "rename") == {
        "state": [{"recipe": "checkpoint object"}]
    }
    assert calls == [({"state": [{"recipe": "checkpoint object"}]}, "flat", "rename")]


def test_framework_adapters_do_not_construct_registered_kernels_directly():
    source_path = (
        Path(__file__).parents[1] / "rl_engine" / "integrations" / "framework_operators.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = {
        "AttentionAblationOp",
        "DeterministicCPAttentionReferenceOp",
        "Qwen3FFNOp",
        "StrictCUDAAttentionRuntime",
        "VocabParallelLogprobOp",
    }
    constructed = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert constructed.isdisjoint(forbidden)


def test_semantic_handle_uses_operator_bridge_and_exposes_instance_provenance():
    handle = SemanticOperatorHandle(
        target="training",
        semantic_op="attention",
        backend_id="rlkernel.attention.deterministic.v1",
    )
    tensor = torch.zeros(1, 1, 1, 8)

    first = handle.get(
        tensor,
        topology={
            "world_size": 1,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        },
    )
    second = handle.get(
        tensor,
        topology={
            "world_size": 1,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        },
    )

    assert first is second
    assert first.backend_id == "rlkernel.attention.deterministic.v1"
    assert handle.provenance is not None
    assert handle.provenance["semantic_op"] == "attention"
    assert handle.provenance["backend_id"] == first.backend_id


def test_plan_environment_is_shared_by_both_framework_installers(monkeypatch, tmp_path):
    for variable in (
        "RL_KERNEL_ATTENTION_CASE",
        "RL_KERNEL_FFN_CASE",
        "RL_KERNEL_LOGP_CASE",
        "RL_KERNEL_READBACK_DIR",
    ):
        monkeypatch.setenv(variable, "previous")
    plan = IntegrationPlan.from_case_ids(attention="P/R", ffn="R/P", logp="R/R")
    configure_integration_environment(plan, readback_dir=str(tmp_path))

    assert integration_plan_from_environment() == plan
    assert Path(os.environ["RL_KERNEL_READBACK_DIR"]) == tmp_path


def test_vllm_rlkernel_attention_overrides_selected_flash_attn_backend(
    monkeypatch,
):
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    plan = IntegrationPlan.from_case_ids(attention="P/R")

    configure_vllm_environment(plan)

    expected = "ROCM_AITER_FA" if torch.version.hip is not None else "FLASH_ATTN"
    assert os.environ["VLLM_ATTENTION_BACKEND"] == expected


def test_vllm_rocm_attention_selects_aiter_metadata_backend(monkeypatch):
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(torch.version, "hip", "7.1")
    plan = IntegrationPlan.from_case_ids(attention="P/R")

    configure_vllm_environment(plan)

    assert os.environ["VLLM_ATTENTION_BACKEND"] == "ROCM_AITER_FA"


def test_vllm_rocm_registration_wraps_aiter_backend(monkeypatch):
    selected = []

    class BackendEnum:
        ROCM_AITER_FA = "ROCM_AITER_FA"
        FLASH_ATTN = "FLASH_ATTN"

    class AiterImpl:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def forward(self, *args, **kwargs):
            return args, kwargs

    class AiterBuilder:
        pass

    class AiterBackend:
        pass

    registry = ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = BackendEnum
    registry.register_backend = lambda backend, path: selected.append((backend, path))
    aiter = ModuleType("vllm.v1.attention.backends.rocm_aiter_fa")
    aiter.AiterFlashAttentionBackend = AiterBackend
    aiter.AiterFlashAttentionImpl = AiterImpl
    aiter.AiterFlashAttentionMetadataBuilder = AiterBuilder
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setitem(sys.modules, aiter.__name__, aiter)
    monkeypatch.setattr(torch.version, "hip", "7.1")

    plan = IntegrationPlan.from_case_ids(attention="P/R")

    class Integration:
        def __init__(self):
            self.plan = plan
            self.installed = {}
            self.hooks = []

        def install_operator(self, module, operator):
            self.installed[module] = operator

        def record_installed_hook(self, module, hook):
            self.hooks.append((module, hook))

    integration = Integration()
    _register_attention_backend(integration)

    assert selected == [
        (
            BackendEnum.ROCM_AITER_FA,
            "rl_engine.integrations.vllm_runtime.RlKernelAttentionBackend",
        )
    ]
    assert "attention" in integration.installed
    assert integration.hooks == [
        (
            "attention",
            "rl_engine.integrations.vllm_runtime.RlKernelAttentionBackend",
        )
    ]


def test_megatron_install_is_idempotent_in_one_actor():
    class Attention:
        def forward(self, value):
            return value

    class FFN:
        def forward(self, value):
            return value

    plan = IntegrationPlan.from_case_ids()
    clear_active_integration("megatron")
    try:
        first = install_megatron_integration(
            plan,
            attention_classes=(Attention,),
            ffn_classes=(FFN,),
        )
        second = install_megatron_integration(
            plan,
            attention_classes=(Attention,),
            ffn_classes=(FFN,),
        )
        assert first is second
    finally:
        clear_active_integration("megatron")


def test_megatron_zigzag_positions_preserve_global_cp_ownership():
    rank_zero = _megatron_zigzag_layout(4, cp_rank=0, cp_world_size=2)
    rank_one = _megatron_zigzag_layout(4, cp_rank=1, cp_world_size=2)

    assert rank_zero == ((0, 1, 6, 7), (0, 3), (0, 6), (0, 2, 4))
    assert rank_one == ((2, 3, 4, 5), (1, 2), (2, 4), (0, 2, 4))
    assert sorted(rank_zero[0] + rank_one[0]) == list(range(8))


def test_packed_layout_recovers_local_offsets_from_global_cu_seqlens():
    packed = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 8, 16], dtype=torch.int32),
    )

    local_offsets, global_lengths = _packed_local_sequence_layout(
        packed,
        cp_world_size=2,
        local_query_tokens=8,
        local_kv_tokens=8,
    )

    assert local_offsets == (0, 4, 8)
    assert global_lengths == (8, 8)


def test_megatron_packed_attention_runs_each_sequence_in_thd_order(monkeypatch):
    calls: list[dict[str, object]] = []

    class Operator:
        def bind_accelerator_runtime(self, tensor, *, process_group=None):
            assert tensor is query
            assert process_group == "cp-group"

        def __call__(self, q, k, v, **kwargs):
            del k, v
            calls.append(kwargs)
            return SimpleNamespace(
                out=q.clone(),
                provenance={
                    "actual_backend": "rlkernel.cuda.attention.fa4_ag_rs.v1",
                    "core_rows": [{"actual_backend": "rlkernel.cuda.attention.fa4.v1"}],
                },
            )

    operator = Operator()

    class Handle:
        provenance = {}

        def get(self, tensor, *, topology):
            assert tensor.shape == (8, 2, 4)
            assert topology["context_parallel_size"] == 2
            return operator

    parallel_state = SimpleNamespace(
        get_context_parallel_world_size=lambda: 2,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: "cp-group",
    )
    monkeypatch.setattr(framework_operators, "_megatron_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        framework_operators,
        "_require_attention_accelerator",
        lambda tensor: "cuda",
    )
    packed = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 8, 16], dtype=torch.int32),
    )
    query = torch.zeros(8, 2, 4, dtype=torch.bfloat16)
    key = torch.zeros(8, 1, 4, dtype=torch.bfloat16)

    output = MegatronAttentionOperator(handle=Handle())(
        SimpleNamespace(softmax_scale=0.5),
        query,
        key,
        key,
        None,
        packed_seq_params=packed,
        num_splits=1,
    )

    assert output.shape == (8, 8)
    assert len(calls) == 1
    assert [call["contract"].sharding.global_sequence_length for call in calls] == [
        8,
    ]
    assert calls[0]["query_position_ids"].tolist() == [
        [0, 1, 6, 7],
        [0, 1, 6, 7],
    ]


def test_megatron_attention_binds_rocm_core_and_schedule(monkeypatch):
    calls = []

    class Operator:
        def bind_accelerator_runtime(self, tensor, *, process_group=None):
            calls.append((tensor, process_group))

        def __call__(self, q, k, v, **kwargs):
            del k, v
            calls.append(kwargs)
            return SimpleNamespace(
                out=q.clone(),
                provenance={
                    "actual_backend": "rlkernel.rocm.attention.aiter_ck_ag_rs.v1",
                    "fallback": False,
                },
            )

    operator = Operator()

    class Handle:
        provenance = {}

        def get(self, tensor, *, topology):
            assert topology["context_parallel_size"] == 1
            return operator

    parallel_state = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setattr(framework_operators, "_megatron_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(
        framework_operators,
        "_require_attention_accelerator",
        lambda tensor: "rocm",
    )
    query = torch.zeros(4, 1, 2, 8, dtype=torch.bfloat16)
    key = torch.zeros(4, 1, 1, 8, dtype=torch.bfloat16)
    adapter = MegatronAttentionOperator(handle=Handle())

    output = adapter(SimpleNamespace(softmax_scale=0.25), query, key, key, None)

    assert output.shape == (4, 1, 16)
    assert calls[0] == (query, None)
    config = calls[1]["config"]
    assert config.strict_core_id == "rlkernel.attention.rocm.aiter_ck_dense_mha.v1"
    assert config.strict_schedule == "single_batch_aiter_ck_dense_mha_no_splitkv"
    assert calls[1]["communication_backend"] == "none"
    assert adapter.provenance["execution"]["runtime_platform"] == "rocm"


def test_vllm_attention_routes_paged_cache_to_rocm_runtime(monkeypatch):
    runtime_calls = []

    class Runtime:
        def forward_paged_with_lse(self, q, k, v, **kwargs):
            runtime_calls.append((q, k, v, kwargs))
            return SimpleNamespace(
                out=q.clone(),
                lse=torch.zeros(q.shape[:-1], dtype=torch.float32),
                provenance={
                    "actual_backend": "rlkernel.rocm.attention.aiter_ck_ag_rs.v1",
                    "fallback": False,
                },
            )

    class Operator:
        def bind_accelerator_runtime(self, tensor, *, process_group=None):
            assert process_group is None
            return Runtime()

    class Handle:
        provenance = {}

        def get(self, tensor, *, topology):
            assert topology["context_parallel_size"] == 1
            return Operator()

    monkeypatch.setattr(
        framework_operators,
        "_require_attention_accelerator",
        lambda tensor: "rocm",
    )
    query = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
    kv_cache = torch.zeros(2, 1, 4, 16, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        block_table=torch.tensor([[0]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        num_actual_tokens=1,
        max_seq_len=1,
    )
    impl = SimpleNamespace(head_size=8, num_heads=2, num_kv_heads=1, scale=8**-0.5)
    adapter = VllmAttentionOperator(handle=Handle())

    output = adapter(impl, object(), query, query, query, kv_cache, metadata)

    assert output.shape == (1, 16)
    assert len(runtime_calls) == 1
    assert runtime_calls[0][0].shape == (1, 2, 1, 8)
    assert torch.equal(runtime_calls[0][3]["cu_seqlens_q"], torch.tensor([0, 1]))
    assert torch.equal(runtime_calls[0][3]["kv_indptr"], torch.tensor([0, 1]))
    assert adapter.provenance["execution"]["runtime_platform"] == "rocm"
    assert (
        adapter.provenance["execution"]["materialization"]
        == "direct_vllm_paged_kv_to_aiter_batch_prefill_ck"
    )
    assert adapter.provenance["execution"]["dense_kv_materialized"] is False


def test_vllm_rocm_metadata_stays_on_device_and_is_reused_across_layers(monkeypatch):
    original_tolist = torch.Tensor.tolist
    tolist_calls = 0

    def counting_tolist(tensor):
        nonlocal tolist_calls
        tolist_calls += 1
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    query = torch.zeros(2, 2, 8, dtype=torch.bfloat16)
    block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        max_seq_len=5,
    )
    adapter = VllmAttentionOperator()
    first_layer = object()
    second_layer = object()

    first, _ = adapter._materialization_groups(
        metadata,
        query=query,
        block_table=block_table,
        block_size=8,
        num_actual=2,
        cache_owner=first_layer,
    )
    second, summary = adapter._materialization_groups(
        metadata,
        query=query,
        block_table=block_table,
        block_size=8,
        num_actual=2,
        cache_owner=second_layer,
    )

    assert first is second
    assert torch.equal(first[0]["seqused_k"], torch.tensor([3, 5], dtype=torch.int32))
    assert torch.equal(first[0]["cu_seqlens_q"], torch.tensor([0, 1, 2], dtype=torch.int32))
    assert torch.equal(first[0]["kv_indptr"], torch.tensor([0, 1, 2], dtype=torch.int32))
    assert tolist_calls == 0
    assert summary["metadata_reused_across_layers"] is True


def test_vllm_current_flash_attention_kv_cache_layout_is_materialized():
    cache = torch.arange(2 * 3 * 4 * 10).reshape(2, 3, 4, 10)

    key, value = _vllm_kv_cache_views(cache, head_size=5)

    assert key.shape == (2, 4, 3, 5)
    assert value.shape == (2, 4, 3, 5)
    assert torch.equal(key, cache.transpose(1, 2)[..., :5])
    assert torch.equal(value, cache.transpose(1, 2)[..., 5:])


def test_vllm_rocm_native_kv_cache_with_two_heads_is_not_treated_as_pair_axis():
    cache = torch.arange(3 * 2 * 4 * 10).reshape(3, 2, 4, 10)

    key, value = _vllm_kv_cache_views(
        cache,
        head_size=5,
        num_kv_heads=2,
        platform="rocm",
    )

    assert key.shape == (3, 4, 2, 5)
    assert value.shape == (3, 4, 2, 5)
    assert torch.equal(key, cache.transpose(1, 2)[..., :5])
    assert torch.equal(value, cache.transpose(1, 2)[..., 5:])


def test_vllm_rocm_rlkernel_token_major_kv_cache_is_zero_copy():
    cache = torch.arange(3 * 4 * 2 * 10).reshape(3, 4, 2, 10)

    key, value = _vllm_kv_cache_views(
        cache,
        head_size=5,
        num_kv_heads=2,
        platform="rocm",
    )

    assert key.shape == (3, 4, 2, 5)
    assert value.shape == (3, 4, 2, 5)
    assert key.data_ptr() == cache.data_ptr()
    assert value.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert torch.equal(key, cache[..., :5])
    assert torch.equal(value, cache[..., 5:])
    assert key.stride(1) >= key.size(2) * key.stride(2)


def test_vllm_rocm_kv_cache_pair_axis_is_materialized():
    cache = torch.arange(3 * 2 * 4 * 1 * 5).reshape(3, 2, 4, 1, 5)

    key, value = _vllm_kv_cache_views(
        cache,
        head_size=5,
        num_kv_heads=1,
        platform="rocm",
    )

    assert key.shape == (3, 4, 1, 5)
    assert value.shape == (3, 4, 1, 5)
    assert torch.equal(key, cache[:, 0])
    assert torch.equal(value, cache[:, 1])


def test_vllm_rocm_lhbnc_kv_cache_is_normalized_to_block_major():
    cache = torch.arange(2 * 1 * 3 * 4 * 5).reshape(2, 1, 3, 4, 5)

    key, value = _vllm_kv_cache_views(cache, head_size=5, num_kv_heads=1)

    assert key.shape == (3, 4, 1, 5)
    assert value.shape == (3, 4, 1, 5)
    assert torch.equal(key, cache[0].permute(1, 2, 0, 3))
    assert torch.equal(value, cache[1].permute(1, 2, 0, 3))


def test_vllm_rocm_flattened_kv_cache_is_unpacked():
    cache = torch.arange(3 * 2 * 4 * 15).reshape(3, 2, 4, 15)

    key, value = _vllm_kv_cache_views(
        cache,
        head_size=5,
        num_kv_heads=3,
        platform="rocm",
    )

    assert key.shape == (3, 4, 3, 5)
    assert value.shape == (3, 4, 3, 5)
    assert torch.equal(key.flatten(2), cache[:, 0])
    assert torch.equal(value.flatten(2), cache[:, 1])


def test_megatron_strict_attention_projections_install_without_debug_environment(
    monkeypatch,
):
    monkeypatch.delenv("RL_KERNEL_MODEL_DEBUG_DIR", raising=False)

    class ColumnLinear:
        def __init__(self):
            self.allreduce_dgrad = True

        def _forward_impl(self, input, weight, *args, **kwargs):
            del args, kwargs
            return input.new_full((*input.shape[:-1], weight.shape[0]), -1)

    class RowLinear:
        def _forward_impl(self, input, weight, *args, **kwargs):
            del args, kwargs
            return input.new_full((*input.shape[:-1], weight.shape[0]), -1)

    class SelfAttention:
        def __init__(self):
            self.linear_qkv = ColumnLinear()
            self.linear_proj = RowLinear()

    _patch_strict_attention_projections(
        self_attention_cls=SelfAttention,
        column_linear_cls=ColumnLinear,
        row_linear_cls=RowLinear,
        det_gemm=lambda lhs, rhs: lhs @ rhs,
    )
    attention = SelfAttention()
    value = torch.tensor([[1.0, 2.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    qkv = attention.linear_qkv._forward_impl(value, weight, bias=None)
    projection = attention.linear_proj._forward_impl(value, weight, bias=None)
    native = ColumnLinear()._forward_impl(value, weight, bias=None)

    assert torch.equal(qkv, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(projection, qkv)
    assert torch.equal(native, torch.full((1, 3), -1.0))
    assert attention.linear_qkv.allreduce_dgrad is False


def test_megatron_te_attention_projection_uses_injected_strict_tp_reduce():
    calls: list[torch.Tensor] = []

    class ColumnLinear:
        def __init__(self):
            self.layer_norm_weight = torch.ones(2)
            self.weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        def _forward_impl(self, input, weight, *args, **kwargs):
            del args, kwargs
            return input @ weight.t()

    class RowLinear:
        def __init__(self):
            self.weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        def _forward_impl(self, input, weight, *args, **kwargs):
            del args, kwargs
            return input @ weight.t()

    class SelfAttention:
        def __init__(self):
            self.linear_qkv = ColumnLinear()
            self.linear_proj = RowLinear()

    def reduce_from_tp(value: torch.Tensor) -> torch.Tensor:
        calls.append(value)
        return value * 4

    _patch_strict_attention_projections(
        self_attention_cls=SelfAttention,
        column_linear_cls=ColumnLinear,
        row_linear_cls=RowLinear,
        det_gemm=lambda lhs, rhs: lhs @ rhs,
        copy_to_tp=lambda value: value,
        reduce_from_tp=reduce_from_tp,
    )
    attention = SelfAttention()
    value = torch.tensor([[1.0, 2.0]])

    output, bias = attention.linear_proj.forward(value)

    assert bias is None
    assert len(calls) == 1
    assert torch.equal(calls[0], value)
    assert torch.equal(output, value * 4)


def test_megatron_deterministic_tp_reduce_keeps_identity_backward(monkeypatch):
    class Collective:
        def all_reduce(self, value):
            return value * 4

    monkeypatch.setattr(
        "rl_engine.distributed.collectives.collective_for_group",
        lambda group, min_size_bytes: Collective(),
    )
    value = torch.tensor([1.0, 2.0], requires_grad=True)

    output = _deterministic_reduce_from_tensor_model_parallel_region(value, object())
    output.sum().backward()

    assert torch.equal(output, value.detach() * 4)
    assert torch.equal(value.grad, torch.ones_like(value))


def test_vllm_qwen3_strict_model_installs_without_debug_environment(monkeypatch):
    monkeypatch.delenv("RL_KERNEL_MODEL_DEBUG_DIR", raising=False)

    class RMSNorm:
        def __init__(self):
            self.variance_size_override = None
            self.has_weight = True
            self.hidden_size = 2
            self.weight = torch.tensor([1.5, 0.5])
            self.variance_epsilon = 1e-6

        def forward_cuda(self, x, residual=None):
            del residual
            return x.new_full(x.shape, -1)

        def forward_native(self, x, residual=None):
            del residual
            return x.new_full(x.shape, -2)

    class LinearLayer:
        def __init__(self):
            self.weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    class LinearMethod:
        def apply(self, layer, x, bias=None):
            del bias
            return x.new_full((*x.shape[:-1], layer.weight.shape[0]), -1)

    class Attention:
        def __init__(self):
            self.qkv_proj = LinearLayer()
            self.o_proj = LinearLayer()

    _patch_qwen3_strict_model(
        rms_norm_cls=RMSNorm,
        linear_method_cls=LinearMethod,
        attention_cls=Attention,
        det_gemm=lambda lhs, rhs: lhs @ rhs,
    )
    attention = Attention()
    method = LinearMethod()
    value = torch.tensor([[1.0, 2.0]])
    norm = RMSNorm()

    assert torch.equal(
        method.apply(attention.qkv_proj, value),
        torch.tensor([[1.0, 2.0, 3.0]]),
    )
    assert torch.equal(
        method.apply(LinearLayer(), value),
        torch.full((1, 3), -1.0),
    )
    assert torch.equal(
        norm.forward_cuda(value),
        torch.nn.functional.rms_norm(value, (2,), norm.weight, 1e-6),
    )


def test_vllm_rocm_rotary_reuses_one_table_for_query_and_key(monkeypatch):
    calls = []

    class FakeOperator:
        def __call__(self, value, positions):
            calls.append(("single", tuple(value.shape), positions.clone()))
            return value + 1

        def forward_pair(self, query, key, positions):
            calls.append(("pair", tuple(query.shape), tuple(key.shape), positions.clone()))
            return query + 2, key + 3

    class Rotary:
        head_size = 4
        rotary_dim = 4

        def forward_cuda(self, positions, query, key=None):
            return query, key

    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(
        "rl_engine.kernels.ops.rocm.rotary_embedding.rope.RocmDeterministicRoPEOp",
        FakeOperator,
    )
    _patch_strict_rocm_rotary_embedding(Rotary)
    rotary = Rotary()
    positions = torch.tensor([7, 2])
    query = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    key = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    query_out, key_out = rotary.forward_cuda(positions, query, key)
    assert torch.equal(query_out, query + 2)
    assert torch.equal(key_out, key + 3)
    assert calls[0][0] == "pair"
    assert calls[0][1:3] == ((3, 2, 4), (1, 2, 4))

    query_only, absent_key = rotary.forward_cuda(positions, query)
    assert torch.equal(query_only, query + 1)
    assert absent_key is None
    assert calls[1][0] == "single"


def test_vllm_logp_replaces_every_duplicate_sampled_token_column():
    logprobs_type = namedtuple(
        "LogprobsTensors",
        ("logprob_token_ids", "logprobs", "selected_token_ranks"),
    )

    @dataclass(frozen=True)
    class SamplerResult:
        sampled_token_ids: torch.Tensor
        logprobs_tensors: object

    operator = VllmLogpOperator(lambda *_args, **_kwargs: None)
    result = SamplerResult(
        sampled_token_ids=torch.tensor([[7], [8]]),
        logprobs_tensors=logprobs_type(
            logprob_token_ids=torch.tensor([[7, 7], [8, 9]]),
            logprobs=torch.tensor([[-0.1, -0.1], [-0.2, -0.3]]),
            selected_token_ranks=torch.tensor([1, 2]),
        ),
    )

    updated = operator._replace_sampled_value(
        result,
        token_ids=torch.tensor([7, 8]),
        selected=torch.tensor([-1.25, -2.5]),
        provenance={},
    )

    assert torch.equal(
        updated.logprobs_tensors.logprobs,
        torch.tensor([[-1.25, -1.25], [-2.5, -0.3]]),
    )


def _cp1_contract(tokens: int) -> AttentionContract:
    return AttentionContract(
        role=AttentionRole.TRAIN,
        mode=AttentionMode.PREFILL,
        dtype=AttentionDType.BF16,
        batch_size=1,
        query_sequence_length=tokens,
        head_dim=4,
        causal=True,
        causal_offsets=(0,),
        sharding=ShardingSpec(
            tp_rank=0,
            tp_world_size=1,
            cp_rank=0,
            cp_world_size=1,
            global_q_heads=2,
            global_kv_heads=1,
            local_q_head_start=0,
            local_q_heads=2,
            local_kv_head_start=0,
            local_kv_heads=1,
            global_sequence_length=tokens,
            local_sequence_length=tokens,
            global_block_indices=(0,),
            global_block_token_starts=(0,),
            local_block_offsets=(0, tokens),
        ),
        reduction=ReductionSpec(),
    )


def test_strict_cuda_runtime_pins_training_to_single_query_prefixes(monkeypatch):
    calls: list[tuple[int, int, bool]] = []

    class Core:
        core_id = STRICT_ATTENTION_PRODUCTION_CORE_ID
        strict_schedule = STRICT_ATTENTION_FA4_SCHEDULE_ID

        def forward_bshd_with_lse(self, q, k, v, *, causal, **kwargs):
            del v, kwargs
            calls.append((q.size(1), k.size(1), causal))
            return SimpleNamespace(
                out=q.clone(),
                lse=torch.zeros((q.size(0), q.size(2), q.size(1)), dtype=torch.float32),
                provenance={"actual_backend": "fake.cuda.fa4"},
            )

    monkeypatch.setattr(
        StrictCUDAAttentionRuntime, "_require_nvidia_cuda", lambda self, tensor: None
    )
    runtime = StrictCUDAAttentionRuntime(core=Core(), communication=object())
    q = torch.zeros(1, 2, 4, 4, dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
    positions = torch.arange(4).unsqueeze(0)

    result = runtime.forward_with_lse(
        q,
        k,
        k,
        contract=_cp1_contract(4),
        causal=True,
        scale=0.5,
        cp_world_size=1,
        query_position_ids=positions,
        key_position_ids=positions,
    )

    assert calls == [(4, 4, True)]
    assert result.provenance["query_schedule"] == "full_sequence_causal_single_launch"


class _ReadbackOperator:
    backend_id = "rlkernel.attention.test"

    def __init__(self, provenance):
        self.provenance = provenance

    def __call__(self, value):
        return value


@pytest.mark.parametrize(
    ("provenance", "match"),
    [
        ({"runtime_platform": "cpu"}, "non-cuda"),
        ({"runtime_platform": "cuda", "actual_backend": "triton.attention"}, "triton"),
        ({"runtime_platform": "cuda", "triton_used": True}, "triton"),
    ],
)
def test_strict_readback_rejects_non_cuda_and_triton(provenance, match):
    plan = IntegrationPlan.from_case_ids(attention="R/R")
    integration = FrameworkOperatorIntegration(
        framework="megatron",
        target="training",
        plan=plan,
        rl_kernel_operators={"attention": _ReadbackOperator(provenance)},
    )
    integration.record_installed_hook("attention", "test.attention")
    integration.execute("attention", lambda value: value, "x")

    with pytest.raises(RuntimeError, match=match):
        integration.assert_strict_ready()


def test_strict_readback_accepts_cuda_without_triton():
    plan = IntegrationPlan.from_case_ids(attention="R/R")
    integration = FrameworkOperatorIntegration(
        framework="megatron",
        target="training",
        plan=plan,
        rl_kernel_operators={
            "attention": _ReadbackOperator(
                {
                    "runtime_platform": "cuda",
                    "actual_backend": "rlkernel.cuda.fa4",
                    "triton_used": False,
                }
            )
        },
    )
    integration.record_installed_hook("attention", "test.attention")
    integration.execute("attention", lambda value: value, "x")

    integration.assert_strict_ready()


def test_strict_readback_accepts_rocm_without_triton():
    plan = IntegrationPlan.from_case_ids(attention="R/R")
    integration = FrameworkOperatorIntegration(
        framework="megatron",
        target="training",
        plan=plan,
        rl_kernel_operators={
            "attention": _ReadbackOperator(
                {
                    "runtime_platform": "rocm",
                    "actual_backend": "rlkernel.rocm.attention.aiter_ck_ag_rs.v1",
                    "triton_used": False,
                }
            )
        },
    )
    integration.record_installed_hook("attention", "test.attention")
    integration.execute("attention", lambda value: value, "x")

    integration.assert_strict_ready()


def test_production_readback_infers_platform_from_real_execution_tensors():
    plan = IntegrationPlan.from_case_ids(attention="P/P")
    integration = FrameworkOperatorIntegration(
        framework="megatron",
        target="training",
        plan=plan,
        rl_kernel_operators={},
    )
    value = torch.zeros(2)

    integration.execute("attention", lambda tensor: tensor + 1, value)

    readback = integration.readback()["operators"]["attention"]
    assert readback["implementation"] == "production"
    assert readback["provenance"]["runtime_platform"] == "cpu"


def test_production_readback_uses_structural_result_provenance():
    plan = IntegrationPlan.from_case_ids(logp="P/P")
    integration = FrameworkOperatorIntegration(
        framework="megatron",
        target="training",
        plan=plan,
        rl_kernel_operators={},
    )
    request = SimpleNamespace(logits=torch.zeros(2, 4), target_ids=torch.zeros(2))

    def native(actual_request):
        return SimpleNamespace(
            logp=actual_request.logits[:, :1],
            provenance={"actual_backend": "production.logp.test"},
        )

    integration.execute("logp", native, request)

    provenance = integration.readback()["operators"]["logp"]["provenance"]
    assert provenance["actual_backend"] == "production.logp.test"
    assert provenance["runtime_platform"] == "cpu"
