# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""vLLM adapter coverage for runtime-scoped ROCm page validation."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from rl_engine.integrations.engines.rollout.vllm import operators as framework_operators
from rl_engine.integrations.engines.rollout.vllm.operators import VllmAttentionOperator


def test_vllm_attention_routes_page_bounds_epoch_to_rocm_runtime(monkeypatch):
    runtime_calls = []
    page_bounds_epoch = object()

    class Runtime:
        def new_page_bounds_epoch(self):
            return page_bounds_epoch

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

    runtime = Runtime()

    class Operator:
        def bind_accelerator_runtime(self, tensor, *, process_group=None):
            assert process_group is None
            return runtime

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

    output = VllmAttentionOperator(handle=Handle())(
        impl,
        object(),
        query,
        query,
        query,
        kv_cache,
        metadata,
    )

    assert output.shape == (1, 16)
    assert len(runtime_calls) == 1
    assert runtime_calls[0][3]["cached_lengths"] == (1,)
    assert runtime_calls[0][3]["page_bounds_epoch"] is page_bounds_epoch


def test_vllm_page_bounds_epoch_is_scoped_to_one_model_forward():
    query = torch.zeros(2, 2, 8, dtype=torch.bfloat16)
    block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        max_seq_len=5,
    )
    adapter = VllmAttentionOperator()
    layers = [object() for _ in range(36)]
    issued_epochs = []

    def issue_epoch():
        epoch = object()
        issued_epochs.append(epoch)
        return epoch

    common = {
        "query": query,
        "block_table": block_table,
        "block_size": 8,
        "num_actual": 2,
        "include_host_lengths": True,
        "page_bounds_epoch_factory": issue_epoch,
    }
    groups_by_layer = []
    summaries = []
    for layer in layers:
        groups, summary = adapter._materialization_groups(
            metadata,
            cache_owner=layer,
            **common,
        )
        groups_by_layer.append(groups)
        summaries.append(summary)
    next_forward, next_summary = adapter._materialization_groups(
        metadata,
        cache_owner=layers[0],
        **common,
    )

    first = groups_by_layer[0]
    assert all(groups is first for groups in groups_by_layer[1:])
    assert summaries[0]["metadata_reused_across_layers"] is False
    assert all(summary["metadata_reused_across_layers"] is True for summary in summaries[1:])
    assert next_forward is not first
    assert next_forward[0]["page_bounds_epoch"] is not first[0]["page_bounds_epoch"]
    assert next_summary["metadata_reused_across_layers"] is False
    assert issued_epochs == [
        first[0]["page_bounds_epoch"],
        next_forward[0]["page_bounds_epoch"],
    ]
