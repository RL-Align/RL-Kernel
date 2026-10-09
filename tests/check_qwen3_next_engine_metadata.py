# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Real vLLM builder semantics; no GPU kernel or capability override is needed."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec

from rl_engine.integrations.qwen3_next_vllm import (
    SharedGDNBackend,
    SharedGDNMetadataBuilder,
    metadata_sequences,
)


def build_metadata(lengths, computed, slots):
    config = SimpleNamespace(
        additional_config={"gdn_prefill_backend": "triton"},
        speculative_config=None,
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(linear_key_head_dim=128)),
    )
    spec = MambaSpec(
        block_size=64,
        shapes=((3, 2048), (8, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_cache_mode="none",
    )
    builder = SharedGDNMetadataBuilder(
        spec, ["model.layers.0.linear_attn"], config, torch.device("cpu")
    )
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    seq = torch.tensor(lengths, dtype=torch.int32) + torch.tensor(computed, dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=cu,
        query_start_loc_cpu=cu,
        seq_lens=seq,
        num_reqs=len(lengths),
        num_actual_tokens=sum(lengths),
        max_query_len=max(lengths),
        max_seq_len=int(seq.max()),
        block_table_tensor=torch.tensor(slots, dtype=torch.int32)[:, None],
        slot_mapping=torch.zeros(sum(lengths), dtype=torch.int64),
    )
    return builder.build(0, common)


def test_fresh_single_token_prompt_resets_reused_slot():
    metadata = build_metadata([1], [0], [3])
    slots, cu, reset = metadata_sequences(metadata, 1)
    assert slots.tolist() == [3] and cu.tolist() == [0, 1] and reset.tolist() == [3]


def test_mixed_step_resets_only_new_prefill_and_preserves_request_order():
    metadata = build_metadata([1, 8, 3], [64, 0, 7], [7, 3, 5])
    slots, cu, reset = metadata_sequences(metadata, 12)
    assert slots.tolist() == [7, 3, 5]
    assert cu.tolist() == [0, 1, 9, 12]
    assert reset.tolist() == [3]


def test_null_padding_has_no_state_owner():
    metadata = build_metadata([1, 0], [64, 0], [7, 0])
    slots, cu, reset = metadata_sequences(metadata, 1)
    assert slots.tolist() == [7] and cu.tolist() == [0, 1] and reset.numel() == 0


@pytest.mark.parametrize("slots", [[0], [-1], [3, 3]])
def test_active_null_negative_or_duplicate_slots_are_rejected(slots):
    metadata = build_metadata([1] * len(slots), [2] * len(slots), slots)
    with pytest.raises(ValueError, match="distinct positive"):
        metadata_sequences(metadata, len(slots))


def test_custom_backend_cannot_bypass_the_batch_invariance_guard():
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

    assert SharedGDNBackend.supports_batch_invariance() is False
    assert GDNAttentionBackend.supports_batch_invariance() is False
    assert SharedGDNBackend.get_name() != GDNAttentionBackend.get_name()


def test_real_selector_rejects_the_unaccepted_custom_backend(monkeypatch):
    from vllm import envs
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum, register_backend
    from vllm.v1.attention.selector import _cached_get_mamba_attn_backend, get_mamba_attn_backend

    from rl_engine.integrations.qwen3_next_vllm import _register_backend

    try:
        previous = MambaAttentionBackendEnum.CUSTOM.get_path()
    except ValueError:
        previous = None
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    _cached_get_mamba_attn_backend.cache_clear()
    try:
        _register_backend()
        with pytest.raises(RuntimeError, match="not supported for RL_KERNEL_QWEN3_NEXT_GDN"):
            get_mamba_attn_backend(MambaAttentionBackendEnum.CUSTOM)
        with pytest.raises(RuntimeError, match="not supported for GDN_ATTN"):
            get_mamba_attn_backend(MambaAttentionBackendEnum.GDN_ATTN)
    finally:
        if previous is None:
            MambaAttentionBackendEnum.CUSTOM.clear_override()
        else:
            register_backend(MambaAttentionBackendEnum.CUSTOM, previous, is_mamba=True)
        _cached_get_mamba_attn_backend.cache_clear()
