# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from rl_engine.executors.paged_kv_baseline import (
    PagedKVScoringBaseline,
    PagedKVScoringConfig,
    reserve_paged_kv_cache,
)
from rl_engine.executors.stateless_executor import StatelessForwardInputs, score_reference_logprobs


class FakeGenerationReferenceModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)
        self.use_cache_calls: list[bool | None] = []

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        del attention_mask
        self.use_cache_calls.append(use_cache)
        batch, seq_len = input_ids.shape
        key = torch.empty(batch, 1, seq_len, 4, device=input_ids.device)
        value = torch.empty_like(key)
        return SimpleNamespace(
            logits=self.fixed_logits[: input_ids.shape[0], : input_ids.shape[1]],
            past_key_values=((key, value),) if use_cache else None,
        )


def _inputs() -> StatelessForwardInputs:
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3, 0],
            [0, 2, 1, 4, 5],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    completion_mask = torch.tensor(
        [
            [False, False, True, True, False],
            [False, True, True, False, False],
        ]
    )
    return StatelessForwardInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
    )


def _logits_for(inputs: StatelessForwardInputs, vocab_size: int = 8) -> torch.Tensor:
    logits = torch.full((*inputs.input_ids.shape, vocab_size), -3.0)
    for batch_index in range(inputs.input_ids.shape[0]):
        for pos in range(inputs.input_ids.shape[1] - 1):
            token = int(inputs.input_ids[batch_index, pos + 1].item())
            logits[batch_index, pos, token] = 4.0 + batch_index + pos
    return logits


def test_reserve_paged_kv_cache_builds_block_table_and_tracks_bytes():
    inputs = _inputs()
    config = PagedKVScoringConfig(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        block_size=2,
        kv_cache_dtype=torch.float32,
        kv_cache_blocks=8,
    )

    reservation = reserve_paged_kv_cache(inputs, config)

    assert reservation.sequence_lengths.tolist() == [5, 3]
    assert reservation.blocks_per_sequence.tolist() == [3, 2]
    assert reservation.required_blocks == 5
    assert reservation.reserved_blocks == 8
    assert reservation.key_cache.shape == (2, 8, 2, 2, 4)
    assert reservation.value_cache.shape == reservation.key_cache.shape
    assert reservation.block_tables.tolist() == [[0, 1, 2], [3, 4, -1]]

    expected_cache_bytes = 2 * reservation.key_cache.numel() * reservation.key_cache.element_size()
    assert reservation.cache_bytes == expected_cache_bytes
    assert reservation.reserved_bytes == reservation.cache_bytes + reservation.metadata_bytes
    assert reservation.reserved_mb > 0.0


def test_paged_kv_baseline_scores_like_reference_path_and_enables_cache():
    inputs = _inputs()
    logits = _logits_for(inputs).requires_grad_()
    model = FakeGenerationReferenceModel(logits)
    baseline = PagedKVScoringBaseline(
        model,
        PagedKVScoringConfig(
            mode="reference",
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            block_size=2,
            kv_cache_dtype=torch.float32,
            return_token_scores=True,
        ),
    )

    result = baseline.score(inputs)
    expected = score_reference_logprobs(logits, inputs)

    assert model.use_cache_calls == [True]
    assert result.reference_logps is not None
    assert torch.allclose(result.reference_logps, expected)
    assert result.reference_logps.requires_grad is False
    assert result.token_scores is not None
    assert result.metrics["baseline_kind"] == "generation_engine_paged_kv_reservation"
    assert result.metrics["use_cache"] is True
    assert result.metrics["use_cache_passed"] is True
    assert result.metrics["paged_kv_required_blocks"] == 5
    assert result.metrics["paged_kv_cache_reserved_mb"] > 0.0
    assert result.metrics["model_kv_cache_output_present"] is True
    assert result.metrics["model_kv_cache_output_tensors"] == 2
    assert result.metrics["model_kv_cache_output_mb"] > 0.0
    assert result.metrics["total_kv_cache_mb"] > result.metrics["paged_kv_cache_reserved_mb"]


def test_paged_kv_reservation_rejects_capacity_smaller_than_batch_needs():
    inputs = _inputs()
    config = PagedKVScoringConfig(
        num_layers=1,
        num_kv_heads=1,
        head_dim=4,
        block_size=2,
        kv_cache_blocks=4,
    )

    with pytest.raises(ValueError, match="kv_cache_blocks=4"):
        reserve_paged_kv_cache(inputs, config)


def test_importing_paged_kv_baseline_does_not_import_heavy_optional_runtimes(monkeypatch):
    for name in ("vllm", "deepspeed", "ray", "flash_attn"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("rl_engine.executors.paged_kv_baseline")

    assert module.PagedKVScoringBaseline is not None
    for name in ("vllm", "deepspeed", "ray", "flash_attn"):
        assert name not in sys.modules
