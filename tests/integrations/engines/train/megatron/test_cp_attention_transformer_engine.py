# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Optional Transformer Engine oracle tests for CP attention merging."""

from __future__ import annotations

import importlib

import pytest
import torch

from rl_engine.reference.attention.cp_attention import (
    AttentionPartialState,
    merge_attention_partial_states,
)


def _te_context_parallel_module():
    try:
        return importlib.import_module(
            "transformer_engine.pytorch.attention.dot_product_attention.context_parallel"
        )
    except (ImportError, OSError, RuntimeError) as exc:
        pytest.skip(f"Transformer Engine context-parallel attention is unavailable: {exc}")


def test_cp_attention_merge_matches_transformer_engine_corrections():
    te_cp = _te_context_parallel_module()
    gen = torch.Generator().manual_seed(238)
    out_a = torch.randn(2, 3, 5, 4, generator=gen)
    out_b = torch.randn(2, 3, 5, 4, generator=gen)
    lse_a = torch.randn(2, 3, 5, generator=gen)
    lse_b = torch.randn(2, 3, 5, generator=gen)

    ours = merge_attention_partial_states(
        [
            AttentionPartialState(out=out_b, lse=lse_b, block_start=5, block_end=9),
            AttentionPartialState(out=out_a, lse=lse_a, block_start=0, block_end=5),
        ]
    )

    te_lse = lse_a.clone()
    te_cp.flash_attn_fwd_softmax_lse_correction(te_lse, lse_b)
    te_out = te_cp.flash_attn_fwd_out_correction_init(out_a.clone(), te_lse, lse_a, seq_dim=2)
    te_cp.flash_attn_fwd_out_correction(te_out, out_b, te_lse, lse_b, seq_dim=2)

    torch.testing.assert_close(ours.lse, te_lse, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(ours.out, te_out, atol=1.0e-6, rtol=0.0)
