# SPDX-License-Identifier: Apache-2.0
"""Reject geometry/masks that would change the original causal selection."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.engines.train.megatron.native_indexer import (
    _require_causal_mask,
    bind_native_fp8_indexer,
)


def test_positive_native_auxiliary_coefficient_is_not_silently_changed():
    core = SimpleNamespace(
        indexer=object(), config=SimpleNamespace(dsa_indexer_loss_coeff=0.1)
    )
    with pytest.raises(ValueError, match="auxiliary coefficient zero"):
        bind_native_fp8_indexer(core, object())


def test_causal_mask_checks_exact_key_availability_and_all_batches():
    t, batch = 12, 2
    future = torch.arange(t // 4)[None, :] >= ((torch.arange(t) + 1) // 4)[:, None]
    mask = torch.where(future, float("-inf"), 0.0)[None].expand(batch, -1, -1).clone()
    _require_causal_mask(mask, t, batch)
    # Exposing a future compressed key in only rank/batch 1 must fail.
    mask[1, 0, 0] = 0
    with pytest.raises(ValueError, match="noncanonical"):
        _require_causal_mask(mask, t, batch)
    with pytest.raises(ValueError, match="actual full-prefix"):
        _require_causal_mask(None, t, batch)
