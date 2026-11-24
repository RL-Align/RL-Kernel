# SPDX-License-Identifier: Apache-2.0
"""INK-73 DSpark/MTP speculative profile tests."""

import pytest
import torch

from rl_engine.models.inkling2.speculative_contracts import (
    SpeculativeContractError,
    SpeculativeProfile,
    accept_speculative_tokens,
    commit_speculative_cache,
)


def test_exact_token_acceptance_tracks_rollback_lengths():
    draft = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    target = torch.tensor([[1, 2, 9, 4], [5, 6, 7, 8]])
    result = accept_speculative_tokens(draft, target)
    assert torch.equal(result.accepted_lengths, torch.tensor([2, 4]))
    assert torch.equal(result.rollback_lengths, torch.tensor([2, 0]))
    assert torch.equal(result.accepted_tokens[0], torch.tensor([1, 2, 0, 0]))


def test_eos_stops_acceptance_and_cache_commit_rolls_back_tail():
    draft = torch.tensor([[10, 11, 2, 12]])
    target = draft.clone()
    result = accept_speculative_tokens(draft, target, eos_token_id=2)
    cache = torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    committed = commit_speculative_cache(cache, result)
    assert torch.equal(result.accepted_lengths, torch.tensor([3]))
    assert committed.shape == (1, 3, 3)
    assert torch.equal(committed[0, 2], cache[0, 2])


def test_profile_requires_rollback_when_enabled():
    profile = SpeculativeProfile(enabled=True)
    profile.validate()
    with pytest.raises(SpeculativeContractError, match="rollback"):
        SpeculativeProfile(enabled=True, rollback_cache=False).validate()


def test_invalid_token_geometry_fails_closed():
    with pytest.raises(SpeculativeContractError, match="shape"):
        accept_speculative_tokens(
            torch.ones(1, 2, dtype=torch.int64), torch.ones(1, 3, dtype=torch.int64)
        )
