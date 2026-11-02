# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU coverage for the optional Vime WS2 selected-logprob adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rl_engine.integrations.orchestrators.vime.providers.logp import (
    SelectedLogprobProviderUnavailable,
    provider,
)


def _request(*, cp_rank: int = 0, with_entropy: bool = False, keep_mask=None):
    logits = torch.tensor(
        [[0.25, -0.5, 1.0, 0.1, -0.3, 0.6, -0.7, 0.4] for _ in range(3)],
        dtype=torch.float32,
        requires_grad=True,
    )
    return SimpleNamespace(
        logits=logits,
        target_ids=torch.tensor([2, 5, 0]),
        tensor_parallel_group=None,
        context_parallel=SimpleNamespace(
            world_size=2,
            rank=cp_rank,
            layout="zigzag",
        ),
        with_entropy=with_entropy,
        with_entropy_grad=with_entropy,
        log_prob_keep_mask=keep_mask,
        metadata={
            "real_vocab_size": 7,
            "padded_vocab_size": 8,
            "tp_rank": 0,
            "tp_world_size": 1,
            "num_vocab_tiles": 4,
        },
    )


def test_provider_runs_locally_with_cp2_row_metadata():
    request = _request(cp_rank=1)

    result = provider(request)
    reference = torch.log_softmax(request.logits[:, :7], dim=-1)[
        torch.arange(request.logits.size(0)), request.target_ids
    ]

    assert result.selected_logprobs.shape == (3, 1)
    torch.testing.assert_close(result.selected_logprobs.squeeze(-1), reference)
    assert result.backend_id == "pytorch-vocab-parallel-logp-ws2"
    assert result.provenance["cp_row_ownership"] == {
        "cp_rank": 1,
        "cp_world_size": 2,
        "layout": "zigzag",
        "local_token_rows": 3,
        "cp_is_merge_axis": False,
    }


def test_provider_entropy_preserves_vime_semantics_and_autograd():
    request = _request(with_entropy=True)

    result = provider(request)
    reference_logits = request.logits.detach().clone().requires_grad_(True)
    log_probs = torch.log_softmax(reference_logits[:, :7], dim=-1)
    reference_logp = log_probs[torch.arange(reference_logits.size(0)), request.target_ids]
    reference_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

    torch.testing.assert_close(result.selected_logprobs.squeeze(-1), reference_logp)
    torch.testing.assert_close(result.entropy, reference_entropy)
    (result.selected_logprobs.sum() + result.entropy.sum()).backward()
    (reference_logp.sum() + reference_entropy.sum()).backward()
    torch.testing.assert_close(request.logits.grad[:, :7], reference_logits.grad[:, :7])
    assert bool((request.logits.grad[:, 7] == 0).all())


def test_provider_rejects_top_p_replay_without_changing_its_semantics():
    request = _request(keep_mask=torch.ones((3, 8), dtype=torch.bool))

    with pytest.raises(SelectedLogprobProviderUnavailable, match="top-p replay"):
        provider(request)


def test_provider_rejects_local_vocab_metadata_that_cannot_describe_tp_ownership():
    request = _request()
    request.metadata["padded_vocab_size"] = 16

    with pytest.raises(SelectedLogprobProviderUnavailable, match="cover padded_vocab_size"):
        provider(request)
