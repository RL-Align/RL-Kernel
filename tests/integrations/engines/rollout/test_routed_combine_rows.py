# SPDX-License-Identifier: Apache-2.0
"""CPU check of the routed-combine row metadata against make_routing_data's rule."""

import torch

from rl_engine.integrations.engines.rollout.vllm.routed_combine import routing_row_metadata


def test_row_metadata_with_nonlocal_slots():
    ids = torch.tensor([[3, 1, -1], [1, -1, 0], [3, 2, 1]], dtype=torch.int32)
    tokens, experts, valid = routing_row_metadata(ids)
    flat = ids.reshape(-1)
    order = torch.sort(flat.clamp(min=0), stable=True).indices
    assert tokens.dtype == experts.dtype == torch.int64
    assert torch.equal(tokens, order // 3)
    assert torch.equal(valid, flat[order] >= 0)
    assert int(valid.sum()) == 7
    # -1 slots sort as expert 0 and are flagged invalid.
    assert (experts[~valid] == 0).all()
    # Valid rows are expert-major, token-ascending within an expert (stable).
    keep = valid.nonzero().flatten()
    pairs = list(zip(experts[keep].tolist(), tokens[keep].tolist()))
    assert pairs == sorted(pairs)


def test_all_local_rows_cover_every_token_slot_once():
    ids = torch.tensor([[5, 2, 9], [2, 7, 5]], dtype=torch.int64)
    tokens, experts, valid = routing_row_metadata(ids)
    assert bool(valid.all())
    assert tokens.bincount().tolist() == [3, 3]
    assert experts.tolist() == sorted(ids.reshape(-1).tolist())
