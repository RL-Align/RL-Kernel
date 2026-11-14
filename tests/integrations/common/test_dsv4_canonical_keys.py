# SPDX-License-Identifier: Apache-2.0
"""CPU metadata/session checks; not GPU execution acceptance."""

import pytest
import torch
from rl_engine.integrations.common.mhc_worker import P1Worker
from rl_engine.integrations.engines.train.megatron.logical_keys import logical_keys_for_input
from rl_engine.ops.autograd.canonical_backward import canonical_backward_session


def cases():
    return [
        dict(id="A", prompt_token_ids=[1, 2], response_token_ids=[3]),
        dict(id="B", prompt_token_ids=[4, 5], response_token_ids=[6]),
    ]


def test_actual_token_rows_keep_logical_identity_when_batch_position_changes():
    a, b = cases()
    single = logical_keys_for_input(torch.tensor([[1, 2, 3]]), [a])
    ab = logical_keys_for_input(torch.tensor([[1, 2, 3], [4, 5, 6]]), [a, b])
    ba = logical_keys_for_input(torch.tensor([[4, 5, 6], [1, 2, 3]]), [b, a])
    assert torch.equal(single, ab[::2]) and torch.equal(single, ba[1::2])
    assert torch.equal(single[:, 1], torch.arange(3))
    with pytest.raises(ValueError, match="exactly one"):
        logical_keys_for_input(torch.tensor([[7, 8, 9]]), [a])


def test_triton_binding_rejects_fused_and_scopes_real_row_metadata():
    with pytest.raises(ValueError):
        P1Worker("verl", "training", backend="rocm_triton", mode="fused")
    worker = P1Worker("verl", "training", backend="rocm_triton", mode="split")
    keys = logical_keys_for_input(torch.tensor([[1, 2, 3]]), [cases()[0]])
    rows = torch.zeros(3, 4, 8, dtype=torch.bfloat16)
    with worker.logical_rows(keys), worker.request(0):
        assert worker._canonical(rows, "model.layers.0.hc_attn")["logical_keys"] is keys
        with pytest.raises(ValueError, match="actual mHC rows"):
            worker._canonical(rows[:1], "model.layers.0.hc_attn")
    assert worker._canonical(rows, "id")["logical_keys"] is None


def test_canonical_session_requires_every_registered_parameter_contribution():
    keys = logical_keys_for_input(torch.tensor([[1, 2, 3]]), [cases()[0]])
    with canonical_backward_session() as session:
        slot = session.register("model.layers.0.hc_attn.scale", keys)
        with pytest.raises(RuntimeError, match="incomplete"):
            session.validate_complete()
        grad = session.submit_rows(
            "model.layers.0.hc_attn.scale", slot, torch.ones(3, 3), lambda rows: rows.sum(0)
        )
        assert torch.equal(grad, torch.tensor([3.0, 3.0, 3.0]))
        session.validate_complete()
