# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.ops.autograd.canonical_backward import CanonicalBackwardSession


def _keys(sequence, rows):
    return torch.stack(
        [torch.full((rows,), sequence, dtype=torch.int64), torch.arange(rows, dtype=torch.int64)], 1
    )


def _reduce(rows, grads):
    return grads.t().double() @ rows.double()


def test_linear_reduction_is_canonical_and_releases_saved_rows():
    torch.manual_seed(0)
    session = CanonicalBackwardSession()
    rows = [torch.randn(5, 3), torch.randn(4, 3)]
    grads = [torch.randn(5, 2), torch.randn(4, 2)]
    # Register sequence 1 first: the reduction must still follow logical-key order.
    slots = [session.register("w", _keys(1, 5)), session.register("w", _keys(0, 4))]
    assert session.submit_linear("w", slots[0], rows[0], grads[0], _reduce) is None
    result = session.submit_linear("w", slots[1], rows[1], grads[1], _reduce)
    expected = _reduce(torch.cat([rows[1], rows[0]]), torch.cat([grads[1], grads[0]]))
    assert torch.equal(result, expected)
    assert all(use.rows is None and use.grads is None for use in session.uses["w"])
    with pytest.raises(RuntimeError, match="duplicate"):
        session.submit_linear("w", slots[0], rows[0], grads[0], _reduce)
    session.validate_complete()


def test_row_reduction_releases_saved_rows_and_rejects_duplicates():
    session = CanonicalBackwardSession()
    slot = session.register("b", _keys(0, 3))
    rows = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    assert torch.equal(session.submit_rows("b", slot, rows, lambda r: r.sum(0)), rows.sum(0))
    assert session.uses["b"][0].rows is None
    with pytest.raises(RuntimeError, match="duplicate"):
        session.submit_rows("b", slot, rows, lambda r: r.sum(0))
