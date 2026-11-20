# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from rl_engine.ops.autograd.canonical_backward import CanonicalBackwardSession


@pytest.mark.parametrize("kind", ["linear", "rows"])
def test_native_1f1b_waits_for_all_declared_uses_and_reduces_in_key_order(kind):
    s = CanonicalBackwardSession()
    s.expect_uses(3)
    observed = []

    def reducer(*args):
        observed.append(args)
        return args[0].sum(0)

    output = None
    for i, key in enumerate([30, 10, 20]):
        s.expect_uses(3)
        slot = s.register("parameter", torch.tensor([[key, 0]]))
        row = torch.tensor([[float(key)]])
        if kind == "linear":
            output = s.submit_linear("parameter", slot, row, torch.ones_like(row), reducer)
        else:
            output = s.submit_rows("parameter", slot, row, reducer)
        if i < 2:
            assert output is None and not observed
    assert output.item() == 60
    assert observed[0][0].flatten().tolist() == [10, 20, 30]
    s.validate_complete()


def test_undeclared_late_forward_still_fails():
    s = CanonicalBackwardSession()
    slot = s.register("x", torch.tensor([[0, 0]]))
    s.submit_rows("x", slot, torch.ones(1, 1), lambda x: x.sum(0))
    with pytest.raises(RuntimeError, match="after backward started"):
        s.register("x", torch.tensor([[1, 0]]))


def test_missing_duplicate_extra_and_changed_schedule_fail():
    s = CanonicalBackwardSession()
    s.expect_uses(2)
    slot = s.register("x", torch.tensor([[0, 0]]))
    s.submit_rows("x", slot, torch.ones(1, 1), lambda x: x.sum(0))
    with pytest.raises(RuntimeError, match="incomplete"):
        s.validate_complete()
    with pytest.raises(RuntimeError, match="duplicate"):
        s.submit_rows("x", slot, torch.ones(1, 1), lambda x: x.sum(0))
    with pytest.raises(RuntimeError, match="changed"):
        s.expect_uses(1)
    s.register("x", torch.tensor([[1, 0]]))
    with pytest.raises(RuntimeError, match="exceed"):
        s.register("x", torch.tensor([[2, 0]]))
