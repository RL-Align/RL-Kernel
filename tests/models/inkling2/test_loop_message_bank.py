# SPDX-License-Identifier: Apache-2.0
"""INK-19 CPU message source, pass order and gradient tests."""

import pytest
import torch

from rl_engine.models.inkling2.message_bank import capture_previous_pass_bank


def _state():
    prelude = torch.ones(3, 8, requires_grad=True)
    first_core = torch.full((3, 8), 2.0, requires_grad=True)
    second_core = torch.full((3, 8), 3.0, requires_grad=True)
    partial = torch.full((3, 8), 4.0, requires_grad=True)
    return prelude, first_core, second_core, partial


@pytest.mark.parametrize("source_pass", [0, 1])
def test_capture_uses_only_last_core_block_and_partial(source_pass):
    prelude, first, last, partial = _state()
    completed = [prelude, first, last]
    bank = capture_previous_pass_bank(
        completed, partial, prelude_blocks=[prelude], source_pass=source_pass
    )
    completed.clear()  # A later pass resets the list, not its tensor sources.
    selected = bank.for_target_pass(source_pass + 1)
    assert selected[0] is last
    assert selected[1] is partial
    assert all(t is not first for t in selected)
    (selected[0].sum() + selected[1].sum()).backward()
    assert torch.equal(last.grad, torch.ones_like(last))
    assert torch.equal(partial.grad, torch.ones_like(partial))
    assert first.grad is None


def test_rejects_nonadjacent_pass_and_unexpected_block_layout():
    prelude, first, last, partial = _state()
    bank = capture_previous_pass_bank(
        [prelude, first, last], partial, prelude_blocks=[prelude], source_pass=0
    )
    with pytest.raises(ValueError, match="immediately preceding"):
        bank.for_target_pass(2)
    with pytest.raises(ValueError, match="only captures"):
        capture_previous_pass_bank(
            [prelude, first, last], partial, prelude_blocks=[prelude], source_pass=2
        )
    with pytest.raises(ValueError, match="three completed"):
        capture_previous_pass_bank(
            [prelude, first], partial, prelude_blocks=[prelude], source_pass=0
        )
    with pytest.raises(ValueError, match="prelude snapshot"):
        capture_previous_pass_bank(
            [prelude.clone(), first, last], partial, prelude_blocks=[prelude], source_pass=0
        )
    with pytest.raises(ValueError, match="same token"):
        capture_previous_pass_bank(
            [prelude, first, last], partial[:2], prelude_blocks=[prelude], source_pass=0
        )
