# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""BF16 DeepSpeed optimizers put the parameters every TP rank holds first in the flat buffer."""

from __future__ import annotations

import torch

from rl_engine.integrations.engines.train.deepspeed.ppo import _layouts_agree, _shared_first


def _named(*names: str) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, torch.nn.Parameter(torch.zeros(1))) for name in names]


def test_leader_only_parameter_moves_after_the_shared_ones():
    # nn.Module lists its own parameters before its children's: the leader-only bias first.
    leader = _named("bias", "backbone.w", "head.weight")
    other = _named("backbone.w", "head.weight")
    names = [[n for n, _ in leader], [n for n, _ in other]]
    order = _shared_first(leader, names)
    assert order == [leader[1][1], leader[2][1], leader[0][1]]
    assert _shared_first(other, names) == [p for _, p in other]


def test_layouts_agree_on_the_common_prefix_only():
    shared = [("w", 0, 8), ("head.weight", 8, 4)]
    assert _layouts_agree([[shared + [("bias", 12, 1)]], [shared]])
    # A one-element parameter in front shifts every shared offset on the leader.
    assert not _layouts_agree([[[("bias", 0, 1), ("w", 1, 8), ("head.weight", 9, 4)]], [shared]])
    assert not _layouts_agree([[shared], [shared, shared]])


def test_equal_sized_shared_parameters_in_swapped_places_do_not_agree():
    first = [("a", 0, 4), ("b", 4, 4)]
    swapped = [("b", 0, 4), ("a", 4, 4)]
    assert not _layouts_agree([[first], [swapped]])
