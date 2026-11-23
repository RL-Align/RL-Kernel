# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixed-loop schedule acceptance for INK-18."""

from collections import Counter

from rl_engine.models.inkling2.loop_schedule import (
    inkling2_mini_visits,
    run_inkling2_mini_schedule,
)


def test_fixed_checkpoint_has_24_physical_layers_and_48_visits():
    visits = inkling2_mini_visits()
    assert len(visits) == 48
    assert [v.logical_index for v in visits] == list(range(48))
    assert [v.physical_layer for v in visits[:6]] == list(range(6))
    assert [v.physical_layer for v in visits[-6:]] == list(range(18, 24))
    assert Counter(v.physical_layer for v in visits) == {
        **{i: 1 for i in range(6)},
        **{i: 3 for i in range(6, 18)},
        **{i: 1 for i in range(18, 24)},
    }
    for p in range(3):
        core = visits[6 + p * 12 : 18 + p * 12]
        assert [v.physical_layer for v in core] == list(range(6, 18))
        assert {v.pass_index for v in core} == {p}
        assert {v.message_source_pass for v in core} == {p - 1 if p else None}
        assert [v.physical_layer for v in core if v.restart_from_prelude] == (
            [6] if p else []
        )


def test_each_core_pass_restarts_from_prelude_and_reads_only_previous_pass():
    prelude_end = tuple(range(6))
    starting_core, message_origins, captured = [], [], []

    def run_layer(visit, state, message):
        if visit.phase == "core" and visit.physical_layer == 6:
            starting_core.append(state)
            message_origins.append(None if message is None else message[0])
        if visit.phase != "core":
            assert message is None
        return (*state, visit.physical_layer)

    def capture_messages(state, pass_index):
        captured.append(pass_index)
        return (pass_index, state)

    final = run_inkling2_mini_schedule((), run_layer, capture_messages)
    assert starting_core == [prelude_end] * 3
    assert message_origins == [None, 0, 1]
    assert captured == [0, 1]
    assert final == (*prelude_end, *range(6, 18), *range(18, 24))
