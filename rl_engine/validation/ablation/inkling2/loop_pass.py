# SPDX-License-Identifier: Apache-2.0
"""INK-44: isolated loop-schedule negative controls, never production switches."""

from collections.abc import Callable

from rl_engine.models.inkling2.loop_schedule import inkling2_mini_visits


def validate_pass_order(order: tuple[int, ...]) -> None:
    if order != (0, 1, 2):
        raise ValueError("released checkpoint requires passes 0, 1, 2 in order")


def run_pass_order_probe(initial, run_layer: Callable, capture: Callable, order=(0, 1, 2)):
    """Execute the fixed scheduler or an explicit reordered negative control.

    The callback receives the logical visit's original identity. This probe
    records by execution slot so a reordered pass cannot masquerade as A0.
    """
    if sorted(order) != [0, 1, 2]:
        raise ValueError("pass count/ownership is incomparable")
    visits = inkling2_mini_visits()
    trace = {}
    state = initial
    for visit in visits[:6]:
        state = run_layer(visit, state, None)
        trace[len(trace)] = state
    prelude = state
    message = None
    for index, pass_id in enumerate(order):
        state = prelude
        for visit in visits[6 + 12 * pass_id : 18 + 12 * pass_id]:
            state = run_layer(visit, state, message)
            trace[len(trace)] = state
        if index + 1 < len(order):
            message = capture(state, pass_id)
    for visit in visits[-6:]:
        state = run_layer(visit, state, None)
        trace[len(trace)] = state
    return trace
