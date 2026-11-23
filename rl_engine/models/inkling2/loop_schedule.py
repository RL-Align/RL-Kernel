# SPDX-License-Identifier: Apache-2.0
"""Fixed three-pass Inkling 2 Mini visit schedule (INK-18).

This is a CPU-safe model reference, not a TorchTitan or SGLang implementation.
The callback owns tensor/layout state and must return a new functional state;
mutating the saved prelude state would invalidate the restart contract.
"""

from dataclasses import dataclass
from typing import Callable, Literal, TypeVar

StateT = TypeVar("StateT")
MessageT = TypeVar("MessageT")
Phase = Literal["prelude", "core", "coda"]


@dataclass(frozen=True)
class LoopVisit:
    logical_index: int
    physical_layer: int
    phase: Phase
    pass_index: int | None = None
    message_source_pass: int | None = None
    restart_from_prelude: bool = False


def inkling2_mini_visits() -> tuple[LoopVisit, ...]:
    """All 48 logical visits for the fixed released Inkling 2 Mini checkpoint.

    The 12 core layers reuse their physical parameters across three passes.
    Only passes 1 and 2 (zero-indexed) read preceding-pass messages.
    """
    visits: list[LoopVisit] = []
    for layer in range(6):
        visits.append(LoopVisit(len(visits), layer, "prelude"))
    for pass_index in range(3):
        for layer in range(6, 18):
            visits.append(
                LoopVisit(
                    len(visits), layer, "core", pass_index,
                    pass_index - 1 if pass_index else None,
                    pass_index > 0 and layer == 6,
                )
            )
    for layer in range(18, 24):
        visits.append(LoopVisit(len(visits), layer, "coda"))
    return tuple(visits)


def run_inkling2_mini_schedule(
    initial_state: StateT,
    run_layer: Callable[[LoopVisit, StateT, MessageT | None], StateT],
    capture_messages: Callable[[StateT, int], MessageT],
) -> StateT:
    """Run the fixed schedule while preserving the prelude restart contract.

    The capture callback receives the previous core pass state and zero-based
    source pass ID. INK-19 defines the two-block message capture contract.
    No message is supplied to the prelude, pass 0 or coda.
    """
    visits = inkling2_mini_visits()
    state = initial_state
    for visit in visits[:6]:
        state = run_layer(visit, state, None)
    prelude_state = state
    last_core_state: StateT | None = None
    for pass_index in range(3):
        message = None
        if pass_index:
            # Capture the old pass before restoring the prelude.
            message = capture_messages(last_core_state, pass_index - 1)
        state = prelude_state
        for visit in visits[6 + 12 * pass_index : 18 + 12 * pass_index]:
            state = run_layer(visit, state, message)
        last_core_state = state
    for visit in visits[-6:]:
        state = run_layer(visit, state, None)
    return state
