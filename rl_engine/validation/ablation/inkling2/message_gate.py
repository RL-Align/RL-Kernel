# SPDX-License-Identifier: Apache-2.0
"""INK-45: isolated rank-64 message-gate fault injection."""

from rl_engine.models.inkling2.message_gates import (
    STREAMS,
    blend_cross_pass_message,
    blend_loop_streams,
)


def message_gate_probe(destinations, messages, controllers, *, stream, fault):
    if stream not in STREAMS or fault not in ("bypass", "misroute"):
        raise ValueError("unknown message gate probe")
    if fault == "bypass":
        return {
            name: destinations[name].float()
            if name == stream
            else blend_cross_pass_message(destinations[name], messages[name], controllers[name])
            for name in STREAMS
        }
    changed = dict(messages)
    changed[stream] = messages[STREAMS[(STREAMS.index(stream) + 1) % len(STREAMS)]]
    return blend_loop_streams(destinations, changed, controllers)
