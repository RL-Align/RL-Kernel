# SPDX-License-Identifier: Apache-2.0
"""Recorded values-only P4 fixtures. No sockets, process groups or reductions."""

from .contract import require
from .oracle import gradient_dispatch, mock_return


def ep_return(plan, rows, context, *, ep=1, placement=0, reverse=True):
    plan.validate(context)
    require(type(ep) is int and ep in (1, 2, 4, 8), "UNSUPPORTED_CAPABILITY", "mock EP")
    require(
        type(placement) is int and 0 <= placement < ep,
        "INVALID_DISCRETE_PLAN",
        "placement rotation",
    )
    peers = [[] for _ in range(ep)]
    for i, (_, slot, valid) in enumerate(plan.inverse_map):
        peers[(slot + placement) % ep if valid else 0].append(i)
    peer_order = list(range(ep))
    if reverse:
        peer_order.reverse()
    arrival = [i for peer in peer_order for i in reversed(peers[peer])]
    return {
        "rows": mock_return(plan, rows, arrival, context),
        "counts": [len(p) for p in peers],
        "arrival": arrival,
        "ep": ep,
        "placement": placement,
        "kind": "mock-values-only",
        "transport_executed": False,
        "plan_fingerprint": plan.fingerprint,
    }


__all__ = ["ep_return", "gradient_dispatch", "mock_return"]
