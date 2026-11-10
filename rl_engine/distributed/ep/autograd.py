# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Autograd nodes for the expert exchange.

Autograd only runs nodes that reach the loss, so a rank whose node does not run leaves
its peers blocked in the collective. Two links keep every rank's nodes in the graph:

* ``anchor``: a scalar that requires grad, so the dispatch node exists even for a frozen
  input;
* ``link``: a scalar from the dispatch node into the return node, so the dispatch
  backward always runs after the return backward, however the expert treats its rows.

An output left out of the loss cannot be handled from inside a node that never runs;
``ep_backward`` rejects that case on all ranks before any payload moves.
"""

from __future__ import annotations

import torch

from rl_engine.distributed.ep.backward import EpGradExchange
from rl_engine.distributed.ep.exchange import EpRoute


class EpDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, anchor, exchange: EpGradExchange, route: EpRoute, holder: dict):
        ctx.exchange, ctx.holder, ctx.dtype = exchange, holder, x.dtype
        rows = exchange.dispatch_tokens(route, x.detach())
        return rows, torch.zeros_like(anchor)

    @staticmethod
    def backward(ctx, grad_rows, _grad_link):
        dx = ctx.exchange.return_dx(ctx.holder["saved"], grad_rows.detach(), ctx.dtype)
        return dx, None, None, None, None


class EpReturn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_rows, link, exchange: EpGradExchange, route: EpRoute, holder: dict):
        ctx.exchange, ctx.holder = exchange, holder
        rows = exchange.return_outputs(route, y_rows.detach()).rows
        holder["saved"] = exchange.save_route(route)
        ctx.grad_link = torch.zeros_like(link)
        return rows

    @staticmethod
    def backward(ctx, grad_rows):
        dy_rows = ctx.exchange.dispatch_dy(ctx.holder["saved"], grad_rows.detach())
        return dy_rows, ctx.grad_link, None, None, None


__all__ = ["EpDispatch", "EpReturn"]
