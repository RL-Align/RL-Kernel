# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Layerwise exact-zero gate for the DSv4.1 model (single device).

DSv4's gate (``deepseek_v4.layerwise_gate``: a continuous-batching rollout against the
actor's packed prefill in another request order and the training forward, compared byte
for byte at every layer, the final hidden state and the selected-token log-probs) on
:class:`Dsv41Model`: the step batch is a :class:`Dsv41Batch`, and each step passes the
Engram n-gram windows of its tokens.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from rl_engine.models.deepseek_v4.layerwise_gate import Trace, _canonical, compare, rollout_schedule
from rl_engine.models.deepseek_v41.attention import Dsv41Batch
from rl_engine.models.deepseek_v41.engram import ngram_windows


class _Engine:
    def __init__(self, model, ids: list[Tensor], seed: int):
        self.model, self.ids = model, ids
        self.caches = model.new_caches(len(ids), model.max_model_len, seed=seed)

    def step(self, parts, *, train: bool = False):
        m = self.model
        dev = m.embed.device
        rows = torch.tensor([r for r, _, _ in parts], device=dev)
        views = [
            type(c)(
                **{
                    k: (v[rows] if k.endswith("table") and v is not None else v)
                    for k, v in c.__dict__.items()
                }
            )
            for c in self.caches
        ]
        ids = torch.cat([self.ids[r][lo:hi] for r, lo, hi in parts])
        seq = [hi for _, _, hi in parts]
        qlen = [hi - lo for _, lo, hi in parts]
        batch = Dsv41Batch.from_lens(seq, qlen, m.config.sliding_window, device=dev)
        windows = None
        if m.hasher is not None:
            pos = [torch.arange(lo, hi, device=dev) for _, lo, hi in parts]
            windows = ngram_windows(
                [self.ids[r] for r, _, _ in parts], pos, m.config.engram_max_ngram_size
            )
        fwd = m.forward_train if train else m.forward_batch
        return fwd(ids, batch, views, engram_windows=windows, return_streams=True)


def _collect(model, ids, steps, lengths, seed, *, train=False) -> Trace:
    eng = _Engine(model, ids, seed)
    outs: list[Tensor] = []
    layers: list[list[Tensor]] = [[] for _ in model.layers]
    for parts in steps:
        if train:
            with torch.enable_grad():
                hidden, per_layer = eng.step(parts, train=True)
            hidden, per_layer = hidden.detach(), [s.detach() for s in per_layer]
        else:
            hidden, per_layer = eng.step(parts)
        outs.append(hidden)
        for acc, s in zip(layers, per_layer):
            acc.append(s)
    where = _canonical(lengths, steps).to(model.embed.device)
    hidden = torch.cat(outs)[where]
    trace = Trace(streams=[torch.cat(acc)[where] for acc in layers], hidden=hidden)
    if model.head is not None:
        rows, targets, base = [], [], 0
        for r, n in enumerate(lengths):
            rows.append(torch.arange(base, base + n - 1))
            targets.append(ids[r][1:n])
            base += n
        keep = torch.cat(rows).to(model.embed.device)
        trace.logp = model.selected_logprob(hidden[keep], torch.cat(targets))[0]
    return trace


def run_rollout(model, ids, prompt_lens, token_budget=256, seed=1) -> Trace:
    lengths = [len(x) for x in ids]
    gen = [n - p for n, p in zip(lengths, prompt_lens)]
    return _collect(model, ids, rollout_schedule(prompt_lens, gen, token_budget), lengths, seed)


def run_packed(model, ids, order=None, seed=2, *, train=False) -> Trace:
    order = list(reversed(range(len(ids)))) if order is None else order
    lengths = [len(x) for x in ids]
    perm_ids = [ids[r] for r in order]
    steps = [[(i, 0, len(perm_ids[i])) for i in range(len(order))]]
    t = _collect(model, perm_ids, steps, [len(x) for x in perm_ids], seed, train=train)
    starts, b = [], 0
    for x in perm_ids:
        starts.append(b)
        b += len(x)
    pos = {r: i for i, r in enumerate(order)}
    idx = torch.cat(
        [torch.arange(starts[pos[r]], starts[pos[r]] + lengths[r]) for r in range(len(ids))]
    ).to(model.embed.device)
    assert t.hidden is not None
    out = Trace(streams=[s[idx] for s in t.streams], hidden=t.hidden[idx])
    if t.logp is not None:
        lstarts, b = [], 0
        for x in perm_ids:
            lstarts.append(b)
            b += len(x) - 1
        lidx = torch.cat(
            [
                torch.arange(lstarts[pos[r]], lstarts[pos[r]] + lengths[r] - 1)
                for r in range(len(ids))
            ]
        ).to(model.embed.device)
        out.logp = t.logp[lidx]
    return out


def layerwise_gate(
    model, ids: list[Tensor], prompt_lens: list[int], token_budget: int = 256, train=True
) -> dict:
    """Rollout against actor recompute (and the training forward): the gate report."""
    rollout = run_rollout(model, ids, prompt_lens, token_budget)
    report: dict[str, Any] = {"tokens": sum(len(x) for x in ids), "requests": len(ids)}
    report["actor"] = compare(rollout, run_packed(model, ids))
    if train:
        report["train"] = compare(rollout, run_packed(model, ids, seed=3, train=True))
    report["passed"] = all(v["passed"] for k, v in report.items() if isinstance(v, dict))
    return report
