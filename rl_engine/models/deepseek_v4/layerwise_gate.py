# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Layerwise exact-zero hidden-state gate for the DSv4 model (single device).

The rollout and the actor must give every token the same bytes at every layer. This
harness runs the same token sequences three ways on one :class:`Dsv4Model`:

- **rollout**: a continuous-batching schedule like vLLM's. Requests arrive staggered,
  prompts are prefilled in chunks under a token budget, and generated tokens are decoded
  one per step. A step can mix decode tokens of running requests with a prefill chunk of
  a new one. Generated tokens are fed teacher-forced (sampling plays no part in the gate).
- **actor**: one packed full prefill of every sequence, in another request order
  (``forward_batch``, the actor recompute).
- **train**: the same packed prefill through ``forward_train`` (the training forward
  under autograd).

It records every layer's output streams per token, then compares layer 0 upward and
reports, per layer, how many tokens differ and the largest difference. The gate passes
only if every layer, the final hidden state and the selected-token log-probs are
byte-identical. The first layer that differs is where to look.

``scripts/check_dsv4_layerwise_gate.py`` runs it on a random checkpoint at the real
dimensions or on the first layers of a real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from rl_engine.models.deepseek_v4.attention import Dsv4Batch


@dataclass
class Trace:
    """Per-token results of one run, each in the canonical order (requests ascending,
    positions ascending)."""

    streams: list[Tensor] = field(default_factory=list)  # per layer BF16 [N, 4, H]
    hidden: Tensor | None = None  # BF16 [N, H]
    logp: Tensor | None = None  # FP32 [N - R] (next-token log-probs, when the head exists)


def _bits(x: Tensor) -> Tensor:
    return x.contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32)


def _canonical(lengths: list[int], parts_per_step: list[list[tuple[int, int, int]]]):
    """Index of each canonical (request, position) token in the concatenated step outputs."""
    base, k = [0], 0
    for n in lengths:
        base.append(base[-1] + n)
    where = torch.empty(base[-1], dtype=torch.long)
    for parts in parts_per_step:
        for r, lo, hi in parts:
            where[base[r] + lo : base[r] + hi] = torch.arange(k, k + hi - lo)
            k += hi - lo
    return where


class _Engine:
    """Steps over the model with caches for ``requests`` (``Runner`` in the model tests)."""

    def __init__(self, model, ids: list[Tensor], seed: int):
        self.model, self.ids = model, ids
        self.caches = model.new_caches(len(ids), model.max_model_len, seed=seed)

    def step(self, parts, *, train: bool = False):
        dev = self.model.embed.device
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
        batch = Dsv4Batch.from_lens(seq, qlen, self.model.config.sliding_window, dev)
        if train:
            return self.model.forward_train(ids, batch, views, return_streams=True)
        return self.model.forward_batch(ids, batch, views, return_streams=True)


def rollout_schedule(
    prompt_lens: list[int], gen_lens: list[int], token_budget: int, arrive_every: int = 2
) -> list[list[tuple[int, int, int]]]:
    """Steps of ``(request, lo, hi)``: request ``r`` arrives at step ``r * arrive_every``.
    Each step first decodes one token for every running request, then fills the rest of
    the budget with prefill chunks of arrived requests (first come, first served)."""
    lengths = [p + g for p, g in zip(prompt_lens, gen_lens)]
    done = [0] * len(lengths)
    steps, t = [], 0
    while any(d < n for d, n in zip(done, lengths)):
        parts, left = [], token_budget
        for r in range(len(lengths)):
            if done[r] >= prompt_lens[r] and done[r] < lengths[r]:
                parts.append((r, done[r], done[r] + 1))
                left -= 1
        for r in range(len(lengths)):
            if left <= 0:
                break
            if r * arrive_every <= t and done[r] < prompt_lens[r]:
                hi = min(prompt_lens[r], done[r] + left)
                parts.append((r, done[r], hi))
                left -= hi - done[r]
        for r, _, hi in parts:
            done[r] = hi
        if parts:
            # The model wants each request once per step and tokens grouped by request.
            parts.sort(key=lambda p: p[0])
            steps.append(parts)
        t += 1
    return steps


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
    trace = Trace(streams=[torch.cat(acc)[where] for acc in layers], hidden=torch.cat(outs)[where])
    if model.head is not None:
        rows, targets, base = [], [], 0
        for r, n in enumerate(lengths):
            rows.append(torch.arange(base, base + n - 1))
            targets.append(ids[r][1:n])
            base += n
        keep = torch.cat(rows).to(model.embed.device)
        hidden = torch.cat(outs)[where]
        trace.logp = model.selected_logprob(hidden[keep], torch.cat(targets))[0]
    return trace


def run_rollout(model, ids, prompt_lens, token_budget=256, seed=1) -> Trace:
    lengths = [len(x) for x in ids]
    gen = [n - p for n, p in zip(lengths, prompt_lens)]
    return _collect(model, ids, rollout_schedule(prompt_lens, gen, token_budget), lengths, seed)


def run_packed(model, ids, order=None, seed=2, *, train=False) -> Trace:
    order = list(reversed(range(len(ids)))) if order is None else order
    lengths = [len(x) for x in ids]
    # The model wants tokens grouped by request in cache-row order; permute the rows.
    perm_ids = [ids[r] for r in order]
    steps = [[(i, 0, len(perm_ids[i])) for i in range(len(order))]]
    t = _collect(model, perm_ids, steps, [len(x) for x in perm_ids], seed, train=train)
    # Back to the canonical request order.
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


def compare(ref: Trace, other: Trace) -> dict:
    """Per layer: tokens whose bytes differ and the largest absolute difference."""
    rows = []
    for i, (a, b) in enumerate(zip(ref.streams, other.streams)):
        diff = (_bits(a) != _bits(b)).flatten(1).any(1)
        mad = float((a.float() - b.float()).abs().max()) if bool(diff.any()) else 0.0
        rows.append({"layer": i, "tokens_differ": int(diff.sum()), "max_abs_diff": mad})
    first = next((r["layer"] for r in rows if r["tokens_differ"]), None)
    hidden_same = torch.equal(_bits(ref.hidden), _bits(other.hidden))
    logp_same = None
    if ref.logp is not None and other.logp is not None:
        logp_same = torch.equal(_bits(ref.logp), _bits(other.logp))
    return {
        "layers": rows,
        "first_layer_differs": first,
        "hidden_same": hidden_same,
        "logp_same": logp_same,
        "passed": first is None and hidden_same and logp_same is not False,
    }


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
