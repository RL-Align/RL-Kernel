# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Batch-invariance checks for row-wise operators.

A row-wise operator takes inputs whose leading dimension is the token count M
and returns outputs with the same leading dimension, where output row i is
computed from input row i (plus shared weights). It is batch invariant when the
bits of row i do not depend on which other rows are in the batch.

An operator is described by a ``RowwiseCase``: a function from a list of row
inputs to a list of named outputs, and a generator of random row inputs. Every
check returns the names of the outputs whose bits differ, in output order, so
the first name is the first output that diverged. Besides the batch, the
checks vary chunking, row order, padding and the inputs' strides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

_BITS = {
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.int64: torch.int64,
    torch.int32: torch.int32,
}


@dataclass
class RowwiseCase:
    name: str
    output_names: Sequence[str]
    # run(rows) -> outputs, each with the same leading dimension as the rows.
    run: Callable[[list[torch.Tensor]], list[torch.Tensor]]
    # make_rows(m, seed) -> inputs with leading dimension m.
    make_rows: Callable[[int, int], list[torch.Tensor]]


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare raw bits, so -0.0 != +0.0 and NaN payloads count."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    bits = _BITS[a.dtype]
    return torch.equal(a.contiguous().view(bits), b.contiguous().view(bits))


def _differing(names, outputs, expected, rows=slice(None)) -> list[str]:
    return [n for n, out, ref in zip(names, outputs, expected) if not bits_equal(out[rows], ref)]


def differing_in_batch(case: RowwiseCase, m: int, *, seed: int = 0) -> list[str]:
    """Rows computed alone (M=1) versus at the first and last row of a batch of M."""
    checked = case.make_rows(2, seed)
    alone = [case.run([t[i : i + 1] for t in checked]) for i in range(2)]
    batch = case.make_rows(m, seed + 1 + m)
    positions = [0] if m == 1 else [0, m - 1]
    for i, pos in enumerate(positions):
        for t, c in zip(batch, checked):
            t[pos] = c[i]
    outputs = case.run(batch)
    differing = []
    for i, pos in enumerate(positions):
        for name in _differing(case.output_names, outputs, alone[i], slice(pos, pos + 1)):
            if name not in differing:
                differing.append(name)
    return differing


def differing_after_chunk_reorder_pad(case: RowwiseCase, m: int, *, seed: int = 0) -> list[str]:
    """Full batch versus two chunks, a row permutation, and zero padding appended."""
    names = case.output_names
    rows = case.make_rows(m, seed)
    full = case.run(rows)

    split = m // 3
    first = case.run([t[:split] for t in rows])
    second = case.run([t[split:] for t in rows])
    chunked = [torch.cat([a, b]) for a, b in zip(first, second)]

    perm = torch.randperm(m, generator=torch.Generator().manual_seed(seed)).to(rows[0].device)
    reordered = case.run([t[perm] for t in rows])

    pad = max(1, m // 10)
    padded = case.run([torch.cat([t, torch.zeros_like(t[:pad])]) for t in rows])

    return (
        [f"chunked {n}" for n in _differing(names, chunked, full)]
        + [f"reordered {n}" for n in _differing(names, reordered, [f[perm] for f in full])]
        + [f"padded {n}" for n in _differing(names, [p[:m] for p in padded], full)]
    )


def _restrided(t: torch.Tensor) -> torch.Tensor:
    """The same values in a non-contiguous view: every other row of a larger buffer,
    and, for 2-D and wider inputs, the first half of a doubled last dimension."""
    if t.dim() == 1:
        buffer = torch.zeros(2 * t.shape[0], dtype=t.dtype, device=t.device)
        view = buffer[::2]
    else:
        shape = (2 * t.shape[0], *t.shape[1:-1], 2 * t.shape[-1])
        buffer = torch.zeros(shape, dtype=t.dtype, device=t.device)
        view = buffer[::2, ..., : t.shape[-1]]
    view.copy_(t)
    return view


def differing_after_restride(case: RowwiseCase, m: int, *, seed: int = 0) -> list[str]:
    """Contiguous inputs versus the same values in non-contiguous views (other strides)."""
    rows = case.make_rows(m, seed)
    full = case.run(rows)
    restrided = case.run([_restrided(t) for t in rows])
    return [f"restrided {n}" for n in _differing(case.output_names, restrided, full)]


def differing_between_runs(case: RowwiseCase, m: int, *, seed: int = 0) -> list[str]:
    """The same batch run twice (run-to-run determinism)."""
    rows = case.make_rows(m, seed)
    return _differing(case.output_names, case.run(rows), case.run(rows))


def differing_between(
    case_a: RowwiseCase, case_b: RowwiseCase, m: int, *, seed: int = 0
) -> list[str]:
    """Two runs of the same rows that must agree, e.g. forward with and without grad.

    Compares the outputs the two cases share by name.
    """
    rows = case_a.make_rows(m, seed)
    out_a = dict(zip(case_a.output_names, case_a.run(rows)))
    out_b = dict(zip(case_b.output_names, case_b.run(rows)))
    return [n for n in case_a.output_names if n in out_b and not bits_equal(out_a[n], out_b[n])]
