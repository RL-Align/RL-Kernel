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
checks vary chunking, row order, padding (zero and random rows), the inputs'
strides, and rows holding NaN, infinities or negative zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

# The integer type with the same width, to compare raw bits: NaN payloads and the
# sign of zero count.
_BITS = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.bfloat16: torch.int16,
    torch.float16: torch.int16,
    torch.int64: torch.int64,
    torch.int32: torch.int32,
    torch.int16: torch.int16,
    torch.int8: torch.int8,
    torch.uint8: torch.uint8,
    torch.bool: torch.uint8,
}

# The distinct rows differing_in_batch computes alone.
DISTINCT_ROWS = 8

# (name, value) of the special values differing_with_special_values places in rows.
SPECIAL_VALUES = (
    ("NaN", float("nan")),
    ("+Inf", float("inf")),
    ("-Inf", float("-inf")),
    ("-0.0", -0.0),
)


@dataclass
class RowwiseCase:
    name: str
    output_names: Sequence[str]
    # run(rows) -> outputs, each with the same leading dimension as the rows.
    run: Callable[[list[torch.Tensor]], list[torch.Tensor]]
    # make_rows(m, seed) -> inputs with leading dimension m.
    make_rows: Callable[[int, int], list[torch.Tensor]]


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype not in _BITS:
        raise TypeError(f"no bit comparison for {t.dtype}; add it to _BITS")
    return t.contiguous().view(_BITS[t.dtype])


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Compare raw bits, so -0.0 != +0.0 and NaN payloads count."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return torch.equal(_bits(a), _bits(b))


def _rows_equal(rows: torch.Tensor, row: torch.Tensor) -> bool:
    """Every row of ``rows`` [R, ...] has the bits of ``row`` [1, ...]."""
    if rows.dtype != row.dtype or rows.shape[1:] != row.shape[1:] or row.shape[0] != 1:
        return False
    return bool((_bits(rows) == _bits(row)).all())


def _differing(names, outputs, expected, rows=slice(None)) -> list[str]:
    return [n for n, out, ref in zip(names, outputs, expected) if not bits_equal(out[rows], ref)]


def differing_in_batch(
    case: RowwiseCase, m: int, *, seed: int = 0, distinct_rows: int = DISTINCT_ROWS
) -> list[str]:
    """Every row of a batch of M versus the same row computed alone (M=1).

    ``min(m, distinct_rows)`` distinct rows are each computed alone once; the batch
    places them at random positions, every one of them at least once, so every row
    of the batch, including the first, middle and last, is compared bit for bit.
    The cost is ``distinct_rows + 1`` runs whatever M is, against M + 1 runs for a
    batch of M rows that are all different. A row that reads another row's data
    still differs: its neighbours are other rows at irregular positions.
    """
    p = min(m, distinct_rows)
    pool = case.make_rows(p, seed)
    alone = [case.run([t[j : j + 1] for t in pool]) for j in range(p)]
    order = torch.randperm(m, generator=torch.Generator().manual_seed(seed + 1 + m))
    assignment = (order % p).to(pool[0].device)
    outputs = case.run([t[assignment] for t in pool])
    differing = []
    for k, (name, out) in enumerate(zip(case.output_names, outputs)):
        if out.shape[0] != m or not all(
            _rows_equal(out[assignment == j], alone[j][k]) for j in range(p)
        ):
            differing.append(name)
    return differing


def differing_after_chunk_reorder_pad(case: RowwiseCase, m: int, *, seed: int = 0) -> list[str]:
    """Full batch versus two chunks, a row permutation, and padding rows appended:
    zero rows, then random rows."""
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
    zero_padded = case.run([torch.cat([t, torch.zeros_like(t[:pad])]) for t in rows])
    padding = case.make_rows(pad, seed + 1)
    random_padded = case.run([torch.cat([t, p]) for t, p in zip(rows, padding)])

    return (
        [f"chunked {n}" for n in _differing(names, chunked, full)]
        + [f"reordered {n}" for n in _differing(names, reordered, [f[perm] for f in full])]
        + [f"zero-padded {n}" for n in _differing(names, [p[:m] for p in zero_padded], full)]
        + [f"random-padded {n}" for n in _differing(names, [p[:m] for p in random_padded], full)]
    )


def differing_with_special_values(
    case: RowwiseCase, m: int, *, seed: int = 0, inputs: Sequence[int] = (0,)
) -> list[str]:
    """Rows holding NaN, +Inf, -Inf or -0.0 inside a batch of M (M >= 4).

    One row per special value, at the first, last and two middle positions. NaN,
    +Inf and -Inf replace the first element of the row; -0.0 fills the whole row.
    The values go into the row inputs at the indices ``inputs`` (floating-point).
    Reports ``other rows <output>`` when the special rows change the bits of the
    other rows, and ``<value> row <output>`` when a special row in the batch does
    not have the bits it has alone.
    """
    count = len(SPECIAL_VALUES)
    if m < count:
        raise ValueError(f"M must be at least {count}, got {m}")
    rows = case.make_rows(m, seed)
    clean = case.run(rows)
    positions = [round(i * (m - 1) / (count - 1)) for i in range(count)]
    special = [t.clone() for t in rows]
    for i in inputs:
        if not special[i].is_floating_point():
            raise TypeError(f"row input {i} is {special[i].dtype}, not floating point")
        for pos, (_, value) in zip(positions, SPECIAL_VALUES):
            row = special[i][pos]
            if value == 0.0:
                row.fill_(value)
            else:
                row.view(-1)[0] = value
    outputs = case.run(special)

    others = torch.ones(m, dtype=torch.bool, device=rows[0].device)
    others[positions] = False
    differing = [
        f"other rows {n}"
        for n, out, ref in zip(case.output_names, outputs, clean)
        if not bits_equal(out[others], ref[others])
    ]
    for pos, (label, _) in zip(positions, SPECIAL_VALUES):
        alone = case.run([t[pos : pos + 1] for t in special])
        differing += [
            f"{label} row {n}"
            for n in _differing(case.output_names, outputs, alone, slice(pos, pos + 1))
        ]
    return differing


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
