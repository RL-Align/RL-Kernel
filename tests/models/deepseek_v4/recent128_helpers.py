# SPDX-License-Identifier: Apache-2.0
"""Test-only scheduler fixtures; production uses framework-owned metadata/pages."""

from types import SimpleNamespace

import torch


def metadata(lengths, queries, *, block_size=64, tables=None, padding=0, device="cpu"):
    if tables is None:
        tables = []
        cursor = 0
        for end in lengths:
            count = (end + block_size - 1) // block_size
            tables.append(list(range(cursor, cursor + count)))
            cursor += count
    width = max(map(len, tables), default=0)
    tables = [list(row) + [-1] * (width - len(row)) for row in tables]
    starts, slots = [0], []
    for end, count, row in zip(lengths, queries, tables, strict=False):
        starts.append(starts[-1] + count)
        for p in range(end - count, end):
            slots.append(row[p // block_size] * block_size + p % block_size)
    slots += [-1] * padding
    return SimpleNamespace(
        block_size=block_size,
        block_table=torch.tensor(tables, dtype=torch.int32, device=device).reshape(
            len(tables),
            width,
        ),
        slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device=device),
    )


def assert_bytes_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def canonical_cache_bytes(cache, table, length, block_size):
    rows = []
    for p in range(length):
        page, offset = table[p // block_size], p % block_size
        rows.append(
            torch.cat(
                (
                    cache[page, offset * 576 : (offset + 1) * 576],
                    cache[
                        page, block_size * 576 + offset * 8 : block_size * 576 + (offset + 1) * 8
                    ],
                )
            )
        )
    return (
        torch.stack(rows) if rows else torch.empty((0, 584), dtype=torch.uint8, device=cache.device)
    )
