# SPDX-License-Identifier: Apache-2.0
"""Input fixtures for prefix invariance; no reduction or expected-result logic."""

import torch


def prefix_values(length: int, dtype: torch.dtype, operation: str) -> torch.Tensor:
    """Make finite values with rounding-sensitive sums or nontrivial products."""
    generator = torch.Generator().manual_seed(939)
    values = torch.randn(length, generator=generator)
    if operation == "sum":
        values *= torch.pow(2.0, torch.randint(-4, 8, (length,), generator=generator))
        if length >= 4:
            values[:4] = torch.tensor([8192.0, 2**-12, -8192.0, 2**-12])
    else:
        values = 1.0 + values * 0.02
    return values.to(dtype)


def prefix_storage(values: torch.Tensor, batch_size: int, width: int, target_row: int):
    """Place the same prefix among other rows, with NaN-poisoned unused storage.

    Each row has the same effective length; changing batch placement and row
    stride must not change the target row's tree. Poison is deliberately not
    the operation's identity: callers must use the logical length to mask it.
    """
    length = values.numel()
    storage = torch.full((batch_size, width), float("nan"), dtype=values.dtype)
    for row in range(batch_size):
        storage[row, :length] = values.roll(row)
    storage[target_row, :length] = values
    return storage


def prefix_layouts(length: int, block_size: int):
    """Vary batch size, target position and physical width without changing data."""
    return (
        (1, max(length, 1), 0),
        (3, max(length, block_size) + 37, 2),
        (4, max(length, 2 * block_size) + 129, 1),
    )
