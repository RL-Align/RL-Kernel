# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shape-independent FP32 reductions shared by the H3 goldens and backwards.

``torch.sum`` picks its reduction tree from the tensor shape, so a row's
result can change with the number of rows next to it. These helpers spell
the order out with elementwise ops only, which keeps every row's bytes a
function of that row alone.
"""

from __future__ import annotations

import torch


def tree_sum_lastdim_fp32(x: torch.Tensor) -> torch.Tensor:
    """Pairwise tree over the last dim: pad to a power of two, halve until 1.

    Level ``l`` adds element ``i`` and ``i + width/2`` for every ``i``. The
    tree depends only on the last-dim size.
    """

    x = x.float()
    width = x.shape[-1]
    if width == 0:
        return x.new_zeros(x.shape[:-1])
    padded = 1 << (width - 1).bit_length()
    if padded != width:
        x = torch.nn.functional.pad(x, (0, padded - width))
    while x.shape[-1] > 1:
        half = x.shape[-1] // 2
        x = x[..., :half] + x[..., half:]
    return x[..., 0]


def fold_rows_fp32(rows: torch.Tensor) -> torch.Tensor:
    """Left fold over dim 0 in ascending row order: ((r0 + r1) + r2) + ..."""

    rows = rows.float()
    if rows.shape[0] == 0:
        return rows.new_zeros(rows.shape[1:])
    acc = rows[0].clone()
    for index in range(1, rows.shape[0]):
        acc = acc + rows[index]
    return acc
