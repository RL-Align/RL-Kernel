# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Declared row-local FP32 VJPs. No batched torch.matmul / cuBLAS.

Each output row is an independent GEMV or outer product. Parameter reductions
walk rows in the caller's order so C10 can re-aggregate by logical token.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import prod

import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE

BACKWARD_IMPL = "row_local_fp32_vjp"


def row_local_linear_dx_fp32(grad_output: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """dX[t] = grad[t] @ weight, one GEMV per row."""

    rows = grad_output.reshape(-1, grad_output.size(-1)).float()
    weight_f = weight.float()
    out_rows = torch.empty(
        (rows.shape[0], weight_f.shape[1]), device=rows.device, dtype=torch.float32
    )
    weight_t = weight_f.t().contiguous()
    for index in range(rows.shape[0]):
        out_rows[index] = torch.mv(weight_t, rows[index])
    return out_rows.reshape(*grad_output.shape[:-1], weight_f.shape[1])


def row_local_linear_dw_fp32(grad_output: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """dW = sum_t outer(grad[t], hidden[t]) in physical row order."""

    grad_rows = grad_output.reshape(-1, grad_output.size(-1)).float()
    hidden_rows = hidden.reshape(-1, hidden.size(-1)).float()
    if grad_rows.shape[0] != hidden_rows.shape[0]:
        raise ValueError(f"grad rows {grad_rows.shape[0]} != hidden rows {hidden_rows.shape[0]}")
    dweight = torch.zeros(
        (grad_rows.shape[1], hidden_rows.shape[1]),
        device=grad_rows.device,
        dtype=torch.float32,
    )
    for index in range(grad_rows.shape[0]):
        dweight.addmm_(grad_rows[index].unsqueeze(1), hidden_rows[index].unsqueeze(0))
    return dweight


def row_local_bias_fp32(grad_output: torch.Tensor) -> torch.Tensor:
    rows = grad_output.reshape(-1, grad_output.size(-1)).float()
    acc = torch.zeros((rows.shape[1],), device=rows.device, dtype=torch.float32)
    for index in range(rows.shape[0]):
        acc = acc + rows[index]
    return acc


def rmsnorm_dweight_rows_fp32(
    x: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    rstd: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-row dweight contributions, shape [..., H]."""

    x32 = x.float()
    grad32 = grad_output.float()
    if rstd is None:
        rstd = torch.rsqrt(x32.square().mean(dim=-1) + float(eps))
    else:
        rstd = rstd.float()
    return grad32 * x32 * rstd.unsqueeze(-1)


def reduce_rows_fp32(rows: torch.Tensor) -> torch.Tensor:
    """Left-fold dim 0 in FP32. Deterministic for a fixed row order."""

    if rows.dim() == 0:
        raise ValueError("rows must have at least one dimension")

    output_shape = tuple(rows.shape[1:])
    columns = prod(output_shape)
    flat = rows.reshape(rows.shape[0], columns).float()

    if (
        flat.is_cuda
        and torch.version.hip is None
        and _EXT_AVAILABLE
        and hasattr(_C, "reduce_rows_fp32_left_fold")
    ):
        reduced = _C.reduce_rows_fp32_left_fold(flat.contiguous())
        return reduced.reshape(output_shape)

    acc = torch.zeros((flat.shape[1],), device=flat.device, dtype=torch.float32)
    for index in range(flat.shape[0]):
        acc = acc + flat[index]
    return acc.reshape(output_shape)


def reduce_keyed_rows_fp32(
    contributions: Mapping[tuple[str, int], torch.Tensor],
) -> torch.Tensor:
    if not contributions:
        raise RuntimeError("no logical-token contributions to reduce")
    keys = sorted(contributions)
    acc = contributions[keys[0]].float().clone()
    for key in keys[1:]:
        acc = acc + contributions[key].float()
    return acc


def reduce_keyed_outers_fp32(
    rows_g: Mapping[tuple[str, int], torch.Tensor],
    rows_x: Mapping[tuple[str, int], torch.Tensor],
) -> torch.Tensor:
    keys = sorted(set(rows_g) | set(rows_x))
    if not keys or set(rows_g) != set(rows_x):
        raise RuntimeError("logical-token sets for outer-product VJP do not match")
    first_g = rows_g[keys[0]].float()
    first_x = rows_x[keys[0]].float()
    acc = torch.outer(first_g, first_x)
    for key in keys[1:]:
        acc = acc + torch.outer(rows_g[key].float(), rows_x[key].float())
    return acc


def merge_keyed(
    target: dict[tuple[str, int], torch.Tensor],
    source: Mapping[tuple[str, int], torch.Tensor],
) -> None:
    overlap = set(target) & set(source)
    if overlap:
        raise RuntimeError(f"logical token collision: {sorted(overlap)[:4]}")
    target.update(source)


__all__ = [
    "BACKWARD_IMPL",
    "merge_keyed",
    "reduce_keyed_outers_fp32",
    "reduce_keyed_rows_fp32",
    "reduce_rows_fp32",
    "rmsnorm_dweight_rows_fp32",
    "row_local_bias_fp32",
    "row_local_linear_dw_fp32",
    "row_local_linear_dx_fp32",
]
