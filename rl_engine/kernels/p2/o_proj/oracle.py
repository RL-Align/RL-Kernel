# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Grouped output projection oracle: inverse RoPE, 8-group wo_a, wo_b."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from rl_engine.kernels.p2.attention.oracle import strict_fp32_math
from rl_engine.kernels.p2.contract import (
    CONCAT_Z_DIM,
    GROUP_FLAT_DIM,
    HEAD_DIM,
    HEADS_PER_GROUP,
    HIDDEN_SIZE,
    N_O_PROJ_GROUPS,
    N_Q_HEADS,
    O_LORA_RANK,
)
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status
from rl_engine.kernels.p2.o_proj.rope_consumer import apply_gptj_interleaved_partial


@dataclass
class OProjForwardTensors:
    y: Tensor
    o: Tensor
    o_tilde: Tensor
    z_groups: tuple[Tensor, ...]
    z: Tensor
    w_a: Tensor
    w_b: Tensor


def sequential_linear(x: Tensor, weight: Tensor) -> Tensor:
    """Y = x @ weight.T with sequential K = 0 .. K-1 FP32 accumulation."""

    if x.shape[-1] != weight.shape[-1]:
        raise P2FailClosedError(
            P2Status.SCHEMA_MISMATCH,
            f"K mismatch x[...,{x.shape[-1]}] vs weight[...,{weight.shape[-1]}]",
        )
    rows = x.reshape(-1, x.shape[-1]).float()
    wf = weight.float()
    out = rows.new_zeros(rows.shape[0], wf.shape[0])
    for k in range(rows.shape[1]):
        out.add_(rows[:, k, None] * wf[None, :, k])
    return out.reshape(*x.shape[:-1], wf.shape[0])


def split_groups(o_tilde: Tensor) -> list[Tensor]:
    tokens, heads, dim = o_tilde.shape
    if heads != N_Q_HEADS or dim != HEAD_DIM:
        raise P2FailClosedError(
            P2Status.SCHEMA_MISMATCH,
            f"O_tilde must be [T,{N_Q_HEADS},{HEAD_DIM}], got {tuple(o_tilde.shape)}",
        )
    groups = []
    for group in range(N_O_PROJ_GROUPS):
        start = group * HEADS_PER_GROUP
        chunk = o_tilde[:, start : start + HEADS_PER_GROUP, :].reshape(tokens, GROUP_FLAT_DIM)
        groups.append(chunk)
    return groups


def concat_groups(z_groups: list[Tensor]) -> Tensor:
    if len(z_groups) != N_O_PROJ_GROUPS:
        raise P2FailClosedError(
            P2Status.SCHEMA_MISMATCH,
            f"expected {N_O_PROJ_GROUPS} groups, got {len(z_groups)}",
        )
    return torch.cat(z_groups, dim=-1)


def o_proj_grouped_fwd(
    o: Tensor,
    w_a: Tensor,
    w_b: Tensor,
    cos: Tensor,
    sin: Tensor,
    *,
    linear=sequential_linear,
) -> OProjForwardTensors:
    """w_a: [8, 1024, 4096], w_b: [4096, 8192]. Inverse RoPE is out-of-place."""

    if o.shape[-2:] != (N_Q_HEADS, HEAD_DIM):
        raise P2FailClosedError(
            P2Status.SCHEMA_MISMATCH,
            f"O must be [T,{N_Q_HEADS},{HEAD_DIM}], got {tuple(o.shape)}",
        )
    if tuple(w_a.shape) != (N_O_PROJ_GROUPS, O_LORA_RANK, GROUP_FLAT_DIM):
        raise P2FailClosedError(
            P2Status.SCHEMA_MISMATCH,
            f"W_a must be [{N_O_PROJ_GROUPS},{O_LORA_RANK},{GROUP_FLAT_DIM}], "
            f"got {tuple(w_a.shape)}",
        )
    if tuple(w_b.shape) != (HIDDEN_SIZE, CONCAT_Z_DIM):
        raise P2FailClosedError(
            P2Status.SCHEMA_MISMATCH,
            f"W_b must be [{HIDDEN_SIZE},{CONCAT_Z_DIM}], got {tuple(w_b.shape)}",
        )
    o_before = o.data_ptr()
    with strict_fp32_math(o.device.type):
        o_tilde = apply_gptj_interleaved_partial(o, cos, sin, inverse=True)
        if o.data_ptr() != o_before:
            raise P2FailClosedError(
                P2Status.IDENTITY_DRIFT,
                "input O storage was replaced; inverse RoPE must be out-of-place",
            )
        groups = split_groups(o_tilde)
        z_groups = [linear(groups[g], w_a[g]) for g in range(N_O_PROJ_GROUPS)]
        z = concat_groups(z_groups)
        y = linear(z, w_b)
    return OProjForwardTensors(
        y=y,
        o=o,
        o_tilde=o_tilde,
        z_groups=tuple(z_groups),
        z=z,
        w_a=w_a,
        w_b=w_b,
    )


def _gemm_prep(tensor: Tensor, gemm_dtype: torch.dtype | None) -> Tensor:
    if gemm_dtype is None:
        return tensor.float()
    return tensor.to(gemm_dtype) if tensor.dtype != gemm_dtype else tensor


def o_proj_grouped_bwd(
    d_y: Tensor,
    saved: OProjForwardTensors,
    cos: Tensor,
    sin: Tensor,
    *,
    linear=sequential_linear,
    gemm_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """dZ = dY @ W_b; dW_b = dY^T @ Z; per-group dO_g / dW_a; then forward RoPE on dO_tilde."""

    with strict_fp32_math(d_y.device.type):
        d_y_g = _gemm_prep(d_y, gemm_dtype)
        w_b = _gemm_prep(saved.w_b, gemm_dtype)
        w_a = _gemm_prep(saved.w_a, gemm_dtype)
        # Y = Z @ W_b.T  =>  dZ = dY @ W_b = linear(dY, W_b.T)
        d_z = linear(d_y_g, w_b.transpose(0, 1).contiguous())
        # dW_b = dY.T @ Z stays sequential FP32 (weight VJP, not DetGemm I/O)
        d_w_b = sequential_linear(
            d_y.float().transpose(0, 1).contiguous(),
            saved.z.float().transpose(0, 1).contiguous(),
        )
        d_o_tilde = torch.zeros(
            saved.o_tilde.shape, dtype=torch.float32, device=saved.o_tilde.device
        )
        d_w_a = torch.zeros(saved.w_a.shape, dtype=torch.float32, device=saved.w_a.device)
        d_z_groups = d_z.split(O_LORA_RANK, dim=-1)
        groups = split_groups(saved.o_tilde)
        for g in range(N_O_PROJ_GROUPS):
            d_zg = d_z_groups[g]
            # Z_g = O_g @ W_a[g].T  so dO_g = dZ_g @ W_a[g] = linear(dZ_g, W_a[g].T)
            d_og = linear(d_zg, w_a[g].transpose(0, 1).contiguous())
            d_w_a[g] = sequential_linear(
                d_zg.float().transpose(0, 1).contiguous(),
                groups[g].float().transpose(0, 1).contiguous(),
            )
            tokens = d_og.shape[0]
            d_o_tilde[:, g * HEADS_PER_GROUP : (g + 1) * HEADS_PER_GROUP, :] = d_og.float().reshape(
                tokens, HEADS_PER_GROUP, HEAD_DIM
            )
        d_o = apply_gptj_interleaved_partial(d_o_tilde, cos, sin, inverse=False)
    return d_o, d_w_a, d_w_b
