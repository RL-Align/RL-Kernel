# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Sequential-tree FP32 oracle for mqa_joint_attention_sink.

QK inner product: d = 0 .. 511.
Softmax: m = max(sink, max_j l); Z = e_sink + sum_j e[j] with j = 0 .. N-1.
PV / backward head reductions: sequential index order. No torch.matmul, no
softmax(), no Split-KV, no sink V column.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

from rl_engine.kernels.p2.candidate_plan import CandidatePlan
from rl_engine.kernels.p2.contract import ATTENTION_SCALE, HEAD_DIM, N_Q_HEADS
from rl_engine.kernels.p2.errors import P2FailClosedError, P2Status


@contextmanager
def strict_fp32_math(device_type: str):
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


@dataclass
class AttentionForwardTensors:
    o: Tensor
    p: Tensor
    p_sink: Tensor
    m: Tensor
    z: Tensor
    logits: Tensor
    e: Tensor
    e_sink: Tensor
    sink: Tensor
    q: Tensor
    k: Tensor
    v: Tensor
    valid: Tensor


@dataclass
class AttentionBackwardState:
    q: Tensor
    k: Tensor
    v: Tensor
    p: Tensor
    p_sink: Tensor
    valid: Tensor


@dataclass
class AttentionBackwardTensors:
    dq: Tensor
    dkv: Tensor
    dsink: Tensor
    dk: Tensor
    dv: Tensor


def _broadcast_sink(sink: Tensor, tokens: int, heads: int) -> Tensor:
    if sink.dim() == 1:
        if sink.shape[0] != heads:
            raise P2FailClosedError(
                P2Status.INVALID_SINK_SEMANTICS,
                f"sink [H] must have H={heads}, got {tuple(sink.shape)}",
            )
        return sink.float().unsqueeze(0).expand(tokens, heads).contiguous()
    if sink.dim() == 2 and sink.shape == (tokens, heads):
        return sink.float().contiguous()
    raise P2FailClosedError(
        P2Status.INVALID_SINK_SEMANTICS,
        f"sink must be [{heads}] or [{tokens}, {heads}], got {tuple(sink.shape)}",
    )


def _validate_q(q: Tensor) -> tuple[int, int, int]:
    if q.dim() != 3:
        raise P2FailClosedError(
            P2Status.SCHEMA_MISMATCH,
            f"Q must be [T, {N_Q_HEADS}, {HEAD_DIM}], got {tuple(q.shape)}",
        )
    tokens, heads, dim = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    if heads != N_Q_HEADS or dim != HEAD_DIM:
        raise P2FailClosedError(
            P2Status.SCHEMA_MISMATCH,
            f"Q must be [T, {N_Q_HEADS}, {HEAD_DIM}], got {tuple(q.shape)}",
        )
    return tokens, heads, dim


def _validate_scale(scale: float) -> None:
    if scale != ATTENTION_SCALE:
        raise P2FailClosedError(
            P2Status.ROUND_POINT_MISMATCH,
            f"scale must be 512^-0.5 ({ATTENTION_SCALE}), got {scale}",
        )


def sequential_qk(
    q: Tensor,
    k: Tensor,
    valid: Tensor,
    scale: float,
    *,
    mask_invalid: bool = True,
) -> Tensor:
    """l[t,h,j] = scale * sum_d Q[t,h,d] * K[j,d]; invalid j -> -inf when masked."""

    tokens, heads, dim = q.shape
    n_cand = int(k.shape[0])
    logits = q.new_zeros(tokens, heads, n_cand, dtype=torch.float32)
    qf = q.float()
    kf = k.float()
    for d in range(dim):
        logits.add_(qf[:, :, d, None] * kf[None, None, :, d])
    logits.mul_(scale)
    if n_cand and mask_invalid:
        logits[:, :, ~valid] = float("-inf")
    return logits


def mqa_joint_attention_sink_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sink: Tensor,
    plan: CandidatePlan,
    *,
    scale: float = ATTENTION_SCALE,
) -> AttentionForwardTensors:
    plan.validate_for_kv(k, v)
    tokens, heads, dim = _validate_q(q)
    _validate_scale(scale)
    if k.device != q.device or v.device != q.device:
        raise P2FailClosedError(P2Status.SCHEMA_MISMATCH, "Q/K/V must share device")
    valid = plan.valid.to(device=q.device)
    sink_th = _broadcast_sink(sink, tokens, heads)
    n_cand = plan.n_candidates

    with strict_fp32_math(q.device.type):
        if n_cand == 0:
            logits = q.new_zeros(tokens, heads, 0, dtype=torch.float32)
            m = sink_th.clone()
            e = logits.clone()
            e_sink = torch.ones(tokens, heads, dtype=torch.float32, device=q.device)
            z = e_sink.clone()
            p = e.clone()
            p_sink = torch.ones(tokens, heads, dtype=torch.float32, device=q.device)
            o = torch.zeros(tokens, heads, dim, dtype=torch.float32, device=q.device)
        else:
            logits = sequential_qk(q, k, valid, scale)
            m = sink_th.clone()
            for j in range(n_cand):
                if bool(valid[j]):
                    m = torch.maximum(m, logits[:, :, j])
            e_sink = torch.exp(sink_th - m)
            z = e_sink.clone()
            e = torch.zeros(tokens, heads, n_cand, dtype=torch.float32, device=q.device)
            for j in range(n_cand):
                if bool(valid[j]):
                    e_j = torch.exp(logits[:, :, j] - m)
                    e[:, :, j] = e_j
                    z = z + e_j
            p = torch.zeros_like(e)
            p_sink = e_sink / z
            for j in range(n_cand):
                if bool(valid[j]):
                    p[:, :, j] = e[:, :, j] / z
            vf = v.float()
            o = torch.zeros(tokens, heads, dim, dtype=torch.float32, device=q.device)
            for j in range(n_cand):
                if bool(valid[j]):
                    o.add_(p[:, :, j, None] * vf[j][None, None, :])
        if not torch.isfinite(o).all() or not torch.isfinite(z).all():
            raise P2FailClosedError(P2Status.NON_FINITE, "forward produced non-finite O or Z")

    return AttentionForwardTensors(
        o=o,
        p=p,
        p_sink=p_sink,
        m=m,
        z=z,
        logits=logits,
        e=e,
        e_sink=e_sink,
        sink=sink_th,
        q=q,
        k=k,
        v=v,
        valid=valid,
    )


def mqa_joint_attention_sink_bwd(
    d_o: Tensor,
    saved: AttentionForwardTensors | AttentionBackwardState,
    *,
    scale: float = ATTENTION_SCALE,
    sink_was_shared: bool = False,
) -> AttentionBackwardTensors:
    _validate_scale(scale)
    tokens, heads, dim = saved.q.shape
    n_cand = int(saved.p.shape[2])
    d_o_f = d_o.float()
    qf = saved.q.float()
    kf = saved.k.float()
    vf = saved.v.float()
    p = saved.p
    p_sink = saved.p_sink
    valid = saved.valid

    with strict_fp32_math(d_o.device.type):
        dq = torch.zeros(tokens, heads, dim, dtype=torch.float32, device=d_o.device)
        dk = torch.zeros(n_cand, dim, dtype=torch.float32, device=d_o.device)
        dv = torch.zeros(n_cand, dim, dtype=torch.float32, device=d_o.device)
        dsink_th = torch.zeros(tokens, heads, dtype=torch.float32, device=d_o.device)

        if n_cand == 0:
            dsink = dsink_th.sum(dim=0) if sink_was_shared else dsink_th
            return AttentionBackwardTensors(dq=dq, dkv=dk, dsink=dsink, dk=dk, dv=dv)

        # dp[t,h,j] = sum_d dO[t,h,d] * V[j,d]  (same sequential-D tree as QK)
        dp = sequential_qk(d_o_f, vf, valid, scale=1.0, mask_invalid=False)
        if n_cand:
            dp[:, :, ~valid] = 0.0
        mu = torch.zeros(tokens, heads, dtype=torch.float32, device=d_o.device)
        for j in range(n_cand):
            if bool(valid[j]):
                mu = mu + p[:, :, j] * dp[:, :, j]
        dsink_th = -p_sink * mu
        dl = torch.zeros_like(p)
        for j in range(n_cand):
            if bool(valid[j]):
                dl[:, :, j] = p[:, :, j] * (dp[:, :, j] - mu)
        for j in range(n_cand):
            if bool(valid[j]):
                dq.add_(scale * dl[:, :, j, None] * kf[j][None, None, :])
        # dK/dV: token then head, sequential logical head tags (not rank id)
        for t in range(tokens):
            for h in range(heads):
                dk.add_(scale * dl[t, h, :, None] * qf[t, h][None, :])
                dv.add_(p[t, h, :, None] * d_o_f[t, h][None, :])

        dkv = dk + dv
        if sink_was_shared:
            dsink = torch.zeros(heads, dtype=torch.float32, device=d_o.device)
            for t in range(tokens):
                dsink.add_(dsink_th[t])
        else:
            dsink = dsink_th
        if not torch.isfinite(dq).all() or not torch.isfinite(dkv).all():
            raise P2FailClosedError(P2Status.NON_FINITE, "backward produced non-finite grads")

    return AttentionBackwardTensors(dq=dq, dkv=dkv, dsink=dsink, dk=dk, dv=dv)
