# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""FP32 CPU oracle for the five P5 operators (P5-1..P5-5; issues #60-#64).

Numeric profile ``oracle-fp32-serial-v1``:

- All accumulations are FP32, serial, in ascending index order.
- Every multiply and add rounds separately (mul-then-add; no fused FMA).
  A strict CUDA kernel must either reproduce this (``__fmul_rn``/``__fadd_rn``)
  or register its own numeric profile.
- Backward is BF16 at operator boundaries (gradients round to BF16 when they
  cross an operator edge) with FP32 accumulators inside, per issue #8.
- Base weights are frozen: no ``dW`` is ever computed (issue #1 s2.5 item 1).
- ``mxfp8_act_quant`` backward is a straight-through estimator (dX = dY).

The oracle favors auditability over speed; use start-kit fixture sizes.
"""

from __future__ import annotations

import sys
from typing import Any

import torch

from rl_engine.moe.contract import (
    GATE_CLAMP_MAX,
    UP_CLAMP_MAX,
    UP_CLAMP_MIN,
    ExpertBatch,
    SharedBatch,
)
from rl_engine.moe.mx_format import (
    MX_BLOCK,
    MXTensor,
    e2m1_decode,
    e4m3_decode,
    e8m0_decode,
    mx_quantize,
    unpack_nibbles,
)
from rl_engine.moe.trace import ExpertTrace


def _serial_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b.T`` with FP32 mul-then-add in ascending-k order.

    a: [M, K], b: [N, K] (any float dtype) -> FP32 [M, N].
    """
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    m, k = a32.shape
    n, kb = b32.shape
    if kb != k:
        raise ValueError(f"serial_dot K mismatch: {k} vs {kb}")
    acc = torch.zeros(m, n, dtype=torch.float32, device=a32.device)
    for kk in range(k):
        acc = acc + a32[:, kk].unsqueeze(1) * b32[:, kk].unsqueeze(0)
    return acc


def _block_scaled_dot(
    a_elems: torch.Tensor,  # FP32 [M, K] decoded elements
    a_scales: torch.Tensor,  # FP32 [M, K/32]
    w_elems: torch.Tensor,  # FP32 [N, K] decoded elements
    w_scales: torch.Tensor,  # FP32 [N, K/32]
) -> torch.Tensor:
    """P5-4 (#61) fixed math: per 32-wide chunk j, ``acc += partial_j * sa_j * sw_j``.

    ``partial_j`` is the serial ascending-k FP32 dot of the decoded elements;
    the scale application order is ``(partial * scale_a) * scale_w``.
    """
    m, k = a_elems.shape
    n = w_elems.shape[0]
    n_blocks = k // MX_BLOCK
    acc = torch.zeros(m, n, dtype=torch.float32, device=a_elems.device)
    for j in range(n_blocks):
        partial = torch.zeros(m, n, dtype=torch.float32, device=a_elems.device)
        for kk in range(j * MX_BLOCK, (j + 1) * MX_BLOCK):
            partial = partial + a_elems[:, kk].unsqueeze(1) * w_elems[:, kk].unsqueeze(0)
        scaled = (partial * a_scales[:, j].unsqueeze(1)) * w_scales[:, j].unsqueeze(0)
        acc = acc + scaled
    return acc


# 1. mxfp8_act_quant — P5-1 (#60)


def mxfp8_act_quant_fwd(x: torch.Tensor) -> MXTensor:
    """BF16 [M, K] -> MXFP8 (block-32 E8M0 scales, row-local amax)."""
    return mx_quantize(x, "e4m3")


def mxfp8_act_quant_bwd(dy: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: dX = dY (not the true derivative)."""
    return dy.clone()


# 2. mxfp8_mxfp4_grouped_gemm — P5-4 (#61)


def mxfp8_mxfp4_grouped_gemm_fwd(
    a: MXTensor, w: MXTensor, expert_offsets: torch.Tensor
) -> torch.Tensor:
    """Frozen-base grouped GEMM: MXFP8 activation x MXFP4 weight -> FP32 [M, N].

    ``w`` holds one [N, K] weight per local expert ([E, N, K]); rows
    ``expert_offsets[e] : expert_offsets[e+1]`` of ``a`` use expert ``e``.
    """
    if a.elem_format != "e4m3" or w.elem_format != "e2m1":
        raise ValueError("grouped GEMM expects e4m3 activation and e2m1 weight")
    m, k = a.shape
    n_experts, n, wk = w.shape
    if wk != k:
        raise ValueError(f"K mismatch: activation {k} vs weight {wk}")
    a_elems = e4m3_decode(a.codes)
    a_scales = e8m0_decode(a.scales)
    w_elems = e2m1_decode(unpack_nibbles(w.codes))
    w_scales = e8m0_decode(w.scales)
    out = torch.zeros(m, n, dtype=torch.float32, device=a_elems.device)
    for e in range(n_experts):
        lo, hi = int(expert_offsets[e]), int(expert_offsets[e + 1])
        if lo == hi:
            continue
        out[lo:hi] = _block_scaled_dot(a_elems[lo:hi], a_scales[lo:hi], w_elems[e], w_scales[e])
    return out


def mxfp8_mxfp4_grouped_gemm_bwd(
    dy: torch.Tensor, w: MXTensor, expert_offsets: torch.Tensor
) -> torch.Tensor:
    """dX = dY @ W, BF16 operands with FP32 accumulator. No dW (frozen base).

    The MXFP4 weight is dequantized to BF16 (exact: <= 2 mantissa bits times a
    power-of-two scale) and the reduction runs serially over ascending n.
    """
    m = dy.shape[0]
    n_experts, n, k = w.shape
    dy_bf16 = dy.to(torch.bfloat16)
    w_elems = e2m1_decode(unpack_nibbles(w.codes))
    w_scales = e8m0_decode(w.scales)
    blocked = w_elems.reshape(n_experts, n, k // MX_BLOCK, MX_BLOCK)
    w_full = (blocked * w_scales.unsqueeze(-1)).reshape(n_experts, n, k)
    w_bf16 = w_full.to(torch.bfloat16)
    dx = torch.zeros(m, k, dtype=torch.float32, device=dy.device)
    for e in range(n_experts):
        lo, hi = int(expert_offsets[e]), int(expert_offsets[e + 1])
        if lo == hi:
            continue
        dx[lo:hi] = _serial_dot(dy_bf16[lo:hi], w_bf16[e].t())
    return dx


# 3. shared_grouped_lora_delta — P5-3 (#62)


def shared_grouped_lora_delta_fwd(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """LoRA delta ``Y = (X @ A.T) @ B.T * alpha`` on the BF16 path.

    Returns ``(y_fp32, u_bf16)``; ``u_bf16`` is the saved inter-GEMM
    activation (the intermediate rounds to BF16 between the two GEMMs).
    """
    u = _serial_dot(x, a)  # [M, r] FP32
    u_bf16 = u.to(torch.bfloat16)
    y = _serial_dot(u_bf16, b) * float(alpha)
    return y, u_bf16


def shared_grouped_lora_delta_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float,
    u_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward of the same graph: returns ``(dX, dA, dB)`` as FP32.

    ``dY' = dY * alpha`` rounds to BF16, then each GEMM runs BF16-in /
    FP32-accumulate, serial ascending order; ``dU`` rounds to BF16 before
    reuse. Association order is frozen as written.
    """
    dys = (dy.to(torch.float32) * float(alpha)).to(torch.bfloat16)
    du = _serial_dot(dys, b.t())  # [M, r]: dY' [M, N] x B [N, r]
    du_bf16 = du.to(torch.bfloat16)
    db = _serial_dot(dys.t(), u_bf16.t())  # [N, r] = dY'.T [N, M] x U.T [r, M] -> a @ b.T
    da = _serial_dot(du_bf16.t(), x.t())  # [r, K]
    dx = _serial_dot(du_bf16, a.t())  # [M, K]
    return dx, da, db


# 4. clamp_swiglu_weighted — P5-2 (#63)


def clamp_swiglu_weighted_fwd(
    gate: torch.Tensor, up: torch.Tensor, p_s: torch.Tensor | None
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``h = SiLU(min(gate, 10)) * clamp(up, -10, 10) * p_s`` — one-round.

    All math in FP32; the only BF16 round is on the output. The association
    order ``(SiLU(g) * u) * p_s`` is frozen. ``p_s=None`` means the unweighted
    shared-expert variant (no clamp is applied in that variant, P5-5 (#64)).
    """
    gate32 = gate.to(torch.float32)
    up32 = up.to(torch.float32)
    if p_s is None:
        g = gate32
        u = up32
    else:
        g = torch.clamp(gate32, max=GATE_CLAMP_MAX)
        u = torch.clamp(up32, min=UP_CLAMP_MIN, max=UP_CLAMP_MAX)
    sig = torch.sigmoid(g)
    silu = g * sig
    prod = silu * u
    h32 = prod if p_s is None else prod * p_s.unsqueeze(1)
    saved = {"gate32": gate32, "up32": up32, "g": g, "u": u, "sig": sig, "silu": silu}
    if p_s is not None:
        saved["p_s"] = p_s
    return h32.to(torch.bfloat16), saved


def clamp_swiglu_weighted_bwd(
    dh: torch.Tensor, saved: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Returns ``(dgate, dup, dp_s)``; ``dp_s`` is FP32 [rows] (or None).

    Clamp subgradients are zero exactly at the bounds (strict inequalities
    pass gradient). ``dp_s`` is a row-local serial sum over ascending n.
    """
    dh32 = dh.to(torch.float32)
    g, u, sig, silu = saved["g"], saved["u"], saved["sig"], saved["silu"]
    gate32, up32 = saved["gate32"], saved["up32"]
    p_s = saved.get("p_s")
    weighted = dh32 if p_s is None else dh32 * p_s.unsqueeze(1)
    dsilu = sig * (1.0 + g * (1.0 - sig))
    if p_s is None:
        gate_mask = torch.ones_like(g)
        up_mask = torch.ones_like(u)
    else:
        gate_mask = (gate32 < GATE_CLAMP_MAX).to(torch.float32)
        up_mask = ((up32 > UP_CLAMP_MIN) & (up32 < UP_CLAMP_MAX)).to(torch.float32)
    dgate = ((weighted * u) * dsilu) * gate_mask
    dup = (weighted * silu) * up_mask
    dp_s: torch.Tensor | None = None
    if p_s is not None:
        rows = dh32.shape[0]
        acc = torch.zeros(rows, dtype=torch.float32, device=dh32.device)
        for n in range(dh32.shape[1]):
            acc = acc + (dh32[:, n] * silu[:, n]) * u[:, n]
        dp_s = acc
    return dgate, dup, dp_s


# 5. shared_expert_mlp — P5-5 (#64)


def shared_expert_mlp_fwd(
    batch: SharedBatch,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Shared expert fc1 -> SwiGLU -> fc2 on every valid token. Returns (y, saved)."""
    z = _serial_dot(batch.x, batch.w_fc1)  # [T, 2F] FP32
    ffn = z.shape[1] // 2
    gate, up = z[:, :ffn], z[:, ffn:]
    h_bf16, sw_saved = clamp_swiglu_weighted_fwd(gate, up, p_s=None)
    y32 = _serial_dot(h_bf16, batch.w_fc2)  # [T, H] FP32
    y = y32.to(torch.bfloat16)
    saved: dict[str, Any] = {"swiglu": sw_saved, "h_bf16": h_bf16}
    return y, saved


def shared_expert_mlp_bwd(
    dy: torch.Tensor, batch: SharedBatch, saved: dict[str, Any]
) -> torch.Tensor:
    """Returns dX (FP32). Shared base weights are frozen: no dW."""
    dy_bf16 = dy.to(torch.bfloat16)
    dh = _serial_dot(dy_bf16, batch.w_fc2.t()).to(torch.bfloat16)  # [T, F]
    dgate, dup, _ = clamp_swiglu_weighted_bwd(dh, saved["swiglu"])
    dz = torch.cat([dgate, dup], dim=1).to(torch.bfloat16)  # [T, 2F]
    dx = _serial_dot(dz, batch.w_fc1.t())  # [T, H] FP32
    return dx


# Routed-expert composition (the full P5 forward/backward chain)


def routed_expert_forward(
    batch: ExpertBatch, trace: ExpertTrace | None = None, ops: Any = None
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Full routed pipeline: quant -> base GEMM + LoRA -> clamp-SwiGLU(p_s)
    -> quant -> base GEMM + LoRA -> BF16 routed output. Returns (y, saved)."""
    ops = ops if ops is not None else sys.modules[__name__]
    batch.validate()
    ffn = batch.ffn
    q1 = ops.mxfp8_act_quant_fwd(batch.x)
    z_base = ops.mxfp8_mxfp4_grouped_gemm_fwd(q1, batch.w1, batch.expert_offsets)
    if batch.lora is not None:
        z_lora, u1_bf16 = ops.shared_grouped_lora_delta_fwd(
            batch.x, batch.lora.a1, batch.lora.b1, batch.lora.alpha
        )
    else:
        z_lora = torch.zeros_like(z_base)
        u1_bf16 = torch.zeros(batch.rows, 0, dtype=torch.bfloat16, device=batch.x.device)
    z = z_base + z_lora
    gate, up = z[:, :ffn], z[:, ffn:]
    h_bf16, sw_saved = ops.clamp_swiglu_weighted_fwd(gate, up, batch.p_s)
    q2 = ops.mxfp8_act_quant_fwd(h_bf16)
    y_base = ops.mxfp8_mxfp4_grouped_gemm_fwd(q2, batch.w2, batch.expert_offsets)
    if batch.lora is not None:
        y_lora, u2_bf16 = ops.shared_grouped_lora_delta_fwd(
            h_bf16, batch.lora.a2, batch.lora.b2, batch.lora.alpha
        )
    else:
        y_lora = torch.zeros_like(y_base)
        u2_bf16 = torch.zeros(batch.rows, 0, dtype=torch.bfloat16, device=batch.x.device)
    y = (y_base + y_lora).to(torch.bfloat16)
    if trace is not None:
        trace.note("act_quant_bwd", "ste")
        trace.record("act_quant1.codes", q1.codes)
        trace.record("act_quant1.scales", q1.scales)
        trace.record("fc1_base", z_base)
        trace.record("fc1_lora", z_lora)
        trace.record("fc1_out", z)
        trace.record("swiglu_h", h_bf16)
        trace.record("act_quant2.codes", q2.codes)
        trace.record("act_quant2.scales", q2.scales)
        trace.record("fc2_base", y_base)
        trace.record("fc2_lora", y_lora)
        trace.record("routed_out", y)
    saved: dict[str, Any] = {
        "h_bf16": h_bf16,
        "swiglu": sw_saved,
        "u1_bf16": u1_bf16,
        "u2_bf16": u2_bf16,
    }
    return y, saved


def routed_expert_backward(
    batch: ExpertBatch,
    saved: dict[str, Any],
    dy: torch.Tensor,
    trace: ExpertTrace | None = None,
    ops: Any = None,
) -> dict[str, torch.Tensor | None]:
    """Backward chain. Returns dx, dp_s and dA1/dB1/dA2/dB2 (None w/o LoRA).

    No base-weight gradient exists anywhere in this function (frozen base).
    """
    ops = ops if ops is not None else sys.modules[__name__]
    dy_bf16 = dy.to(torch.bfloat16)
    dh_base = ops.mxfp8_mxfp4_grouped_gemm_bwd(dy_bf16, batch.w2, batch.expert_offsets)
    if batch.lora is not None:
        dh_lora, da2, db2 = ops.shared_grouped_lora_delta_bwd(
            dy_bf16,
            saved["h_bf16"],
            batch.lora.a2,
            batch.lora.b2,
            batch.lora.alpha,
            saved["u2_bf16"],
        )
    else:
        dh_lora, da2, db2 = torch.zeros_like(dh_base), None, None
    dh = ops.mxfp8_act_quant_bwd((dh_base + dh_lora).to(torch.bfloat16))  # STE
    dgate, dup, dp_s = ops.clamp_swiglu_weighted_bwd(dh, saved["swiglu"])
    dz = torch.cat([dgate, dup], dim=1).to(torch.bfloat16)
    dx_base = ops.mxfp8_mxfp4_grouped_gemm_bwd(dz, batch.w1, batch.expert_offsets)
    if batch.lora is not None:
        dx_lora, da1, db1 = ops.shared_grouped_lora_delta_bwd(
            dz,
            batch.x,
            batch.lora.a1,
            batch.lora.b1,
            batch.lora.alpha,
            saved["u1_bf16"],
        )
    else:
        dx_lora, da1, db1 = torch.zeros_like(dx_base), None, None
    dx = ops.mxfp8_act_quant_bwd(dx_base + dx_lora)  # STE; FP32 accumulator output
    if trace is not None:
        trace.record("bwd.dh", dh)
        trace.record("bwd.dp_s", dp_s if dp_s is not None else torch.zeros(0))
        trace.record("bwd.dz", dz)
        trace.record("bwd.dx", dx)
    return {"dx": dx, "dp_s": dp_s, "da1": da1, "db1": db1, "da2": da2, "db2": db2}
