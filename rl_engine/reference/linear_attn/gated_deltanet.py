# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Gated DeltaNet reference op (Qwen3-Next linear-attention layers).

Qwen3-Next replaces 3 of every 4 transformer layers with a linear-attention
layer built on the gated delta rule (Gated DeltaNet). This module is the
pure-PyTorch golden for that primitive, in the same role
``NativeAttentionOp`` plays for full attention. This normalized-input
reference is distinct from the upstream vllm_decode_v1 numerical profile:
state is K-major FP32, g is already log decay, and beta is still a logit.
Provider adapters must explicitly convert gates and state layout; comparison
with that profile is tolerance-based, not a claim of bitwise equivalence.

Canonical recurrence (per head, everything in fp32):

    k_t, q_t <- l2norm(k_t), l2norm(q_t)          # Qwen3-Next / FLA convention
    gamma_t   = exp(g_t)                          # per-head decay gate, (0, 1]
    S        <- gamma_t * S                       # decay retained memory
    S        <- S + beta_t * k_t (v_t - k_t^T S)^T  # un-decayed delta write
    o_t       = (Dk**-0.5 * q_t)^T S                           # read AFTER the write at t

with S in R^{Dk x Dv} (per head), beta_t = sigmoid(beta_logit_t). The delta
read-out uses the already-decayed state, matching the FLA recurrent
formulation used by the HF ``Qwen3Next`` implementation. The output at t
sees the write at t (causal-inclusive), as in delta-rule fast-weight
attention.

This file deliberately implements the *sequential* form: it is the
sequential reference. The chunked WY-representation matmul form (chunk
size 64, intra-chunk UT transform) is the design target of the AscendC
kernel and must be validated against this reference on the NPU host.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from rl_engine.reference.activation.sigmoid import length_invariant_sigmoid

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_dtype(x: torch.Tensor, name: str) -> None:
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {x.dtype}.")


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """Per-head L2 normalization over the last dim, fp32, zero-safe.

    ``x`` is usually a strided per-timestep view (``q[:, :, t]``); fp32
    input makes ``.float()`` an identity that would keep that layout, and
    TensorIterator picks different reduction orders for strided vs
    contiguous inputs depending on shape — which would break bitwise batch
    invariance. Materialize a contiguous fp32 copy first.
    """
    x_f = x.float().contiguous()
    return x_f * torch.rsqrt((x_f * x_f).sum(dim=-1, keepdim=True) + 1e-6)


class NativeGatedDeltaNetOp(nn.Module):
    """
    Pure PyTorch Gated DeltaNet reference (sequential golden).

    Shapes (head-major, matching the Qwen3-Next dataflow; GQA-style
    grouped heads as in FLA's gated_delta_rule — q/k carry ``Hk`` key
    heads, v/o/beta/g and the state carry ``Hv`` value heads with
    ``Hv % Hk == 0``; value head ``h`` reads key head ``h // (Hv/Hk)``):
      - q:      [B, Hk, T, Dk]
      - k:      [B, Hk, T, Dk]
      - v:      [B, Hv, T, Dv]
      - beta:   [B, Hv, T]  sigmoid LOGITS (sigmoid applied inside)
      - g:      [B, Hv, T]  log decay (exp(g) used as gamma) or None
      - initial_state: [B, Hv, Dk, Dv] fp32 or None (None -> zeros)
      - returns o: [B, Hv, T, Dv] (+ final_state fp32 when requested)

    Batch slice/padding invariance is regression-tested on the available
    host. Transcendental and reduction last bits are not qualified across
    devices, CPU ISAs, PyTorch versions or provider numerical profiles.
    """

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------ #
    # Canonical entries
    # ------------------------------------------------------------------ #
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        *,
        scale: Optional[float] = None,
        return_final_state: bool = False,
        normalize_qk: bool = True,
    ):
        """
        Canonical entry: accumulate in fp32, cast outputs back to q.dtype.
        """
        return self._run(
            q,
            k,
            v,
            beta,
            g,
            initial_state,
            output_dtype=q.dtype,
            scale=scale,
            return_final_state=return_final_state,
            normalize_qk=normalize_qk,
        )

    def forward_fp32(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        *,
        scale: Optional[float] = None,
        return_final_state: bool = False,
        normalize_qk: bool = True,
    ):
        """Ground-truth: accumulate in fp32 and force fp32 outputs."""
        return self._run(
            q,
            k,
            v,
            beta,
            g,
            initial_state,
            output_dtype=torch.float32,
            scale=scale,
            return_final_state=return_final_state,
            normalize_qk=normalize_qk,
        )

    def decode_step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: Optional[torch.Tensor],
        state: torch.Tensor,
        *,
        scale: Optional[float] = None,
        normalize_qk: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stateful single-token step for incremental decode.

        q/k/v: [B, H, 1, Dk/Dk/Dv]; beta: [B, H, 1]; g: [B, H, 1] or None;
        state: [B, H, Dk, Dv] fp32. Returns (o [B, H, 1, Dv] in q.dtype,
        new_state fp32). Same math as the prefill loop at t = T+1.
        """
        if q.ndim != 4 or q.shape[2] != 1:
            raise ValueError("decode_step requires exactly one token")
        o, new_state = self._run(
            q,
            k,
            v,
            beta,
            g,
            state,
            output_dtype=q.dtype,
            scale=scale,
            return_final_state=True,
            normalize_qk=normalize_qk,
        )
        return o, new_state

    # ------------------------------------------------------------------ #
    # Sequential core
    # ------------------------------------------------------------------ #
    def _run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        g: Optional[torch.Tensor],
        initial_state: Optional[torch.Tensor],
        *,
        output_dtype: torch.dtype,
        scale: Optional[float],
        return_final_state: bool,
        normalize_qk: bool,
    ):
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"q/k/v must be [B, H, T, D], got {tuple(q.shape)}, "
                f"{tuple(k.shape)}, {tuple(v.shape)}"
            )
        _validate_dtype(q, "q")
        _validate_dtype(k, "k")
        _validate_dtype(v, "v")
        _validate_dtype(beta, "beta")
        if g is not None:
            _validate_dtype(g, "g")

        batch, key_heads, seqlen, dim_k = q.shape
        if k.shape != q.shape:
            raise ValueError(f"k must share q's shape, got {tuple(k.shape)} vs {tuple(q.shape)}")
        if v.dim() != 4 or v.shape[0] != batch or v.shape[2] != seqlen:
            raise ValueError(
                f"v must be [B, Hv, T, Dv] with B={batch}, T={seqlen}, got {tuple(v.shape)}"
            )
        if key_heads <= 0 or dim_k <= 0 or v.shape[1] <= 0 or v.shape[-1] <= 0:
            raise ValueError("head counts and head dimensions must be positive")
        for name, tensor in (
            ("k", k),
            ("v", v),
            ("beta", beta),
            ("g", g),
            ("initial_state", initial_state),
        ):
            if tensor is not None and tensor.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")
        if initial_state is not None and initial_state.dtype != torch.float32:
            raise TypeError("initial_state must be fp32")
        value_heads = v.shape[1]
        if value_heads % key_heads != 0:
            raise ValueError(
                f"GQA grouping requires Hv % Hk == 0, got Hv={value_heads}, Hk={key_heads}"
            )
        group = value_heads // key_heads
        dim_v = v.shape[-1]
        if beta.shape != (batch, value_heads, seqlen):
            raise ValueError(
                f"beta must be [B, Hv, T] = [{batch}, {value_heads}, {seqlen}], "
                f"got {tuple(beta.shape)}"
            )
        if g is not None and g.shape != (batch, value_heads, seqlen):
            raise ValueError(
                f"g must be [B, Hv, T] = [{batch}, {value_heads}, {seqlen}], got {tuple(g.shape)}"
            )
        if initial_state is not None and initial_state.shape != (batch, value_heads, dim_k, dim_v):
            raise ValueError(
                f"initial_state must be [B, Hv, {dim_k}, {dim_v}], got {tuple(initial_state.shape)}"
            )

        q_scale = dim_k**-0.5 if scale is None else float(scale)

        if not math.isfinite(q_scale):
            raise ValueError("scale must be finite")

        if initial_state is None:
            state = q.new_zeros(batch, value_heads, dim_k, dim_v, dtype=torch.float32)
        else:
            state = initial_state.float()

        outputs = []
        beta_sig = length_invariant_sigmoid(beta.float())  # [B, Hv, T]
        gamma = g.float().exp() if g is not None else None  # [B, Hv, T]

        # Normalize and expand grouped heads once, rather than launching the
        # same pointwise/copy kernels for every decode step in prefill.
        q_f = _l2norm(q) if normalize_qk else q.float().contiguous()
        k_f = _l2norm(k) if normalize_qk else k.float().contiguous()
        if group > 1:
            q_f = q_f.repeat_interleave(group, dim=1)
            k_f = k_f.repeat_interleave(group, dim=1)
        v_f = v.float()

        for t in range(seqlen):
            q_t = q_f[:, :, t]
            k_t = k_f[:, :, t]
            v_t = v_f[:, :, t]

            if gamma is not None:
                state = state * gamma[:, :, t].unsqueeze(-1).unsqueeze(-1)

            # All reductions below are broadcast-multiplies followed by sums
            # over a fixed-length dim (Dk) — never BLAS calls — so the
            # reduction order cannot vary with batch size (determinism
            # contract; see the batch-invariance tests).
            # delta read-out on the decayed state: k^T S -> [B, H, Dv]
            kv = (k_t.unsqueeze(-1) * state).sum(dim=-2)
            # un-decayed delta write: S += beta * k (v - k^T S)^T
            state = state + beta_sig[:, :, t].unsqueeze(-1).unsqueeze(-1) * (
                k_t.unsqueeze(-1) * (v_t - kv).unsqueeze(-2)
            )
            # causal-inclusive read AFTER the write
            o_t = ((q_t * q_scale).unsqueeze(-1) * state).sum(dim=-2)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=2) if outputs else v.float() + state.sum() * 0
        result = out.to(output_dtype)
        if return_final_state:
            return result, state
        return result
