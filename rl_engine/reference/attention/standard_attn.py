# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from typing import Optional

import torch


class NativeAttentionOp:
    """
    Pure PyTorch native standard-softmax attention reference.
    out = softmax(Q Kᵀ * scale + masks) @ V

    Hand-written naive softmax -- deliberately NOT
    ``F.scaled_dot_product_attention`` / flash / mem-efficient attention, whose
    reduction order is unspecified and would break the batch-invariance (Axis-A)
    contract. This op defines the *correct answer* the fused kernels align to.

    Qwen3-8B shapes: q ``[B, 32, Sq, 128]``, k/v ``[B, 8, Skv, 128]`` (GQA group
    g = 32/8 = 4), scale = 1/sqrt(head_dim) = 1/sqrt(128). Heads precede seq in
    the layout. This op covers ONLY the softmax attention; QK-Norm and RoPE are
    applied *before* the call (see the chain test) -- the q,k passed in are
    already normalized and rotated.

    This is a reduction over the key dimension (Skv): the low-precision
    ``forward`` path accumulates in the input dtype and therefore drifts from
    the fp32 ``forward_fp32`` ground truth, so Axis-B accuracy uses a tolerance
    (``torch.allclose``), not bitwise equality. Axis-A batch invariance still
    holds bitwise within a single dtype (each query row reduces over the keys
    independently of how many sequences share the batch).

    Masking conventions:
      * causal=True  -> upper-triangular -inf at diagonal Skv-Sq+1, valid for
        both prefill (Sq==Skv) and decode (Sq<Skv).
      * key_padding_mask ``[B, Skv]`` bool, True=valid / False=padding -> padded
        key columns set to -inf (matches reference_ops.py: True=keep, False=mask).
    """

    def __init__(self) -> None:
        """No state; the op is a pure function over (q, k, v, ...)."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for ``forward`` so the op is callable like a module."""
        return self.forward(q, k, v, causal=causal, scale=scale, key_padding_mask=key_padding_mask)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Canonical entry: attend in the input dtype, output the input dtype.
        This is the dtype-behavior path used as the Axis-B accuracy candidate.
        """
        return self._attention(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            compute_dtype=q.dtype,
            output_dtype=q.dtype,
        )

    def forward_fp32(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ground truth: upcast to fp32, accumulate in fp32, force fp32 output.

        The whole score->softmax->value path is wrapped to disable autocast and
        TF32 so this stays a true fp32 reference regardless of the caller's
        ambient precision context.
        """
        return self._attention(
            q,
            k,
            v,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
            compute_dtype=torch.float32,
            output_dtype=torch.float32,
            strict_fp32=True,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        scale: Optional[float],
        key_padding_mask: Optional[torch.Tensor],
        compute_dtype: torch.dtype,
        output_dtype: torch.dtype,
        strict_fp32: bool = False,
    ) -> torch.Tensor:
        """Core softmax attention: cast to ``compute_dtype``, score, mask, softmax,
        weighted-sum over V, cast out. ``strict_fp32`` disables autocast/TF32 so the
        fp32 reference is not silently downcast by the caller's ambient context.
        """
        Hq, Sq, D = q.shape[1], q.shape[2], q.shape[3]
        Hkv, Skv = k.shape[1], k.shape[2]
        if Hq % Hkv != 0:
            raise ValueError(f"Hq={Hq} not divisible by Hkv={Hkv} (GQA group)")

        ctx = NativeAttentionOp._strict_fp32_math(q.device.type) if strict_fp32 else nullcontext()
        with ctx:
            qf = q.to(compute_dtype)
            kf = k.to(compute_dtype)
            vf = v.to(compute_dtype)

            # GQA: replicate each KV head g=Hq//Hkv times (Qwen3: 32/8 -> 4).
            # repeat_interleave (not repeat) keeps each KV head's copies adjacent
            # so query head h maps to KV head h // g.
            if Hkv != Hq:
                r = Hq // Hkv
                kf = kf.repeat_interleave(r, dim=1)
                vf = vf.repeat_interleave(r, dim=1)

            # scale defaults to 1/sqrt(head_dim); `is not None` so an explicit 0.0 is kept.
            scale = scale if scale is not None else (1.0 / math.sqrt(D))
            scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale  # [B, Hq, Sq, Skv]

            # Causal: offset Skv-Sq+1 covers prefill (Sq==Skv) and decode (Sq<Skv).
            if causal:
                causal_mask = torch.triu(
                    torch.ones(Sq, Skv, dtype=torch.bool, device=q.device),
                    diagonal=Skv - Sq + 1,
                )
                scores = scores.masked_fill(causal_mask, float("-inf"))

            # key_padding_mask [B, Skv]: True=valid; False columns -> -inf.
            if key_padding_mask is not None:
                pad = ~key_padding_mask
                scores = scores.masked_fill(pad[:, None, None, :], float("-inf"))

            probs = torch.softmax(scores, dim=-1)  # subtracts row max internally
            if key_padding_mask is not None:
                # A query whose every valid key is padded out has an all -inf row;
                # softmax would emit NaN. Define such fully-masked rows as 0 (no key
                # contributes), keeping outputs/grads finite. Only key_padding_mask can
                # fully mask a row -- causal alone always leaves a query its own key --
                # so this guard lives in the padding branch and the no-pad path is
                # untouched. Row-independent, so Axis-A batch invariance still holds.
                all_masked = ~torch.isfinite(scores).any(dim=-1, keepdim=True)
                probs = torch.where(all_masked, torch.zeros_like(probs), probs)
            out = torch.matmul(probs, vf)  # [B, Hq, Sq, D]
            return out.to(output_dtype)

    @staticmethod
    @contextmanager
    def _strict_fp32_math(device_type: str):
        """Disable autocast and TF32 for a true fp32 path, restoring state after."""
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with torch.autocast(device_type=device_type, enabled=False):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
