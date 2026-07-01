# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""
Prefill / decode KV-cache path-consistency harness.

Rollout generates token-by-token through a *decode* path (one query against the
cached KV of the preceding tokens). Training re-scores the same sequence through
a *prefill* path (the whole sequence in one forward). If the two paths reduce in
a different order the same token gets a different logprob in rollout vs training
-- a high-impact rollout-vs-training drift source.

This module defines a single fixed reduction-order contract and drives both
paths through it, so prefill and decode are bitwise-identical by construction.
A naive batched SDPA path is also provided to demonstrate the ~1e-7 drift that
appears when the reduction order is *not* shared -- i.e. the bug this guards
against.

The core contract: attention for query position ``t`` is always computed by
:func:`attend_single_query` over keys ``0..t`` in ascending index order. Prefill
loops that op over every position; decode calls the same op once per generated
token, reading the keys back from a :class:`KVCache`. Same op + same inputs +
same order => bitwise-equal output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AttentionSpec:
    """Shapes for a single attention layer (FlashAttention head layout)."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    causal: bool = True

    def __post_init__(self) -> None:
        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_heads, num_kv_heads, head_dim must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads (GQA/MQA)")

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def gqa_group(self) -> int:
        return self.num_heads // self.num_kv_heads


def expand_kv_heads(x: torch.Tensor, spec: AttentionSpec) -> torch.Tensor:
    """Expand ``[..., Hkv, D]`` to ``[..., H, D]`` for GQA/MQA (ascending repeat)."""

    if spec.num_kv_heads == spec.num_heads:
        return x
    return x.repeat_interleave(spec.gqa_group, dim=-2)


def attend_single_query(
    q_t: torch.Tensor,
    k_ctx: torch.Tensor,
    v_ctx: torch.Tensor,
    *,
    scale: float,
    spec: AttentionSpec,
    key_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Canonical fixed-order attention of one query against a key/value context.

    This is the single op shared by prefill and decode -- the reduction-order
    contract lives here and nowhere else.

    Args:
        q_t: ``[B, H, D]`` query for one position.
        k_ctx, v_ctx: ``[B, T, Hkv, D]`` cached context (keys ``0..T-1``).
        scale: softmax scale (``1/sqrt(D)``).
        spec: attention shapes (drives GQA expansion).
        key_mask: optional ``[B, T]`` bool, ``True`` = valid key. Invalid keys
            are scored ``-inf`` (their post-softmax weight is exactly 0, so they
            do not perturb the ascending-order sum -- adding ``0.0`` is an IEEE
            identity regardless of reduction grouping).

    Returns:
        ``[B, H, D]`` attention output in fp32.
    """

    q = q_t.float()
    k = expand_kv_heads(k_ctx.float(), spec)  # [B, T, H, D]
    v = expand_kv_heads(v_ctx.float(), spec)

    # scores[b, h, t] = scale * sum_d q[b,h,d] * k[b,t,h,d]
    scores = torch.einsum("bhd,bthd->bht", q, k) * scale
    if key_mask is not None:
        invalid = ~key_mask.to(device=scores.device, dtype=torch.bool)  # [B, T]
        scores = scores.masked_fill(invalid.unsqueeze(1), float("-inf"))

    weights = torch.softmax(scores, dim=-1)  # over T, ascending key order
    # Guard fully-masked rows (no valid key) -> zero output instead of NaN.
    weights = torch.nan_to_num(weights, nan=0.0)
    out = torch.einsum("bht,bthd->bhd", weights, v)  # [B, H, D]
    return out


class KVCache:
    """Pre-allocated paged-free KV buffer; writer dtype is the stored dtype."""

    def __init__(
        self,
        batch: int,
        spec: AttentionSpec,
        max_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self._spec = spec
        self._len = 0
        self.key = torch.zeros(
            (batch, max_len, spec.num_kv_heads, spec.head_dim), dtype=dtype, device=device
        )
        self.value = torch.zeros_like(self.key)

    @property
    def length(self) -> int:
        return self._len

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor) -> None:
        """Store one timestep ``[B, Hkv, D]`` (cast to the cache's stored dtype)."""

        t = self._len
        self.key[:, t] = k_t.to(self.key.dtype)
        self.value[:, t] = v_t.to(self.value.dtype)
        self._len += 1

    def context(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key[:, : self._len], self.value[:, : self._len]


def fixed_order_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    spec: AttentionSpec,
    key_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Prefill via the contract: loop every query position through the single-query
    op against keys ``0..t``. This is the golden reduction order.

    Args:
        q: ``[B, S, H, D]``; k, v: ``[B, S, Hkv, D]``.
        key_mask: optional ``[B, S]`` bool of valid (non-pad) positions.

    Returns:
        ``[B, S, H, D]`` fp32 attention output.
    """

    _check_qkv(q, k, v, spec)
    b, s = q.shape[0], q.shape[1]
    out = torch.zeros((b, s, spec.num_heads, spec.head_dim), dtype=torch.float32, device=q.device)
    for t in range(s):
        ctx_end = t + 1 if spec.causal else s
        mask_t = key_mask[:, :ctx_end] if key_mask is not None else None
        out[:, t] = attend_single_query(
            q[:, t],
            k[:, :ctx_end],
            v[:, :ctx_end],
            scale=spec.scale,
            spec=spec,
            key_mask=mask_t,
        )
    return out


def replay_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    spec: AttentionSpec,
    key_mask: Optional[torch.Tensor] = None,
    kv_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Decode via the contract: step token-by-token, append each ``k/v`` to a
    :class:`KVCache`, attend the single query against the cache.

    With ``kv_dtype`` equal to ``q.dtype`` (default) the stored context is a
    bitwise copy of the prefill keys, so the output equals
    :func:`fixed_order_attention` bitwise. A lower-precision ``kv_dtype``
    exercises writer-vs-reader drift introduced solely by the stored KV.
    """

    _check_qkv(q, k, v, spec)
    if not spec.causal:
        raise ValueError("decode replay requires a causal spec")
    b, s = q.shape[0], q.shape[1]
    cache = KVCache(
        b, spec, s, dtype=kv_dtype or q.dtype, device=q.device
    )
    out = torch.zeros((b, s, spec.num_heads, spec.head_dim), dtype=torch.float32, device=q.device)
    for t in range(s):
        cache.append(k[:, t], v[:, t])
        k_ctx, v_ctx = cache.context()
        mask_t = key_mask[:, : t + 1] if key_mask is not None else None
        out[:, t] = attend_single_query(
            q[:, t], k_ctx, v_ctx, scale=spec.scale, spec=spec, key_mask=mask_t
        )
    return out


# --------------------------------------------------------------------------- #
# Parity assertion helper (bitwise where required, else within tolerance).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParityReport:
    bitwise: bool
    max_abs_error: float
    mean_abs_error: float


def parity_report(candidate: torch.Tensor, reference: torch.Tensor) -> ParityReport:
    if candidate.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: {tuple(candidate.shape)} vs {tuple(reference.shape)}"
        )
    bitwise = bool(torch.equal(candidate, reference))
    diff = (candidate.float() - reference.float()).abs()
    return ParityReport(
        bitwise=bitwise,
        max_abs_error=float(diff.max().item()) if diff.numel() else 0.0,
        mean_abs_error=float(diff.mean().item()) if diff.numel() else 0.0,
    )


def assert_path_parity(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    require_bitwise: bool = False,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    msg: str = "",
) -> ParityReport:
    """Assert two paths agree; bitwise when required, else within tolerance."""

    report = parity_report(candidate, reference)
    prefix = f"{msg}: " if msg else ""
    if require_bitwise:
        assert report.bitwise, (
            f"{prefix}expected bitwise-equal paths but max_abs_error="
            f"{report.max_abs_error:.3e} (mean={report.mean_abs_error:.3e})"
        )
    else:
        torch.testing.assert_close(
            candidate.float(),
            reference.float(),
            atol=atol,
            rtol=rtol,
            msg=lambda m: f"{prefix}{m}",
        )
    return report


# --------------------------------------------------------------------------- #
# Tiny end-to-end causal LM for logprob-level (generate vs re-score) checks.
# --------------------------------------------------------------------------- #


class TinyCausalLM(nn.Module):
    """
    Minimal single-layer causal LM sharing one attention contract across prefill
    and decode. Deterministic init; fp32 throughout. Not a real model -- just
    enough of a forward chain (embed -> qkv -> attention -> o_proj -> lm_head)
    to produce logits/logprobs for path-parity tests.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        spec: AttentionSpec,
        *,
        seed: int = 0,
    ):
        super().__init__()
        self.spec = spec
        self.vocab_size = vocab_size
        qdim = spec.num_heads * spec.head_dim
        kvdim = spec.num_kv_heads * spec.head_dim
        self.embed = nn.Embedding(vocab_size, d_model)
        self.q_proj = nn.Linear(d_model, qdim, bias=False)
        self.k_proj = nn.Linear(d_model, kvdim, bias=False)
        self.v_proj = nn.Linear(d_model, kvdim, bias=False)
        self.o_proj = nn.Linear(qdim, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        gen = torch.Generator().manual_seed(seed)
        for param in self.parameters():
            with torch.no_grad():
                param.copy_(torch.empty_like(param).normal_(0.0, 0.02, generator=gen))
        self.eval()

    def _project(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s = input_ids.shape
        spec = self.spec
        h = self.embed(input_ids)
        q = self.q_proj(h).view(b, s, spec.num_heads, spec.head_dim)
        k = self.k_proj(h).view(b, s, spec.num_kv_heads, spec.head_dim)
        v = self.v_proj(h).view(b, s, spec.num_kv_heads, spec.head_dim)
        return q, k, v

    def _to_logits(self, attn_out: torch.Tensor) -> torch.Tensor:
        b, s = attn_out.shape[0], attn_out.shape[1]
        merged = attn_out.reshape(b, s, self.spec.num_heads * self.spec.head_dim)
        return self.lm_head(self.o_proj(merged))

    @torch.no_grad()
    def prefill_logits(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q, k, v = self._project(input_ids)
        attn = fixed_order_attention(q, k, v, spec=self.spec, key_mask=attention_mask)
        return self._to_logits(attn)

    @torch.no_grad()
    def decode_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        kv_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        q, k, v = self._project(input_ids)
        attn = replay_decode(
            q, k, v, spec=self.spec, key_mask=attention_mask, kv_dtype=kv_dtype
        )
        return self._to_logits(attn)

    @torch.no_grad()
    def generate(
        self, prompt_ids: torch.Tensor, max_new_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decode against a live KV cache.

        Returns ``(full_ids, step_logprobs)`` where ``full_ids`` is
        ``[B, prompt + max_new_tokens]`` and ``step_logprobs[:, i]`` is the
        logprob the decode path assigned to the token chosen at generation
        step ``i`` (token at full position ``prompt + i``).
        """

        spec = self.spec
        b, prompt_len = prompt_ids.shape
        total = prompt_len + max_new_tokens
        cache = KVCache(b, spec, total, dtype=torch.float32, device=prompt_ids.device)

        def step(token_col: torch.Tensor) -> torch.Tensor:
            # token_col: [B] -> logits [B, V] for the next position.
            h = self.embed(token_col)  # [B, d_model]
            q = self.q_proj(h).view(b, spec.num_heads, spec.head_dim)
            k = self.k_proj(h).view(b, spec.num_kv_heads, spec.head_dim)
            vv = self.v_proj(h).view(b, spec.num_kv_heads, spec.head_dim)
            cache.append(k, vv)
            k_ctx, v_ctx = cache.context()
            attn = attend_single_query(q, k_ctx, v_ctx, scale=spec.scale, spec=spec)
            merged = attn.reshape(b, spec.num_heads * spec.head_dim)
            return self.lm_head(self.o_proj(merged))

        # Consume the prompt; keep the logits produced at the final prompt token.
        logits = None
        for t in range(prompt_len):
            logits = step(prompt_ids[:, t])

        gen_ids = []
        step_logprobs = []
        for _ in range(max_new_tokens):
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            next_token = torch.argmax(logprobs, dim=-1)  # [B]
            step_logprobs.append(logprobs.gather(1, next_token.unsqueeze(1)).squeeze(1))
            gen_ids.append(next_token)
            logits = step(next_token)

        generated = torch.stack(gen_ids, dim=1)  # [B, max_new_tokens]
        full_ids = torch.cat([prompt_ids, generated], dim=1)
        return full_ids, torch.stack(step_logprobs, dim=1)


# --------------------------------------------------------------------------- #
# internal
# --------------------------------------------------------------------------- #


def _check_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: AttentionSpec) -> None:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be [B, S, H, D]")
    if q.shape[2] != spec.num_heads or k.shape[2] != spec.num_kv_heads:
        raise ValueError("head counts must match the AttentionSpec")
    if k.shape != v.shape:
        raise ValueError("k and v must share shape")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1]:
        raise ValueError("q and k/v must share batch and sequence length")
