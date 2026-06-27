# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.kernels.ops.pytorch.attention.standard_attn import NativeAttentionOp


class NativeKVCacheAttnOp:
    """
    Pure PyTorch native KV-cache attention reference (ISSUE #108 WS1).

    Decode/incremental attention: the past keys/values live in a cache and the
    step's new keys/values are appended before attending::

        k_full = cat([k_cache, k_new], dim=2)   # along the seq axis
        v_full = cat([v_cache, v_new], dim=2)
        out    = NativeAttentionOp().forward_fp32(q, k_full, v_full, ...)

    The whole point is that it delegates to the *same* ``NativeAttentionOp`` used
    for full-sequence (prefill) attention: prefill and decode therefore share one
    reduction path, which is what makes rollout (decode) numerically consistent
    with training (prefill). Re-implementing the softmax here would defeat that.

    Qwen3-8B shapes (synthetic tensors, no checkpoint): q ``[B, 32, Sq, 128]``,
    cache/new k,v ``[B, 8, S_past/S_new, 128]`` (GQA group g = 32/8 = 4). Heads
    precede seq in the layout; the GQA KV replication happens inside
    ``NativeAttentionOp``, not here. This is a reduction over the full key length
    Skv = S_past + S_new.

    Alignment assumption: q's ``Sq`` rows are the *last* Sq positions of the full
    sequence -- i.e. the queries for ``k_new`` -- so callers pass ``Sq == S_new``
    (decode: Sq == S_new == 1). The causal offset ``Skv - Sq + 1`` inside
    ``NativeAttentionOp`` then lets each new query see the whole cache plus the
    new tokens up to and including itself, for both decode and chunked prefill.

    Masking conventions (forwarded verbatim to ``NativeAttentionOp``):
      * causal=True -> upper-triangular -inf at diagonal Skv-Sq+1.
      * key_padding_mask ``[B, Skv]`` bool over the *concatenated* length
        (S_past + S_new), True=valid / False=padding.

    Only the attention output is returned; producing an updated cache is a
    caller/runtime concern and is not part of this numerical contract.
    """

    def __init__(self) -> None:
        """No state; the op is a pure function over (q, k_cache, v_cache, k_new, v_new, ...)."""
        self._attn = NativeAttentionOp()

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for ``forward`` so the op is callable like a module."""
        return self.forward(
            q,
            k_cache,
            v_cache,
            k_new,
            v_new,
            causal=causal,
            scale=scale,
            key_padding_mask=key_padding_mask,
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Canonical entry: concat cache+new, then attend in the input dtype.
        Delegates to ``NativeAttentionOp.forward`` (the Axis-B dtype path).
        """
        k_full, v_full = self._concat_kv(k_cache, v_cache, k_new, v_new)
        return self._attn.forward(
            q, k_full, v_full, causal=causal, scale=scale, key_padding_mask=key_padding_mask
        )

    def forward_fp32(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        *,
        causal: bool = True,
        scale: Optional[float] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ground truth: concat cache+new, then attend in strict fp32.
        Delegates to ``NativeAttentionOp.forward_fp32`` so the fp32 golden path
        is identical to prefill's.
        """
        k_full, v_full = self._concat_kv(k_cache, v_cache, k_new, v_new)
        return self._attn.forward_fp32(
            q, k_full, v_full, causal=causal, scale=scale, key_padding_mask=key_padding_mask
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _concat_kv(
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the step's new K/V to the cache along the seq axis (dim=2).

        Layout is [B, Hkv, S, D], so the sequence axis is dim=2. Pure (no
        in-place writes into the passed-in cache tensors).
        """
        k_full = torch.cat([k_cache, k_new], dim=2)
        v_full = torch.cat([v_cache, v_new], dim=2)
        return k_full, v_full
