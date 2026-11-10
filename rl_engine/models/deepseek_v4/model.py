# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4 decoder layer and model forward on the batch-invariant SM90 kernel set.

Follows vLLM's ``DeepseekV4Model`` in its batch-invariant mode (unfused
pre / block / post, ``ISEEKYAN/vllm@1a6542c``)::

    h        = embed[input_ids]                                   BF16 [T, H]
    streams  = h copied to 4 streams                              BF16 [T, 4, H]
    per layer:
        x, post, comb = mhc_pre(streams, hc_attn_*, attn_norm)    dsv4-mhc-v1, norm fused
        streams       = mhc_post(attention(x), streams, post, comb)
        x, post, comb = mhc_pre(streams, hc_ffn_*, ffn_norm)
        streams       = mhc_post(moe(x, input_ids), streams, post, comb)
    out      = hc_head(streams, hc_head_*, norm)                  BF16 [T, H], final norm fused

The LM head (selected-token log-probs) is P7 and not part of this module.

Every operator is batch-invariant and per-token, so a token's output bytes do not
depend on which other tokens share its step, nor on whether its request runs as one
prefill, chunked prefill or decode steps (``tests/models/deepseek_v4/test_dsv4_model.py``).

The first layer reads the embedding as four identical streams. vLLM's broadcast
kernel instead sums ``hc_attn_fn`` over the streams first; the two are equal in exact
arithmetic, not in rounding, and this model keeps the ``dsv4-mhc-v1`` contract.

Forward only (rollout and actor recompute). The attention backward does not exist
yet; the mHC and MoE operators have their own dX backward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.mhc_dsv4 import hc_head_cuda, mhc_post_cuda, mhc_pre_cuda
from rl_engine.models.deepseek_v4.attention import Dsv4Attention, Dsv4Batch, LayerCaches, new_caches
from rl_engine.models.deepseek_v4.config import Dsv4Config
from rl_engine.models.deepseek_v4.moe import Dsv4MoE


@dataclass
class MhcParams:
    """One mHC site: ``fn`` FP32 ``[24, 4H]`` (``[4, 4H]`` for the head), ``scale``,
    ``base`` FP32, and the fused RMSNorm weight BF16 ``[H]``."""

    fn: Tensor
    scale: Tensor
    base: Tensor
    norm: Tensor


@dataclass
class Dsv4DecoderLayer:
    index: int
    attn: Dsv4Attention
    moe: Dsv4MoE
    hc_attn: MhcParams
    hc_ffn: MhcParams
    hash_routing: bool = False

    def forward_batch(
        self, streams: Tensor, batch: Dsv4Batch, caches: LayerCaches, input_ids: Tensor
    ) -> Tensor:
        """``streams`` BF16 ``[T, 4, H]`` -> BF16 ``[T, 4, H]``."""
        p = self.hc_attn
        x, post, comb = mhc_pre_cuda(streams, p.fn, p.scale, p.base, p.norm)[:3]
        streams = mhc_post_cuda(self.attn.forward_batch(x, batch, caches), streams, post, comb)
        p = self.hc_ffn
        x, post, comb = mhc_pre_cuda(streams, p.fn, p.scale, p.base, p.norm)[:3]
        y = self.moe.forward(x, None, input_ids if self.hash_routing else None)
        return mhc_post_cuda(y, streams, post, comb)


@dataclass
class Dsv4Model:
    config: Dsv4Config
    embed: Tensor  # BF16 [vocab, H]
    layers: list[Dsv4DecoderLayer]
    hc_head: MhcParams  # norm = the final norm
    head: Tensor | None = None  # BF16 [vocab, H], for P7
    max_model_len: int = 65536
    first_layer: int = 0
    extra: dict = field(default_factory=dict)

    def new_caches(self, requests: int, max_len: int | None = None, seed: int = 0):
        """Paged caches for every layer (shuffled physical blocks, as in the layer tests)."""
        max_len = self.max_model_len if max_len is None else max_len
        return [
            new_caches(layer.attn.c, requests, max_len, device=self.embed.device, seed=seed + i)
            for i, layer in enumerate(self.layers)
        ]

    def embed_streams(self, input_ids: Tensor) -> Tensor:
        h = self.embed.index_select(0, input_ids)
        return h.unsqueeze(1).expand(-1, self.config.hc_mult, -1).contiguous()

    @torch.no_grad()
    def forward_batch(
        self,
        input_ids: Tensor,
        batch: Dsv4Batch,
        caches: list[LayerCaches],
        *,
        return_streams: bool = False,
    ):
        """Final hidden BF16 ``[T, H]`` (after hc_head and the final norm).

        ``input_ids`` int64 ``[T]``, tokens grouped by request in ascending position
        (the order :class:`Dsv4Batch` describes). With ``return_streams`` also returns
        each layer's output streams BF16 ``[T, 4, H]`` (for the layerwise gate).
        """
        input_ids = input_ids.long()
        streams = self.embed_streams(input_ids)
        per_layer = []
        for layer, cache in zip(self.layers, caches):
            streams = layer.forward_batch(streams, batch, cache, input_ids)
            if return_streams:
                per_layer.append(streams)
        p = self.hc_head
        out = hc_head_cuda(streams, p.fn, p.scale, p.base, p.norm)[0]
        return (out, per_layer) if return_streams else out

    def forward(self, input_ids: Tensor, seq_lens: list[int], query_lens: list[int], caches):
        """Convenience form: builds the step's :class:`Dsv4Batch` on the host."""
        batch = Dsv4Batch.from_lens(
            seq_lens, query_lens, self.config.sliding_window, self.embed.device
        )
        return self.forward_batch(input_ids, batch, caches)
