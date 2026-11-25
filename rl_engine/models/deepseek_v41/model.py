# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DeepSeek-V4.1-Flash forward on the batch-invariant SM90 kernel set.

As DeepSeek's reference (``inference/model.py`` ``Transformer.forward`` / ``Block.forward``)::

    streams = embed[input_ids] copied to 4 streams                    BF16 [T, 4, H]
    pre     = None (the first sublayer mixes stream 0)
    per layer i:
        streams              = engram_i(streams)                      Engram layers only
        x, post, comb, a_pre = mhc_pre(streams, pre, hc_attn, attn_norm)   pre of the previous
        streams              = mhc_post(attention_i(x), streams, post, comb)   sublayer
        x, post, comb, pre   = mhc_pre(streams, a_pre, hc_ffn, ffn_norm)
        streams              = mhc_post(moe_i(x), streams, post, comb)
    hidden = norm(collapse(streams, pre))                             no learned hc_head

with mHC ``dsv41-mhc-v1`` (``eps = rms_norm_eps``, 1e-20 in the release), the attention of
``attention.py`` (KV / index sharing through ``shared``), DSv4's MoE with a BF16 shared
expert, and Engram (``engram.py``). The first layer's controller runs on the expanded
residual (four copies of the embedding).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.mhc import mhc_post_cuda
from rl_engine.kernels.ops.cuda.mhc.dsv41_mhc import hc_collapse_cuda, mhc_pre_delayed_cuda
from rl_engine.models.deepseek_v4.model import MhcParams
from rl_engine.models.deepseek_v4.moe import Dsv4MoE
from rl_engine.models.deepseek_v41.attention import (
    Dsv41Attention,
    Dsv41Batch,
    LayerCaches,
    new_group_caches,
)
from rl_engine.models.deepseek_v41.config import Dsv41Config
from rl_engine.models.deepseek_v41.engram import EngramWeights, NgramHasher, engram_forward


@dataclass
class Dsv41DecoderLayer:
    index: int
    attn: Dsv41Attention
    moe: Dsv4MoE
    hc_attn: MhcParams
    hc_ffn: MhcParams
    eps: float
    engram: EngramWeights | None = None

    def forward_batch(
        self,
        streams: Tensor,
        pre: Tensor | None,
        batch: Dsv41Batch,
        caches: LayerCaches,
        shared: dict,
        hashes: Tensor | None = None,
        engram_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """``streams`` BF16 ``[T, 4, H]`` and the carried pre ``[T, 4]`` (None at the model
        entry) -> the layer's output streams and the pre for the next layer."""
        if self.engram is not None:
            if hashes is None:
                raise ValueError(f"layer {self.index} has an Engram: pass engram windows")
            streams = engram_forward(self.engram, streams, hashes, engram_mask, self.eps)
        p = self.hc_attn
        x, post, comb, attn_pre = mhc_pre_delayed_cuda(
            streams, p.fn, p.scale, p.base, pre, p.norm, self.eps
        )[:4]
        streams = mhc_post_cuda(
            self.attn.forward_batch(x, batch, caches, shared), streams, post, comb
        )
        p = self.hc_ffn
        x, post, comb, ffn_pre = mhc_pre_delayed_cuda(
            streams, p.fn, p.scale, p.base, attn_pre, p.norm, self.eps
        )[:4]
        y = self.moe.forward(x, None, None)
        return mhc_post_cuda(y, streams, post, comb), ffn_pre


@dataclass
class Dsv41Model:
    config: Dsv41Config
    embed: Tensor  # BF16 [vocab, H]
    layers: list[Dsv41DecoderLayer]
    norm: Tensor  # BF16 [H], the final norm
    head: Tensor | None = None  # BF16 [vocab, H]
    max_model_len: int = 65536
    hasher: NgramHasher | None = None
    extra: dict = field(default_factory=dict)

    def new_caches(self, requests: int, max_len: int | None = None, seed: int = 0):
        """Paged caches for every layer; compressed layers alias their KV source's caches."""
        max_len = self.max_model_len if max_len is None else max_len
        caches: list[LayerCaches] = []
        for i, layer in enumerate(self.layers):
            role = self.config.layer_role(i)
            group = caches[role.kv_source] if role.ratio and role.kv_source != i else None
            caches.append(
                new_group_caches(
                    layer.attn.c, requests, max_len, self.embed.device, seed + i, group=group
                )
            )
        return caches

    def embed_streams(self, input_ids: Tensor) -> Tensor:
        h = self.embed.index_select(0, input_ids)
        return h.unsqueeze(1).expand(-1, self.config.hc_mult, -1).contiguous()

    def engram_hashes(self, windows: Tensor | None) -> Tensor | None:
        if not any(layer.engram is not None for layer in self.layers):
            return None
        if windows is None or self.hasher is None:
            raise ValueError("the model has Engram layers: pass engram_windows and a hasher")
        return self.hasher(windows)

    @torch.no_grad()
    def forward_batch(
        self,
        input_ids: Tensor,
        batch: Dsv41Batch,
        caches: list[LayerCaches],
        *,
        engram_windows: Tensor | None = None,
        engram_mask: Tensor | None = None,
        return_streams: bool = False,
    ):
        """Final hidden BF16 ``[T, H]`` (collapse by the last pre, then the final norm).

        ``input_ids`` int64 ``[T]`` in :class:`Dsv41Batch` order; ``engram_windows`` int64
        ``[T, max_ngram]`` (``engram.ngram_windows``) when the model has Engram layers.
        ``return_streams`` also returns each layer's output streams.
        """
        input_ids = input_ids.long()
        hashes = self.engram_hashes(engram_windows)
        streams = self.embed_streams(input_ids)
        pre: Tensor | None = None
        shared: dict = {}
        per_layer = []
        for layer, cache in zip(self.layers, caches):
            streams, pre = layer.forward_batch(
                streams, pre, batch, cache, shared, hashes, engram_mask
            )
            if return_streams:
                per_layer.append(streams)
        assert pre is not None
        out = hc_collapse_cuda(streams, pre, self.norm, self.config.rms_norm_eps)
        return (out, per_layer) if return_streams else out

    def lora_parameters(self) -> list[Tensor]:
        """Every adapter's ``A`` / ``B``, layer by layer (attention, then MoE)."""
        params: list[Tensor] = []
        for layer in self.layers:
            for ad in layer.attn.w.lora.values():
                params += [ad.a, ad.b]
            params += layer.moe.lora_params()
        return params

    def forward_train(
        self,
        input_ids: Tensor,
        batch: Dsv41Batch,
        caches: list[LayerCaches],
        *,
        engram_windows: Tensor | None = None,
        engram_mask: Tensor | None = None,
        return_streams: bool = False,
    ):
        """:meth:`forward_batch` under autograd, byte-equal to it (full-prefill batches).

        Gradients reach the LoRA adapters (``deepseek_v4.model.attach_lora`` works on this
        model); base weights and the embedding are frozen. A KV source's attention returns
        its group's handle and the other layers of the group take it, so their gradient on
        the shared compressed KV reaches the source (``attention_backward``).
        """
        from rl_engine.kernels.ops.cuda.mhc import mhc_post
        from rl_engine.kernels.ops.cuda.mhc.dsv41_mhc import hc_collapse, mhc_pre_delayed
        from rl_engine.models.deepseek_v41.attention_backward import attention_train
        from rl_engine.models.deepseek_v41.engram import engram_forward_train

        input_ids = input_ids.long()
        hashes = self.engram_hashes(engram_windows)
        eps = self.config.rms_norm_eps
        streams = self.embed_streams(input_ids)
        pre: Tensor | None = None
        shared: dict = {}
        handles: dict = {}
        per_layer = []
        for i, (layer, cache) in enumerate(zip(self.layers, caches)):
            role = self.config.layer_role(i)
            if layer.engram is not None:
                assert hashes is not None
                streams = engram_forward_train(layer.engram, streams, hashes, engram_mask, eps)
            p = layer.hc_attn
            x, post, comb, attn_pre = mhc_pre_delayed(
                streams, p.fn, p.scale, p.base, pre, p.norm, eps
            )
            group_in = handles.get(role.kv_source) if role.ratio and role.kv_source != i else None
            y, handle = attention_train(layer.attn, x, batch, cache, shared, group_in)
            if layer.attn.c.is_kv_source:
                handles[i] = handle
            streams = mhc_post(y, streams, post, comb)
            p = layer.hc_ffn
            x, post, comb, pre = mhc_pre_delayed(
                streams, p.fn, p.scale, p.base, attn_pre, p.norm, eps
            )
            streams = mhc_post(layer.moe.forward(x, None, None), streams, post, comb)
            if return_streams:
                per_layer.append(streams)
        assert pre is not None
        out = hc_collapse(streams, pre, self.norm, eps)
        return (out, per_layer) if return_streams else out
