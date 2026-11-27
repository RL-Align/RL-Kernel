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

    def forward_train(
        self,
        streams: Tensor,
        pre: Tensor | None,
        batch: Dsv41Batch,
        caches: LayerCaches,
        shared: dict,
        hashes: Tensor | None = None,
        engram_mask: Tensor | None = None,
        group_in: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """:meth:`forward_batch` under autograd (same bytes): ``(streams, pre, handle)``;
        ``handle`` is the KV group's (a KV source) or an empty tensor, ``group_in`` the
        source's handle for the other layers of its group."""
        from rl_engine.kernels.ops.cuda.mhc import mhc_post
        from rl_engine.kernels.ops.cuda.mhc.dsv41_mhc import mhc_pre_delayed
        from rl_engine.models.deepseek_v41.attention_backward import attention_train
        from rl_engine.models.deepseek_v41.engram import engram_forward_train

        if self.engram is not None:
            assert hashes is not None
            streams = engram_forward_train(self.engram, streams, hashes, engram_mask, self.eps)
        p = self.hc_attn
        x, post, comb, attn_pre = mhc_pre_delayed(
            streams, p.fn, p.scale, p.base, pre, p.norm, self.eps
        )
        y, handle = attention_train(self.attn, x, batch, caches, shared, group_in)
        streams = mhc_post(y, streams, post, comb)
        p = self.hc_ffn
        x, post, comb, ffn_pre = mhc_pre_delayed(
            streams, p.fn, p.scale, p.base, attn_pre, p.norm, self.eps
        )
        return mhc_post(self.moe.forward(x, None, None), streams, post, comb), ffn_pre, handle

    def lora_parameters(self) -> list[Tensor]:
        """This layer's adapter tensors (``A`` and ``B``), attention first, then MoE."""
        params = []
        for ad in self.attn.w.lora.values():
            params += [ad.a, ad.b]
        return params + self.moe.lora_params()


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

    def selected_logprob(self, hidden: Tensor, targets: Tensor, temperature: float = 1.0):
        """``(logp, lse)`` FP32 ``[T]`` of ``targets`` under ``softmax(head(hidden) / T)``
        (DSv4's logprob kernel, ``dsv4-logprob-v1``); differentiable in ``hidden``."""
        from rl_engine.kernels.ops.cuda.logprob_dsv4 import selected_logprob

        if self.head is None:
            raise RuntimeError("the model was loaded without its LM head (load_head=False)")
        return selected_logprob(hidden, self.head, targets, temperature)

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
        recompute: bool = False,
    ):
        """:meth:`forward_batch` under autograd, byte-equal to it (full-prefill batches).

        Gradients reach the LoRA adapters (``deepseek_v4.model.attach_lora`` works on this
        model); base weights and the embedding are frozen. A KV source's attention returns
        its group's handle and the other layers of the group take it, so their gradient on
        the shared compressed KV reaches the source (``attention_backward``).

        ``recompute``: activation recompute per layer (:class:`_RecomputedLayer`): autograd
        keeps each layer's input streams and carried pre and reruns the layer in the
        backward. Same forward bytes, same gradients (up to the attention backward's atomics).
        """
        from rl_engine.kernels.ops.cuda.mhc.dsv41_mhc import hc_collapse

        input_ids = input_ids.long()
        hashes = self.engram_hashes(engram_windows)
        streams = self.embed_streams(input_ids)
        pre: Tensor | None = None
        shared: dict = {}
        handles: dict = {}
        per_layer = []
        for i, (layer, cache) in enumerate(zip(self.layers, caches)):
            role = self.config.layer_role(i)
            group_in = handles.get(role.kv_source) if role.ratio and role.kv_source != i else None
            args = (batch, cache, shared, hashes, engram_mask)
            if recompute:
                streams, pre, handle = _RecomputedLayer.apply(
                    streams, pre, group_in, layer, *args, *layer.lora_parameters()
                )
            else:
                streams, pre, handle = layer.forward_train(streams, pre, *args, group_in)
            if layer.attn.c.is_kv_source:
                handles[i] = handle
            if return_streams:
                per_layer.append(streams)
        assert pre is not None
        out = hc_collapse(streams, pre, self.norm, self.config.rms_norm_eps)
        return (out, per_layer) if return_streams else out


class _RecomputedLayer(torch.autograd.Function):
    """One decoder layer with activation recompute (DSv4's ``_RecomputedLayer`` with the
    carried pre and the KV group handle).

    The forward runs :meth:`Dsv41DecoderLayer.forward_batch` without autograd and keeps the
    layer's input streams and carried pre; a KV source also returns its group handle. The
    backward reruns :meth:`Dsv41DecoderLayer.forward_train` on them and backpropagates the
    gradients of all three outputs, the handle's being the group's summed compressed-entry
    gradient from the later layers. A layer of a group gets its source's handle as input and
    returns its gradient. The rerun reads what the first forward left: the group's caches
    and ``shared`` (top-k ids, candidates), which a rerun rewrites with the same bytes.
    """

    @staticmethod
    def forward(ctx, streams, pre, group_in, layer, batch, caches, shared, hashes, mask, *lora):
        ctx.layer, ctx.args = layer, (batch, caches, shared, hashes, mask)
        ctx.has_pre, ctx.has_group = pre is not None, group_in is not None
        ctx.save_for_backward(streams, pre, group_in)
        out, new_pre = layer.forward_batch(streams, pre, batch, caches, shared, hashes, mask)
        c = layer.attn.c
        if c.is_kv_source:
            width = max(batch.entries[c.ratio][3], 1)
            handle = torch.zeros(
                batch.seq_lens.shape[0], width, c.head_dim, dtype=torch.float32, device=out.device
            )
        else:
            handle = torch.empty(0, device=out.device)
            ctx.mark_non_differentiable(handle)
        return out, new_pre, handle

    @staticmethod
    def backward(ctx, d_streams, d_pre, d_handle):
        streams, pre, group_in = ctx.saved_tensors
        layer = ctx.layer
        x = streams.detach().requires_grad_(ctx.needs_input_grad[0])
        p = pre.detach().requires_grad_(ctx.needs_input_grad[1]) if ctx.has_pre else None
        g = group_in.detach().requires_grad_(True) if ctx.has_group else None
        params = layer.lora_parameters()
        wanted = [q for q, need in zip(params, ctx.needs_input_grad[9:]) if need]
        inputs = [t for t in (x, p, g) if t is not None and t.requires_grad] + wanted
        with torch.enable_grad():
            out, new_pre, handle = layer.forward_train(x, p, *ctx.args, g)
        outs, grads_out = [out, new_pre], [d_streams, d_pre]
        if layer.attn.c.is_kv_source:
            outs.append(handle)
            grads_out.append(d_handle)
        pairs = [(o, d) for o, d in zip(outs, grads_out) if d is not None]
        grads = torch.autograd.grad(
            [o for o, _ in pairs], inputs, [d for _, d in pairs], allow_unused=True
        )
        ctx.layer = ctx.args = None
        it = iter(grads)
        dx = next(it) if x.requires_grad else None
        dp = next(it) if p is not None and p.requires_grad else None
        dg = next(it) if g is not None else None
        dparams = [next(it) if need else None for need in ctx.needs_input_grad[9:]]
        return (dx, dp, dg, None, None, None, None, None, None, *dparams)
