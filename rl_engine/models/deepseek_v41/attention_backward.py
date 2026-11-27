# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Backward of the DSv4.1 attention layer: ``dL/dhidden`` (base weights frozen).

DSv4's attention backward (``deepseek_v4.attention_backward``) with DSv4.1's changes:

- No per-head q-norm: the q path is the RoPE transpose only.
- BF16 linears (D12): ``Bf16Linear.backward_dx``; the MXFP8 rounding of their inputs, the
  FP8 KV round trip and the rounding of ``o`` before ``wo_a`` are straight-through.
- **KV sharing.** A layer's sparse attention reads its KV group's compressed entries, which
  only the group's KV source computes (from its own input). The source's
  :class:`Dsv41AttentionFunction` therefore returns a second output, the group handle
  ``[R, E, 512]`` (zeros; it carries no data, only the gradient of the compressed entries
  in request-local order), and every other layer of the group takes it as an input and
  returns its workspace's compressed-row gradient for it. Autograd sums the group's
  gradients before the source's backward, which adds its own and runs the compressor
  backward (ratio 2: pooling softmax over the pair; ratio 1: no gate) into the source's
  ``dL/dhidden``.
- The indexers' top-k ids and candidate blocks are selections: no gradient.

Full-prefill batches only (``seq_lens == query_lens``, fresh caches). Not deterministic:
cuDNN's sparse-attention backward accumulates with atomics (owner decision).
"""

from __future__ import annotations

import importlib

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.attention.dsa_backward import sparse_attention_backward
from rl_engine.models.deepseek_v4.attention_backward import _norm_rope_bwd


def _layout(attn, batch) -> tuple[int, int]:
    """``(compressed rows per request slab, slab rows)`` of the prefill workspace."""
    c = attn.c
    n_comp = c.max_model_len // c.ratio if c.ratio else 0
    return n_comp, n_comp + batch.max_gather


def attention_backward(attn, hidden: Tensor, batch, saved: dict, dy: Tensor):
    """``(dL/dhidden FP32 [T, H], {target: (dA, dB)}, d_entries FP32 [R, E, 512] or None)``.

    ``d_entries`` is the gradient of the group's compressed entries this layer read (its
    workspace's compressed rows, request-local order), for compressed layers.
    """
    from rl_engine.models.deepseek_v4.lora import lora_backward

    c, w = attn.c, attn.w
    lora = w.lora
    grads: dict = {}
    tokens = hidden.shape[0]
    positions = batch.positions
    d = c.head_dim

    # Output projection: wo_b, per-group wo_a, inverse-RoPE transpose.
    dy = dy.to(torch.bfloat16).contiguous()
    dz = w.o_proj.wo_b.backward_dx(dy)
    if "wo_b" in lora:
        dx_l, da, db = lora_backward(dy, saved["z"], lora["wo_b"])
        grads["wo_b"] = (da, db)
        dz = (dz.float() + dx_l).to(torch.bfloat16)
    rank = w.o_proj.o_lora_rank
    d_rot = torch.cat(
        [
            lin.backward_dx(dz[:, g * rank : (g + 1) * rank].contiguous())
            for g, lin in enumerate(w.o_proj.wo_a)
        ],
        dim=1,
    )
    out = saved["out"]
    padded = out.shape[1]
    d_out = torch.empty(out.shape, dtype=torch.bfloat16, device=out.device)
    if padded > c.heads:
        d_out[:, c.heads :].zero_()
    _norm_rope_bwd(
        d_rot.view(-1, d),
        d_out.view(-1, d),
        tokens * c.heads,
        positions=positions,
        heads=c.heads,
        cos_sin=attn.cos_sin,
        rope_sign=1,
        out_heads=padded,
    )

    d_qpad, d_ws = sparse_attention_backward(
        saved["q_pad"],
        saved["ws"],
        saved["idx"],
        saved["lens"],
        w.attn_sink,
        attn.scale,
        out,
        saved["lse"],
        d_out,
    )

    # q path: RoPE transpose (no per-head norm), wq_b, q_norm.
    src = None
    if padded != c.heads:
        r = torch.arange(tokens * c.heads, device=out.device)
        src = (r // c.heads) * padded + r % c.heads
    dq = torch.empty(tokens * c.heads, d, dtype=torch.bfloat16, device=out.device)
    _norm_rope_bwd(
        d_qpad.view(-1, d),
        dq,
        tokens * c.heads,
        positions=positions,
        heads=c.heads,
        cos_sin=attn.cos_sin,
        rope_sign=-1,
        src_rows=src,
    )
    dq = dq.view(tokens, -1)
    dqr = w.wq_b.backward_dx(dq)
    if "wq_b" in lora:
        dx_l, da, db = lora_backward(dq, saved["qr"], lora["wq_b"])
        grads["wq_b"] = (da, db)
        dqr = dqr.float() + dx_l
    qr_kv = saved["qr_kv"]
    q_rank = c.q_lora_rank
    d_raw = torch.empty(tokens, qr_kv.shape[1], dtype=torch.bfloat16, device=out.device)
    _norm_rope_bwd(
        dqr.contiguous(), d_raw[:, :q_rank], tokens, x=qr_kv[:, :q_rank], w=w.q_norm, eps=c.eps
    )

    # Window rows of the workspace -> tokens; RoPE transpose; kv_norm.
    n_comp, slab = _layout(attn, batch)
    req = batch.token_req.long()
    _norm_rope_bwd(
        d_ws.view(-1, d),
        d_raw[:, q_rank:],
        tokens,
        x=qr_kv[:, q_rank:],
        w=w.kv_norm,
        positions=positions,
        heads=1,
        cos_sin=attn.cos_sin,
        rope_sign=-1,
        eps=c.eps,
        src_rows=req * slab + n_comp + positions.long(),
    )
    dh = w.fused_wqa_wkv.backward_dx(d_raw).float()
    for name, d_part in (("wq_a", d_raw[:, :q_rank]), ("wkv", d_raw[:, q_rank:])):
        if name in lora:
            dx_l, da, db = lora_backward(d_part, hidden, lora[name])
            grads[name] = (da, db)
            dh = dh + dx_l

    d_entries = None
    if c.ratio:
        width = batch.entries[c.ratio][3]
        requests = batch.seq_lens.shape[0]
        e = torch.arange(max(width, 1), device=out.device)
        rows = torch.arange(requests, device=out.device)[:, None] * slab + e[None]
        d_entries = d_ws.view(-1, d)[rows.view(-1)].float().view(requests, -1, d)
        count = (batch.seq_lens.long() // c.ratio)[:, None]
        d_entries = d_entries * (e[None] < count)[..., None]
    return dh, grads, d_entries


def compressor_backward(attn, hidden: Tensor, batch, d_entries: Tensor) -> Tensor:
    """``dL/dhidden`` FP32 of a KV source through its compressor, from the group's summed
    entry gradients ``[R, E, 512]`` (``_C.dsv4_compressor_backward``, ratios 1 and 2)."""
    from rl_engine.kernels.ops.cuda.matmul.deepgemm import require_deepgemm

    c, w = attn.c, attn.w
    ratio, d = c.ratio, c.head_dim
    tokens = hidden.shape[0]
    out = torch.zeros(tokens, hidden.shape[1], dtype=torch.float32, device=hidden.device)
    width = d_entries.shape[1]
    meta, start = [], 0
    for b, length in enumerate(batch.seq_lens.tolist()):
        meta += [(b * width + e, start, e) for e in range(length // ratio)]
        start += length
    if not meta or not tokens:
        return out
    if ratio == 2:
        score = w.comp_wkv(hidden, torch.float32)  # [T, 2d]: kv | gate, as the forward
        ape = torch.zeros(2, d, dtype=torch.float32, device=hidden.device)
    else:  # the forward's BF16 wkv rows
        score = w.comp_wkv(hidden).float().contiguous()
        ape = torch.zeros(1, d, dtype=torch.float32, device=hidden.device)
    d_score = torch.zeros(score.shape, dtype=torch.bfloat16, device=hidden.device)
    importlib.import_module("rl_engine._C").dsv4_compressor_backward(
        score,
        ape,
        d_entries.reshape(-1, d).contiguous(),
        torch.tensor(meta, dtype=torch.int64).to(hidden.device, non_blocking=True),
        w.comp_norm.float().contiguous(),
        attn.cos_sin,
        ratio,
        float(c.eps),
        d_score,
    )
    dg, _ = require_deepgemm(hidden.device)
    dg.bf16_gemm_nn(d_score, w.comp_wkv.weight, out)
    return out


def _lora_order(attn) -> list[str]:
    from rl_engine.models.deepseek_v4.lora import TARGETS

    return [name for name in TARGETS if name in attn.w.lora]


class Dsv41AttentionFunction(torch.autograd.Function):
    """``(y, group handle) = attn.forward_batch(hidden)`` with the backward above.

    ``group_in``: the KV source's handle for a layer of its group (None otherwise). The
    second output is the handle for a KV source, an empty non-differentiable tensor for
    any other layer.
    """

    @staticmethod
    def forward(ctx, hidden, group_in, attn, batch, caches, shared, *lora_params):
        if batch.decode or not torch.equal(batch.seq_lens, batch.query_lens):
            raise ValueError("the attention backward needs seq_lens == query_lens (full prefill)")
        saved: dict = {}
        y = attn.forward_batch(hidden, batch, caches, shared, saved=saved)
        ctx.attn, ctx.batch, ctx.saved = attn, batch, saved
        ctx.save_for_backward(hidden)
        c = attn.c
        if c.is_kv_source:
            width = max(batch.entries[c.ratio][3], 1)
            handle = torch.zeros(
                batch.seq_lens.shape[0], width, c.head_dim, dtype=torch.float32, device=y.device
            )
        else:
            handle = torch.empty(0, device=y.device)
            ctx.mark_non_differentiable(handle)
        return y, handle

    @staticmethod
    def backward(ctx, dy, d_handle):
        (hidden,) = ctx.saved_tensors
        attn = ctx.attn
        dh, grads, d_entries = attention_backward(attn, hidden, ctx.batch, ctx.saved, dy)
        d_group_in = None
        if attn.c.is_kv_source:
            total = d_entries if d_handle is None else d_entries + d_handle
            dh = dh + compressor_backward(attn, hidden, ctx.batch, total)
        elif attn.c.ratio:
            d_group_in = d_entries
        ctx.saved = ctx.attn = ctx.batch = None
        lora_grads = []
        for name in _lora_order(attn):
            da, db = grads[name]
            ad = attn.w.lora[name]
            lora_grads += [da.to(ad.a.dtype), db.to(ad.b.dtype)]
        return (dh.to(hidden.dtype), d_group_in, None, None, None, None, *lora_grads)


def attention_train(attn, hidden: Tensor, batch, caches, shared: dict, group_in=None):
    """Differentiable attention layer: ``(y, group handle)``; dL/dhidden, the group's
    compressed-entry gradients and the LoRA adapters' gradients."""
    params = []
    for name in _lora_order(attn):
        ad = attn.w.lora[name]
        params += [ad.a, ad.b]
    return Dsv41AttentionFunction.apply(hidden, group_in, attn, batch, caches, shared, *params)
