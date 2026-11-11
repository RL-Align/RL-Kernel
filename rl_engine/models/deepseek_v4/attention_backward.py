# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Backward of the DSv4 attention layer: ``dL/dhidden`` (base weights frozen).

:class:`Dsv4AttentionFunction` runs the forward with :meth:`Dsv4Attention.forward_batch`,
so the output bytes equal rollout and recompute. Its backward is written out here,
consuming the forward's saved intermediates. It covers full-sequence prefill batches
only (``seq_lens == query_lens``, fresh caches), which is what training runs.

Straight-through at every FP8 point:
- the KV cache round trip (``fp8_ds_mla``);
- ``o`` before ``wo_a``;
- the activations of each FP8 linear (their dgrads are DeepGEMM FP8, ``Fp8Linear.backward_dx``).
The indexer's selection, ``attn_sink``, ``ape``, the norm weights and the base projections
are constants. LoRA adapters on ``wq_a``, ``wkv``, ``wq_b`` and ``wo_b`` (``lora.py``) get
their ``A`` / ``B`` gradients, and their dX joins each projection's dgrad.

Backward order, for dy -> dh:

    wo_b dX; per-group wo_a dX; inverse-RoPE transpose        -> d out
    sparse attention (cuDNN dsa_bwd_sm90)                      -> d q_pad, d workspace
    q: RoPE transpose, weightless per-head RMSNorm, wq_b dX, q_norm  -> d qr
    workspace window rows -> tokens; RoPE transpose; kv_norm         -> d kv
    workspace compressed rows -> entries; RoPE transpose; comp_norm;
        gated softmax pooling; [d kv | d score] @ wkv_wgate (BF16) -> d h (compressor)
    [d qr | d kv] through fused_wqa_wkv dX                            -> d h

**Not deterministic yet:** cuDNN accumulates d workspace with FP32 atomics
(``rl_engine/kernels/ops/cuda/dsa_backward.py``). Everything else here is gathers and
row-local FP32 math, so it is deterministic. The small steps run in PyTorch for now.
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_engine.kernels.ops.cuda.dsa_backward import sparse_attention_backward
from rl_engine.kernels.ops.cuda.fp8_linear import Fp8Linear

NOPE, ROPE = 448, 64


def _cos_sin(cos_sin: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
    cs = cos_sin[positions.long()]
    return cs[:, : ROPE // 2], cs[:, ROPE // 2 :]


def _rotate(x: Tensor, cos: Tensor, sin: Tensor, sign: float) -> Tensor:
    """GPT-J pairs on the last 64 dims, in place on ``x`` (FP32):
    ``(x0 c - sign x1 s, x1 c + sign x0 s)``."""
    pairs = x[..., NOPE:].unflatten(-1, (ROPE // 2, 2))
    while cos.dim() < pairs.dim() - 1:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    x0, x1 = pairs[..., 0].clone(), pairs[..., 1].clone()
    pairs[..., 0] = x0 * cos - sign * x1 * sin
    pairs[..., 1] = x1 * cos + sign * x0 * sin
    return x


def rope_transpose(dy: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Gradient through RoPE (rotation by +theta): rotate by -theta. FP32 copy of ``dy``."""
    return _rotate(dy.float().clone(), cos, sin, -1.0)


def inverse_rope_transpose(dy: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Gradient through the inverse RoPE (rotation by -theta): rotate by +theta."""
    return _rotate(dy.float().clone(), cos, sin, 1.0)


def rms_norm_backward(x: Tensor, dy: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """``dL/dx`` FP32 of ``y = x * rsqrt(mean(x^2) + eps) [* w]`` over the last dim."""
    x = x.float()
    g = dy.float() if weight is None else dy.float() * weight.float()
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    xh = x * r
    return r * (g - xh * (g * xh).mean(-1, keepdim=True))


def _wo_a_linears(o_proj) -> list[Fp8Linear]:
    lin = getattr(o_proj, "_wo_a_linears", None)
    if lin is None:
        lin = [Fp8Linear(w) for w in o_proj.wo_a]
        o_proj._wo_a_linears = lin
    return lin


def _compressor_backward(attn, hidden: Tensor, batch, d_entries: list[Tensor]) -> Tensor:
    """``dL/dhidden`` FP32 through the main-KV compressor from each request's entry grads."""
    from rl_engine.models.deepseek_v4.attention import _bf16_gemm

    c, w = attn.c, attn.w
    ratio, d = c.compress_ratio, c.head_dim
    coff = c.coff
    width = coff * d
    score = _bf16_gemm(hidden, w.wkv_wgate, torch.float32)  # [T, 2 * width]
    kv_all, gate_all = score[:, :width], score[:, width:]
    pos_all = batch.positions.long()
    gate_all = gate_all + w.ape[pos_all % ratio]
    d_score = torch.zeros_like(score)
    span = coff * ratio  # tokens per entry window
    start = 0
    for b, length in enumerate(batch.seq_lens.tolist()):
        n_entries = length // ratio
        if n_entries == 0:
            start += length
            continue
        e = torch.arange(n_entries, device=hidden.device)
        slot = torch.arange(span, device=hidden.device)
        tok = e[:, None] * ratio + ratio - span + slot[None]  # positions in the window
        valid = tok >= 0
        tok_c = tok.clamp(min=0) + start
        half = (slot >= ratio).long() if coff == 2 else torch.zeros_like(slot)
        cols = half[:, None] * d + torch.arange(d, device=hidden.device)[None]  # [span, d]
        kvw = kv_all[tok_c[:, :, None], cols[None]]  # [E, span, d]
        sw = gate_all[tok_c[:, :, None], cols[None]].masked_fill(~valid[:, :, None], float("-inf"))
        p = torch.softmax(sw, dim=1)
        x = (p * kvw).sum(1)  # [E, d]
        cos, sin = _cos_sin(attn.cos_sin, e * ratio)
        dn = rope_transpose(d_entries[b].float(), cos, sin)
        dx = rms_norm_backward(x, dn, w.comp_norm, c.eps)
        dkvw = p * dx[:, None]
        dsw = p * (kvw - x[:, None]) * dx[:, None]
        # Each (token, half) belongs to exactly one (entry, slot) window position, so the
        # gradient is a gather: token position q, half 1 (or the single half) from entry
        # q // ratio at slot span - ratio + q % ratio; with overlap, half 0 from entry
        # q // ratio + 1 at slot q % ratio.
        qpos = torch.arange(length, device=hidden.device)
        for half_id, (ent, slot_i) in enumerate(
            [
                (qpos // ratio, span - ratio + qpos % ratio),
                (qpos // ratio + 1, qpos % ratio),
            ][:coff]
        ):
            ok = ent < n_entries
            ent_c, rows = ent.clamp(max=n_entries - 1), start + qpos
            half = coff - 1 - half_id if coff == 2 else 0
            sl = slice(half * d, half * d + d)
            g_kv = dkvw[ent_c, slot_i] * ok[:, None]
            g_s = dsw[ent_c, slot_i] * ok[:, None]
            d_score[rows, sl] = g_kv
            d_score[rows, width + half * d : width + half * d + d] = g_s
        start += length
    out = torch.empty(hidden.shape[0], hidden.shape[1], dtype=torch.float32, device=hidden.device)
    if hidden.shape[0]:
        from rl_engine.kernels.ops.cuda.deepgemm import require_deepgemm

        dg, _ = require_deepgemm(hidden.device)
        dg.bf16_gemm_nn(d_score.bfloat16().contiguous(), w.wkv_wgate.contiguous(), out)
    return out


def attention_backward(attn, hidden: Tensor, batch, saved: dict, dy: Tensor):
    """``(dL/dhidden BF16 [T, hidden], {target: (dA, dB) FP32})`` from ``dy`` BF16.

    The second item holds the LoRA adapters' gradients (``lora.py``), one entry per
    adapter in ``attn.w.lora``.
    """
    from rl_engine.models.deepseek_v4.lora import lora_backward

    c, w = attn.c, attn.w
    lora = w.lora
    grads: dict = {}
    tokens = hidden.shape[0]
    positions = batch.positions
    cos, sin = _cos_sin(attn.cos_sin, positions)

    # Output projection.
    dy = dy.to(torch.bfloat16).contiguous()
    dz = w.o_proj.wo_b.backward_dx(dy)  # [T, G * rank]
    if "wo_b" in lora:
        dx_l, da, db = lora_backward(dy, saved["z"], lora["wo_b"])
        grads["wo_b"] = (da, db)
        dz = (dz.float() + dx_l).to(torch.bfloat16)
    rank = w.o_proj.o_lora_rank
    d_rot = torch.cat(
        [
            lin.backward_dx(dz[:, g * rank : (g + 1) * rank].contiguous())
            for g, lin in enumerate(_wo_a_linears(w.o_proj))
        ],
        dim=1,
    ).view(tokens, c.heads, c.head_dim)
    d_out = inverse_rope_transpose(d_rot.float(), cos, sin)
    out = saved["out"]
    if out.shape[1] > c.heads:  # padded heads carry no gradient
        d_full = torch.zeros(out.shape, dtype=torch.float32, device=out.device)
        d_full[:, : c.heads] = d_out
        d_out = d_full
    d_out = d_out.to(torch.bfloat16)

    # Sparse attention.
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

    # q path: per-head weightless RMSNorm + RoPE, wq_b, q_norm.
    dq_n = rope_transpose(d_qpad[:, : c.heads].float(), cos, sin)
    dq = rms_norm_backward(saved["q"].float(), dq_n, None, c.eps)
    dq = dq.reshape(tokens, -1)
    dqr = w.wq_b.backward_dx(dq.to(torch.bfloat16))
    if "wq_b" in lora:
        dx_l, da, db = lora_backward(dq, saved["qr"], lora["wq_b"])
        grads["wq_b"] = (da, db)
        dqr = dqr.float() + dx_l
    qr_kv = saved["qr_kv"]
    dqr_raw = rms_norm_backward(qr_kv[:, : c.q_lora_rank], dqr, w.q_norm, c.eps)

    # kv path: window rows of the workspace -> tokens; RoPE; kv_norm.
    ratio = c.compress_ratio
    n_comp = c.max_model_len // ratio if ratio > 1 else 0
    slab = n_comp + batch.max_gather
    req = batch.token_req.long()
    rows = req * slab + n_comp + positions.long()  # full prefill: gather starts at 0
    d_kv_rope = d_ws.index_select(0, rows).float()
    d_kv_n = rope_transpose(d_kv_rope, cos, sin)
    dkv_raw = rms_norm_backward(qr_kv[:, c.q_lora_rank :], d_kv_n, w.kv_norm, c.eps)

    dh = w.fused_wqa_wkv.backward_dx(torch.cat([dqr_raw, dkv_raw], dim=1).to(torch.bfloat16))
    dh = dh.float()
    for name, d_out in (("wq_a", dqr_raw), ("wkv", dkv_raw)):
        if name in lora:
            dx_l, da, db = lora_backward(d_out, hidden, lora[name])
            grads[name] = (da, db)
            dh = dh + dx_l

    # Compressor: compressed rows of the workspace -> entries.
    if ratio > 1:
        d_entries = []
        for b, length in enumerate(batch.seq_lens.tolist()):
            n = length // ratio
            d_entries.append(d_ws[b * slab : b * slab + n])
        dh = dh + _compressor_backward(attn, hidden, batch, d_entries)
    return dh.to(hidden.dtype), grads


class Dsv4AttentionFunction(torch.autograd.Function):
    """``y = attn.forward_batch(hidden)`` with the backward above (full-prefill batches).

    ``lora_params`` are the adapters' ``A`` / ``B`` tensors in ``_lora_order(attn)``
    order; they are inputs so autograd returns their gradients.
    """

    @staticmethod
    def forward(ctx, hidden, attn, batch, caches, *lora_params):
        if batch.decode or not torch.equal(batch.seq_lens, batch.query_lens):
            raise ValueError("the attention backward needs seq_lens == query_lens (full prefill)")
        saved: dict = {}
        need = any(ctx.needs_input_grad[i] for i in [0] + list(range(4, 4 + len(lora_params))))
        y = attn.forward_batch(hidden, batch, caches, saved=saved if need else None)
        if need:
            ctx.attn, ctx.batch, ctx.saved = attn, batch, saved
            ctx.save_for_backward(hidden)
        return y

    @staticmethod
    def backward(ctx, dy):
        (hidden,) = ctx.saved_tensors
        dh, grads = attention_backward(ctx.attn, hidden, ctx.batch, ctx.saved, dy)
        ctx.saved = None
        lora_grads = []
        for name in _lora_order(ctx.attn):
            da, db = grads[name]
            ad = ctx.attn.w.lora[name]
            lora_grads += [da.to(ad.a.dtype), db.to(ad.b.dtype)]
        return (dh if ctx.needs_input_grad[0] else None, None, None, None, *lora_grads)


def _lora_order(attn) -> list[str]:
    from rl_engine.models.deepseek_v4.lora import TARGETS

    return [name for name in TARGETS if name in attn.w.lora]


def attention_train(attn, hidden: Tensor, batch, caches) -> Tensor:
    """Differentiable attention layer for training: dL/dhidden and the LoRA adapters'
    gradients (their ``A`` / ``B`` tensors get ``.grad``); base weights are frozen."""
    params = []
    for name in _lora_order(attn):
        ad = attn.w.lora[name]
        params += [ad.a, ad.b]
    return Dsv4AttentionFunction.apply(hidden, attn, batch, caches, *params)
