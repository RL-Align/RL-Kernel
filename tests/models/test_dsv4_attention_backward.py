# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 attention layer backward (dL/dhidden) against an FP32 autograd reference.

The reference rebuilds the layer forward in FP32 PyTorch from the dequantized weights
(no FP8 rounding anywhere, matching the straight-through backward), over the same
workspace layout and the forward's index lists (the indexer's selection is a constant),
and differentiates it with autograd.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.cuda.deepgemm import deepgemm_available
from rl_engine.kernels.ops.cuda.flashmla import flashmla_available


def _ready() -> bool:
    if not (torch.cuda.is_available() and deepgemm_available() and flashmla_available()):
        return False
    from rl_engine.kernels.ops.cuda.dsa_backward import cudnn_dsa_available

    return cudnn_dsa_available()


pytestmark = pytest.mark.skipif(
    not _ready(), reason="SM90 + pinned DeepGEMM, FlashMLA and cudnn-frontend"
)

MAX_LEN = 4096


def _cos_sin():
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2).double() / 64))
    ang = torch.arange(MAX_LEN).double()[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.sin()], -1).float().cuda()


def _rms(x, w, eps):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if w is None else y * w.float()


def _rope(x, cs, pos, sign=1.0):
    c, s = cs[pos][..., :32], cs[pos][..., 32:]
    while c.dim() < x[..., 448::2].dim():
        c, s = c.unsqueeze(-2), s.unsqueeze(-2)
    x0, x1 = x[..., 448::2], x[..., 449::2]
    r0, r1 = x0 * c - sign * x1 * s, x1 * c + sign * x0 * s
    rot = torch.stack([r0, r1], -1).flatten(-2)
    return torch.cat([x[..., :448], rot], -1)


def _reference(attn, h, batch, idx, lens, dy):
    c, w = attn.c, attn.w
    cs = attn.cos_sin
    pos = batch.positions.long()
    x = h.float().requires_grad_(True)
    qr_kv = x @ w.fused_wqa_wkv.weight.dequantize().float().t()
    qr = _rms(qr_kv[:, : c.q_lora_rank], w.q_norm, c.eps)
    kvn = _rms(qr_kv[:, c.q_lora_rank :], w.kv_norm, c.eps)
    q = (qr @ w.wq_b.weight.dequantize().float().t()).view(-1, c.heads, c.head_dim)
    q_pad = _rope(_rms(q, None, c.eps), cs, pos)
    kv_r = _rope(kvn, cs, pos)

    ratio = c.compress_ratio
    n_comp = c.max_model_len // ratio if ratio > 1 else 0
    slab = n_comp + batch.max_gather
    rows = batch.token_req.long() * slab + n_comp + pos
    total = batch.seq_lens.shape[0] * slab
    ws = torch.zeros(total, c.head_dim, device="cuda").index_put((rows,), kv_r)
    if ratio > 1:
        coff, d = c.coff, c.head_dim
        width = coff * d
        score = x @ w.wkv_wgate.float().t()
        kv_all, gate_all = score[:, :width], score[:, width:] + w.ape[pos % ratio]
        start = 0
        for b, length in enumerate(batch.seq_lens.tolist()):
            for e in range(length // ratio):
                boundary = e * ratio + ratio - 1
                parts_kv, parts_s = [], []
                for p in range(boundary - coff * ratio + 1, boundary + 1):
                    if p < 0:
                        continue
                    half = 1 if (coff == 2 and p > boundary - ratio) else 0
                    sl = slice(half * d, half * d + d)
                    parts_kv.append(kv_all[start + p, sl])
                    parts_s.append(gate_all[start + p, sl])
                kvw, sw = torch.stack(parts_kv), torch.stack(parts_s)
                pooled = (torch.softmax(sw, 0) * kvw).sum(0)
                n = _rms(pooled, w.comp_norm, c.eps)
                entry = _rope(n[None], cs, torch.tensor([e * ratio], device="cuda"))[0]
                ws = ws.index_put((torch.tensor([b * slab + e], device="cuda"),), entry[None])
            start += length

    ids = idx.view(idx.shape[0], -1).long()
    k = ws[ids.clamp(min=0)]
    valid = (ids >= 0) & (torch.arange(ids.shape[1], device="cuda")[None] < lens[:, None])
    s = torch.einsum("thd,twd->thw", q_pad, k) * attn.scale
    s = s.masked_fill(~valid[:, None], float("-inf"))
    lse = torch.logsumexp(s, -1)
    o = torch.einsum("thw,twd->thd", torch.softmax(s, -1), k)
    o = o * torch.sigmoid(lse - w.attn_sink[: c.heads])[..., None]
    o_rot = _rope(o, cs, pos, sign=-1.0)
    proj = w.o_proj
    groups = proj.groups
    a = o_rot.reshape(o.shape[0], groups, -1)
    z = torch.cat([a[:, g] @ proj.wo_a[g].dequantize().float().t() for g in range(groups)], dim=1)
    y = z @ proj.wo_b.weight.dequantize().float().t()
    (y * dy.float()).sum().backward()
    return y.detach(), x.grad


def _layer(ratio):
    from rl_engine.models.deepseek_v4.attention import (
        Dsv4Attention,
        Dsv4AttentionConfig,
        random_layer,
    )

    config = Dsv4AttentionConfig(compress_ratio=ratio, max_model_len=MAX_LEN)
    return Dsv4Attention(config, random_layer(config, seed=ratio + 3), _cos_sin())


@pytest.mark.parametrize("ratio", [1, 4, 128])
def test_attention_backward_matches_reference(ratio):
    from rl_engine.models.deepseek_v4.attention import Dsv4Batch, new_caches
    from rl_engine.models.deepseek_v4.attention_backward import attention_train

    attn = _layer(ratio)
    lens = [150, 300] if ratio == 128 else [37, 200]
    g = torch.Generator(device="cuda").manual_seed(ratio)
    h = (torch.randn(sum(lens), attn.c.hidden, generator=g, device="cuda")).bfloat16()
    dy = (torch.randn(sum(lens), attn.c.hidden, generator=g, device="cuda") * 0.1).bfloat16()
    batch = Dsv4Batch.from_lens(lens, lens, attn.c.window)

    x = h.clone().requires_grad_(True)
    y = attention_train(attn, x, batch, new_caches(attn.c, len(lens), MAX_LEN, seed=1))
    y.backward(dy)
    dh = x.grad
    assert dh.dtype == torch.bfloat16 and torch.isfinite(dh.float()).all()

    # The forward is the BI layer itself.
    y_plain = attn.forward_batch(h, batch, new_caches(attn.c, len(lens), MAX_LEN, seed=2))
    assert torch.equal(y.view(torch.int16), y_plain.view(torch.int16))

    saved: dict = {}
    attn.forward_batch(h, batch, new_caches(attn.c, len(lens), MAX_LEN, seed=3), saved=saved)
    y_ref, dh_ref = _reference(attn, h, batch, saved["idx"], saved["lens"], dy)
    rel_y = float((y.float() - y_ref).norm() / y_ref.norm())
    rel = float((dh.float() - dh_ref).norm() / dh_ref.norm())
    cos = float(torch.nn.functional.cosine_similarity(dh.float().flatten(), dh_ref.flatten(), 0))
    print(f"ratio {ratio}: forward rel {rel_y:.3e}; dh rel {rel:.3e}, cos {cos:.6f}")
    assert rel_y < 0.08
    assert rel < 0.1 and cos > 0.995, (rel, cos)


def test_small_backward_pieces_match_autograd():
    from rl_engine.models.deepseek_v4.attention_backward import _cos_sin as bwd_cos_sin
    from rl_engine.models.deepseek_v4.attention_backward import (
        inverse_rope_transpose,
        rms_norm_backward,
        rope_transpose,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(7, 3, 512, generator=g, device="cuda", dtype=torch.float64)
    w = torch.randn(512, generator=g, device="cuda", dtype=torch.float64)
    dy = torch.randn(7, 3, 512, generator=g, device="cuda", dtype=torch.float64)
    for weight in (None, w):
        xr = x.clone().requires_grad_(True)
        _rms(xr, weight, 1e-6).backward(dy)
        got = rms_norm_backward(
            x.float(), dy.float(), None if weight is None else weight.float(), 1e-6
        )
        assert torch.allclose(got.double(), xr.grad, rtol=1e-4, atol=1e-5)

    _cos_sin_table = _cos_sin()
    cs = _cos_sin_table
    pos = torch.randint(0, MAX_LEN, (7,), generator=g, device="cuda")
    cos, sin = bwd_cos_sin(cs, pos)
    xf, dyf = x.float(), dy.float()
    for sign, fn in ((1.0, rope_transpose), (-1.0, inverse_rope_transpose)):
        xr = xf.clone().requires_grad_(True)
        _rope(xr, _cos_sin_table, pos, sign).backward(dyf)
        assert torch.allclose(fn(dyf, cos, sin), xr.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressor_backward_matches_autograd(ratio):
    """The compressor alone: entries' gradients -> dL/dhidden."""
    from rl_engine.models.deepseek_v4.attention import Dsv4Batch
    from rl_engine.models.deepseek_v4.attention_backward import _compressor_backward

    attn = _layer(ratio)
    c, w = attn.c, attn.w
    lens = [300, 260]
    batch = Dsv4Batch.from_lens(lens, lens, c.window)
    g = torch.Generator(device="cuda").manual_seed(5)
    h = torch.randn(sum(lens), c.hidden, generator=g, device="cuda").bfloat16()
    d_entries = [
        torch.randn(n // ratio, c.head_dim, generator=g, device="cuda").bfloat16() for n in lens
    ]
    got = _compressor_backward(attn, h, batch, d_entries)

    x = h.float().requires_grad_(True)
    pos = batch.positions.long()
    coff, d = c.coff, c.head_dim
    width = coff * d
    score = x @ w.wkv_wgate.float().t()
    kv_all, gate_all = score[:, :width], score[:, width:] + w.ape[pos % ratio]
    loss = 0.0
    start = 0
    for b, length in enumerate(lens):
        for e in range(length // ratio):
            boundary = e * ratio + ratio - 1
            kvs, ss = [], []
            for p in range(boundary - coff * ratio + 1, boundary + 1):
                if p < 0:
                    continue
                half = 1 if (coff == 2 and p > boundary - ratio) else 0
                kvs.append(kv_all[start + p, half * d : half * d + d])
                ss.append(gate_all[start + p, half * d : half * d + d])
            pooled = (torch.softmax(torch.stack(ss), 0) * torch.stack(kvs)).sum(0)
            n = _rms(pooled, w.comp_norm, c.eps)
            entry = _rope(n[None], attn.cos_sin, torch.tensor([e * ratio], device="cuda"))[0]
            loss = loss + (entry * d_entries[b][e].float()).sum()
        start += length
    loss.backward()
    rel = float((got - x.grad).norm() / x.grad.norm())
    assert rel < 2e-2, rel
