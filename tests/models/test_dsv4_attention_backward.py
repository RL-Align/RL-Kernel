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


def _reference(attn, h, batch, idx, lens, dy, lora=None):
    c, w = attn.c, attn.w
    cs = attn.cos_sin
    pos = batch.positions.long()
    x = h.float().requires_grad_(True)
    lp = {
        k: (
            v.a.detach().float().requires_grad_(True),
            v.b.detach().float().requires_grad_(True),
            v.scale,
        )
        for k, v in (lora or {}).items()
    }

    def delta(name, inp):
        a, b, s = lp[name]
        return s * ((inp @ a.t()) @ b.t())

    qr_kv = x @ w.fused_wqa_wkv.weight.dequantize().float().t()
    qa, kvp = qr_kv[:, : c.q_lora_rank], qr_kv[:, c.q_lora_rank :]
    if "wq_a" in lp:
        qa = qa + delta("wq_a", x)
    if "wkv" in lp:
        kvp = kvp + delta("wkv", x)
    qr = _rms(qa, w.q_norm, c.eps)
    kvn = _rms(kvp, w.kv_norm, c.eps)
    q = qr @ w.wq_b.weight.dequantize().float().t()
    if "wq_b" in lp:
        q = q + delta("wq_b", qr)
    q = q.view(-1, c.heads, c.head_dim)
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
    if "wo_b" in lp:
        y = y + delta("wo_b", z)
    (y * dy.float()).sum().backward()
    if lora:
        return y.detach(), x.grad, {k: (v[0].grad, v[1].grad) for k, v in lp.items()}
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


@pytest.mark.parametrize("ratio", [4, 1])
def test_attention_lora_backward_matches_reference(ratio):
    """LoRA on wq_a, wkv, wq_b, wo_b: dL/dhidden and every adapter's dA / dB."""
    from rl_engine.models.deepseek_v4.attention import Dsv4Batch, new_caches
    from rl_engine.models.deepseek_v4.attention_backward import attention_train
    from rl_engine.models.deepseek_v4.lora import LoraAdapter

    attn = _layer(ratio)
    c = attn.c
    shapes = {
        "wq_a": (c.hidden, c.q_lora_rank),
        "wkv": (c.hidden, c.head_dim),
        "wq_b": (c.q_lora_rank, c.heads * c.head_dim),
        "wo_b": (c.o_groups * c.o_lora_rank, c.hidden),
    }
    attn.w.lora = {
        name: LoraAdapter.init(i, o, rank=16, alpha=32, seed=k, zero_b=False)
        for k, (name, (i, o)) in enumerate(shapes.items())
    }
    for ad in attn.w.lora.values():
        ad.a.requires_grad_(True)
        ad.b.requires_grad_(True)
    lens = [37, 200]
    g = torch.Generator(device="cuda").manual_seed(ratio + 10)
    h = torch.randn(sum(lens), c.hidden, generator=g, device="cuda").bfloat16()
    dy = (torch.randn(sum(lens), c.hidden, generator=g, device="cuda") * 0.1).bfloat16()
    batch = Dsv4Batch.from_lens(lens, lens, c.window)
    try:
        x = h.clone().requires_grad_(True)
        y = attention_train(attn, x, batch, new_caches(c, len(lens), MAX_LEN, seed=1))
        y.backward(dy)
        saved: dict = {}
        with torch.no_grad():
            attn.forward_batch(h, batch, new_caches(c, len(lens), MAX_LEN, seed=3), saved=saved)
        y_ref, dh_ref, g_ref = _reference(
            attn, h, batch, saved["idx"], saved["lens"], dy, lora=attn.w.lora
        )
    finally:
        lora = attn.w.lora
        attn.w.lora = {}

    def rel(a, b):
        return float((a.float() - b).norm() / b.norm())

    assert rel(y, y_ref) < 0.08
    assert rel(x.grad, dh_ref) < 0.1, rel(x.grad, dh_ref)
    for name, (da_ref, db_ref) in g_ref.items():
        ad = lora[name]
        assert ad.a.grad is not None and ad.b.grad is not None, name
        assert rel(ad.a.grad, da_ref) < 0.1, (name, rel(ad.a.grad, da_ref))
        assert rel(ad.b.grad, db_ref) < 0.1, (name, rel(ad.b.grad, db_ref))


def test_norm_rope_kernel_matches_torch():
    """The fused helper (csrc/cuda/attention/dsv4_attn_bwd.cu) against the PyTorch steps."""
    from rl_engine.models.deepseek_v4.attention_backward import _cos_sin as bwd_cos_sin
    from rl_engine.models.deepseek_v4.attention_backward import (
        _norm_rope_bwd,
        inverse_rope_transpose,
        rms_norm_backward,
        rope_transpose,
    )

    g = torch.Generator(device="cuda").manual_seed(3)
    cs = _cos_sin()
    t, heads = 37, 4
    pos = torch.randint(0, MAX_LEN, (t,), generator=g, device="cuda")
    cos, sin = bwd_cos_sin(cs, pos)
    dy = torch.randn(t, heads, 512, generator=g, device="cuda").bfloat16()
    x = torch.randn(t, heads, 512, generator=g, device="cuda").bfloat16()
    w = (1 + 0.1 * torch.randn(512, generator=g, device="cuda")).bfloat16()

    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm())

    # o side: inverse-RoPE transpose into a padded-head layout.
    out = torch.zeros(t, heads + 2, 512, dtype=torch.bfloat16, device="cuda")
    _norm_rope_bwd(dy.view(-1, 512), out.view(-1, 512), t * heads, positions=pos,
                   heads=heads, cos_sin=cs, rope_sign=1, out_heads=heads + 2)  # fmt: skip
    assert rel(out[:, :heads], inverse_rope_transpose(dy, cos, sin)) < 1e-2
    assert float(out[:, heads:].abs().max()) == 0
    # q side: RoPE transpose + weightless RMSNorm backward.
    out = torch.empty(t * heads, 512, dtype=torch.float32, device="cuda")
    _norm_rope_bwd(dy.view(-1, 512), out, t * heads, x=x.view(-1, 512), positions=pos,
                   heads=heads, cos_sin=cs, rope_sign=-1)  # fmt: skip
    want = rms_norm_backward(x.float(), rope_transpose(dy, cos, sin), None, 1e-6)
    assert rel(out.view(t, heads, 512), want) < 1e-4
    # kv side: gathered rows, RoPE transpose, weighted RMSNorm backward, strided output.
    src = torch.randperm(t * heads, generator=g, device="cuda")[:t]
    buf = torch.zeros(t, 1536, dtype=torch.float32, device="cuda")
    _norm_rope_bwd(dy.view(-1, 512), buf[:, 1024:], t, x=x[:, 0], w=w, positions=pos,
                   heads=1, cos_sin=cs, rope_sign=-1, src_rows=src)  # fmt: skip
    gathered = dy.view(-1, 512)[src]
    want = rms_norm_backward(x[:, 0].float(), rope_transpose(gathered, cos, sin), w, 1e-6)
    assert rel(buf[:, 1024:], want) < 1e-4 and float(buf[:, :1024].abs().max()) == 0
    # q_norm: D = 1024, no RoPE.
    dy2 = torch.randn(t, 1024, generator=g, device="cuda")
    x2 = torch.randn(t, 1024, generator=g, device="cuda").bfloat16()
    w2 = (1 + 0.1 * torch.randn(1024, generator=g, device="cuda")).bfloat16()
    out = torch.empty(t, 1024, dtype=torch.float32, device="cuda")
    _norm_rope_bwd(dy2, out, t, x=x2, w=w2)
    assert rel(out, rms_norm_backward(x2.float(), dy2, w2, 1e-6)) < 1e-4
