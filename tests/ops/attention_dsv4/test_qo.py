# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 attention query / output side: CUDA bytes against the golden, batch invariance."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.cuda.deepgemm import deepgemm_available
from rl_engine.kernels.ops.pytorch.attention_dsv4 import inv_rope_fp8, kv_entry, q_norm_rope


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "dsv4_q_norm_rope_kv_insert")


pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA rl_engine._C with dsv4_qo required")

D, BS, ROW = 512, 64, 576 + 8


def _cos_sin(max_pos=1 << 15):
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2).double() / 64))
    ang = torch.arange(max_pos).double()[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.sin()], -1).float().cuda()


def _entry(cache, slot):
    block = cache[slot // BS]
    i = slot % BS
    row = block[i * 576 : (i + 1) * 576]
    return row[:448], block[BS * 576 + i * 8 : BS * 576 + (i + 1) * 8], row[448:]


@pytest.mark.parametrize("scale", [1.0, 1e-3, 300.0])
@pytest.mark.parametrize("heads,padded", [(64, 64), (40, 64)])
def test_q_norm_rope_kv_insert_matches_golden(scale, heads, padded):
    from rl_engine.kernels.ops.cuda.dsv4_attention import q_norm_rope_kv_insert

    gen = torch.Generator().manual_seed(int(scale * 10) + heads)
    t = 300
    q = (torch.randn(t, heads, D, generator=gen) * scale).to(torch.bfloat16).cuda()
    kv = (torch.randn(t, D, generator=gen) * scale).to(torch.bfloat16).cuda()
    pos = torch.randint(0, 1 << 15, (t,), generator=gen).cuda()
    cs = _cos_sin()
    cache = torch.zeros(8, BS * ROW, dtype=torch.uint8, device="cuda")
    slots = torch.randperm(8 * BS, generator=gen)[:t].cuda()
    slots[::7] = -1
    q_out = q_norm_rope_kv_insert(q, kv, pos, cs, cache, BS, slots, padded, 1e-6)
    want = q_norm_rope(q, pos, cs, 1e-6, padded)
    assert torch.equal(q_out.view(torch.int16), want.view(torch.int16))
    fq, fs, fr = kv_entry(kv, pos, cs)
    for i in range(t):
        s = int(slots[i])
        if s < 0:
            continue
        gq, gs, gr = _entry(cache, s)
        assert torch.equal(gq, fq[i].view(torch.uint8)), i
        assert torch.equal(gs, fs[i]), i
        assert torch.equal(gr, fr[i].view(torch.uint8)), i


@pytest.mark.parametrize("scale", [1.0, 1e-4, 500.0])
def test_inv_rope_fp8_matches_golden(scale):
    from rl_engine.kernels.ops.cuda import dsv4_attention as core

    gen = torch.Generator().manual_seed(3)
    t = 257
    o = (torch.randn(t, 64, D, generator=gen) * scale).to(torch.bfloat16).cuda()
    pos = torch.randint(0, 1 << 15, (t,), generator=gen).cuda()
    cs = _cos_sin()
    a, sc = core.inv_rope_fp8(o, pos, cs, 8)
    wa, wsc = inv_rope_fp8(o, pos, cs, 8)
    assert torch.equal(a.view(torch.uint8), wa.view(torch.uint8))
    assert torch.equal(sc, wsc)
    assert sc.stride(1) == 1  # MN-major for DeepGEMM


@pytest.mark.skipif(not deepgemm_available(), reason="pinned DeepGEMM required")
def test_output_projection_composition_and_batch_invariance():
    import deep_gemm

    from rl_engine.kernels.ops.cuda.dsv4_attention import OutputProjection
    from rl_engine.kernels.ops.cuda.fp8_linear import Fp8Linear, quantize_weight

    gen = torch.Generator().manual_seed(5)
    groups, rank, hidden = 8, 1024, 4096
    wo_a_bf16 = (torch.randn(groups * rank, 8 * D, generator=gen) / 64).to(torch.bfloat16).cuda()
    wo_b_bf16 = (torch.randn(hidden, groups * rank, generator=gen) / 90).to(torch.bfloat16).cuda()
    proj = OutputProjection(quantize_weight(wo_a_bf16), Fp8Linear.from_bf16(wo_b_bf16), groups)
    t = 300
    o = torch.randn(t, 64, D, generator=gen).to(torch.bfloat16).cuda()
    pos = torch.randint(0, 1 << 15, (t,), generator=gen).cuda()
    cs = _cos_sin()
    y = proj(o, pos, cs)
    # Golden composition: golden A operands, per-group DeepGEMM, then wo_b.
    a, sc = inv_rope_fp8(o, pos, cs, groups)
    z = torch.empty(groups, t, rank, dtype=torch.bfloat16, device="cuda")
    for g in range(groups):
        sc_g = deep_gemm.get_mn_major_tma_aligned_tensor(sc[g].contiguous())
        deep_gemm.fp8_gemm_nt((a[g], sc_g), proj.wo_a[g].pair(), z[g], disable_ue8m0_cast=True)
    want = proj.wo_b(z.transpose(0, 1).reshape(t, -1))
    assert torch.equal(y.view(torch.int16), want.view(torch.int16))
    for lo, hi in ((0, 1), (5, 6), (100, 229), (299, 300)):
        sub = proj(o[lo:hi].contiguous(), pos[lo:hi].contiguous(), cs)
        assert torch.equal(sub.view(torch.int16), y[lo:hi].view(torch.int16)), (lo, hi)
    # Large T runs wo_a as per-group GEMMs instead of one fp8_einsum: same bytes.
    proj.EINSUM_MAX_TOKENS = 1
    try:
        for lo, hi in ((0, 300), (5, 6)):
            sub = proj(o[lo:hi].contiguous(), pos[lo:hi].contiguous(), cs)
            assert torch.equal(sub.view(torch.int16), y[lo:hi].view(torch.int16)), (lo, hi)
    finally:
        del proj.EINSUM_MAX_TOKENS
    # Accuracy against FP32 math on the dequantized weights.
    x = o.float()
    half = cs[pos][:, None, :32]
    sin = cs[pos][:, None, 32:]
    e, od = x[..., 448::2].clone(), x[..., 449::2].clone()
    x[..., 448::2], x[..., 449::2] = e * half + od * sin, od * half - e * sin
    xg = x.reshape(t, groups, 8 * D)
    wa = proj.wo_a
    zf = torch.stack([xg[:, g] @ wa[g].dequantize().float().t() for g in range(groups)], 1)
    ref = zf.reshape(t, -1) @ proj.wo_b.weight.dequantize().float().t()
    assert float((y.float() - ref).norm() / ref.norm()) < 0.08
