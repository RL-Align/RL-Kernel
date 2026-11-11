# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Semantics of the indexer_q_rope_fp8 golden (vLLM CuTe-DSL IndexerQFp8Kernel)."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.indexer import indexer_q_rope_fp8
from rl_engine.kernels.ops.pytorch.indexer.indexer_q import _fma_f32

SOFTMAX, HEAD = 128**-0.5, 64**-0.5


def _cache(max_pos=64, rope=64):
    inv = 1.0 / (10000 ** (torch.arange(0, rope, 2).double() / rope))
    ang = torch.arange(max_pos).double()[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.sin()], -1).float()


def _f32(x):
    return torch.tensor([x], dtype=torch.float32)


def test_fma_is_correctly_rounded_on_a_double_rounding_midpoint():
    # a*b = 2^-24 - 2^-70; c + a*b sits 2^-70 below the FP32 midpoint between
    # 1+2^-23 and 1+2^-22, but rounds onto it in FP64. A correct fma rounds down.
    a, b, c = _f32(2.0**-24 * (1 + 2.0**-23)), _f32(1 - 2.0**-23), _f32(1 + 2.0**-23)
    assert _fma_f32(a, b, c).item() == 1 + 2.0**-23
    assert (a.double() * b.double() + c.double()).float().item() == 1 + 2.0**-22  # naive


def test_fma_matches_exact_rational_rounding_on_random_values():
    from fractions import Fraction

    gen = torch.Generator().manual_seed(0)
    a, b, c = (torch.randn(2000, generator=gen) * 4 for _ in range(3))
    got = _fma_f32(a, b, c)
    for i in range(0, 2000, 7):
        exact = Fraction(a[i].item()) * Fraction(b[i].item()) + Fraction(c[i].item())
        lo = torch.tensor(float(exact), dtype=torch.float32)  # candidate, then fix up
        cands = [
            torch.nextafter(lo, torch.tensor(-float("inf"))),
            lo,
            torch.nextafter(lo, torch.tensor(float("inf"))),
        ]
        best = min(
            cands, key=lambda t: (abs(Fraction(t.item()) - exact), t.view(torch.int32).item() & 1)
        )
        assert got[i].item() == best.item(), i


def test_position_zero_is_identity_rope_and_scale_is_a_power_of_two():
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(3, 4, 128, generator=gen).to(torch.bfloat16)
    w = torch.randn(3, 4, generator=gen).to(torch.bfloat16)
    q8, wo = indexer_q_rope_fp8(q, torch.zeros(3, dtype=torch.long), _cache(), w, SOFTMAX, HEAD)
    amax = q.float().abs().amax(-1)
    scale = 2.0 ** torch.ceil(torch.log2(amax * _f32(1 / 448.0)))
    assert torch.equal(q8.float(), (q.float() / scale[..., None]).to(torch.float8_e4m3fn).float())
    combined = _f32(SOFTMAX * HEAD)
    assert torch.equal(wo, (w.float() * combined) * scale)


def test_amax_floor_nan_and_saturation():
    q = torch.zeros(1, 3, 128)
    q[0, 1, :64] = 1e-7  # below the 1e-4 floor
    # amax = inf: e = 255, so scale = +inf and the inverse bits are -inf
    # (vLLM's bit math); values flip sign and saturate, zeros become NaN.
    q[0, 2, 0] = float("inf")
    q[0, 2, 1] = float("nan")
    q[0, 2, 2] = -2.0
    q = q.to(torch.bfloat16)
    w = torch.ones(1, 3, dtype=torch.bfloat16)
    q8, wo = indexer_q_rope_fp8(q, torch.zeros(1, dtype=torch.long), _cache(), w, 1.0, 1.0)
    # all-zero row: scale = 2^ceil(log2(1e-4 / 448))
    assert wo[0, 0].item() == 2.0**-22
    assert wo[0, 1].item() == 2.0**-22
    assert q8[0, 1, 0].float().item() == pytest.approx(1e-7 / 2.0**-22, rel=0.07)
    assert wo[0, 2].item() == float("inf")
    assert q8[0, 2, 0].float().item() == -448.0 and q8[0, 2, 2].float().item() == 448.0
    assert q8[0, 2, 3].float().isnan()


def test_rows_are_independent():
    gen = torch.Generator().manual_seed(2)
    q = torch.randn(9, 8, 128, generator=gen).to(torch.bfloat16)
    pos = torch.randint(0, 64, (9,), generator=gen)
    w = torch.randn(9, 8, generator=gen).to(torch.bfloat16)
    full = indexer_q_rope_fp8(q, pos, _cache(), w, SOFTMAX, HEAD)
    part = indexer_q_rope_fp8(q[3:5], pos[3:5], _cache(), w[3:5], SOFTMAX, HEAD)
    assert torch.equal(full[0][3:5].view(torch.uint8), part[0].view(torch.uint8))
    assert torch.equal(full[1][3:5], part[1])


def test_validation():
    q = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
    pos = torch.zeros(2, dtype=torch.long)
    w = torch.zeros(2, 4, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        indexer_q_rope_fp8(q.float(), pos, _cache(), w, 1.0, 1.0)
    with pytest.raises(TypeError):
        indexer_q_rope_fp8(q, pos, _cache(), w.float(), 1.0, 1.0)
    with pytest.raises(ValueError):
        indexer_q_rope_fp8(q, pos, _cache(rope=40), w, 1.0, 1.0)
