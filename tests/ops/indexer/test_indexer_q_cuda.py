# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CUDA indexer_q_rope_fp8: byte-equal to the golden and batch-invariant."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.indexer import indexer_q_rope_fp8


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "indexer_q_rope_fp8_forward")


pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA rl_engine._C with indexer_q required")

SOFTMAX, HEAD = 128**-0.5, 64**-0.5
MAX_POS = 1 << 16


def _cuda(*args):
    from rl_engine.kernels.ops.cuda.indexer import indexer_q_rope_fp8_cuda

    return indexer_q_rope_fp8_cuda(*args)


def _cache(rope=64, dtype=torch.float32):
    inv = 1.0 / (10000 ** (torch.arange(0, rope, 2).double() / rope))
    ang = torch.arange(MAX_POS).double()[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.sin()], -1).to(dtype).cuda()


def _inputs(tokens, heads=64, scale=1.0, seed=0, pos_dtype=torch.int64):
    gen = torch.Generator().manual_seed(seed)
    q = (torch.randn(tokens, heads, 128, generator=gen) * scale).to(torch.bfloat16).cuda()
    pos = torch.randint(0, MAX_POS, (tokens,), generator=gen).to(pos_dtype).cuda()
    w = torch.randn(tokens, heads, generator=gen).to(torch.bfloat16).cuda()
    return q, pos, w


def _assert_same(got, want):
    assert torch.equal(got[0].view(torch.uint8), want[0].view(torch.uint8))
    assert torch.equal(got[1].view(torch.int32), want[1].view(torch.int32))


@pytest.mark.parametrize("scale", [1.0, 1e-3, 1e-6, 300.0, 3e4])
@pytest.mark.parametrize("cs_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pos_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("tokens", [333, 700])  # one head per group / four per group
def test_matches_golden(scale, cs_dtype, pos_dtype, tokens):
    q, pos, w = _inputs(tokens, scale=scale, seed=int(scale * 7) % 97, pos_dtype=pos_dtype)
    cache = _cache(dtype=cs_dtype)
    _assert_same(
        _cuda(q, pos, cache, w, SOFTMAX, HEAD), indexer_q_rope_fp8(q, pos, cache, w, SOFTMAX, HEAD)
    )


@pytest.mark.parametrize("rope", [16, 32, 64, 128])
@pytest.mark.parametrize("heads", [8, 3])  # paired heads / one head per group
def test_rope_widths(rope, heads):
    q, pos, w = _inputs(65, heads=heads, seed=rope)
    cache = _cache(rope=rope)
    _assert_same(_cuda(q, pos, cache, w, 1.0, 1.0), indexer_q_rope_fp8(q, pos, cache, w, 1.0, 1.0))


def test_mixed_magnitudes_and_infinities_outside_rope():
    q, pos, w = _inputs(64, seed=3)
    q[:, :, 0] = q[:, :, 0] * torch.logspace(-30, 30, 64, device="cuda").to(torch.bfloat16)[:, None]
    q[5, 3, 7] = float("inf")
    q[6, 4, 1] = -float("inf")
    q[7] = 0
    cache = _cache()
    got = _cuda(q, pos, cache, w, SOFTMAX, HEAD)
    want = indexer_q_rope_fp8(q, pos, cache, w, SOFTMAX, HEAD)
    # Rows whose inverse scale is -inf turn zeros into NaN; compare NaN as NaN.
    gq, wq = got[0].float(), want[0].float()
    assert torch.equal(gq.isnan(), wq.isnan())
    assert torch.equal(torch.nan_to_num(gq), torch.nan_to_num(wq))
    assert torch.equal(got[1], want[1])


def test_batch_invariance_and_strided_q():
    q, pos, w = _inputs(1000, seed=4)
    cache = _cache()
    full = _cuda(q, pos, cache, w, SOFTMAX, HEAD)
    for lo, hi in ((0, 1), (1, 2), (511, 640), (999, 1000)):
        _assert_same(
            _cuda(q[lo:hi], pos[lo:hi], cache, w[lo:hi], SOFTMAX, HEAD),
            (full[0][lo:hi], full[1][lo:hi]),
        )
    # q as a head-strided view (every other head of a wider projection output)
    wide = torch.randn(50, 128, 128, device="cuda").to(torch.bfloat16)
    sub, wsub = wide[:, ::2], w[:50]
    _assert_same(
        _cuda(sub, pos[:50], cache, wsub, SOFTMAX, HEAD),
        _cuda(sub.contiguous(), pos[:50], cache, wsub, SOFTMAX, HEAD),
    )


def test_empty_and_validation():
    q, pos, w = _inputs(0)
    out = _cuda(q, pos, _cache(), w, SOFTMAX, HEAD)
    assert out[0].shape == (0, 64, 128) and out[1].shape == (0, 64)
    q, pos, w = _inputs(2, heads=4)
    with pytest.raises(RuntimeError):
        _cuda(q.float(), pos, _cache(), w, 1.0, 1.0)
    with pytest.raises(RuntimeError):
        _cuda(q, pos, _cache(), w.float(), 1.0, 1.0)
    with pytest.raises(RuntimeError):
        _cuda(q[..., :64], pos, _cache(), w, 1.0, 1.0)
