# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Backward of the DSv4 sparse attention core (cuDNN-frontend dsa_bwd_sm90) against an
FP32 autograd reference. dq is deterministic; dkv is not (FP32 atomics), so only its
values are checked."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.cuda.flashmla import flashmla_available


def _ready() -> bool:
    if not (torch.cuda.is_available() and flashmla_available()):
        return False
    from rl_engine.kernels.ops.cuda.dsa_backward import cudnn_dsa_available

    return cudnn_dsa_available()


pytestmark = pytest.mark.skipif(not _ready(), reason="SM90, pinned FlashMLA and cudnn-frontend")

H, D = 64, 512
SCALE = D**-0.5


def _case(t, s, w, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(t, H, D, generator=g, device="cuda").bfloat16()
    ws = torch.randn(s, 1, D, generator=g, device="cuda").bfloat16()
    lens = torch.randint(1, min(w, s) + 1, (t,), generator=g, device="cuda").int()
    idx = torch.full((t, 1, w), -1, dtype=torch.int32, device="cuda")
    for i in range(t):
        n = int(lens[i])
        idx[i, 0, :n] = torch.randperm(s, generator=g, device="cuda")[:n].int()
    sink = torch.randn(H, generator=g, device="cuda")
    dout = torch.randn(t, H, D, generator=g, device="cuda").bfloat16()
    return q, ws, idx, lens, sink, dout


def _reference(q, ws, idx, lens, sink, dout):
    qf = q.float().requires_grad_(True)
    kf = ws.float().view(-1, D).requires_grad_(True)
    ids = idx.view(q.shape[0], -1).long()
    k = kf[ids.clamp(min=0)]  # [T, W, D]
    s = torch.einsum("thd,twd->thw", qf, k) * SCALE
    valid = (ids >= 0) & (torch.arange(ids.shape[1], device="cuda")[None] < lens[:, None])
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    lse = torch.logsumexp(s, -1)
    o = torch.einsum("thw,twd->thd", torch.softmax(s, -1), k)
    o = o * torch.sigmoid(lse - sink)[..., None]
    (o * dout.float()).sum().backward()
    return o.detach(), lse.detach(), qf.grad, kf.grad


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("t,s,w", [(96, 1024, 256), (64, 300, 640)])
def test_backward_matches_reference(t, s, w):
    from rl_engine.kernels.ops.cuda import dsv4_attention as core
    from rl_engine.kernels.ops.cuda.dsa_backward import sparse_attention_backward

    q, ws, idx, lens, sink, dout = _case(t, s, w, seed=t + s)
    out, lse = core.sparse_attention(q, ws, idx, lens, sink, SCALE)
    o_ref, lse_ref, dq_ref, dws_ref = _reference(q, ws, idx, lens, sink, dout)
    assert _rel(out, o_ref) < 1e-2
    assert torch.allclose(lse.float(), lse_ref, atol=2e-3, rtol=0)

    dq, dws = sparse_attention_backward(q, ws, idx, lens, sink, SCALE, out, lse, dout)
    assert dq.shape == q.shape and dws.shape == (s, D)
    assert _rel(dq, dq_ref) < 2e-2, _rel(dq, dq_ref)
    assert _rel(dws, dws_ref) < 2e-2, _rel(dws, dws_ref)

    dq2, dws2 = sparse_attention_backward(q, ws, idx, lens, sink, SCALE, out, lse, dout)
    assert torch.equal(dq.view(torch.int16), dq2.view(torch.int16))  # dq is deterministic
    # dkv uses FP32 atomics: report how many entries differ between runs (not asserted).
    diff = int((dws.view(torch.int16) != dws2.view(torch.int16)).sum())
    print(f"t={t} s={s} w={w}: dkv entries differing run to run: {diff} of {dws.numel()}")
