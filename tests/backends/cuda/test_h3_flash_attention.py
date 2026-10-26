# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""h3-flash-attention-v1: FP64 accuracy, bitwise repeat, batch and head-shard invariance."""

from __future__ import annotations

import pytest
import torch

from rl_engine.backends.shared.triton.attention import h3_flash as hf

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None or not hf.HAS_TRITON,
    reason="needs NVIDIA CUDA and Triton",
)

H, D = hf.HEADS, hf.HEAD_DIM


def _qkv(b, s, seed=0, grad=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(b, s, H, D, device="cuda", dtype=torch.bfloat16, generator=g).requires_grad_(
            grad
        )
        for _ in range(3)
    ]


def _fp64(q, k, v, grad):
    q, k, v = (x.detach().double().requires_grad_() for x in (q, k, v))
    s = torch.einsum("bshd,bthd->bhst", q, k) * hf.SCALE
    out = torch.einsum("bhst,bthd->bshd", s.softmax(-1), v)
    out.backward(grad.double())
    return out, q.grad, k.grad, v.grad


def _run(q, k, v, grad):
    q, k, v = (x.detach().clone().requires_grad_() for x in (q, k, v))
    out = hf.H3FlashAttentionOp()(q, k, v)
    out.backward(grad)
    return out.detach(), q.grad, k.grad, v.grad


def _rel(a, ref):
    return ((a.double() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("s", [2, 12, 65, 200])
def test_accuracy_vs_fp64(s):
    q, k, v = _qkv(1, s)
    grad = torch.randn(1, s, H, D, device="cuda", dtype=torch.bfloat16)
    got = _run(q, k, v, grad)
    ref = _fp64(q, k, v, grad)
    # one BF16 rounding of each result is the floor: 2^-9 relative per element
    for name, a, r in zip(("out", "dq", "dk", "dv"), got, ref):
        assert _rel(a, r) < 4e-3, (name, _rel(a, r))


def test_bitwise_repeatable():
    q, k, v = _qkv(1, 130)
    grad = torch.randn(1, 130, H, D, device="cuda", dtype=torch.bfloat16)
    a, b = _run(q, k, v, grad), _run(q, k, v, grad)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_batch_invariance():
    q, k, v = _qkv(3, 97)
    grad = torch.randn(3, 97, H, D, device="cuda", dtype=torch.bfloat16)
    batched = _run(q, k, v, grad)
    for i in range(3):
        single = _run(q[i : i + 1], k[i : i + 1], v[i : i + 1], grad[i : i + 1])
        assert all(torch.equal(x[i : i + 1], y) for x, y in zip(batched, single))


def test_head_shard_invariance():
    q, k, v = _qkv(1, 77)
    grad = torch.randn(1, 77, H, D, device="cuda", dtype=torch.bfloat16)
    full = _run(q, k, v, grad)
    # emulate a head-sharded layout: run the same kernel on a contiguous 14-head slice
    # (head-independent kernel: shards must reproduce the same bytes padded back to 56 heads)
    sub = []
    for x in (q, k, v, grad):
        pad = torch.zeros_like(x)
        pad[:, :, :14] = x[:, :, :14]
        sub.append(pad)
    shard = _run(*sub)
    assert all(torch.equal(f[:, :, :14], s_[:, :, :14]) for f, s_ in zip(full, shard))


def test_fail_closed():
    q, k, v = _qkv(1, 8)
    with pytest.raises(ValueError):
        hf.H3FlashAttentionOp()(q.float(), k.float(), v.float())
    with pytest.raises(ValueError):
        hf.H3FlashAttentionOp()(
            q[:, :, :8].contiguous(), k[:, :, :8].contiguous(), v[:, :, :8].contiguous()
        )


def test_delta_uses_fp32_output():
    # V rows 1 and 1.0078125 make BF16-rounded O differ from FP32 O; with Q=0 and
    # K=1 every score ties, so exact dQ is zero while a BF16 delta gives ~0.044.
    q = torch.zeros(1, 2, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.ones_like(q)
    v = torch.ones_like(q)
    v[:, 1] = 1.0078125
    grad = torch.ones_like(q)
    _, dq, _, _ = _run(q, k, v, grad)
    _, ref_dq, _, _ = _fp64(q, k, v, grad)
    assert ref_dq.abs().max().item() < 1e-6
    assert dq.abs().max().item() < 1e-3, dq.abs().max().item()


def test_rejects_int32_offset_overflow():
    seq = 2**31 // (H * D) + 1  # first length whose BF16 tensor reaches 2**31 elements
    q = torch.empty(1, seq, H, D, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        hf._validate(q, q, q)


def test_backward_validates_aux_tensors():
    q, k, v = _qkv(1, 4, grad=False)
    out, out32, lse = hf.flash_forward(q, k, v)
    grad = torch.ones_like(q)
    with pytest.raises(ValueError):
        hf.flash_backward(grad, q, k, v, out32.bfloat16(), lse)
    with pytest.raises(ValueError):
        hf.flash_backward(grad, q, k, v, out32, lse[:, :, :2].contiguous())
