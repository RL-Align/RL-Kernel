# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from rl_engine.models.inkling2.local_attention import local_attention

DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_packed_gqa_reference_and_backward(device):
    torch.manual_seed(14)
    q = torch.randn(6, 4, 5, device=device, requires_grad=True)
    k = torch.randn(6, 2, 5, device=device, requires_grad=True)
    v = torch.randn(6, 2, 5, device=device, requires_grad=True)
    bias = torch.randn(6, 4, 3, device=device, requires_grad=True)
    ids = torch.arange(6, device=device)
    docs = torch.tensor([0, 0, 0, 1, 1, 1], device=device)
    out, lse = local_attention(q, k, v, bias, ids, docs, window=2, query_chunk=2)
    lag = ids[:, None] - ids[None, :]
    mask = (lag >= 0) & (lag < 2) & (docs[:, None] == docs[None, :])
    scores = torch.einsum("qhd,khd->qhk", q, k.repeat_interleave(2, dim=1)) * (5**-0.5)
    scores = scores + bias.gather(-1, lag.clamp(0, 2)[:, None, :].expand(-1, 4, -1))
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    expected = torch.einsum("qhk,khd->qhd", scores.softmax(-1), v.repeat_interleave(2, dim=1))
    torch.testing.assert_close(out, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(lse, scores.logsumexp(-1), atol=2e-6, rtol=2e-5)
    args = q, k, v, bias
    grad = torch.randn_like(out)
    for a, b in zip(torch.autograd.grad(out, args, grad),
                    torch.autograd.grad(expected, args, grad), strict=True):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=4e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_prefill_decode_and_unrelated_document_invariance(device):
    torch.manual_seed(15)
    q = torch.randn(8, 4, 8, device=device).bfloat16()
    k = torch.randn(8, 2, 8, device=device).bfloat16()
    v = torch.randn_like(k)
    b = torch.randn(8, 4, 4, device=device).bfloat16()
    ids = torch.arange(8, device=device)
    docs = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device)
    out, lse = local_attention(q, k, v, b, ids, docs, window=4, query_chunk=3)
    for i in range(8):
        one, one_lse = local_attention(q[i:i+1], k[:i+1], v[:i+1], b[i:i+1],
                                       ids[i:i+1], docs[:i+1], window=4)
        assert torch.equal(one, out[i:i+1])
        assert torch.equal(one_lse, lse[i:i+1])
    isolated, isolated_lse = local_attention(q[4:], k[4:], v[4:], b[4:],
                                               ids[:4], docs[4:], window=4)
    assert torch.equal(isolated, out[4:]) and torch.equal(isolated_lse, lse[4:])


@pytest.mark.parametrize("device", DEVICES)
def test_1024_window_includes_current_and_1023_predecessors(device):
    q = torch.zeros(1, 1, 1, device=device)
    k = torch.zeros(1025, 1, 1, device=device)
    v = torch.ones_like(k)
    v[0] = 100000.0
    v[-1] = 1025.0
    b = torch.zeros(1, 1, 1024, device=device)
    out, lse = local_attention(q, k, v, b, torch.tensor([1024], device=device),
                                torch.zeros(1025, dtype=torch.long, device=device))
    assert out.item() == 2.0
    torch.testing.assert_close(lse, torch.tensor([[1024.0]], device=device).log())


def test_invalid_query_index():
    x = torch.ones(1, 1, 2)
    with pytest.raises(ValueError, match="existing KV"):
        local_attention(x, x, x, torch.zeros(1, 1, 4), torch.tensor([1]), torch.tensor([0]))
