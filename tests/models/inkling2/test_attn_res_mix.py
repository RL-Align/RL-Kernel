# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from rl_engine.models.inkling2.attn_res_mix import block_attention_residual

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/ROCm device required"
))]


@pytest.mark.parametrize("device", DEVICES)
def test_attention_residual_math_and_gradients(device):
    torch.manual_seed(3)
    xs = [torch.randn(3, 17, device=device, requires_grad=True) for _ in range(3)]
    q = torch.randn(17, device=device, requires_grad=True)
    addend = torch.randn_like(xs[0], requires_grad=True)
    y = block_attention_residual(xs[:2], xs[2], q, partial_addend=addend)
    stack = torch.stack([xs[0], xs[1], xs[2] + addend])
    rms = torch.rsqrt(stack.square().mean(-1) + 1e-6)
    logits = torch.einsum("ntd,d,nt->tn", stack, q * (17**-0.5), rms)
    reference = torch.einsum("tn,ntd->td", logits.softmax(-1), stack * rms[..., None])
    torch.testing.assert_close(y, reference, atol=2e-6, rtol=2e-5)
    grad = torch.randn_like(y)
    params = (*xs, q, addend)
    actual = torch.autograd.grad(y, params, grad)
    expected = torch.autograd.grad(reference, params, grad)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_attention_residual_chunk_and_single_source(device):
    torch.manual_seed(4)
    x = torch.randn(7, 17, device=device).bfloat16()
    p = torch.randn_like(x)
    q = torch.randn(17, device=device)
    full = block_attention_residual([x], p, q)
    split = torch.cat([block_attention_residual([x[i:i+1]], p[i:i+1], q)
                       for i in range(x.shape[0])])
    assert torch.equal(full, split)
    one = block_attention_residual([x], None, q)
    assert one.dtype == torch.float32
    torch.testing.assert_close(one, x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + 1e-6))


def test_invalid_residual_state():
    x = torch.ones(2, 4)
    with pytest.raises(ValueError, match="partial addend"):
        block_attention_residual([x], None, torch.ones(4), partial_addend=x)
