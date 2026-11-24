# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from rl_engine.models.inkling2.norm_clip import inkling_l2_clip, inkling_rmsnorm

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/ROCm device required"
))]


@pytest.mark.parametrize("device", DEVICES)
def test_rmsnorm_fp32_affine_and_backward(device):
    torch.manual_seed(10)
    x = torch.randn(3, 17, device=device, requires_grad=True)
    w = torch.randn(17, device=device, requires_grad=True)
    y = inkling_rmsnorm(x, w, out_dtype=torch.float32)
    ref = F.rms_norm(x, (17,), w, 1e-6)
    torch.testing.assert_close(y, ref, rtol=2e-5, atol=2e-6)
    g = torch.randn_like(y)
    for a, b in zip(torch.autograd.grad(y, (x, w), g),
                    torch.autograd.grad(ref, (x, w), g), strict=True):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
    assert inkling_rmsnorm(x).dtype == torch.bfloat16


@pytest.mark.parametrize("device", DEVICES)
def test_clip_cast_order_and_chunks(device):
    x = torch.tensor([[[3.0, 4.0]], [[0.5, -1.0]]], device=device).bfloat16()
    scale = torch.tensor([[1.0, 1.0]], device=device)
    tau = torch.tensor([1.23, 2.1], device=device)
    y = inkling_l2_clip(x, threshold=2.0, scale=scale, output_scale=tau)
    val = x.float() * scale
    clipped = (val * (2.0 / (val.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1.0))
    reference = (clipped.bfloat16().float() * tau[:, None, None]).bfloat16()
    assert torch.equal(y, reference)
    chunks = torch.cat([inkling_l2_clip(x[i:i+1], threshold=2.0, scale=scale,
                                       output_scale=tau[i:i+1]) for i in range(2)])
    assert torch.equal(y, chunks)


@pytest.mark.parametrize("device", DEVICES)
def test_clip_zero_gradient(device):
    x = torch.zeros(2, 3, 4, device=device, requires_grad=True)
    y = inkling_l2_clip(x, threshold=8.0)
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_invalid_norm_shapes():
    with pytest.raises(ValueError, match="weight"):
        inkling_rmsnorm(torch.ones(2, 4), torch.ones(3))
    with pytest.raises(ValueError, match="output scale"):
        inkling_l2_clip(torch.ones(2, 3, 4), threshold=8.0, output_scale=torch.ones(3))
