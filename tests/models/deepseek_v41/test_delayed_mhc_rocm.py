# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

from rl_engine.backends.rocm.mhc.v41_delayed import collapse, inverse_rms, split_sinkhorn
from rl_engine.models.deepseek_v41.bitwise import compare_bits


@pytest.fixture(autouse=True, scope="module")
def require_device():
    if not torch.version.hip or not torch.cuda.is_available():
        if os.environ.get("RLK_REQUIRE_ROCM") == "1":
            pytest.fail("real ROCm device required")
        pytest.skip("ROCm device required")


def reference(m, s, b, iters):
    pre = torch.sigmoid(m[..., :4] * s[0] + b[:4]) + 1e-6
    post = 2 * torch.sigmoid(m[..., 4:8] * s[1] + b[4:8])
    z = (m[..., 8:] * s[2] + b[8:]).reshape(*m.shape[:-1], 4, 4).softmax(-1) + 1e-6
    z = z / (z.sum(-2, keepdim=True) + 1e-6)
    for _ in range(iters - 1):
        z = z / (z.sum(-1, keepdim=True) + 1e-6)
        z = z / (z.sum(-2, keepdim=True) + 1e-6)
    return pre, post, z


@pytest.mark.parametrize("iters", [1, 20])
def test_split_all_gradients_and_replay(iters):
    torch.manual_seed(40)
    operands = [
        torch.randn(2, 3, 24, device="cuda") * 0.2,
        torch.randn(3, device="cuda") * 0.3,
        torch.randn(24, device="cuda") * 0.3,
    ]

    def run(kernel):
        xs = [t.clone().requires_grad_() for t in operands]
        ys = kernel(*xs)
        cotangents = [torch.linspace(-1, 1, y.numel(), device="cuda").reshape(y.shape) for y in ys]
        gs = torch.autograd.grad(ys, xs, cotangents)
        return (*ys, *gs)

    actual = run(lambda *xs: split_sinkhorn(*xs, iters=iters))
    expected = run(lambda *xs: reference(*xs, iters))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=2e-6)
    for a, b in zip(actual, run(lambda *xs: split_sinkhorn(*xs, iters=iters)), strict=True):
        assert compare_bits(a, b)["status"] == "PASS"
    with torch.inference_mode():
        outputs = split_sinkhorn(*operands, iters=iters)
    for a, b in zip(actual[:3], outputs, strict=True):
        assert compare_bits(a, b)["status"] == "PASS"


@pytest.mark.parametrize("d", [128, 5120])
def test_delayed_collapse_gradients(d):
    torch.manual_seed(51)
    x = torch.randn(2, 3, 4, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    pre = torch.rand(2, 3, 4, device="cuda", requires_grad=True)
    y = collapse(x, pre)
    dy = torch.randn_like(y)
    gs = torch.autograd.grad(y, (x, pre), dy)
    ref = (x.float() * pre[..., None]).sum(-2).bfloat16()
    expected = torch.autograd.grad(ref, (x, pre), dy)
    torch.testing.assert_close(y, ref, rtol=0.01, atol=0.01)
    for a, b in zip(gs, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0.01, atol=0.001)


def test_controller_rms_width_20480_and_gradient():
    x = torch.randn(3, 20480, device="cuda", requires_grad=True)
    r = inverse_rms(x, eps=1e-20)
    ref = torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20)
    actual = torch.autograd.grad(r, x, torch.ones_like(r))[0]
    expected = torch.autograd.grad(ref, x, torch.ones_like(ref))[0]
    torch.testing.assert_close(r, ref, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-8)
