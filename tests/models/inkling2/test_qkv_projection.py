# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from rl_engine.models.inkling2.qkv_projection import QKVGeometry, project_qkv

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/ROCm device required"
))]


@pytest.mark.parametrize("device", DEVICES)
def test_distinct_qkv_streams_and_gradients(device):
    spec = QKVGeometry(hidden=16, q_heads=4, kv_heads=2, head_dim=8)
    torch.manual_seed(12)
    x = tuple(torch.randint(-4, 5, (3, 16), device=device).bfloat16().requires_grad_()
              for _ in range(3))
    w = tuple(torch.randint(-4, 5, (n, 16), device=device).float().requires_grad_()
              for n in (32, 16, 16))
    out = project_qkv(*x, *w, geometry=spec)
    ref = tuple(F.linear(a.float(), b.bfloat16().float()).bfloat16().reshape_as(y)
                for a, b, y in zip(x, w, out, strict=True))
    assert [tuple(y.shape) for y in out] == [(3, 4, 8), (3, 2, 8), (3, 2, 8)]
    for y, expected in zip(out, ref, strict=True):
        assert torch.equal(y, expected)
    grads = tuple(torch.ones_like(y) for y in out)
    actual = torch.autograd.grad(out, (*x, *w), grads)
    expected = torch.autograd.grad(ref, (*x, *w), grads)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_released_geometry_and_batch_row_invariance(device):
    torch.manual_seed(13)
    x = tuple(torch.randn(3, 2304, device=device).bfloat16() for _ in range(3))
    w = tuple(torch.randn(n, 2304, device=device).bfloat16() / 48
              for n in (3072, 1024, 1024))
    full = project_qkv(*x, *w)
    one = project_qkv(*(a[1:2] for a in x), *w)
    assert [tuple(y.shape) for y in full] == [(3, 24, 128), (3, 8, 128), (3, 8, 128)]
    for a, b in zip(full, one, strict=True):
        if device == "cuda":
            assert torch.equal(a[1:2], b)
        else:
            torch.testing.assert_close(a[1:2], b, atol=0.01, rtol=0.01)


def test_invalid_qkv_geometry():
    with pytest.raises(ValueError, match="divisible"):
        QKVGeometry(q_heads=3, kv_heads=2)
    x = torch.zeros(1, 2304, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="weight"):
        project_qkv(x, x, x, x, x, x)
