# SPDX-License-Identifier: Apache-2.0
"""INK-27 ROCm grouped expert GEMM tests (MI300X)."""

import pytest
import torch

from rl_engine.backends.extension import _C, _EXT_AVAILABLE
from rl_engine.models.inkling2.grouped_gemm import grouped_expert_gemm_rocm

pytestmark = pytest.mark.skipif(
    torch.version.hip is None
    or not torch.cuda.is_available()
    or not _EXT_AVAILABLE
    or not hasattr(_C, "inkling2_grouped_gemm_forward"),
    reason="ROCm device and Inkling 2 grouped GEMM extension required",
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _ref(tokens, experts, weights, bias):
    """Sequential-order FP32 reference (h ascending, FMA chain, bias first)."""
    t = tokens.float()
    w = weights.float()
    out = torch.empty(tokens.shape[0], weights.shape[1], device=tokens.device)
    for row in range(tokens.shape[0]):
        e = int(experts[row])
        acc = bias[e].float().clone() if bias is not None else torch.zeros_like(out[0])
        for h in range(tokens.shape[1]):
            acc = torch.addcmul(acc, t[row, h], w[e, :, h])
        out[row] = acc
    return out.to(tokens.dtype)


def _make(tokens_n, hidden, experts_n, out, dtype, seed=0, bias=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens = torch.randn(tokens_n, hidden, device="cuda", generator=g).to(dtype)
    idx = torch.randint(0, experts_n, (tokens_n,), device="cuda", generator=g)
    w = (torch.randn(experts_n, out, hidden, device="cuda", generator=g) * 0.05).to(dtype)
    b = torch.randn(experts_n, out, device="cuda", generator=g).to(dtype) if bias else None
    return tokens, idx, w, b


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 3, 2, 5), (5, 3, 3, 2), (37, 70, 5, 67), (130, 129, 4, 65)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_forward_matches_sequential_reference(dtype, shape, use_bias):
    tokens, idx, w, b = _make(*shape, dtype, bias=use_bias)
    out = grouped_expert_gemm_rocm(tokens, idx, w, b)
    ref = _ref(tokens, idx, w, b)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2 if dtype != torch.float32 else 1e-4, rtol=0)


def test_forward_real_inkling_shape_tolerance_and_determinism():
    tokens, idx, w, b = _make(96, 2304, 16, 1152, torch.bfloat16, seed=3)
    a = grouped_expert_gemm_rocm(tokens, idx, w, b)
    c = grouped_expert_gemm_rocm(tokens, idx, w, b)
    assert torch.equal(a, c)
    ref = torch.stack(
        [(tokens[i].float() @ w[idx[i]].float().T + b[idx[i]].float()) for i in range(0, 96, 7)]
    )
    torch.testing.assert_close(a[::7].float(), ref.to(torch.bfloat16).float(), atol=5e-2, rtol=0)


def test_empty_and_expert_range():
    w = torch.randn(3, 4, 5, device="cuda")
    out = grouped_expert_gemm_rocm(
        torch.empty(0, 5, device="cuda"), torch.empty(0, dtype=torch.long, device="cuda"), w
    )
    assert out.shape == (0, 4)
    with pytest.raises(RuntimeError, match="outside"):
        grouped_expert_gemm_rocm(
            torch.randn(2, 5, device="cuda"), torch.tensor([0, 3], device="cuda"), w
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_token_permutation_and_batch_invariance_bitwise(dtype):
    tokens, idx, w, b = _make(53, 70, 6, 67, dtype, seed=5)
    base = grouped_expert_gemm_rocm(tokens, idx, w, b)
    perm = torch.randperm(53, device="cuda")
    shuf = grouped_expert_gemm_rocm(tokens[perm], idx[perm], w, b)
    restored = torch.empty_like(shuf)
    restored[perm] = shuf
    assert torch.equal(restored, base)
    # a row is unchanged when the batch is truncated / other rows change
    sub = grouped_expert_gemm_rocm(tokens[:11], idx[:11], w, b)
    assert torch.equal(sub, base[:11])
    other = tokens.clone()
    other[20:] = torch.randn_like(other[20:])
    assert torch.equal(grouped_expert_gemm_rocm(other, idx, w, b)[:20], base[:20])


@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_matches_reference_and_is_deterministic(dtype):
    tokens, idx, w, b = _make(41, 70, 5, 67, dtype, seed=9)
    tokens.requires_grad_(True)
    w.requires_grad_(True)
    b.requires_grad_(True)
    out = grouped_expert_gemm_rocm(tokens, idx, w, b)
    dy = torch.randn_like(out)
    out.backward(dy)
    grads = [tokens.grad.clone(), w.grad.clone(), b.grad.clone()]
    tokens.grad = w.grad = b.grad = None
    out2 = grouped_expert_gemm_rocm(tokens, idx, w, b)
    out2.backward(dy)
    for g, g2 in zip(grads, (tokens.grad, w.grad, b.grad), strict=True):
        assert torch.equal(g, g2)

    tol = 1e-3 if dtype == torch.float32 else 5e-2
    tf, wf, bf = (x.detach().float().requires_grad_(True) for x in (tokens, w, b))
    ref = torch.stack([tf[i] @ wf[idx[i]].T + bf[idx[i]] for i in range(41)])
    ref.backward(dy.float())
    for g, r in zip(grads, (tf, wf, bf), strict=True):
        torch.testing.assert_close(g.float(), r.grad, atol=tol, rtol=tol)
