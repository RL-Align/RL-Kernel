# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 MoE router: golden contract, CUDA bytes, ties / NaN / hash, backward.

The golden is defined on CUDA (libdevice expf / log1pf), so these tests need a GPU.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.router import (
    route_backward_dx,
    route_backward_logits,
    route_topk,
    sqrt_softplus,
)


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "dsv4_router_topk")


pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA rl_engine._C with the router required")

SCALE = 1.5


def _logits(t, e, seed, spread=4.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(t, e, generator=g, device="cuda") * spread
    if t:  # exercise the softplus branches: large positive / negative values
        x[0, :8] = torch.tensor([25.0, 20.0, 19.9, -30.0, -90.0, 0.0, -0.0, 88.0])
    return x


def _bias(e, seed):
    g = torch.Generator(device="cuda").manual_seed(seed + 1000)
    return torch.randn(e, generator=g, device="cuda") * 0.1


def _cuda():
    from rl_engine.kernels.ops.cuda.router import route_backward_dx_cuda, route_topk_cuda

    return route_topk_cuda, route_backward_dx_cuda


def _bits_equal(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


def test_golden_matches_float64_model():
    x = _logits(64, 256, 0)
    bias = _bias(256, 0)
    w, ids = route_topk(x, bias, SCALE)
    x64 = x.double()
    s64 = torch.sqrt(torch.nn.functional.softplus(x64, threshold=1e9))
    ref_ids = torch.topk(s64 + bias.double(), 6, dim=-1).indices
    assert torch.equal(ids.long(), ref_ids)
    sel = s64.gather(1, ref_ids)
    ref = sel / sel.sum(-1, keepdim=True) * SCALE
    torch.testing.assert_close(w.double(), ref, rtol=1e-5, atol=0)
    torch.testing.assert_close(sqrt_softplus(x).double(), s64, rtol=2e-6, atol=1e-30)


def test_ties_go_to_smallest_id_and_nan_is_last():
    x = torch.zeros(3, 256, device="cuda")
    bias = torch.zeros(256, device="cuda")
    x[1, 200] = float("nan")
    x[1, :6] = float("nan")
    x[2, 250] = 1.0
    x[2, 3] = float("nan")
    route_topk_cuda, _ = _cuda()
    for fn in (route_topk, route_topk_cuda):
        w, ids = fn(x, bias, SCALE)
        assert ids[0].tolist() == [0, 1, 2, 3, 4, 5]
        assert ids[1].tolist() == [6, 7, 8, 9, 10, 11]
        assert ids[2].tolist() == [250, 0, 1, 2, 4, 5]
        assert torch.isfinite(w).all()


@pytest.mark.parametrize("experts", [256, 384])
@pytest.mark.parametrize("tokens", [0, 1, 7, 1000])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cuda_topk_byte_equal(experts, tokens, seed):
    route_topk_cuda, _ = _cuda()
    x = _logits(tokens, experts, seed)
    bias = _bias(experts, seed)
    w_ref, ids_ref = route_topk(x, bias, SCALE)
    w, ids = route_topk_cuda(x, bias, SCALE)
    assert torch.equal(ids, ids_ref)
    assert _bits_equal(w, w_ref)


def test_cuda_topk_quantized_logits_many_ties():
    route_topk_cuda, _ = _cuda()
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randint(-3, 4, (2048, 256), generator=g, device="cuda").float()
    bias = torch.zeros(256, device="cuda")
    w_ref, ids_ref = route_topk(x, bias, SCALE)
    w, ids = route_topk_cuda(x, bias, SCALE)
    assert torch.equal(ids, ids_ref)
    assert _bits_equal(w, w_ref)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cuda_hash_routing_byte_equal(dtype):
    route_topk_cuda, _ = _cuda()
    vocab, tokens = 1000, 513
    g = torch.Generator(device="cuda").manual_seed(3)
    tid2eid = torch.randint(0, 256, (vocab, 6), generator=g, device="cuda", dtype=dtype)
    input_ids = torch.randint(0, vocab, (tokens,), generator=g, device="cuda", dtype=dtype)
    x = _logits(tokens, 256, 3)
    w_ref, ids_ref = route_topk(x, None, SCALE, input_ids=input_ids, tid2eid=tid2eid)
    w, ids = route_topk_cuda(x, None, SCALE, input_ids=input_ids, tid2eid=tid2eid)
    assert torch.equal(ids, tid2eid[input_ids.long()].int())
    assert torch.equal(ids, ids_ref)
    assert _bits_equal(w, w_ref)


def test_cuda_topk_is_batch_invariant():
    route_topk_cuda, _ = _cuda()
    x = _logits(777, 256, 5)
    bias = _bias(256, 5)
    w, ids = route_topk_cuda(x, bias, SCALE)
    for lo, hi in ((0, 1), (5, 6), (100, 357), (776, 777)):
        w2, ids2 = route_topk_cuda(x[lo:hi].clone(), bias, SCALE)
        assert torch.equal(ids2, ids[lo:hi])
        assert _bits_equal(w2, w[lo:hi])


def test_backward_logits_matches_autograd():
    x = _logits(32, 256, 11).double()
    bias = _bias(256, 11)
    _, ids = route_topk(x.float(), bias, SCALE)
    d = torch.randn(32, 6, device="cuda", dtype=torch.float64)
    xs = x.gather(1, ids.long()).requires_grad_(True)
    s = torch.sqrt(torch.nn.functional.softplus(xs, threshold=1e9))
    out = s / s.sum(-1, keepdim=True) * SCALE
    (out * d).sum().backward()
    got = route_backward_logits(x.float(), ids, d.float(), SCALE)
    torch.testing.assert_close(got.double(), xs.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [0, 1, 300])
@pytest.mark.parametrize("hidden", [4096, 1024])
def test_cuda_backward_dx_byte_equal(out_dtype, tokens, hidden):
    _, route_backward_dx_cuda = _cuda()
    x = _logits(tokens, 256, 13)
    bias = _bias(256, 13)
    _, ids = route_topk(x, bias, SCALE)
    g = torch.Generator(device="cuda").manual_seed(14)
    d = torch.randn(tokens, 6, generator=g, device="cuda")
    gate = (torch.randn(256, hidden, generator=g, device="cuda") * 0.02).bfloat16()
    ref = route_backward_dx(x, ids, d, gate, SCALE, out_dtype=out_dtype)
    got = route_backward_dx_cuda(x, ids, d, gate, SCALE, out_dtype=out_dtype)
    view = torch.int16 if out_dtype == torch.bfloat16 else torch.int32
    assert got.dtype == out_dtype
    assert torch.equal(got.view(view), ref.view(view))
    if tokens:  # equals the dense dgrad of the gate GEMM up to FP32 rounding
        dsel = route_backward_logits(x, ids, d, SCALE)
        dense = torch.zeros(tokens, 256, device="cuda").scatter(1, ids.long(), dsel)
        torch.testing.assert_close(
            got.float(), (dense.double() @ gate.double()).float(), rtol=2e-2, atol=1e-6
        )
