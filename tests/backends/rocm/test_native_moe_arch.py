# SPDX-License-Identifier: Apache-2.0
"""Native DSV4 MoE forward/VJP and batch invariance on gfx942/gfx950."""

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from rl_engine.backends.rocm.moe.native_hash_route import native_hash_route
from rl_engine.backends.rocm.moe.packed_w4a16 import packed_w4a16
from rl_engine.backends.rocm.moe.router_projection import router_projection

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm GPU"
)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_router_cpu_oracle_leaf_vjp_and_batch_invariance(dtype):
    torch.manual_seed(42)
    x = torch.randn(17, 4096, device="cuda", dtype=torch.bfloat16).requires_grad_()
    w = (torch.randn(256, 4096, device="cuda", dtype=dtype) * 0.01).requires_grad_()
    y = router_projection(x, w)
    for indices in ([0], [16, 0, 8], list(range(17))):
        sub = router_projection(x[indices], w)
        assert torch.equal(sub.view(torch.int32), y[indices].view(torch.int32))
    xx = x.detach().cpu().double().requires_grad_()
    ww = w.detach().cpu().double().requires_grad_()
    ref = xx @ ww.T
    torch.testing.assert_close(y.cpu().double(), ref, rtol=0.0001, atol=0.00001)
    grad = torch.randn_like(y)
    y.backward(grad)
    ref.backward(grad.cpu().double())
    for actual, expected in ((x.grad, xx.grad), (w.grad, ww.grad)):
        assert bool(actual.isfinite().all()) and bool(actual.ne(0).any())
        torch.testing.assert_close(
            actual.cpu().double(), expected.to(actual.dtype).double(), rtol=0.01, atol=0.0001
        )


def test_hash_route_ids_probabilities_vjp_and_batch_invariance():
    torch.manual_seed(43)
    logits = torch.randn(17, 256, device="cuda", requires_grad=True)
    table = torch.stack([torch.randperm(256)[:6] for _ in range(19)]).cuda()
    tokens = torch.arange(17, device="cuda", dtype=torch.int64)
    probs, ids = native_hash_route(logits, tokens, table, 1.5)
    assert torch.equal(ids.long(), table[tokens])
    for indices in ([0], [16, 0, 8], list(range(17))):
        pp, ii = native_hash_route(logits[indices], tokens[indices], table, 1.5)
        assert torch.equal(ii, ids[indices])
        assert torch.equal(pp.view(torch.int32), probs[indices].view(torch.int32))
    xx = logits.detach().cpu().double().requires_grad_()
    scores = F.softplus(xx.gather(1, ids.cpu().long())).sqrt()
    ref = scores / scores.sum(-1, keepdim=True) * 1.5
    torch.testing.assert_close(probs.cpu().double(), ref, rtol=0.0001, atol=0.000001)
    grad = torch.randn_like(probs)
    probs.backward(grad)
    ref.backward(grad.cpu().double())
    torch.testing.assert_close(logits.grad.cpu().double(), xx.grad, rtol=0.0001, atol=0.000001)


@pytest.mark.parametrize("projection", ("gate_up", "down"))
def test_checkpoint_w4a16_forward_input_vjp_route_vjp_and_frozen_bytes(projection):
    directory = os.getenv("RLK_DSV4_CHECKPOINT")
    if not directory:
        pytest.skip("set RLK_DSV4_CHECKPOINT to the original checkpoint")
    from safetensors import safe_open

    names = ("w1", "w3") if projection == "gate_up" else ("w2",)
    with safe_open(str(next(Path(directory).glob("*.safetensors"))), framework="pt") as source:
        codes = torch.cat(
            [
                source.get_tensor(f"layers.0.ffn.experts.0.{n}.weight").view(torch.uint8)
                for n in names
            ]
        )[None]
        scales = torch.cat(
            [
                source.get_tensor(f"layers.0.ffn.experts.0.{n}.scale").view(torch.uint8)
                for n in names
            ]
        )[None]
    # Decode original E2M1/E8M0 bytes independently on the CPU; no dense weight
    # is supplied to the production operator or stored in the checkpoint.
    nib = torch.stack((codes & 15, codes >> 4), -1).flatten(-2)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    decoded = lut[(nib & 7).long()] * torch.where(nib < 8, 1.0, -1.0)
    weight = decoded * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    torch.manual_seed(44)
    x = torch.randn(17, weight.shape[-1], device="cuda", dtype=torch.bfloat16).requires_grad_()
    gamma = torch.rand(17, device="cuda").requires_grad_() if projection == "down" else None
    cc, ss = codes.cuda(), scales.cuda()
    offsets = torch.tensor([0, 17], device="cuda", dtype=torch.int64)
    y = packed_w4a16(x, cc, ss, offsets, gamma)
    for indices in ([0], [16, 0, 8]):
        sub = packed_w4a16(
            x[indices].contiguous(),
            cc,
            ss,
            torch.tensor([0, len(indices)], device="cuda", dtype=torch.int64),
            None if gamma is None else gamma[indices].contiguous(),
        )
        assert torch.equal(sub.view(torch.int16), y[indices].view(torch.int16))
    raw = x.detach().cpu().float() @ weight[0].T
    ref = raw if gamma is None else raw * gamma.detach().cpu()[:, None]
    torch.testing.assert_close(y.cpu().float(), ref, rtol=0.01, atol=0.003)
    grad = torch.randn_like(y)
    y.backward(grad)
    staged_grad = (
        grad.cpu()
        if gamma is None
        else (grad.cpu().float() * gamma.detach().cpu()[:, None]).bfloat16()
    )
    dx = staged_grad.float() @ weight[0]
    torch.testing.assert_close(x.grad.cpu().float(), dx, rtol=0.01, atol=0.003)
    if gamma is not None:
        dg = (grad.cpu().float() * raw).sum(-1)
        torch.testing.assert_close(gamma.grad.cpu(), dg, rtol=0.0001, atol=0.0001)
    assert torch.equal(cc.cpu(), codes) and torch.equal(ss.cpu(), scales)
