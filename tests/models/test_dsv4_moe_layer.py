# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 MoE block (EP = 1): router -> plan -> P5 routed experts -> combine + shared.

Byte checks: batch invariance across splits, fused == two-launch. Accuracy: an
FP32 reference on the same MX-dequantized operands.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.moe.mx_format import mx_dequantize, mx_quantize


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    if torch.cuda.get_device_capability()[0] != 9:
        return False
    try:
        import rl_engine._C as ext
        from rl_engine.kernels.ops.cuda.deepgemm import deepgemm_available
    except ImportError:
        return False
    need = ("moe_slot_sum", "dsv4_router_topk", "sm90_moe_fused_mlp_forward")
    return all(hasattr(ext, n) for n in need) and deepgemm_available()


pytestmark = pytest.mark.skipif(
    not _ready(), reason="SM90, DeepGEMM, and a KERNEL_ALIGN_MOE_SM90=1 build"
)

E, H, F, FS, SCALE = 256, 1024, 256, 256, 1.5


@pytest.fixture(scope="module")
def moe():
    from rl_engine.models.deepseek_v4.moe import Dsv4MoE
    from rl_engine.models.deepseek_v4.router import Dsv4Router

    g = torch.Generator(device="cuda").manual_seed(0)
    gate = (torch.randn(E, H, generator=g, device="cuda") / H**0.5).bfloat16()
    bias = torch.randn(E, generator=g, device="cuda") * 0.05
    w1 = torch.randn(E, 2 * F, H, generator=g, device="cuda") / H**0.5
    w2 = torch.randn(E, H, F, generator=g, device="cuda") / F**0.5
    s1 = (torch.randn(2 * FS, H, generator=g, device="cuda") / H**0.5).bfloat16()
    s2 = (torch.randn(H, FS, generator=g, device="cuda") / FS**0.5).bfloat16()
    return Dsv4MoE(
        router=Dsv4Router(gate, bias=bias, routed_scaling_factor=SCALE),
        w1=mx_quantize(w1, "e2m1"),
        w2=mx_quantize(w2, "e2m1"),
        shared_fc1=s1,
        shared_fc2=s2,
    )


def _x(t, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(t, H, generator=g, device="cuda").bfloat16()
    res = torch.randn(t, H, generator=g, device="cuda").bfloat16()
    return x, res


def test_moe_block_batch_invariant(moe):
    x, res = _x(300, 1)
    full = moe.forward(x, res)
    for lo, hi in ((0, 1), (7, 8), (64, 193), (299, 300)):
        part = moe.forward(x[lo:hi].clone(), res[lo:hi].clone())
        assert torch.equal(part.view(torch.int16), full[lo:hi].view(torch.int16))
    assert torch.equal(moe.forward(x, res).view(torch.int16), full.view(torch.int16))


def test_moe_block_fused_equals_two_launch(moe):
    x, res = _x(129, 2)
    a = moe.forward(x, res)
    moe.path = "fused"
    try:
        b = moe.forward(x, res)
    finally:
        moe.path = "two_launch"
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_moe_block_close_to_fp32_reference(moe):
    from rl_engine.kernels.ops.pytorch.router import route_topk
    from rl_engine.models.deepseek_v4.router import gate_logits

    x, res = _x(64, 3)
    out = moe.forward(x, res)
    w, ids = route_topk(gate_logits(x, moe.router.gate_weight), moe.router.bias, SCALE)
    a = mx_dequantize(mx_quantize(x, "e4m3"))
    w1, w2 = mx_dequantize(moe.w1), mx_dequantize(moe.w2)
    routed = torch.zeros(64, H, device="cuda")
    for t in range(64):
        for s in range(6):
            e = int(ids[t, s])
            z = a[t] @ w1[e].t()
            gg = torch.clamp(z[:F], max=10.0)
            u = torch.clamp(z[F:], min=-10.0, max=10.0)
            routed[t] += (gg * torch.sigmoid(gg) * u * w[t, s]) @ w2[e].t()
    shared, _ = moe._shared.forward(x, moe._shared_w)
    ref = routed + shared.float() + res.float()
    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < 3e-2, err


def _grad(moe, x, res, dout):
    x = x.clone().requires_grad_(True)
    res = res.clone().requires_grad_(True)
    out = moe.forward(x, res)
    out.backward(dout)
    return out, x.grad, res.grad


def test_moe_block_backward_batch_invariant_and_deterministic(moe):
    if not moe._routed.capabilities()["backward"]:
        pytest.skip("P5 backward kernels not built")
    x, res = _x(300, 4)
    dout = torch.randn(300, H, device="cuda").bfloat16()
    _, dx, dres = _grad(moe, x, res, dout)
    assert dx.dtype == torch.bfloat16 and torch.equal(dres, dout)
    _, dx2, _ = _grad(moe, x, res, dout)
    assert torch.equal(dx.view(torch.int16), dx2.view(torch.int16))
    for lo, hi in ((0, 1), (7, 8), (64, 193), (299, 300)):
        _, part, _ = _grad(moe, x[lo:hi], res[lo:hi], dout[lo:hi].contiguous())
        assert torch.equal(part.view(torch.int16), dx[lo:hi].view(torch.int16))


@pytest.mark.parametrize("with_shared", [False, True])
def test_moe_block_backward_close_to_fp32_reference(moe, with_shared):
    """Routed + router path: BF16 dgrad on exact weights. With the shared expert its
    FP8 dgrad (DeepGEMM, tested in its own suite) dominates the error."""
    if not moe._routed.capabilities()["backward"]:
        pytest.skip("P5 backward kernels not built")
    from rl_engine.models.deepseek_v4.moe import Dsv4MoE

    full = moe
    if not with_shared:
        moe = Dsv4MoE(full.router, full.w1, full.w2)
    from rl_engine.kernels.ops.pytorch.router import route_topk
    from rl_engine.models.deepseek_v4.router import gate_logits

    x, res = _x(48, 5)
    dout = torch.randn(48, H, device="cuda").bfloat16()
    _, dx, _ = _grad(moe, x, res, dout)

    _, ids = route_topk(gate_logits(x, moe.router.gate_weight), moe.router.bias, SCALE)
    w1, w2 = mx_dequantize(moe.w1), mx_dequantize(moe.w2)
    xr = x.float().requires_grad_(True)
    xq = mx_dequantize(mx_quantize(x, "e4m3"))
    a = xr + (xq - xr).detach()  # straight-through quantization
    logits = xr @ moe.router.gate_weight.float().t()
    s = torch.sqrt(torch.nn.functional.softplus(logits, threshold=1e9)).gather(1, ids.long())
    w = s / s.sum(-1, keepdim=True) * SCALE
    rows = []
    for t in range(48):
        acc = torch.zeros(H, device="cuda")
        for k in range(6):
            e = int(ids[t, k])
            z = a[t] @ w1[e].t()
            g = torch.clamp(z[:F], max=10.0)
            u = torch.clamp(z[F:], min=-10.0, max=10.0)
            acc = acc + (g * torch.sigmoid(g) * u * w[t, k]) @ w2[e].t()
        rows.append(acc)
    ref = torch.stack(rows)
    if with_shared:
        zs = xr @ full.shared_fc1.float().t()
        gs = torch.clamp(zs[:, :FS], max=10.0)
        us = torch.clamp(zs[:, FS:], min=-10.0, max=10.0)
        ref = ref + (gs * torch.sigmoid(gs) * us) @ full.shared_fc2.float().t()
    ref.backward(dout.float())
    diff = dx.float() - xr.grad
    rel_l2 = (diff.norm() / xr.grad.norm()).item()
    rel_max = (diff.abs().max() / xr.grad.abs().max()).item()
    print(f"with_shared={with_shared} rel_l2={rel_l2:.3g} rel_max={rel_max:.3g}")
    if with_shared:
        assert rel_l2 < 8e-2, (rel_l2, rel_max)
    else:
        assert rel_l2 < 2e-2 and rel_max < 5e-2, (rel_l2, rel_max)
