# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 MoE LoRA (routed fc1 / fc2 shared by all experts, shared-expert fc1 / fc2).

- The standalone clamp-SwiGLU -> MXFP8 kernel equals the fused fc1 epilogue bytewise.
- Zero adapters (B = 0) give exactly the base block's bytes.
- The forward with adapters is batch-invariant.
- dL/dx and every adapter's dA / dB match an FP32 autograd reference.
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
    need = ("moe_swiglu_quant", "sm90_moe_fc1_swiglu_quant_forward", "sm90_moe_bwd_dh_swiglu")
    return all(hasattr(ext, n) for n in need) and deepgemm_available()


pytestmark = pytest.mark.skipif(
    not _ready(), reason="SM90, DeepGEMM, and a KERNEL_ALIGN_MOE_SM90=1 build"
)

E, H, F, FS, SCALE, RANK = 256, 1024, 256, 256, 1.5, 16


def _moe(lora=None, seed=0):
    from rl_engine.models.deepseek_v4.moe import Dsv4MoE
    from rl_engine.models.deepseek_v4.router import Dsv4Router

    g = torch.Generator(device="cuda").manual_seed(seed)
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
        lora=lora,
    )


def _lora(zero_b=False, seed=1):
    from rl_engine.models.deepseek_v4.lora import LoraAdapter, MoeLora

    return MoeLora(
        fc1=LoraAdapter.init(H, 2 * F, RANK, 32, seed=seed, zero_b=zero_b),
        fc2=LoraAdapter.init(F, H, RANK, 32, seed=seed + 1, zero_b=zero_b),
        shared_fc1=LoraAdapter.init(H, 2 * FS, RANK, 32, seed=seed + 2, zero_b=zero_b),
        shared_fc2=LoraAdapter.init(FS, H, RANK, 32, seed=seed + 3, zero_b=zero_b),
    )


def _x(t, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(t, H, generator=g, device="cuda").bfloat16()
    res = torch.randn(t, H, generator=g, device="cuda").bfloat16()
    return x, res


def _same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_swiglu_quant_equals_fused_epilogue():
    from rl_engine.kernels.ops.cuda.moe import build_plan_cuda, mxfp8_act_quant_fwd_cuda
    from rl_engine.kernels.ops.cuda.router import route_topk_cuda
    from rl_engine.models.deepseek_v4.moe import _Rows
    from rl_engine.models.deepseek_v4.router import gate_logits

    moe = _moe()
    for t in (1, 37, 300):
        x, _ = _x(t, t)
        w, ids = route_topk_cuda(gate_logits(x, moe.router.gate_weight), moe.router.bias, SCALE)
        plan = build_plan_cuda(ids, E)
        q = mxfp8_act_quant_fwd_cuda(x)
        rows = _Rows(
            q.codes.index_select(0, plan.token),
            q.scales.index_select(0, plan.token),
            w.reshape(-1).float().index_select(0, plan.perm),
            plan.expert_offsets,
        )
        batch, x_q = moe._expert_batch(rows, H)
        fused = moe._routed.fc1_swiglu_quant(batch, x_q)
        h_q, h = moe._routed.swiglu_quant(moe._routed.fc1_z(batch, x_q), rows.p_s)
        assert torch.equal(h_q.codes, fused.codes) and torch.equal(h_q.scales, fused.scales), t
        assert h.dtype == torch.bfloat16 and h.shape == (t * 6, F)


def test_zero_adapters_give_base_bytes():
    x, res = _x(200, 2)
    base = _moe().forward(x, res)
    with_zero = _moe(lora=_lora(zero_b=True)).forward(x, res)
    assert _same(with_zero, base)


def test_lora_forward_batch_invariant():
    moe = _moe(lora=_lora())
    x, res = _x(300, 3)
    full = moe.forward(x, res)
    assert not _same(full, _moe().forward(x, res))  # the adapters change the output
    for lo, hi in ((0, 1), (7, 8), (64, 193), (299, 300)):
        assert _same(moe.forward(x[lo:hi].clone(), res[lo:hi].clone()), full[lo:hi]), (lo, hi)


def test_lora_backward_matches_reference():
    from rl_engine.kernels.ops.pytorch.router import route_topk
    from rl_engine.models.deepseek_v4.router import gate_logits

    lora = _lora()
    for _, ad in lora.adapters():
        ad.a.requires_grad_(True)
        ad.b.requires_grad_(True)
    moe = _moe(lora=lora)
    t = 48
    x, res = _x(t, 5)
    dout = torch.randn(t, H, device="cuda").bfloat16()
    xg = x.clone().requires_grad_(True)
    moe.forward(xg, res).backward(dout)

    # FP32 reference on the MX-dequantized routed weights, BF16 shared weights.
    _, ids = route_topk(gate_logits(x, moe.router.gate_weight), moe.router.bias, SCALE)
    w1, w2 = mx_dequantize(moe.w1), mx_dequantize(moe.w2)
    p = {
        n: (ad.a.detach().float().requires_grad_(True), ad.b.detach().float().requires_grad_(True))
        for n, ad in lora.adapters()
    }

    def delta(name, inp):
        a, b = p[name]
        return getattr(lora, name).scale * ((inp @ a.t()) @ b.t())

    xr = x.float().requires_grad_(True)
    xq = mx_dequantize(mx_quantize(x, "e4m3"))
    aq = xr + (xq - xr).detach()  # straight-through activation quantization
    logits = xr @ moe.router.gate_weight.float().t()
    s = torch.sqrt(torch.nn.functional.softplus(logits, threshold=1e9)).gather(1, ids.long())
    w = s / s.sum(-1, keepdim=True) * SCALE
    z_l = delta("fc1", xr)
    rows = []
    for i in range(t):
        acc = torch.zeros(H, device="cuda")
        for k in range(6):
            e = int(ids[i, k])
            z = aq[i] @ w1[e].t() + z_l[i]
            g = torch.clamp(z[:F], max=10.0)
            u = torch.clamp(z[F:], min=-10.0, max=10.0)
            h = g * torch.sigmoid(g) * u * w[i, k]
            acc = acc + h @ w2[e].t() + delta("fc2", h[None])[0]
        rows.append(acc)
    ref = torch.stack(rows)
    zs = xr @ moe.shared_fc1.float().t() + delta("shared_fc1", xr)
    gs = torch.clamp(zs[:, :FS], max=10.0)
    us = torch.clamp(zs[:, FS:], min=-10.0, max=10.0)
    hs = gs * torch.sigmoid(gs) * us
    ref = ref + hs @ moe.shared_fc2.float().t() + delta("shared_fc2", hs)
    ref.backward(dout.float())

    def rel(a, b):
        return float((a.float() - b).norm() / b.norm())

    assert rel(xg.grad, xr.grad) < 8e-2, rel(xg.grad, xr.grad)
    for name, ad in lora.adapters():
        da_ref, db_ref = p[name][0].grad, p[name][1].grad
        assert rel(ad.a.grad, da_ref) < 8e-2, (name, rel(ad.a.grad, da_ref))
        assert rel(ad.b.grad, db_ref) < 8e-2, (name, rel(ad.b.grad, db_ref))
