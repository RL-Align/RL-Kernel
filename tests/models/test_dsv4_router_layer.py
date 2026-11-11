# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 router layer: DeepGEMM gate GEMM + routing, batch invariance and autograd."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.router import route_backward_dx, route_topk


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
    return hasattr(ext, "dsv4_router_topk") and deepgemm_available()


pytestmark = pytest.mark.skipif(not _ready(), reason="SM90, DeepGEMM and the router kernels")

H, E, SCALE = 4096, 256, 1.5


def _router(hash_layer: bool, seed: int = 0):
    from rl_engine.models.deepseek_v4.router import Dsv4Router

    g = torch.Generator(device="cuda").manual_seed(seed)
    gate = (torch.randn(E, H, generator=g, device="cuda") * 0.02).bfloat16()
    if hash_layer:
        tid2eid = torch.randint(0, E, (5000, 6), generator=g, device="cuda", dtype=torch.int32)
        return Dsv4Router(gate, tid2eid=tid2eid, routed_scaling_factor=SCALE)
    bias = torch.randn(E, generator=g, device="cuda") * 0.1
    return Dsv4Router(gate, bias=bias, routed_scaling_factor=SCALE)


@pytest.mark.parametrize("hash_layer", [False, True])
def test_router_layer_batch_invariant(hash_layer):
    router = _router(hash_layer)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(1029, H, generator=g, device="cuda").bfloat16()
    ids_in = torch.randint(0, 5000, (1029,), generator=g, device="cuda")
    w, ids = router(x, ids_in)
    for lo, hi in ((0, 1), (3, 4), (64, 193), (1000, 1029)):
        w2, ids2 = router(x[lo:hi].clone(), ids_in[lo:hi].clone())
        assert torch.equal(ids2, ids[lo:hi])
        assert torch.equal(w2.view(torch.int32), w[lo:hi].view(torch.int32))


@pytest.mark.parametrize("hash_layer", [False, True])
def test_router_layer_forward_backward_match_golden(hash_layer):
    from rl_engine.models.deepseek_v4.router import gate_logits

    router = _router(hash_layer, seed=2)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(300, H, generator=g, device="cuda").bfloat16().requires_grad_(True)
    ids_in = torch.randint(0, 5000, (300,), generator=g, device="cuda")
    w, ids = router(x, ids_in)
    d = torch.randn(300, 6, generator=g, device="cuda")
    (w * d).sum().backward()

    logits = gate_logits(x.detach(), router.gate_weight)
    ref_w, ref_ids = route_topk(
        logits,
        router.bias,
        SCALE,
        input_ids=ids_in if hash_layer else None,
        tid2eid=router.tid2eid,
    )
    assert torch.equal(ids, ref_ids)
    assert torch.equal(w.view(torch.int32), ref_w.view(torch.int32))
    ref_dx = route_backward_dx(logits, ids, d, router.gate_weight, SCALE, torch.bfloat16)
    assert torch.equal(x.grad.view(torch.int16), ref_dx.view(torch.int16))
    # The gate GEMM itself is close to an FP32 reference.
    ref_logits = x.detach().float() @ router.gate_weight.float().T
    torch.testing.assert_close(logits, ref_logits, rtol=1e-4, atol=1e-4)
