# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""DSv4 MoE route plan: dispatch order, combine / fan-in bytes, P6 policy."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.pytorch.moe.route_plan import (
    build_plan,
    combine,
    combine_backward,
    dispatch,
    dispatch_backward,
    slot_sum,
    weights_backward,
)


def _ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import rl_engine._C as ext
    except ImportError:
        return False
    return hasattr(ext, "moe_slot_sum")


pytestmark = pytest.mark.skipif(not _ready(), reason="CUDA rl_engine._C with moe_slot_sum")

E = 256


def _ids(t, seed, experts=E):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Six distinct experts per token, like the router.
    return torch.rand(t, experts, generator=g, device="cuda").topk(6, dim=-1).indices.int()


def test_plan_order_offsets_and_inverse():
    ids = _ids(333, 0)
    plan = build_plan(ids, E)
    flat = ids.reshape(-1).long()
    keys = flat[plan.perm] * (1 << 32) + plan.perm  # (expert, token * 6 + slot)
    assert torch.all(keys[1:] > keys[:-1])
    counts = torch.bincount(flat, minlength=E)
    assert torch.equal(plan.expert_offsets.long()[1:] - plan.expert_offsets.long()[:-1], counts)
    assert plan.expert_offsets[0].item() == 0 and plan.expert_offsets[-1].item() == 333 * 6
    assert torch.equal(plan.perm[plan.inv.reshape(-1)], torch.arange(333 * 6, device="cuda"))
    assert torch.equal(ids.long().reshape(-1)[plan.perm][plan.inv.reshape(-1)], flat)


def test_dispatch_and_gradients_are_exact():
    ids = _ids(64, 1)
    plan = build_plan(ids, E)
    x = torch.randn(64, 256, device="cuda").bfloat16()
    w = torch.rand(64, 6, device="cuda")
    rows, p_s, slot = dispatch(x, w, plan)
    for r in (0, 17, 383):
        t, s = divmod(int(slot[r]), 6)
        assert (
            torch.equal(rows[r], x[t])
            and p_s[r] == w[t, s]
            and ids[t, s] == ids.reshape(-1)[slot[r]]
        )
    assert torch.equal(weights_backward(p_s, plan), w)
    dy = torch.randn(64, 256, device="cuda")
    assert torch.equal(combine_backward(dy, plan), dy[plan.token])


def test_combine_follows_p6_policy():
    ids = _ids(16, 2)
    plan = build_plan(ids, E)
    g = torch.Generator(device="cuda").manual_seed(3)
    y = torch.randn(96, 64, generator=g, device="cuda").bfloat16()
    shared = torch.randn(16, 64, generator=g, device="cuda").bfloat16()
    res = torch.randn(16, 64, generator=g, device="cuda").bfloat16()
    out = combine(y, plan, shared, res)
    for t in (0, 9):
        acc = y[plan.inv[t, 0]].float()
        for s in range(1, 6):
            acc = acc + y[plan.inv[t, s]].float()
        assert torch.equal(out[t], ((acc + shared[t].float()) + res[t].float()).bfloat16())
    # Slot-0 row is copied, not added to zero: -0.0 survives.
    from rl_engine.kernels.ops.cuda.moe import slot_sum_cuda

    y.fill_(-0.0)  # zero-initialised accumulation would give +0
    neg_zero = torch.full((16, 64), -0.0, device="cuda").view(torch.int32)
    for fn in (slot_sum, slot_sum_cuda):
        z = fn(y, plan, out_dtype=torch.float32)
        assert torch.equal(z.view(torch.int32), neg_zero)


@pytest.mark.parametrize("row_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shared_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens,hidden", [(0, 4096), (1, 4096), (257, 4096), (31, 2056)])
def test_cuda_slot_sum_byte_equal(
    row_dtype, shared_dtype, with_residual, out_dtype, tokens, hidden
):
    from rl_engine.kernels.ops.cuda.moe import slot_sum_cuda

    ids = _ids(tokens, 4)
    plan = build_plan(ids, E)
    g = torch.Generator(device="cuda").manual_seed(5)
    rows = (torch.randn(tokens * 6, hidden, generator=g, device="cuda") * 3).to(row_dtype)
    shared = None
    if shared_dtype is not None:
        shared = torch.randn(tokens, hidden, generator=g, device="cuda").to(shared_dtype)
    res = (
        torch.randn(tokens, hidden, generator=g, device="cuda").bfloat16()
        if with_residual
        else None
    )
    ref = slot_sum(rows, plan, shared, res, out_dtype)
    got = slot_sum_cuda(rows, plan, shared, res, out_dtype)
    view = torch.int16 if out_dtype == torch.bfloat16 else torch.int32
    assert got.dtype == out_dtype and torch.equal(got.view(view), ref.view(view))


def test_cuda_fanin_matches_golden_and_is_batch_invariant():
    from rl_engine.kernels.ops.cuda.moe import dispatch_backward_cuda

    ids = _ids(200, 6)
    plan = build_plan(ids, E)
    g = torch.Generator(device="cuda").manual_seed(7)
    dx_rows = torch.randn(1200, 4096, generator=g, device="cuda")
    dx_shared = torch.randn(200, 4096, generator=g, device="cuda")
    full = dispatch_backward_cuda(dx_rows, plan, dx_shared)
    assert torch.equal(full, dispatch_backward(dx_rows, plan, dx_shared))
    # Tokens 50..59 alone: their rows, in their own plan's order.
    sub = build_plan(ids[50:60], E)
    sub_rows = dx_rows[plan.inv[50:60].reshape(-1)][sub.perm]
    part = dispatch_backward_cuda(sub_rows, sub, dx_shared[50:60].contiguous())
    assert torch.equal(part, full[50:60])


def _plan_cuda_ready() -> bool:
    import rl_engine._C as ext

    return hasattr(ext, "moe_route_plan")


@pytest.mark.skipif(not _ready() or not _plan_cuda_ready(), reason="moe_route_plan build")
@pytest.mark.parametrize("t", [0, 1, 7, 85, 86, 333, 4096, 32768])
@pytest.mark.parametrize("experts", [256, 384])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_cuda_plan_equals_golden(t, experts, dtype):
    from rl_engine.kernels.ops.cuda.moe import build_plan_cuda

    ids = _ids(t, t + experts, experts).to(dtype)
    want = build_plan(ids, experts)
    got = build_plan_cuda(ids, experts)
    assert torch.equal(got.perm, want.perm)
    assert torch.equal(got.inv, want.inv)
    assert torch.equal(got.expert_offsets, want.expert_offsets)
    assert torch.equal(got.token, want.perm // 6)
    assert (got.num_tokens, got.num_experts) == (t, experts)


@pytest.mark.skipif(not _ready() or not _plan_cuda_ready(), reason="moe_route_plan build")
def test_cuda_plan_skewed_routing():
    """Hot experts (every token on experts 0-5) and a hash-like repeated pattern."""
    from rl_engine.kernels.ops.cuda.moe import build_plan_cuda

    hot = torch.arange(6, device="cuda", dtype=torch.int32).repeat(5000, 1)
    for ids in (hot, hot.flip(1), torch.randint(256, (777, 6), device="cuda")):
        want = build_plan(ids, 256)
        got = build_plan_cuda(ids, 256)
        assert torch.equal(got.perm, want.perm) and torch.equal(got.inv, want.inv)
        assert torch.equal(got.expert_offsets, want.expert_offsets)
