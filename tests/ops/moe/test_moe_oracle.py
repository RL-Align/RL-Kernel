# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Self-consistency tests for the P5 FP32 oracle (P5-1..P5-5; issues #60-#64)."""

from __future__ import annotations

import torch

from rl_engine.moe import fixtures, oracle
from rl_engine.moe.contract import tensor_sha256


def test_act_quant_bwd_is_ste() -> None:
    dy = torch.randn(4, 32)
    dx = oracle.mxfp8_act_quant_bwd(dy)
    assert torch.equal(dx, dy) and dx is not dy


def test_swiglu_bwd_matches_autograd_away_from_clamps() -> None:
    g = torch.Generator().manual_seed(3)
    gate = (torch.randn(4, 32, generator=g) * 2.0).requires_grad_(True)
    up = (torch.randn(4, 32, generator=g) * 2.0).requires_grad_(True)
    p_s = torch.rand(4, generator=g).requires_grad_(True)
    ref = (
        torch.nn.functional.silu(torch.clamp(gate, max=10.0))
        * torch.clamp(up, -10.0, 10.0)
        * p_s.unsqueeze(1)
    )
    dh = torch.randn(4, 32, generator=g).to(torch.bfloat16)
    ref.backward(dh.to(torch.float32))
    h, saved = oracle.clamp_swiglu_weighted_fwd(gate.detach(), up.detach(), p_s.detach())
    dgate, dup, dp_s = oracle.clamp_swiglu_weighted_bwd(dh, saved)
    assert torch.allclose(dgate, gate.grad, atol=1e-5)
    assert torch.allclose(dup, up.grad, atol=1e-5)
    assert torch.allclose(dp_s, p_s.grad, atol=1e-4)


def test_swiglu_clamp_subgradient_zero_at_bounds() -> None:
    gate = torch.tensor([[10.0, 10.5, 9.5]])
    up = torch.tensor([[-10.0, 10.0, 5.0]])
    p_s = torch.ones(1)
    _, saved = oracle.clamp_swiglu_weighted_fwd(gate, up, p_s)
    dh = torch.ones(1, 3, dtype=torch.bfloat16)
    dgate, dup, _ = oracle.clamp_swiglu_weighted_bwd(dh, saved)
    assert dgate[0, 0] == 0.0 and dgate[0, 1] == 0.0 and dgate[0, 2] != 0.0
    assert dup[0, 0] == 0.0 and dup[0, 1] == 0.0 and dup[0, 2] != 0.0


def test_route_weight_applied_exactly_once() -> None:
    gate = torch.full((2, 32), 1.5)
    up = torch.full((2, 32), 2.0)
    h1, _ = oracle.clamp_swiglu_weighted_fwd(gate, up, torch.tensor([1.0, 1.0]))
    h2, _ = oracle.clamp_swiglu_weighted_fwd(gate, up, torch.tensor([2.0, 2.0]))
    assert torch.allclose(h2.float(), h1.float() * 2.0, rtol=1e-2)


def test_lora_bwd_matches_autograd() -> None:
    g = torch.Generator().manual_seed(4)
    x = torch.randn(6, 32, generator=g).to(torch.bfloat16)
    a = (torch.randn(4, 32, generator=g) * 0.2).to(torch.bfloat16).requires_grad_(True)
    b = (torch.randn(16, 4, generator=g) * 0.2).to(torch.bfloat16).requires_grad_(True)
    xg = x.detach().clone().requires_grad_(True)
    y_ref = (xg.float() @ a.float().t() @ b.float().t()) * 0.5
    dy = torch.randn(6, 16, generator=g).to(torch.bfloat16)
    y_ref.backward(dy.float())
    y, u = oracle.shared_grouped_lora_delta_fwd(x, a.detach(), b.detach(), 0.5)
    dx, da, db = oracle.shared_grouped_lora_delta_bwd(dy, x, a.detach(), b.detach(), 0.5, u)

    def _close(got: torch.Tensor, want: torch.Tensor) -> bool:
        # BF16 inter-GEMM rounding => compare normalized to the tensor scale.
        return bool((got - want).abs().max() <= 2e-2 * want.abs().max() + 1e-6)

    assert _close(y, y_ref)
    assert _close(dx, xg.grad.float())
    assert _close(da, a.grad.float())
    assert _close(db, b.grad.float())
    assert bool(da.abs().sum() > 0) and bool(db.abs().sum() > 0)


def test_geometry_gate_one_row_equals_packed() -> None:
    """P5-4 (#61) acceptance: row-count=1 and packed multi-row give equal bytes."""
    import dataclasses

    batch = fixtures.make_expert_batch("base_plus_lora")
    y_packed, _ = oracle.routed_expert_forward(batch)
    offsets = batch.expert_offsets.tolist()
    for row in range(batch.rows):
        expert = sum(1 for o in offsets[1:-1] if o <= row)
        single = dataclasses.replace(
            batch,
            x=batch.x[row : row + 1],
            p_s=batch.p_s[row : row + 1],
            output_slot=batch.output_slot[row : row + 1],
            expert_offsets=torch.tensor(
                [0] * (expert + 1) + [1] * (len(offsets) - expert - 1), dtype=torch.int32
            ),
            row_geometry="one-row",
        )
        y_one, _ = oracle.routed_expert_forward(single)
        assert torch.equal(y_one[0], y_packed[row]), f"row {row} diverges from one-row"


def test_backward_has_no_dw_and_leaves_base_untouched() -> None:
    batch = fixtures.make_expert_batch("base_plus_lora")
    before = tensor_sha256(batch.w1.codes)
    y, saved = oracle.routed_expert_forward(batch)
    dy = fixtures.make_grad_output("t", tuple(y.shape))
    grads = oracle.routed_expert_backward(batch, saved, dy)
    assert set(grads) == {"dx", "dp_s", "da1", "db1", "da2", "db2"}
    assert tensor_sha256(batch.w1.codes) == before
    assert grads["dp_s"] is not None and grads["dp_s"].shape == (batch.rows,)
    assert grads["dp_s"].dtype == torch.float32
    for key in ("da1", "db1", "da2", "db2"):
        grad = grads[key]
        assert grad is not None and torch.isfinite(grad).all() and bool(grad.abs().sum() > 0)


def test_base_only_has_no_lora_grads() -> None:
    batch = fixtures.make_expert_batch("base_only_packed")
    y, saved = oracle.routed_expert_forward(batch)
    grads = oracle.routed_expert_backward(
        batch, saved, fixtures.make_grad_output("t2", tuple(y.shape))
    )
    assert grads["da1"] is None and grads["db2"] is None


def test_shared_expert_batch_invariant() -> None:
    import dataclasses

    batch = fixtures.make_shared_batch("shared_t16")
    y_full, _ = oracle.shared_expert_mlp_fwd(batch)
    one = dataclasses.replace(batch, x=batch.x[5:6])
    y_one, _ = oracle.shared_expert_mlp_fwd(one)
    assert torch.equal(y_one[0], y_full[5])


def test_shared_bwd_matches_autograd() -> None:
    batch = fixtures.make_shared_batch("shared_t16")
    x = batch.x.float().requires_grad_(True)
    z = x @ batch.w_fc1.float().t()
    ffn = z.shape[1] // 2
    y_ref = (torch.nn.functional.silu(z[:, :ffn]) * z[:, ffn:]) @ batch.w_fc2.float().t()
    y, saved = oracle.shared_expert_mlp_fwd(batch)
    dy = fixtures.make_grad_output("sg", tuple(y.shape))
    y_ref.backward(dy.float())
    dx = oracle.shared_expert_mlp_bwd(dy, batch, saved)
    assert (dx - x.grad).abs().max() <= 2e-2 * x.grad.abs().max()
