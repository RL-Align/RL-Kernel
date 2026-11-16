# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""NativeGatedDeltaNetOp tests (Qwen3-Next linear-attention reference).

Covers:
- Vectorized recurrence vs an independent fp64 nested-loop golden
- Prefill + decode continuation == one-shot prefill
- initial_state passthrough; g=None (no decay) path
- Axis A batch invariance (slice + padding)
- Shape/dtype guards
- Registry dispatch
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.reference.linear_attn.gated_deltanet import NativeGatedDeltaNetOp
from rl_engine.runtime.registry import OpBackend, kernel_registry

# Qwen3-Next GDN head geometry (80B-A3B): head_k_dim = head_v_dim = 128 is
# the production value; smaller here so the fp64 golden stays fast.
_B, _H, _T = 2, 3, 9


def _golden_gdn_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta_logits: torch.Tensor,
    g_logits: torch.Tensor | None,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Independent transcription of the canonical recurrence, plain loops.

    GQA grouping: q/k carry Hk heads, v/o/beta/g carry Hv heads; value
    head h reads key head h // (Hv // Hk).
    """
    batch, key_heads, seqlen, dim_k = q.shape
    value_heads = v.shape[1]
    group = value_heads // key_heads
    dim_v = v.shape[-1]
    q, k, v = q.double(), k.double(), v.double()
    outs = torch.zeros(batch, value_heads, seqlen, dim_v, dtype=torch.float64, device=q.device)
    for b in range(batch):
        for h in range(value_heads):
            hk = h // group
            s = (
                initial_state[b, h].double().clone()
                if initial_state is not None
                else torch.zeros(dim_k, dim_v, dtype=torch.float64, device=q.device)
            )
            for t in range(seqlen):
                qn = q[b, hk, t] / (q[b, hk, t].square().sum() + 1e-6).sqrt()
                kn = k[b, hk, t] / (k[b, hk, t].square().sum() + 1e-6).sqrt()
                if g_logits is not None:
                    s = s * g_logits[b, h, t].double().exp()
                beta = torch.sigmoid(beta_logits[b, h, t].double())
                kv = kn @ s
                s = s + beta * torch.outer(kn, v[b, h, t] - kv)
                outs[b, h, t] = (qn * dim_k**-0.5) @ s
    return outs


@pytest.fixture
def op() -> NativeGatedDeltaNetOp:
    return NativeGatedDeltaNetOp()


def _make_inputs(seed: int, batch: int = _B, heads: int = _H, seqlen: int = _T):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, seqlen, 16, generator=gen)
    k = torch.randn(batch, heads, seqlen, 16, generator=gen)
    v = torch.randn(batch, heads, seqlen, 24, generator=gen)
    beta = torch.randn(batch, heads, seqlen, generator=gen)
    g = -torch.rand(batch, heads, seqlen, generator=gen)
    return q, k, v, beta, g


def test_matches_golden(op: NativeGatedDeltaNetOp) -> None:
    q, k, v, beta, g = _make_inputs(0)
    expected = _golden_gdn_fp64(q, k, v, beta, g)
    out = op.forward_fp32(q, k, v, beta, g)
    torch.testing.assert_close(out, expected.float(), rtol=0, atol=1e-4)


def test_matches_golden_no_decay(op: NativeGatedDeltaNetOp) -> None:
    q, k, v, beta, _ = _make_inputs(1)
    expected = _golden_gdn_fp64(q, k, v, beta, None)
    out = op.forward_fp32(q, k, v, beta, None)
    torch.testing.assert_close(out, expected.float(), rtol=0, atol=1e-4)


def test_initial_state_passthrough(op: NativeGatedDeltaNetOp) -> None:
    q, k, v, beta, g = _make_inputs(2)
    state = torch.randn(_B, _H, 16, 24)
    expected = _golden_gdn_fp64(q, k, v, beta, g, initial_state=state)
    out = op.forward_fp32(q, k, v, beta, g, initial_state=state)
    torch.testing.assert_close(out, expected.float(), rtol=0, atol=1e-4)


def test_prefill_then_decode_equals_one_shot(op: NativeGatedDeltaNetOp) -> None:
    """Prefill T-1 tokens, decode the last one: must equal one-shot prefill."""
    q, k, v, beta, g = _make_inputs(3)
    one_shot, final = op.forward_fp32(q, k, v, beta, g, initial_state=None, return_final_state=True)
    assert final.dtype == torch.float32

    _, state = op.forward_fp32(
        q[:, :, :-1],
        k[:, :, :-1],
        v[:, :, :-1],
        beta[:, :, :-1],
        g[:, :, :-1],
        return_final_state=True,
    )
    step, _ = op.decode_step(
        q[:, :, -1:], k[:, :, -1:], v[:, :, -1:], beta[:, :, -1:], g[:, :, -1:], state
    )
    torch.testing.assert_close(step, one_shot[:, :, -1:].float(), rtol=0, atol=1e-5)


def test_split_prefill_equals_one_shot(op: NativeGatedDeltaNetOp) -> None:
    q, k, v, beta, g = _make_inputs(4, seqlen=12)
    one_shot = op.forward_fp32(q, k, v, beta, g)
    first, state = op.forward_fp32(
        q[:, :, :5], k[:, :, :5], v[:, :, :5], beta[:, :, :5], g[:, :, :5], return_final_state=True
    )
    second = op.forward_fp32(
        q[:, :, 5:], k[:, :, 5:], v[:, :, 5:], beta[:, :, 5:], g[:, :, 5:], initial_state=state
    )
    torch.testing.assert_close(torch.cat([first, second], dim=2), one_shot, rtol=0, atol=1e-5)


def test_batch_invariance_slice_and_padding(op: NativeGatedDeltaNetOp) -> None:
    """Axis A: batch rows never interact; slicing/padding cannot change a row."""
    q, k, v, beta, g = _make_inputs(5, batch=6)
    full = op.forward_fp32(q, k, v, beta, g)

    sliced = op.forward_fp32(q[2:5], k[2:5], v[2:5], beta[2:5], g[2:5])
    torch.testing.assert_close(sliced, full[2:5], rtol=0, atol=0)

    zeros = lambda t: torch.zeros_like(t[2:5])  # noqa: E731
    padded = op.forward_fp32(
        torch.cat([q[2:5], zeros(q)]),
        torch.cat([k[2:5], zeros(k)]),
        torch.cat([v[2:5], zeros(v)]),
        torch.cat([beta[2:5], zeros(beta)]),
        torch.cat([g[2:5], zeros(g)]),
    )
    torch.testing.assert_close(padded[:3], full[2:5], rtol=0, atol=0)


def test_batch_invariance_by_construction(op: NativeGatedDeltaNetOp) -> None:
    """Padding with garbage rows (not zeros) still leaves real rows unchanged."""
    q, k, v, beta, g = _make_inputs(6, batch=3)
    full = op.forward_fp32(q, k, v, beta, g)
    tail = _make_inputs(7, batch=2)
    padded = op.forward_fp32(
        torch.cat([q, tail[0]]),
        torch.cat([k, tail[1]]),
        torch.cat([v, tail[2]]),
        torch.cat([beta, tail[3]]),
        torch.cat([g, tail[4]]),
    )
    torch.testing.assert_close(padded[:3], full, rtol=0, atol=0)


def test_gqa_grouping_matches_golden(op: NativeGatedDeltaNetOp) -> None:
    """Qwen3-Next GQA: q/k Hk=2 heads shared by Hv=6 value heads (group 3)."""
    gen = torch.Generator().manual_seed(11)
    B, Hk, Hv, T = 2, 2, 6, 7
    q = torch.randn(B, Hk, T, 16, generator=gen)
    k = torch.randn(B, Hk, T, 16, generator=gen)
    v = torch.randn(B, Hv, T, 24, generator=gen)
    beta = torch.randn(B, Hv, T, generator=gen)
    g = -torch.rand(B, Hv, T, generator=gen)
    state = torch.randn(B, Hv, 16, 24, generator=gen)

    expected = _golden_gdn_fp64(q, k, v, beta, g, initial_state=state)
    out, final = op.forward_fp32(q, k, v, beta, g, initial_state=state, return_final_state=True)
    torch.testing.assert_close(out, expected.float(), rtol=0, atol=1e-4)
    assert final.shape == (B, Hv, 16, 24)

    # GQA prefill + decode continuation still equals one-shot prefill.
    _, state2 = op.forward_fp32(
        q[:, :, :-1],
        k[:, :, :-1],
        v[:, :, :-1],
        beta[:, :, :-1],
        g[:, :, :-1],
        initial_state=state,
        return_final_state=True,
    )
    step, _ = op.decode_step(
        q[:, :, -1:], k[:, :, -1:], v[:, :, -1:], beta[:, :, -1:], g[:, :, -1:], state2
    )
    torch.testing.assert_close(step, out[:, :, -1:].float(), rtol=0, atol=1e-5)


def test_backward_gradient_flow(op: NativeGatedDeltaNetOp) -> None:
    q, k, v, beta, g = _make_inputs(8, seqlen=5)
    q = q.requires_grad_(True)
    v = v.requires_grad_(True)
    out = op(q, k, v, beta, g)
    out.sum().backward()
    assert q.grad is not None and q.grad.shape == q.shape
    assert v.grad is not None and v.grad.shape == v.shape


def test_dtype_cast_back(op: NativeGatedDeltaNetOp) -> None:
    q, k, v, beta, g = _make_inputs(9)
    out = op(q.bfloat16(), k.bfloat16(), v.bfloat16(), beta.bfloat16(), g.bfloat16())
    assert out.dtype == torch.bfloat16


def test_guards(op: NativeGatedDeltaNetOp) -> None:
    q, k, v, beta, g = _make_inputs(10)
    with pytest.raises(ValueError):
        op(q[:1], k, v, beta, g)  # q batch mismatch -> k.shape != q.shape
    with pytest.raises(ValueError):
        op(q, k, v[:1], beta, g)  # v batch mismatch
    with pytest.raises(ValueError):
        # Hv=4 is not divisible by Hk=3 (slicing the 3-head v can't produce it)
        op(q, k, torch.randn(_B, 4, _T, 24), torch.randn(_B, 4, _T), torch.randn(_B, 4, _T))
    with pytest.raises(ValueError):
        op(q, k, v, beta[:, :, :-1], g)  # beta shape
    with pytest.raises(ValueError):
        op(q, k, v, beta, g[:, :, :-1])  # g shape
    with pytest.raises(ValueError):
        op(q, k, v, beta, g, initial_state=torch.zeros(_B, _H, 8, 24))  # state shape
    with pytest.raises(TypeError):
        op(q.double(), k.double(), v.double(), beta.double(), g.double())


def test_registry_dispatch_cpu() -> None:
    resolved = kernel_registry.get_op("gated_deltanet", device="cpu")
    assert isinstance(resolved, NativeGatedDeltaNetOp)
    candidates = kernel_registry._priority_map["cpu"]["gated_deltanet"]
    assert candidates == [OpBackend.PYTORCH_GATED_DELTANET]


@pytest.mark.parametrize("dim_k", [7, 33])
def test_matches_upstream_recurrent_reference(op, dim_k):
    """Compare against the newly merged provider reference, including gate conversion."""
    from torch.nn import functional as F

    from rl_engine.reference.linear_attn import GatedDeltaRuleRecurrentStepOp

    gen = torch.Generator().manual_seed(652)
    batch, hk, hv, dim_v = 2, 2, 4, 5
    q = torch.randn(batch, hk, 1, dim_k, generator=gen)
    k = torch.randn(batch, hk, 1, dim_k, generator=gen)
    v = torch.randn(batch, hv, 1, dim_v, generator=gen)
    a, b = (torch.randn(batch, hv, generator=gen) for _ in range(2))
    A_log, dt_bias = (torch.randn(hv, generator=gen) for _ in range(2))
    bank = torch.randn(batch + 1, hv, dim_v, dim_k, generator=gen)
    packed = torch.cat(
        [q[:, :, 0].flatten(1), k[:, :, 0].flatten(1), v[:, :, 0].flatten(1)], dim=-1
    )
    expected, expected_bank = GatedDeltaRuleRecurrentStepOp().forward_fp32(
        packed,
        a,
        b,
        A_log,
        dt_bias,
        bank,
        torch.arange(1, batch + 1),
        scale=dim_k**-0.5,
        num_k_heads=hk,
    )
    log_decay = -A_log.exp() * F.softplus(a + dt_bias)
    actual, state = op.forward_fp32(
        q,
        k,
        v,
        b.unsqueeze(-1),
        log_decay.unsqueeze(-1),
        bank[1:].transpose(-1, -2),
        return_final_state=True,
    )
    torch.testing.assert_close(actual.transpose(1, 2), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(state.transpose(-1, -2), expected_bank[1:], rtol=2e-5, atol=2e-6)


def test_all_input_gradients_match_fp64_oracle(op):
    tensors = [x.requires_grad_() for x in _make_inputs(19, batch=1, heads=1, seqlen=3)]
    state = torch.randn(1, 1, 16, 24, requires_grad=True)
    actual = op.forward_fp32(*tensors, initial_state=state)
    expected = _golden_gdn_fp64(*tensors, initial_state=state)
    inputs = tensors + [state]
    actual_grads = torch.autograd.grad(actual.sum(), inputs)
    expected_grads = torch.autograd.grad(expected.sum(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-4, atol=2e-6)


def test_empty_sequence_preserves_state(op):
    q, k, v, beta, g = _make_inputs(21, seqlen=0)
    state = torch.randn(_B, _H, 16, 24, requires_grad=True)
    out, updated = op(q, k, v, beta, g, state, return_final_state=True)
    assert out.shape == v.shape
    torch.testing.assert_close(updated, state, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(updated.sum(), state)[0], torch.ones_like(state))


def test_decode_requires_one_token(op):
    q, k, v, beta, g = _make_inputs(22)
    with pytest.raises(ValueError, match="one token"):
        op.decode_step(q, k, v, beta, g, torch.zeros(_B, _H, 16, 24))


def test_extreme_beta_has_finite_gradient(op):
    q, k, v, beta, g = _make_inputs(23)
    beta = torch.full_like(beta, -100.0).requires_grad_()
    grad = torch.autograd.grad(op(q, k, v, beta, g).sum(), beta)[0]
    assert torch.isfinite(grad).all()
