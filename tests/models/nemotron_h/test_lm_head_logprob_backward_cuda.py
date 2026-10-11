# SPDX-License-Identifier: Apache-2.0
"""Nemotron-H strict LM-head selected-token logprob: backward.

* ``grad_hidden`` rows are bitwise batch invariant (aligned ``grad_logp`` rows);
* ``grad_weight`` is bitwise reproducible, and the row/vocab chunk policy is
  bit-neutral (single-pass == chunked);
* the strict dlogits kernel equals the same formula evaluated with torch on the
  forward kernel's own FP32 logits;
* both gradients match the independent FP64 VJP at the WS1 logprob
  ``gradient_accuracy`` tolerance.
"""
from __future__ import annotations

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.models.nemotron_h.reference import selected_logprob_vjp_fp64
from rl_engine.ops.autograd.backward_runtime import snapshot_backward_runtime

from .conftest import VOCAB, make_hidden, make_targets, requires_nemotron_sm90

pytestmark = requires_nemotron_sm90

_TOL = resolve_tolerance(
    load_contract(), judgment="gradient_accuracy", op_class="logprob", dtype=torch.bfloat16
)


def _grad_logp(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, generator=gen, device="cuda", dtype=torch.float32)


def _run_backward(op, hidden, weight, target, grad_logp, temperature=None):
    hidden = hidden.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    logp, _lse = op.logprob(hidden, weight, target, temperature=temperature)
    logp.backward(grad_logp)
    return hidden.grad, weight.grad


@pytest.mark.parametrize("use_temperature", [False, True])
def test_gradients_match_fp64_vjp(strict_op, lm_head_weight, use_temperature):
    hidden = make_hidden(64, seed=30)
    target = make_targets(64, seed=31)
    grad_logp = _grad_logp(64, seed=32)
    temperature = (
        torch.linspace(0.6, 1.4, 64, device="cuda", dtype=torch.float32)
        if use_temperature
        else None
    )

    grad_hidden, grad_weight = _run_backward(
        strict_op, hidden, lm_head_weight, target, grad_logp, temperature
    )
    ref_hidden, ref_weight = selected_logprob_vjp_fp64(
        hidden, lm_head_weight, target, grad_logp, temperature=temperature
    )

    assert grad_hidden.dtype == torch.bfloat16 and grad_weight.dtype == torch.bfloat16
    torch.testing.assert_close(grad_hidden.double(), ref_hidden, atol=_TOL.atol, rtol=_TOL.rtol)
    torch.testing.assert_close(grad_weight.double(), ref_weight, atol=_TOL.atol, rtol=_TOL.rtol)
    event = snapshot_backward_runtime()["nemotron_h_lm_head_logprob"]
    assert event["impl"] == "cuda_strict_dlogits_det_gemm"


def test_grad_hidden_rows_are_batch_invariant(strict_op, lm_head_weight):
    rows = 300
    hidden = make_hidden(rows, seed=40)
    target = make_targets(rows, seed=41)
    grad_logp = _grad_logp(rows, seed=42)
    ref_hidden, _ = _run_backward(strict_op, hidden, lm_head_weight, target, grad_logp)

    for batch in (1, 7, 64, 257):
        gen = torch.Generator(device="cpu").manual_seed(batch)
        index = torch.randperm(rows, generator=gen)[:batch].to("cuda")
        grad_hidden, _ = _run_backward(
            strict_op, hidden[index], lm_head_weight, target[index], grad_logp[index]
        )
        assert torch.equal(grad_hidden, ref_hidden[index]), f"batch={batch}"


def test_grad_weight_is_bitwise_reproducible(strict_op, lm_head_weight):
    hidden = make_hidden(96, seed=50)
    target = make_targets(96, seed=51)
    grad_logp = _grad_logp(96, seed=52)

    first = _run_backward(strict_op, hidden, lm_head_weight, target, grad_logp)
    second = _run_backward(strict_op, hidden, lm_head_weight, target, grad_logp)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_chunk_policy_is_bit_neutral(strict_op, lm_head_weight):
    from rl_engine.backends.cuda.logprob.nemotron_h_lm_head_logp import strict_lm_head_logp_backward

    rows = 300
    hidden = make_hidden(rows, seed=60)
    target = make_targets(rows, seed=61)
    grad_logp = _grad_logp(rows, seed=62)
    temperature = torch.full((rows,), 0.85, device="cuda", dtype=torch.float32)
    _logp, lse = strict_op.logprob(hidden, lm_head_weight, target, temperature=temperature)

    single = strict_lm_head_logp_backward(
        grad_logp, hidden, lm_head_weight, target, lse, temperature
    )
    # 64 * V elements forces row chunks of 256 and five vocab chunks.
    chunked = strict_lm_head_logp_backward(
        grad_logp, hidden, lm_head_weight, target, lse, temperature, budget_elems=64 * VOCAB
    )

    assert torch.equal(single[0], chunked[0])
    assert torch.equal(single[1], chunked[1])


def test_strict_dlogits_kernel_matches_torch_formula(strict_op, lm_head_weight):
    from rl_engine.backends.extension import _C

    rows = 48
    hidden = make_hidden(rows, seed=70)
    target = make_targets(rows, seed=71)
    grad_logp = _grad_logp(rows, seed=72)
    temperature = torch.linspace(0.7, 1.3, rows, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        _logp, lse, logits = strict_op.logprob_with_logits(
            hidden, lm_head_weight, target, temperature=temperature
        )
    start, stop = 8192, 8192 + 4096

    dlogits = _C.fused_linear_logp_sm90_strict_dlogits(
        hidden, lm_head_weight[start:stop], target, grad_logp, lse, temperature, start, -1
    )

    inv_t = (1.0 / temperature)[:, None]
    expected = -torch.exp(logits[:, start:stop] * inv_t - lse[:, None])
    hit = (target >= start) & (target < stop)
    expected[hit.nonzero().squeeze(1), (target[hit] - start)] += 1.0
    expected = (expected * grad_logp[:, None] * inv_t).to(torch.bfloat16)
    assert torch.equal(dlogits, expected)
