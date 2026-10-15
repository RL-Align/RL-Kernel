# SPDX-License-Identifier: Apache-2.0
"""Nemotron-H strict LM-head selected-token logprob: backward.

* ``grad_hidden`` rows are bitwise batch invariant (aligned ``grad_logp`` rows);
* ``grad_weight`` is bitwise reproducible, and the row/vocab chunk policy is
  bit-neutral (single-pass == chunked);
* the default budget really takes the chunked path at training sizes and
  still matches the single-pass result bitwise;
* the untied embedding ``grad_weight`` (sorted-segment reduction over repeated
  token ids) is bitwise reproducible; its FP64 parity is a known gap (BF16
  accumulation) tracked by a strict xfail;
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

from .conftest import (
    HIDDEN,
    VOCAB,
    make_hidden,
    make_targets,
    requires_nemotron_sm90,
    requires_strict_det_gemm,
)

pytestmark = requires_nemotron_sm90

_TOL = resolve_tolerance(
    load_contract(), judgment="gradient_accuracy", op_class="logprob", dtype=torch.bfloat16
)
# grad_weight is a GEMM reduction over tokens whose entries reach O(10); at that
# magnitude BF16 output rounding alone (half-ulp 0.031 in [8, 16)) nearly exhausts
# the logprob cell's absolute-only budget, so it is judged with the reduction cell.
_WEIGHT_TOL = resolve_tolerance(
    load_contract(), judgment="gradient_accuracy", op_class="reduction", dtype=torch.bfloat16
)
_EMBED_TOL = resolve_tolerance(
    load_contract(), judgment="gradient_accuracy", op_class="elementwise", dtype=torch.bfloat16
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


@requires_strict_det_gemm
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
    torch.testing.assert_close(
        grad_weight.double(), ref_weight, atol=_WEIGHT_TOL.atol, rtol=_WEIGHT_TOL.rtol
    )
    event = snapshot_backward_runtime()["nemotron_h_lm_head_logprob"]
    assert event["impl"] == "cuda_strict_dlogits_det_gemm"


@requires_strict_det_gemm
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


@requires_strict_det_gemm
def test_grad_weight_is_bitwise_reproducible(strict_op, lm_head_weight):
    hidden = make_hidden(96, seed=50)
    target = make_targets(96, seed=51)
    grad_logp = _grad_logp(96, seed=52)

    first = _run_backward(strict_op, hidden, lm_head_weight, target, grad_logp)
    second = _run_backward(strict_op, hidden, lm_head_weight, target, grad_logp)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


@requires_strict_det_gemm
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


@requires_strict_det_gemm
def test_default_budget_chunked_path_matches_single_pass(strict_op, lm_head_weight):
    from rl_engine.backends.cuda.logprob.nemotron_h_lm_head_logp import (
        DLOGITS_BUDGET_ELEMS,
        strict_lm_head_logp_backward,
    )

    # 4096 * V = 2^29 > the default 2^28 budget, so no override is needed to chunk.
    rows = 4096
    assert rows * VOCAB > DLOGITS_BUDGET_ELEMS
    hidden = make_hidden(rows, seed=80)
    target = make_targets(rows, seed=81)
    grad_logp = _grad_logp(rows, seed=82)
    with torch.no_grad():
        _logp, lse = strict_op.logprob(hidden, lm_head_weight, target)

    chunked = strict_lm_head_logp_backward(grad_logp, hidden, lm_head_weight, target, lse, None)
    single = strict_lm_head_logp_backward(
        grad_logp, hidden, lm_head_weight, target, lse, None, budget_elems=rows * VOCAB
    )

    assert torch.equal(chunked[0], single[0])
    assert torch.equal(chunked[1], single[1])


def _embedding_backward_case():
    gen = torch.Generator(device="cuda").manual_seed(90)
    # 16 distinct ids over 2 * 64 tokens: every id repeats, exercising the segment reduction.
    vocab_ids = torch.randint(0, VOCAB, (16,), generator=gen, device="cuda")
    token_ids = vocab_ids[torch.randint(0, 16, (2, 64), generator=gen, device="cuda")]
    grad_out = torch.randn(2, 64, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16)
    return token_ids, grad_out


def _embedding_grad_weight(op, embed_weight, token_ids, grad_out):
    weight = embed_weight.detach().clone().requires_grad_(True)
    op.embed(token_ids, weight).backward(grad_out)
    return weight.grad


def test_embedding_grad_weight_is_bitwise_reproducible(strict_op, embed_weight):
    token_ids, grad_out = _embedding_backward_case()

    first = _embedding_grad_weight(strict_op, embed_weight, token_ids, grad_out)
    second = _embedding_grad_weight(strict_op, embed_weight, token_ids, grad_out)

    assert first.dtype == torch.bfloat16
    assert torch.equal(first, second)
    event = snapshot_backward_runtime()["embedding"]
    assert event["impl"] == "cuda_sorted_segment_dweight"


@pytest.mark.xfail(
    strict=True,
    reason="SM90EmbeddingOp accumulates repeated-id gradients in BF16; FP32 accumulation "
    "is needed to meet the elementwise gradient_accuracy tolerance",
)
def test_embedding_grad_weight_matches_fp64(strict_op, embed_weight):
    token_ids, grad_out = _embedding_backward_case()

    grad_weight = _embedding_grad_weight(strict_op, embed_weight, token_ids, grad_out)
    reference = torch.zeros(VOCAB, HIDDEN, device="cuda", dtype=torch.float64)
    reference.index_add_(0, token_ids.reshape(-1), grad_out.reshape(-1, HIDDEN).double())

    torch.testing.assert_close(
        grad_weight.double(), reference, atol=_EMBED_TOL.atol, rtol=_EMBED_TOL.rtol
    )


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
