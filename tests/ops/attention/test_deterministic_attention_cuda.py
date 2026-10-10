# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Deterministic standard-softmax Attention tests for CUDA and ROCm.

Covers (per §7 and §8 of the implementation plan):
- Forward correctness via #108 harness (run_operator_suite)
- Backward correctness via #108 harness (check_grad=True, grad_mode="random")
- LSE correctness
- Batch invariance — Axis-A bitwise (slice, position permutation, batch-chunk)
- Sequence-dim chunked-prefill invariance (§7.4)
- Prefill/decode handoff (§7.5)
- KV-cache cat handoff (§7.5.2)
- Scale: None / 0.0 / custom
- GQA dK/dV order validation
- FP64 high-precision gradient comparison
- Gradient batch invariance
- Valid-only vs padded accuracy (not bitwise)
"""

import math

import pytest
import torch

from rl_engine.platforms.device import device_ctx

IS_ROCM = device_ctx.is_rocm
IS_CUDA = device_ctx.device_type == "cuda"
IS_GPU = IS_CUDA or IS_ROCM
BACKEND = "rocm" if IS_ROCM else "cuda"

pytestmark = pytest.mark.skipif(not IS_GPU, reason="CUDA/ROCm GPU not available")
ROCM_ONLY = pytest.mark.skipif(not IS_ROCM, reason="ROCm-only acceptance check")

try:
    from rl_engine.backends.cuda.attention.deterministic_attn import (
        DeterministicAttentionOp,
        RLKernelDeterministicAttentionCore,
    )
    from rl_engine.contracts.operators.attention import SplitKVSpec
    from rl_engine.reference.attention.standard_attn import NativeAttentionOp
    from rl_engine.validation.operators.op_checks import (
        CandidateSpec,
        OperatorCase,
        run_operator_suite,
    )

    _OP_AVAILABLE = True
except (ImportError, RuntimeError):
    _OP_AVAILABLE = False

if IS_ROCM:
    from rl_engine.backends.rocm.rope.rope import RocmDeterministicRoPEOp

pytestmark = [
    pytestmark,
    pytest.mark.skipif(not _OP_AVAILABLE, reason=f"{BACKEND} attention op not built"),
]

DEVICE = device_ctx.device
D = 128


@pytest.fixture
def attention_op():
    return DeterministicAttentionOp()


@pytest.fixture
def gold_op():
    return NativeAttentionOp()


def _tol(dtype):
    if dtype == torch.bfloat16:
        return 5e-2, 2e-2
    return 1e-3, 1e-3


# =============================================================================
# §8.4 — #108 Harness integration: forward + backward via run_operator_suite
# =============================================================================


def _make_case(name, dtype, hq, hkv, sq, skv, causal, scale=None, padding=False):
    torch.manual_seed(42)
    B = 2
    inputs = {
        "q": torch.randn(B, hq, sq, D, device=DEVICE, dtype=dtype),
        "k": torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype),
        "v": torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype),
        "causal": causal,
    }
    if scale is not None:
        inputs["scale"] = scale
    if padding:
        mask = torch.ones(B, skv, device=DEVICE, dtype=torch.bool)
        mask[0, skv // 2 :] = False
        mask[1, skv * 3 // 4 :] = False
        inputs["key_padding_mask"] = mask
    gold = NativeAttentionOp()
    return OperatorCase(
        name=name,
        op_class="attention",
        dtype=dtype,
        inputs=inputs,
        gold_fn=gold.forward_fp32,
        grad_input_names=("q", "k", "v"),
    )


def _build_harness_cases():
    return [
        _make_case("bf16-gqa4x1-16x32-causal", torch.bfloat16, 4, 1, 16, 32, True),
        _make_case("bf16-gqa32x8-16x32-causal", torch.bfloat16, 32, 8, 16, 32, True),
        _make_case("bf16-gqa1x1-3x31-nocausal", torch.bfloat16, 1, 1, 3, 31, False),
        _make_case("fp16-gqa4x2-17x33-causal", torch.float16, 4, 2, 17, 33, True),
        _make_case("fp16-gqa32x8-1x64-decode", torch.float16, 32, 8, 1, 64, True),
        _make_case("bf16-gqa4x1-16x16-nocausal", torch.bfloat16, 4, 1, 16, 16, False),
        _make_case("bf16-gqa32x8-64x65-causal", torch.bfloat16, 32, 8, 64, 65, True),
        _make_case("fp16-gqa4x1-65x127-causal", torch.float16, 4, 1, 65, 127, True),
        _make_case("bf16-pad-gqa4x1-8x16", torch.bfloat16, 4, 1, 8, 16, True, padding=True),
        _make_case("bf16-scale0-4x1-8x16", torch.bfloat16, 4, 1, 8, 16, True, scale=0.0),
        _make_case("fp16-scale-custom-4x1-8x16", torch.float16, 4, 1, 8, 16, True, scale=0.05),
    ]


def test_harness_forward():
    """§8.4: run_operator_suite forward — candidate vs gold (accuracy tolerance)."""
    attention = DeterministicAttentionOp()
    candidate = CandidateSpec(name=f"{BACKEND}-attention", fn=attention, backend=BACKEND)
    report = run_operator_suite("attention", candidates=[candidate], cases=_build_harness_cases())
    for cr in report.candidates:
        for case in cr.cases:
            if not case.passed:
                msgs = [
                    o.message or f"idx={o.output_index} max_abs={o.max_abs_error}"
                    for o in case.outputs
                    if not o.passed
                ]
                pytest.fail(f"Forward failed {case.case_name}: {msgs}")
    assert report.passed


def test_harness_backward():
    """§8.4: run_operator_suite backward — grad comparison vs gold fp32 autograd."""
    attention = DeterministicAttentionOp()
    candidate = CandidateSpec(name=f"{BACKEND}-attention", fn=attention, backend=BACKEND)
    report = run_operator_suite(
        "attention",
        candidates=[candidate],
        cases=_build_harness_cases(),
        check_grad=True,
        grad_mode="random",
    )
    for cr in report.candidates:
        for case in cr.cases:
            if not case.passed:
                msgs = [
                    o.message or f"idx={o.output_index} max_abs={o.max_abs_error}"
                    for o in case.outputs
                    if not o.passed
                ]
                pytest.fail(f"Backward failed {case.case_name}: {msgs}")
    assert report.passed


# =============================================================================
# §7.1 — Full correctness sweep (covers Sq=65, Skv=65/127, non-causal sq>skv)
# =============================================================================

SWEEP_CONFIGS = []
for dtype in [torch.bfloat16, torch.float16]:
    for hq, hkv in [(1, 1), (4, 1), (4, 2), (32, 8)]:
        for sq in [1, 3, 16, 17, 64, 65]:
            for skv in [1, 31, 32, 33, 64, 65, 127]:
                for causal in [True, False]:
                    if causal and sq > skv:
                        continue
                    SWEEP_CONFIGS.append((dtype, hq, hkv, sq, skv, causal))


@pytest.mark.parametrize("dtype,hq,hkv,sq,skv,causal", SWEEP_CONFIGS)
def test_forward_sweep(attention_op, gold_op, dtype, hq, hkv, sq, skv, causal):
    B = 2
    torch.manual_seed(42)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype)

    out_actual = attention_op.forward(q, k, v, causal=causal)
    out_gold = gold_op.forward_fp32(q, k, v, causal=causal)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out_actual.float(), out_gold.float(), atol=atol, rtol=rtol)


# =============================================================================
# §7.1 — Scale tests (None, 0.0, custom)
# =============================================================================


@pytest.mark.parametrize("scale", [None, 0.0, 0.05])
def test_scale(attention_op, gold_op, scale):
    B, hq, hkv, sq, skv = 2, 4, 1, 8, 16
    torch.manual_seed(42)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    out_actual = attention_op.forward(q, k, v, causal=True, scale=scale)
    out_gold = gold_op.forward_fp32(q, k, v, causal=True, scale=scale)

    atol, rtol = _tol(torch.bfloat16)
    torch.testing.assert_close(out_actual.float(), out_gold.float(), atol=atol, rtol=rtol)


# =============================================================================
# Padding and fully-masked row
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_forward_with_padding(attention_op, gold_op, dtype):
    B, hq, hkv, sq, skv = 2, 4, 1, 8, 16
    torch.manual_seed(7)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=dtype)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=dtype)
    mask = torch.ones(B, skv, device=DEVICE, dtype=torch.bool)
    mask[0, 10:] = False
    mask[1, 12:] = False

    out_actual = attention_op.forward(q, k, v, causal=True, key_padding_mask=mask)
    out_gold = gold_op.forward_fp32(q, k, v, causal=True, key_padding_mask=mask)

    atol, rtol = _tol(dtype)
    torch.testing.assert_close(out_actual.float(), out_gold.float(), atol=atol, rtol=rtol)


def test_fully_masked_row(attention_op):
    B, hq, hkv, sq, skv = 1, 1, 1, 2, 4
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    mask = torch.zeros(B, skv, device=DEVICE, dtype=torch.bool)

    out, lse = attention_op.forward_with_lse(q, k, v, causal=False, key_padding_mask=mask)
    assert (out == 0).all()
    assert (lse == float("-inf")).all()


# =============================================================================
# §7.2/8.5 — LSE correctness
# =============================================================================


def test_lse_correctness(attention_op):
    B, hq, hkv, sq, skv = 2, 4, 1, 8, 16
    torch.manual_seed(99)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    _, lse_actual = attention_op.forward_with_lse(q, k, v, causal=True)

    scale = 1.0 / math.sqrt(D)
    g = hq // hkv
    k_exp = k.float().repeat_interleave(g, dim=1)
    scores = scale * torch.einsum("bhqd,bhkd->bhqk", q.float(), k_exp)
    sq_idx = torch.arange(sq, device=DEVICE).unsqueeze(1)
    kv_idx = torch.arange(skv, device=DEVICE).unsqueeze(0)
    causal_mask = kv_idx <= (skv - sq + sq_idx)
    scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    lse_gold = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(lse_actual, lse_gold, atol=5e-2, rtol=2e-2)


# =============================================================================
# §7.2 — Valid-only vs padded: near-equal, NOT bitwise
# =============================================================================


def test_valid_only_vs_padded_accuracy(attention_op):
    """Padding changes reduction width; result is near-equal, not bitwise.

    We use causal=False here so that all valid keys are equally visible
    regardless of Skv. The padded path has extra columns masked to -inf,
    which changes the sum-exp denominator and can shift results slightly.
    """
    B, hq, hkv, sq, skv_valid = 1, 4, 1, 4, 8
    skv_padded = 12
    torch.manual_seed(77)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k_valid = torch.randn(B, hkv, skv_valid, D, device=DEVICE, dtype=torch.bfloat16)
    v_valid = torch.randn(B, hkv, skv_valid, D, device=DEVICE, dtype=torch.bfloat16)

    k_padded = torch.cat(
        [
            k_valid,
            torch.zeros(B, hkv, skv_padded - skv_valid, D, device=DEVICE, dtype=torch.bfloat16),
        ],
        dim=2,
    )
    v_padded = torch.cat(
        [
            v_valid,
            torch.zeros(B, hkv, skv_padded - skv_valid, D, device=DEVICE, dtype=torch.bfloat16),
        ],
        dim=2,
    )
    mask = torch.ones(B, skv_padded, device=DEVICE, dtype=torch.bool)
    mask[:, skv_valid:] = False

    out_valid = attention_op.forward(q, k_valid, v_valid, causal=False)
    out_padded = attention_op.forward(q, k_padded, v_padded, causal=False, key_padding_mask=mask)

    atol, rtol = _tol(torch.bfloat16)
    torch.testing.assert_close(out_valid.float(), out_padded.float(), atol=atol, rtol=rtol)


# =============================================================================
# §7.3 — Batch invariance (Axis-A bitwise)
# =============================================================================


def test_batch_invariance_single(attention_op):
    """Same sample in full batch vs extracted single — bitwise."""
    B, hq, hkv, sq, skv = 4, 4, 1, 8, 16
    torch.manual_seed(11)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    out_full, lse_full = attention_op.forward_with_lse(q, k, v, causal=True)

    for i in range(B):
        out_single, lse_single = attention_op.forward_with_lse(
            q[i : i + 1], k[i : i + 1], v[i : i + 1], causal=True
        )
        assert torch.equal(out_full[i : i + 1], out_single), f"Output batch invariance failed i={i}"
        assert torch.equal(lse_full[i : i + 1], lse_single), f"LSE batch invariance failed i={i}"


def test_batch_invariance_position_permutation(attention_op):
    """Same sample at different batch positions — bitwise."""
    B, hq, hkv, sq, skv = 4, 32, 8, 8, 16
    torch.manual_seed(12)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    out_full = attention_op.forward(q, k, v, causal=True)

    perm = [2, 0, 3, 1]
    q_perm = q[perm]
    k_perm = k[perm]
    v_perm = v[perm]
    out_perm = attention_op.forward(q_perm, k_perm, v_perm, causal=True)

    for new_pos, orig_pos in enumerate(perm):
        assert torch.equal(
            out_full[orig_pos], out_perm[new_pos]
        ), f"Position permutation invariance failed: orig={orig_pos} new={new_pos}"


def test_batch_invariance_chunk(attention_op):
    """Batch-dim chunking — bitwise."""
    B, hq, hkv, sq, skv = 4, 4, 1, 8, 16
    torch.manual_seed(13)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    out_full = attention_op.forward(q, k, v, causal=True)
    out_chunk = torch.cat(
        [
            attention_op.forward(q[:2], k[:2], v[:2], causal=True),
            attention_op.forward(q[2:], k[2:], v[2:], causal=True),
        ],
        dim=0,
    )
    assert torch.equal(out_full, out_chunk), "Batch-chunk invariance failed"


# =============================================================================
# §7.4 — Sequence-dim chunked-prefill invariance
# =============================================================================


@pytest.mark.parametrize(
    "chunk_size,hq,hkv",
    [
        (1, 4, 1),
        (3, 4, 1),
        (8, 4, 1),
        (1, 32, 8),
        (3, 32, 8),
    ],
)
def test_chunked_prefill(attention_op, chunk_size, hq, hkv):
    B, T = 1, 16
    torch.manual_seed(22)
    q = torch.randn(B, hq, T, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)

    out_full = attention_op.forward(q, k, v, causal=True)

    outs = []
    for t in range(0, T, chunk_size):
        c = min(chunk_size, T - t)
        outs.append(
            attention_op.forward(
                q[:, :, t : t + c], k[:, :, : t + c], v[:, :, : t + c], causal=True
            )
        )
    out_chunked = torch.cat(outs, dim=2)
    assert torch.equal(
        out_full, out_chunked
    ), f"Chunked-prefill invariance failed chunk={chunk_size} hq={hq} hkv={hkv}"


@pytest.mark.parametrize("chunk_size", [1, 3, 8])
def test_chunked_prefill_with_padding(attention_op, chunk_size):
    """§7.4: chunked-prefill with key_padding_mask (mask sliced with Skv)."""
    B, hq, hkv, T = 1, 4, 1, 16
    torch.manual_seed(23)
    q = torch.randn(B, hq, T, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)
    mask = torch.ones(B, T, device=DEVICE, dtype=torch.bool)
    mask[0, 12:] = False

    out_full = attention_op.forward(q, k, v, causal=True, key_padding_mask=mask)

    outs = []
    for t in range(0, T, chunk_size):
        c = min(chunk_size, T - t)
        outs.append(
            attention_op.forward(
                q[:, :, t : t + c],
                k[:, :, : t + c],
                v[:, :, : t + c],
                causal=True,
                key_padding_mask=mask[:, : t + c],
            )
        )
    out_chunked = torch.cat(outs, dim=2)
    assert torch.equal(
        out_full, out_chunked
    ), f"Chunked-prefill with padding failed chunk={chunk_size}"


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_chunked_prefill_lse(attention_op, chunk_size):
    """§7.4: LSE chunked-prefill invariance."""
    B, hq, hkv, T = 1, 4, 1, 12
    torch.manual_seed(24)
    q = torch.randn(B, hq, T, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, T, D, device=DEVICE, dtype=torch.bfloat16)

    _, lse_full = attention_op.forward_with_lse(q, k, v, causal=True)

    lses = []
    for t in range(0, T, chunk_size):
        c = min(chunk_size, T - t)
        _, lse_chunk = attention_op.forward_with_lse(
            q[:, :, t : t + c], k[:, :, : t + c], v[:, :, : t + c], causal=True
        )
        lses.append(lse_chunk)
    lse_chunked = torch.cat(lses, dim=2)
    assert torch.equal(lse_full, lse_chunked), "LSE chunked-prefill invariance failed"


# =============================================================================
# §7.5 — Prefill/decode handoff
# =============================================================================


def test_prefill_decode_slice(attention_op):
    """§7.5.1: prefill[:, :, -1:] == decode(q[-1:], k_full, v_full)."""
    B, hq, hkv, sq, skv = 1, 4, 1, 8, 16
    torch.manual_seed(33)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)

    prefill = attention_op.forward(q, k, v, causal=True)
    decode = attention_op.forward(q[:, :, -1:], k, v, causal=True)
    assert torch.equal(prefill[:, :, -1:], decode)


@pytest.mark.parametrize(
    "S_new,hq,hkv",
    [
        (1, 4, 1),
        (1, 32, 8),
        (3, 4, 1),
        (3, 32, 8),
    ],
)
def test_kv_cache_handoff(attention_op, S_new, hq, hkv):
    """§7.5.2: cat(k_cache, k_new) handoff == prefill tail."""
    B, S_past = 1, 12
    torch.manual_seed(44)
    q_full = torch.randn(B, hq, S_past + S_new, D, device=DEVICE, dtype=torch.bfloat16)
    k_full = torch.randn(B, hkv, S_past + S_new, D, device=DEVICE, dtype=torch.bfloat16)
    v_full = torch.randn(B, hkv, S_past + S_new, D, device=DEVICE, dtype=torch.bfloat16)

    q_new = q_full[:, :, -S_new:]
    prefill_tail = attention_op.forward(q_full, k_full, v_full, causal=True)[:, :, -S_new:]
    decode_path = attention_op.forward(q_new, k_full, v_full, causal=True)
    assert torch.equal(decode_path, prefill_tail)


def test_kv_cache_handoff_with_padding(attention_op):
    """§7.5.2: cat handoff with padding mask."""
    B, hq, hkv, S_past, S_new = 1, 4, 1, 12, 1
    Skv = S_past + S_new
    torch.manual_seed(45)
    q_full = torch.randn(B, hq, Skv, D, device=DEVICE, dtype=torch.bfloat16)
    k_full = torch.randn(B, hkv, Skv, D, device=DEVICE, dtype=torch.bfloat16)
    v_full = torch.randn(B, hkv, Skv, D, device=DEVICE, dtype=torch.bfloat16)
    mask = torch.ones(B, Skv, device=DEVICE, dtype=torch.bool)
    mask[0, 8:10] = False

    q_new = q_full[:, :, -S_new:]
    prefill_tail = attention_op.forward(q_full, k_full, v_full, causal=True, key_padding_mask=mask)[
        :, :, -S_new:
    ]
    decode_path = attention_op.forward(q_new, k_full, v_full, causal=True, key_padding_mask=mask)
    assert torch.equal(decode_path, prefill_tail)


# =============================================================================
# §7.6 — Backward
# =============================================================================


def test_backward_smoke(attention_op):
    """Backward runs and produces gradients with correct shapes."""
    B, hq, hkv, sq, skv = 1, 4, 1, 4, 8
    torch.manual_seed(55)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    out = attention_op.forward(q, k, v, causal=True)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None and q.grad.shape == q.shape
    assert k.grad is not None and k.grad.shape == k.shape
    assert v.grad is not None and v.grad.shape == v.shape


def test_backward_fp64_reference(attention_op):
    """§7.6: FP64 high-precision gradient comparison."""
    B, hq, hkv, sq, skv = 1, 4, 1, 4, 8
    torch.manual_seed(56)
    q_bf16 = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k_bf16 = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    v_bf16 = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

    grad_out = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.float32)

    out_actual = attention_op.forward(q_bf16, k_bf16, v_bf16, causal=True)
    out_actual.backward(grad_out.to(out_actual.dtype))
    dq_actual = q_bf16.grad.float()
    dk_actual = k_bf16.grad.float()
    dv_actual = v_bf16.grad.float()

    scale = 1.0 / math.sqrt(D)
    g = hq // hkv
    q64 = q_bf16.detach().double().requires_grad_(True)
    k64 = k_bf16.detach().double().requires_grad_(True)
    v64 = v_bf16.detach().double().requires_grad_(True)
    k64_exp = k64.repeat_interleave(g, dim=1)
    v64_exp = v64.repeat_interleave(g, dim=1)
    scores64 = scale * torch.einsum("bhqd,bhkd->bhqk", q64, k64_exp)
    sq_idx = torch.arange(sq, device=DEVICE).unsqueeze(1)
    kv_idx = torch.arange(skv, device=DEVICE).unsqueeze(0)
    causal_mask = kv_idx <= (skv - sq + sq_idx)
    scores64 = scores64.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    P64 = torch.softmax(scores64, dim=-1)
    out64 = torch.einsum("bhqk,bhkd->bhqd", P64, v64_exp)
    out64.backward(grad_out.double())
    dq_gold = q64.grad.float()
    dk_gold = k64.grad.float()
    dv_gold = v64.grad.float()

    torch.testing.assert_close(dq_actual, dq_gold, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dk_actual, dk_gold, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(dv_actual, dv_gold, atol=5e-2, rtol=2e-2)


def test_gradient_batch_invariance(attention_op):
    """§7.6: dQ/dK/dV bitwise identical for same sample at different batch positions."""
    B, hq, hkv, sq, skv = 3, 4, 1, 4, 8
    torch.manual_seed(57)
    q = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(B, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    grad_out = torch.randn(B, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)

    q_full = q.clone().requires_grad_(True)
    k_full = k.clone().requires_grad_(True)
    v_full = v.clone().requires_grad_(True)
    out_full = attention_op.forward(q_full, k_full, v_full, causal=True)
    out_full.backward(grad_out)

    for i in range(B):
        qi = q[i : i + 1].clone().requires_grad_(True)
        ki = k[i : i + 1].clone().requires_grad_(True)
        vi = v[i : i + 1].clone().requires_grad_(True)
        out_i = attention_op.forward(qi, ki, vi, causal=True)
        out_i.backward(grad_out[i : i + 1])
        assert torch.equal(q_full.grad[i : i + 1], qi.grad), f"dQ batch invariance failed i={i}"
        assert torch.equal(k_full.grad[i : i + 1], ki.grad), f"dK batch invariance failed i={i}"
        assert torch.equal(v_full.grad[i : i + 1], vi.grad), f"dV batch invariance failed i={i}"


def test_gqa_dk_dv_order(attention_op):
    """§7.6/§4.1: GQA dK/dV must follow fixed (hq_local, query_index) order.

    Two checks:
    1. Batch-size invariance: same sample at B=1 vs B=4 gives bitwise-identical dK/dV
       (catches unordered atomics or grid-shape-dependent accumulation).
    2. Correctness vs FP64 gold: with asymmetric per-head Q, a reversed local order
       would produce different numerical results. This catches "deterministic but wrong".
    """
    hq, hkv, sq, skv = 32, 8, 4, 8
    torch.manual_seed(58)
    q_data = torch.randn(1, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)
    k_data = torch.randn(1, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    v_data = torch.randn(1, hkv, skv, D, device=DEVICE, dtype=torch.bfloat16)
    grad_data = torch.randn(1, hq, sq, D, device=DEVICE, dtype=torch.bfloat16)

    # Check 1: batch-size invariance
    q1 = q_data.clone().requires_grad_(True)
    k1 = k_data.clone().requires_grad_(True)
    v1 = v_data.clone().requires_grad_(True)
    attention_op.forward(q1, k1, v1, causal=True).backward(grad_data)

    q_batch = q_data.expand(4, -1, -1, -1).contiguous().clone().requires_grad_(True)
    k_batch = k_data.expand(4, -1, -1, -1).contiguous().clone().requires_grad_(True)
    v_batch = v_data.expand(4, -1, -1, -1).contiguous().clone().requires_grad_(True)
    grad_batch = grad_data.expand(4, -1, -1, -1).contiguous()
    attention_op.forward(q_batch, k_batch, v_batch, causal=True).backward(grad_batch)

    assert torch.equal(k1.grad, k_batch.grad[0:1]), "dK GQA order depends on batch size"
    assert torch.equal(v1.grad, v_batch.grad[0:1]), "dV GQA order depends on batch size"

    # Check 2: correctness vs FP64 reference (catches reversed hq_local order)
    scale = 1.0 / math.sqrt(D)
    g = hq // hkv
    q64 = q_data.detach().double().requires_grad_(True)
    k64 = k_data.detach().double().requires_grad_(True)
    v64 = v_data.detach().double().requires_grad_(True)
    k64_exp = k64.repeat_interleave(g, dim=1)
    v64_exp = v64.repeat_interleave(g, dim=1)
    scores64 = scale * torch.einsum("bhqd,bhkd->bhqk", q64, k64_exp)
    sq_idx = torch.arange(sq, device=DEVICE).unsqueeze(1)
    kv_idx = torch.arange(skv, device=DEVICE).unsqueeze(0)
    causal_mask = kv_idx <= (skv - sq + sq_idx)
    scores64 = scores64.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    P64 = torch.softmax(scores64, dim=-1)
    out64 = torch.einsum("bhqk,bhkd->bhqd", P64, v64_exp)
    out64.backward(grad_data.double())
    dk_gold = k64.grad.float()
    dv_gold = v64.grad.float()

    torch.testing.assert_close(k1.grad.float(), dk_gold, atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(v1.grad.float(), dv_gold, atol=5e-2, rtol=2e-2)


# =============================================================================
# Strict core acceptance and ROCm HIP RoPE acceptance
# =============================================================================


def _strict_qkv(*, batch: int = 2, sequence: int = 7):
    generator = torch.Generator(device="cpu").manual_seed(942)
    q = torch.randn(batch, 4, sequence, D, dtype=torch.bfloat16, generator=generator).to(DEVICE)
    k = torch.randn(batch, 1, sequence, D, dtype=torch.bfloat16, generator=generator).to(DEVICE)
    v = torch.randn(batch, 1, sequence, D, dtype=torch.bfloat16, generator=generator).to(DEVICE)
    return q, k, v


def test_strict_attention_core_is_repeat_bitwise_and_no_fallback():
    q, k, v = _strict_qkv()
    positions = torch.arange(q.size(2), device=q.device).expand(q.size(0), -1)
    core = RLKernelDeterministicAttentionCore()
    first = core.forward_with_lse(
        q,
        k,
        v,
        query_position_ids=positions,
        key_position_ids=positions,
    )
    second = core.forward_with_lse(
        q,
        k,
        v,
        query_position_ids=positions,
        key_position_ids=positions,
    )
    assert torch.equal(first.out, second.out)
    assert torch.equal(first.lse, second.lse)
    assert first.provenance["attention_backend"] == f"rlkernel.{BACKEND}.deterministic_attention"
    assert first.provenance["fallback"] is False
    assert first.provenance["split_kv"]["actual_split_kv_policy"] == "disabled"


def test_strict_attention_forward_backward_train_rollout_bitwise():
    q, k, v = (tensor.requires_grad_() for tensor in _strict_qkv(batch=1, sequence=5))
    positions = torch.arange(q.size(2), device=q.device).expand(q.size(0), -1)
    core = RLKernelDeterministicAttentionCore()
    train = core.forward_with_lse(
        q,
        k,
        v,
        query_position_ids=positions,
        key_position_ids=positions,
    )
    grad = torch.randn(train.out.shape, dtype=train.out.dtype, device="cpu").to(DEVICE)
    (train.out.float() * grad.float()).sum().backward()
    train_grads = tuple(tensor.grad.detach().clone() for tensor in (q, k, v))

    q2, k2, v2 = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    rollout = core.forward_with_lse(
        q2,
        k2,
        v2,
        query_position_ids=positions,
        key_position_ids=positions,
    )
    (rollout.out.float() * grad.float()).sum().backward()
    assert torch.equal(train.out, rollout.out)
    assert torch.equal(train.lse, rollout.lse)
    assert all(
        torch.equal(expected, actual.grad)
        for expected, actual in zip(train_grads, (q2, k2, v2), strict=True)
    )


@ROCM_ONLY
def test_rocm_rope_is_batch_invariant_and_backward_repeat_bitwise():
    generator = torch.Generator(device="cpu").manual_seed(714)
    x = torch.randn(3, 4, 6, D, dtype=torch.bfloat16, generator=generator).to(DEVICE)
    positions = torch.arange(6, device=x.device).expand(3, -1)
    rope = RocmDeterministicRoPEOp()
    together = rope(x, positions)
    separate = torch.cat([rope(x[index : index + 1], positions[index]) for index in range(3)])
    assert torch.equal(together, separate)
    assert torch.equal(together, rope(x, positions))

    grad = torch.randn(together.shape, dtype=together.dtype, device="cpu").to(DEVICE)
    x1 = x.detach().clone().requires_grad_()
    x2 = x.detach().clone().requires_grad_()
    (rope(x1, positions).float() * grad.float()).sum().backward()
    (rope(x2, positions).float() * grad.float()).sum().backward()
    assert torch.equal(x1.grad, x2.grad)


@ROCM_ONLY
def test_rocm_rope_matches_fp32_rotate_half_reference():
    x = torch.randn(1, 2, 4, D, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.arange(4, device=x.device).expand(1, -1)
    actual = RocmDeterministicRoPEOp()(x, positions)
    half = x.size(-1) // 2
    inv_freq = 1.0 / (
        1_000_000.0 ** (torch.arange(half, dtype=torch.float32, device=x.device) / half)
    )
    frequency = positions.float().unsqueeze(-1) * inv_freq
    cos = frequency.cos().unsqueeze(1)
    sin = frequency.sin().unsqueeze(1)
    reference = torch.cat(
        (
            x[..., :half].float() * cos - x[..., half:].float() * sin,
            x[..., half:].float() * cos + x[..., :half].float() * sin,
        ),
        dim=-1,
    ).to(x.dtype)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@ROCM_ONLY
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("tokens", (1, 2, 7, 32))
def test_rocm_rope_pair_matches_two_single_calls_raw_bytes(dtype, tokens):
    generator = torch.Generator(device="cpu").manual_seed(810 + tokens)
    query = torch.randn(8, tokens, D, dtype=dtype, generator=generator).to(DEVICE)
    key = torch.randn(2, tokens, D, dtype=dtype, generator=generator).to(DEVICE)
    position_storage = torch.arange(tokens * 2, device=DEVICE, dtype=torch.int64)
    positions = position_storage[::2]
    assert not positions.is_contiguous() or tokens == 1
    rope = RocmDeterministicRoPEOp()

    expected = (rope(query, positions), rope(key, positions))
    actual = rope.forward_pair(query, key, positions)
    repeated = rope.forward_pair(query, key, positions)
    for expected_tensor, actual_tensor, repeated_tensor in zip(
        expected,
        actual,
        repeated,
        strict=True,
    ):
        assert torch.equal(
            expected_tensor.contiguous().view(torch.uint8),
            actual_tensor.contiguous().view(torch.uint8),
        )
        assert torch.equal(
            actual_tensor.contiguous().view(torch.uint8),
            repeated_tensor.contiguous().view(torch.uint8),
        )


@ROCM_ONLY
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("tokens", (1, 2, 32))
def test_rocm_rope_pair_inference_bypasses_autograd_raw_bytes(monkeypatch, dtype, tokens):
    from rl_engine.backends.rocm.rope.rope import _RocmRoPEPairFunction

    generator = torch.Generator(device="cpu").manual_seed(813 + tokens)
    position_storage = torch.arange(tokens * 2, device=DEVICE, dtype=torch.int64)
    positions = position_storage[::2]
    query = torch.randn(8, tokens, D, dtype=dtype, generator=generator).to(DEVICE)
    key = torch.randn(2, tokens, D, dtype=dtype, generator=generator).to(DEVICE)
    query.requires_grad_()
    key.requires_grad_()
    rope = RocmDeterministicRoPEOp()

    with torch.inference_mode():
        expected = _RocmRoPEPairFunction.apply(query, key, positions, 1_000_000.0)

    def unexpected_apply(*args, **kwargs):
        raise AssertionError("no-grad paired RoPE entered its autograd Function")

    monkeypatch.setattr(_RocmRoPEPairFunction, "apply", unexpected_apply)
    for mode in (torch.no_grad, torch.inference_mode):
        with mode():
            actual = rope.forward_pair(query, key, positions)
            repeated = rope.forward_pair(query, key, positions)

        for expected_tensor, actual_tensor, repeated_tensor in zip(
            expected,
            actual,
            repeated,
            strict=True,
        ):
            assert actual_tensor.requires_grad is False
            assert actual_tensor.grad_fn is None
            assert torch.equal(expected_tensor.view(torch.uint8), actual_tensor.view(torch.uint8))
            assert torch.equal(actual_tensor.view(torch.uint8), repeated_tensor.view(torch.uint8))


@ROCM_ONLY
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_rocm_rope_pair_inference_reuses_only_position_independent_frequency(dtype):
    from rl_engine.backends.rocm.rope.rope import _forward_rope_pair

    generator = torch.Generator(device="cpu").manual_seed(814)
    query = torch.randn(8, 3, D, dtype=dtype, generator=generator).to(DEVICE)
    key = torch.randn(2, 3, D, dtype=dtype, generator=generator).to(DEVICE)
    first_positions = torch.tensor([0, 17, 2048], device=DEVICE)
    second_positions = torch.tensor([4095, 3, 29], device=DEVICE)
    rope = RocmDeterministicRoPEOp()

    with torch.inference_mode():
        first = rope.forward_pair(query, key, first_positions)
        assert len(rope._inference_inv_freq_cache) == 1
        cached_inv_freq = next(iter(rope._inference_inv_freq_cache.values()))
        expected_second = _forward_rope_pair(
            query,
            key,
            second_positions,
            1_000_000.0,
        )
        cached_second = _forward_rope_pair(
            query,
            key,
            second_positions,
            1_000_000.0,
            inv_freq=cached_inv_freq,
        )
    with torch.no_grad():
        second = rope.forward_pair(query, key, second_positions)

    assert len(rope._inference_inv_freq_cache) == 1
    assert next(iter(rope._inference_inv_freq_cache.values())) is cached_inv_freq
    assert not torch.equal(first[0].view(torch.uint8), second[0].view(torch.uint8))
    for expected_tensor, cached_tensor in zip(expected_second, cached_second, strict=True):
        assert torch.equal(expected_tensor.view(torch.uint8), cached_tensor.view(torch.uint8))
    for expected_tensor, actual_tensor in zip(expected_second[:2], second, strict=True):
        assert torch.equal(expected_tensor.view(torch.uint8), actual_tensor.view(torch.uint8))


@ROCM_ONLY
def test_rocm_rope_pair_inference_frequency_cache_keys_shape_and_theta():
    from rl_engine.backends.rocm.rope.rope import _forward_rope_pair

    rope = RocmDeterministicRoPEOp()
    positions = torch.tensor([1, 2], device=DEVICE)

    def run(head_dim, theta):
        query = torch.randn(2, 2, head_dim, dtype=torch.bfloat16, device=DEVICE)
        key = torch.randn(1, 2, head_dim, dtype=torch.bfloat16, device=DEVICE)
        with torch.inference_mode():
            expected = _forward_rope_pair(query, key, positions, theta)[:2]
            actual = rope.forward_pair(query, key, positions, theta=theta)
        for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
            assert torch.equal(expected_tensor.view(torch.uint8), actual_tensor.view(torch.uint8))
        return rope._cached_inference_inv_freq(query, theta)

    grad_query = torch.randn(2, 2, D, dtype=torch.bfloat16, device=DEVICE).requires_grad_()
    grad_key = torch.randn(1, 2, D, dtype=torch.bfloat16, device=DEVICE).requires_grad_()
    rope.forward_pair(grad_query, grad_key, positions)
    assert not rope._inference_inv_freq_cache

    initial = run(D, 1_000_000.0)
    assert run(D, 1_000_000.0) is initial
    changed_theta = run(D, 10_000.0)
    assert changed_theta is not initial
    changed_shape = run(D // 2, 10_000.0)
    assert changed_shape is not changed_theta
    assert changed_shape.numel() == D // 4

    cache_entries = len(rope._inference_inv_freq_cache)
    trained_query = grad_query.detach().clone().requires_grad_()
    trained_key = grad_key.detach().clone().requires_grad_()
    trained_out = rope.forward_pair(trained_query, trained_key, positions)
    (trained_out[0].float().sum() + trained_out[1].float().sum()).backward()
    assert trained_query.grad is not None
    assert trained_key.grad is not None
    assert len(rope._inference_inv_freq_cache) == cache_entries


@ROCM_ONLY
def test_rocm_rope_pair_inference_frequency_cache_is_stream_local():
    from rl_engine.backends.rocm.rope.rope import _forward_rope_pair

    rope = RocmDeterministicRoPEOp()
    query = torch.randn(2, 2, D, dtype=torch.bfloat16, device=DEVICE)
    key = torch.randn(1, 2, D, dtype=torch.bfloat16, device=DEVICE)
    positions = torch.tensor([7, 31], device=DEVICE)
    streams = (torch.cuda.Stream(device=DEVICE), torch.cuda.Stream(device=DEVICE))
    outputs = []
    repeated_outputs = []
    expected = []
    torch.cuda.synchronize(DEVICE)

    for stream in streams:
        theta = 1_000_000.0
        with torch.cuda.stream(stream), torch.inference_mode():
            expected.append(_forward_rope_pair(query, key, positions, theta)[:2])
            outputs.append(rope.forward_pair(query, key, positions, theta=theta))
            repeated_outputs.append(rope.forward_pair(query, key, positions, theta=theta))
    torch.cuda.synchronize(DEVICE)

    assert len(rope._inference_inv_freq_cache) == 2
    assert len({id(value) for value in rope._inference_inv_freq_cache.values()}) == 2
    for expected_pair, actual_pair, repeated_pair in zip(
        expected,
        outputs,
        repeated_outputs,
        strict=True,
    ):
        for expected_tensor, actual_tensor, repeated_tensor in zip(
            expected_pair,
            actual_pair,
            repeated_pair,
            strict=True,
        ):
            assert torch.equal(expected_tensor.view(torch.uint8), actual_tensor.view(torch.uint8))
            assert torch.equal(actual_tensor.view(torch.uint8), repeated_tensor.view(torch.uint8))


@ROCM_ONLY
@pytest.mark.parametrize("tokens", (2, 32))
def test_vllm_rocm_rope_pair_adapter_matches_legacy_raw_bytes(tokens):
    from rl_engine.integrations.engines.rollout.vllm.runtime import (
        _patch_strict_rocm_rotary_embedding,
    )

    generator = torch.Generator(device="cpu").manual_seed(812 + tokens)
    query = torch.randn(tokens, 8 * D, dtype=torch.bfloat16, generator=generator).to(DEVICE)
    key = torch.randn(tokens, 2 * D, dtype=torch.bfloat16, generator=generator).to(DEVICE)
    positions = torch.arange(tokens, device=DEVICE, dtype=torch.int64)
    rope = RocmDeterministicRoPEOp()

    def legacy(value):
        heads = value.shape[1] // D
        head_major = value.view(tokens, heads, D).permute(1, 0, 2).contiguous()
        return rope(head_major, positions).permute(1, 0, 2).reshape_as(value).contiguous()

    class Rotary:
        head_size = D
        rotary_dim = D

        def forward_cuda(self, positions, query, key=None):
            return query, key

    expected = legacy(query), legacy(key)
    _patch_strict_rocm_rotary_embedding(Rotary)
    actual = Rotary().forward_cuda(positions, query, key)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.equal(expected_tensor.view(torch.uint8), actual_tensor.view(torch.uint8))


@ROCM_ONLY
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_rocm_rope_pair_backward_matches_two_single_calls_raw_bytes(dtype):
    generator = torch.Generator(device="cpu").manual_seed(811)
    positions = torch.tensor([9, 2, 41, 2, 77, 5, 103], device=DEVICE)
    query = torch.randn(8, positions.numel(), D, dtype=dtype, generator=generator)
    key = torch.randn(2, positions.numel(), D, dtype=dtype, generator=generator)
    query = query.to(DEVICE)
    key = key.to(DEVICE)
    query_grad = torch.randn(query.shape, dtype=query.dtype, generator=generator).to(DEVICE)
    key_grad = torch.randn(key.shape, dtype=key.dtype, generator=generator).to(DEVICE)
    rope = RocmDeterministicRoPEOp()

    expected_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in (query, key))
    expected_outputs = tuple(rope(tensor, positions) for tensor in expected_inputs)
    (expected_outputs[0].float() * query_grad.float()).sum().backward()
    (expected_outputs[1].float() * key_grad.float()).sum().backward()

    actual_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in (query, key))
    actual_outputs = rope.forward_pair(actual_inputs[0], actual_inputs[1], positions)
    loss = (actual_outputs[0].float() * query_grad.float()).sum()
    loss = loss + (actual_outputs[1].float() * key_grad.float()).sum()
    loss.backward()

    for expected_input, actual_input in zip(expected_inputs, actual_inputs, strict=True):
        assert torch.equal(
            expected_input.grad.contiguous().view(torch.uint8),
            actual_input.grad.contiguous().view(torch.uint8),
        )


@ROCM_ONLY
@pytest.mark.parametrize("query_requires_grad", (False, True))
def test_rocm_rope_pair_preserves_independent_autograd_semantics(query_requires_grad):
    positions = torch.tensor([1, 3], device=DEVICE)
    query = torch.randn(8, 2, D, device=DEVICE, dtype=torch.bfloat16)
    key = torch.randn(2, 2, D, device=DEVICE, dtype=torch.bfloat16)
    query.requires_grad_(query_requires_grad)
    key.requires_grad_(not query_requires_grad)

    query_out, key_out = RocmDeterministicRoPEOp().forward_pair(query, key, positions)

    assert query_out.requires_grad is query_requires_grad
    assert key_out.requires_grad is (not query_requires_grad)


@ROCM_ONLY
@pytest.mark.parametrize("cached_length", (32, 127, 512))
def test_rocm_aiter_decode_direct_output_matches_staged_raw_bytes(cached_length):
    from rl_engine.backends.rocm.attention.flash_attn import StrictRocmAiterCKAttentionCore

    generator = torch.Generator(device="cpu").manual_seed(820 + cached_length)
    q = torch.randn(1, 4, 1, D, dtype=torch.bfloat16, generator=generator).to(DEVICE)
    k = torch.randn(1, 1, cached_length, D, dtype=torch.bfloat16, generator=generator).to(DEVICE)
    v = torch.randn(1, 1, cached_length, D, dtype=torch.bfloat16, generator=generator).to(DEVICE)
    core = StrictRocmAiterCKAttentionCore()

    with torch.inference_mode():
        staged = core.forward_with_lse(q, k, v, causal=False)
        caller_out = torch.empty_like(q)
        direct = core.forward_decode_with_lse_into(q, k, v, out=caller_out)
        repeated_out = torch.empty_like(q)
        repeated = core.forward_decode_with_lse_into(q, k, v, out=repeated_out)

    assert direct.out is caller_out
    assert direct.out.data_ptr() == caller_out.data_ptr()
    assert direct.provenance["core_output_staging"] == "aiter_direct_caller_group"
    for expected, actual, replay in (
        (staged.out, direct.out, repeated.out),
        (staged.lse, direct.lse, repeated.lse),
    ):
        expected_bytes = expected.contiguous().view(torch.uint8)
        actual_bytes = actual.contiguous().view(torch.uint8)
        replay_bytes = replay.contiguous().view(torch.uint8)
        assert torch.equal(expected_bytes, actual_bytes)
        assert torch.equal(actual_bytes, replay_bytes)


def test_strict_core_rejects_split_k():
    with pytest.raises(ValueError, match="Split-KV"):
        RLKernelDeterministicAttentionCore(split_kv=SplitKVSpec.fixed(32))
