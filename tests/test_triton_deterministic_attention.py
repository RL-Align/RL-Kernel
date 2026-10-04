# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Bitwise parity between the Triton and native deterministic Attention cores.

The Triton core in ``rl_engine.kernels.ops.triton.attention.deterministic_attn`` is a
port of ``csrc/cuda/attention/deterministic_attention.cu``. Its contract is stronger
than "numerically close": every tensor it returns must be bit-identical to the native
kernel's. These tests pin that contract, plus the two properties the port depends on:
the vendor-exact ``expf``/``logf`` helpers, and batch invariance.
"""

import math

import pytest
import torch

from rl_engine.platforms.device import device_ctx

IS_ROCM = device_ctx.is_rocm
IS_GPU = device_ctx.device_type == "cuda" or IS_ROCM

pytestmark = pytest.mark.skipif(not IS_GPU, reason="CUDA/ROCm GPU not available")

try:
    from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
    from rl_engine.kernels.ops.cuda.attention.deterministic_attn import DeterministicAttentionOp
    from rl_engine.kernels.ops.triton.attention.deterministic_attn import (
        BITWISE_LIBM_PARITY,
        TritonDeterministicAttentionOp,
        triton_deterministic_attention_backward,
        triton_deterministic_attention_forward,
    )

    _IMPORTED = True
except (ImportError, RuntimeError):  # pragma: no cover - import guard
    _IMPORTED = False

_NATIVE = _IMPORTED and _EXT_AVAILABLE and hasattr(_C, "deterministic_attention_forward")

needs_native = pytest.mark.skipif(
    not _NATIVE, reason="native deterministic attention kernel not built"
)
needs_bitwise_libm = pytest.mark.skipif(
    not (_IMPORTED and BITWISE_LIBM_PARITY),
    reason="bitwise expf/logf sequence is only ported for ROCm",
)

DEVICE = device_ctx.device
D = 128

# (B, Hq, Hkv, Sq, Skv, causal, mask_kind, scale)
_CASES = [
    (1, 1, 1, 1, 1, True, None, None),
    (1, 2, 2, 1, 64, True, None, None),  # decode step
    (1, 8, 2, 64, 64, True, None, None),  # GQA, group 4
    (2, 4, 1, 128, 128, True, None, None),  # MQA
    (1, 2, 2, 256, 256, True, None, None),  # exactly one softmax lane chunk
    (1, 2, 2, 257, 257, True, None, None),  # one past the chunk boundary
    (1, 2, 2, 100, 512, False, None, None),  # two full lane chunks
    (2, 4, 2, 64, 700, True, None, None),  # ragged multi-chunk
    (1, 2, 2, 32, 32, True, "right", None),
    (1, 2, 2, 32, 32, False, "left", None),
    (2, 2, 2, 16, 300, True, "right", None),
    (2, 2, 2, 8, 8, True, "allfalse", None),  # fully masked row -> lse == -inf
    (1, 2, 2, 16, 16, True, None, 0.0),
    (1, 2, 2, 16, 16, True, None, 3.7),
    (1, 2, 2, 16, 16, False, None, -1.25),
]


def _make_inputs(case, dtype, seed):
    b, hq, hkv, sq, skv, causal, mask_kind, scale = case
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    q = torch.randn(b, hq, sq, D, device=DEVICE, dtype=dtype, generator=gen)
    k = torch.randn(b, hkv, skv, D, device=DEVICE, dtype=dtype, generator=gen)
    v = torch.randn(b, hkv, skv, D, device=DEVICE, dtype=dtype, generator=gen)

    mask = None
    if mask_kind is not None:
        mask = torch.ones(b, skv, device=DEVICE, dtype=torch.bool)
        if mask_kind == "right":
            mask[:, skv // 2 :] = False
        elif mask_kind == "left":
            mask[:, : skv // 3] = False
        elif mask_kind == "allfalse":
            mask[0, :] = False
        else:  # pragma: no cover - guards the parametrisation itself
            raise AssertionError(f"unknown mask kind {mask_kind}")

    resolved_scale = 1.0 / math.sqrt(D) if scale is None else scale
    return q, k, v, causal, resolved_scale, mask


def _case_id(case):
    b, hq, hkv, sq, skv, causal, mask_kind, scale = case
    return f"b{b}_hq{hq}_hkv{hkv}_sq{sq}_skv{skv}_causal{int(causal)}_{mask_kind}_{scale}"


@needs_native
@needs_bitwise_libm
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("case", _CASES, ids=[_case_id(c) for c in _CASES])
def test_triton_forward_is_bitwise_identical_to_native(case, dtype):
    q, k, v, causal, scale, mask = _make_inputs(case, dtype, seed=11)

    ref_out, ref_lse, ref_p = _C.deterministic_attention_forward(q, k, v, causal, scale, mask)
    out, lse, p = triton_deterministic_attention_forward(q, k, v, causal, scale, mask, False)

    assert torch.equal(out, ref_out)
    assert torch.equal(lse, ref_lse)
    assert torch.equal(p, ref_p)


@needs_native
@needs_bitwise_libm
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("case", _CASES, ids=[_case_id(c) for c in _CASES])
def test_triton_backward_is_bitwise_identical_to_native(case, dtype):
    q, k, v, causal, scale, mask = _make_inputs(case, dtype, seed=12)

    _ref_out, _ref_lse, ref_p = _C.deterministic_attention_forward(q, k, v, causal, scale, mask)
    _out, _lse, p = triton_deterministic_attention_forward(q, k, v, causal, scale, mask, False)
    grad_out = torch.randn_like(q)

    ref_dq, ref_dk, ref_dv = _C.deterministic_attention_backward(
        grad_out, q, k, v, ref_p, causal, scale, mask
    )
    dq, dk, dv = triton_deterministic_attention_backward(grad_out, q, k, v, p, scale)

    assert torch.equal(dq, ref_dq)
    assert torch.equal(dk, ref_dk)
    assert torch.equal(dv, ref_dv)


@needs_native
@needs_bitwise_libm
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_triton_autograd_matches_native_autograd_bitwise(dtype):
    b, hq, hkv, sq, skv = 2, 8, 2, 96, 300
    gen = torch.Generator(device=DEVICE).manual_seed(13)
    base = (
        torch.randn(b, hq, sq, D, device=DEVICE, dtype=dtype, generator=gen),
        torch.randn(b, hkv, skv, D, device=DEVICE, dtype=dtype, generator=gen),
        torch.randn(b, hkv, skv, D, device=DEVICE, dtype=dtype, generator=gen),
    )
    grad_out = torch.randn(b, hq, sq, D, device=DEVICE, dtype=dtype, generator=gen)

    results = []
    for op in (DeterministicAttentionOp(), TritonDeterministicAttentionOp()):
        tensors = [t.clone().requires_grad_(True) for t in base]
        out, lse = op.forward_with_lse(*tensors, causal=True)
        out.backward(grad_out)
        results.append((out.detach(), lse, [t.grad for t in tensors]))

    native, triton_result = results
    assert torch.equal(triton_result[0], native[0])
    assert torch.equal(triton_result[1], native[1])
    for triton_grad, native_grad in zip(triton_result[2], native[2]):
        assert torch.equal(triton_grad, native_grad)


@needs_bitwise_libm
def test_triton_fp32_output_downcasts_to_the_native_dtype_result():
    """``output_fp32=True`` must expose the same accumulator the bf16 path rounds."""
    q, k, v, causal, scale, mask = _make_inputs(_CASES[6], torch.bfloat16, seed=14)

    out, _lse, _p = triton_deterministic_attention_forward(q, k, v, causal, scale, mask, False)
    out_fp32, _lse32, _p32 = triton_deterministic_attention_forward(
        q, k, v, causal, scale, mask, True
    )

    assert out_fp32.dtype is torch.float32
    assert torch.equal(out_fp32.to(torch.bfloat16), out)


@needs_bitwise_libm
def test_triton_fully_masked_row_is_zero_with_neg_inf_lse():
    q, k, v, causal, scale, mask = _make_inputs(_CASES[11], torch.bfloat16, seed=15)

    out, lse, p = triton_deterministic_attention_forward(q, k, v, causal, scale, mask, False)

    assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert torch.equal(p[0], torch.zeros_like(p[0]))
    assert torch.isneginf(lse[0]).all()
    assert torch.isfinite(lse[1]).all()


@needs_bitwise_libm
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_triton_batch_slice_is_bitwise_invariant(dtype):
    """A row's result must not depend on what else was in the batch."""
    b, hq, hkv, sq, skv = 4, 4, 2, 48, 192
    gen = torch.Generator(device=DEVICE).manual_seed(16)
    q = torch.randn(b, hq, sq, D, device=DEVICE, dtype=dtype, generator=gen)
    k = torch.randn(b, hkv, skv, D, device=DEVICE, dtype=dtype, generator=gen)
    v = torch.randn(b, hkv, skv, D, device=DEVICE, dtype=dtype, generator=gen)
    scale = 1.0 / math.sqrt(D)

    batched, batched_lse, _ = triton_deterministic_attention_forward(
        q, k, v, True, scale, None, False
    )
    single, single_lse, _ = triton_deterministic_attention_forward(
        q[2:3], k[2:3], v[2:3], True, scale, None, False
    )

    assert torch.equal(single[0], batched[2])
    assert torch.equal(single_lse[0], batched_lse[2])


@needs_bitwise_libm
def test_expf_and_logf_match_the_vendor_libm_bitwise():
    """The softmax parity rests on these two helpers; pin them independently."""
    import triton

    from rl_engine.kernels.ops.triton.attention import deterministic_attn as mod

    import triton.language as tl  # isort: skip

    @triton.jit
    def _exp_probe(x_ptr, out_ptr, n_elem, EXPF: tl.constexpr):
        offs = tl.program_id(0) * 256 + tl.arange(0, 256)
        keep = offs < n_elem
        value = tl.load(x_ptr + offs, mask=keep, other=0.0)
        tl.store(out_ptr + offs, EXPF(value), mask=keep)

    gen = torch.Generator(device=DEVICE).manual_seed(17)
    edge = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 88.72283935546875, 88.73, -103.2789306640625, -104.0],
        device=DEVICE,
        dtype=torch.float32,
    )
    xs = torch.cat(
        [torch.rand(1 << 18, device=DEVICE, generator=gen) * 240.0 - 130.0, edge]
    ).contiguous()
    got = torch.empty_like(xs)
    _exp_probe[(triton.cdiv(xs.numel(), 256),)](xs, got, xs.numel(), EXPF=mod._expf)
    assert torch.equal(got, torch.exp(xs))

    positives = torch.cat(
        [
            torch.rand(1 << 18, device=DEVICE, generator=gen) * 1e3,
            torch.rand(1 << 12, device=DEVICE, generator=gen) * 1e-38,  # subnormal inputs
            torch.tensor([1.0, 1e-45, 3.4e38], device=DEVICE),
        ]
    ).contiguous()
    got = torch.empty_like(positives)
    _exp_probe[(triton.cdiv(positives.numel(), 256),)](
        positives, got, positives.numel(), EXPF=mod._logf
    )
    assert torch.equal(got, torch.log(positives))


def test_op_refuses_to_run_without_a_bitwise_libm():
    """On a platform with no ported expf/logf the op must fail loudly, not silently."""
    if BITWISE_LIBM_PARITY:
        op = TritonDeterministicAttentionOp()
        assert op.bitwise_libm is True
    else:  # pragma: no cover - exercised on CUDA
        with pytest.raises(RuntimeError, match="bitwise-identical"):
            TritonDeterministicAttentionOp()
        assert TritonDeterministicAttentionOp(require_bitwise_libm=False).bitwise_libm is False


@needs_bitwise_libm
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"head_dim": 64}, "head dim D must be 128"),
        ({"dtype": torch.float32}, "only FP16/BF16 supported"),
        ({"gqa_mismatch": True}, "not divisible"),
    ],
)
def test_triton_op_validation_mirrors_the_native_op(kwargs, message):
    head_dim = kwargs.get("head_dim", D)
    dtype = kwargs.get("dtype", torch.bfloat16)
    hkv = 3 if kwargs.get("gqa_mismatch") else 2
    q = torch.randn(1, 2, 8, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(1, hkv, 8, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(1, hkv, 8, head_dim, device=DEVICE, dtype=dtype)

    with pytest.raises(ValueError, match=message):
        TritonDeterministicAttentionOp().forward(q, k, v, causal=True)
