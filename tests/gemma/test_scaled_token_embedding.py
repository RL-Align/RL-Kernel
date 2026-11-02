# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU reference checks and real CUDA/ROCm tests for the Gemma scaled token embedding.

Forward is held to bitwise equality with the HF expression. Backward is held to the
contract oracle (stable sort by id, fp32 left fold of ``g * s``, one final cast), which
differs from HF autograd on purpose: HF rounds ``g * s`` to bf16 before summing.
"""

import argparse

import pytest
import torch
import torch.nn.functional as F

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.embedding.scaled_embedding import (
    NativeScaledTokenEmbeddingOp,
    embedding_scale_for_dtype,
)
from rl_engine.runtime.registry import KernelRegistry, OpBackend
from rl_engine.validation.operators.op_checks import run_operator_suite
from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case

_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_HIDDEN = 5376
_EMBED_SCALE = _HIDDEN**0.5
_GEMMA_VOCAB = 262144
_CONTRACT = load_contract()
# Three bf16 rows for one id whose fp32 products cancel: the contract sum rounds to
# -7.90625, while rounding each product to bf16 first (HF) sums to -8.1875.
_CANCELLING_ROWS = [[-0.85546875], [0.94140625], [-0.193359375]]


def _rand_weight(vocab, hidden, *, dtype, device="cpu", seed=415):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(vocab, hidden, generator=generator).to(device=device, dtype=dtype)


def _rand_ids(shape, vocab, *, device="cpu", seed=416):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, shape, generator=generator).to(device)


def _rand_grad(shape, *, dtype, device="cpu", seed=417):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)


def _hf_forward(token_ids, weight, embed_scale=_EMBED_SCALE):
    """The ``Gemma4TextScaledWordEmbedding`` expression, kept literal."""
    scale = torch.tensor(embed_scale, device=weight.device).to(weight.dtype)
    return F.embedding(token_ids.long(), weight) * scale


def _hf_grad_weight(token_ids, weight, grad_out, embed_scale=_EMBED_SCALE):
    leaf = weight.detach().clone().requires_grad_(True)
    (grad_weight,) = torch.autograd.grad(_hf_forward(token_ids, leaf, embed_scale), leaf, grad_out)
    return grad_weight


def _contract_grad_weight(token_ids, grad_out, *, vocab, scale, dtype):
    """Section-2 oracle written out in Python; ``torch.sum`` would not fix the order."""
    ids = token_ids.reshape(-1).long().cpu()
    rows = grad_out.reshape(ids.numel(), -1).float().cpu()
    folded = {}
    for position in torch.argsort(ids, stable=True).tolist():
        token = int(ids[position])
        term = rows[position] * scale
        folded[token] = term if token not in folded else folded[token] + term
    result = torch.zeros((vocab, rows.shape[1]), dtype=torch.float32)
    for token, value in folded.items():
        result[token] = value
    return result.to(dtype)


def _evaluate(op, token_ids, weight, grad_out, embed_scale=_EMBED_SCALE):
    leaf = weight.detach().clone().requires_grad_(True)
    out = op(token_ids, leaf, embed_scale)
    (grad_weight,) = torch.autograd.grad(out, leaf, grad_outputs=grad_out)
    assert out.shape == (*token_ids.shape, weight.shape[1]) and out.dtype == weight.dtype
    assert grad_weight.shape == weight.shape and grad_weight.dtype == weight.dtype
    return out.detach(), grad_weight.detach()


def _duplicate_heavy_ids(vocab, *, device="cpu"):
    ids = _rand_ids((4, 16), vocab, device=device)
    ids[0, 0] = 0
    ids[0, 1] = vocab - 1
    ids[1, :8] = 7
    ids[2, ::2] = vocab - 1
    return ids


def _assert_within_contract(actual, expected, judgment):
    tolerance = resolve_tolerance(
        _CONTRACT, judgment=judgment, op_class="elementwise", dtype=actual.dtype
    )
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.device).float(),
        atol=tolerance.atol,
        rtol=tolerance.rtol,
    )


def _max_ulp_gap(actual, expected):
    bits = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}
    delta = actual.view(bits[actual.dtype]).long() - expected.view(bits[expected.dtype]).long()
    return int(delta.abs().max().item()) if delta.numel() else 0


# --------------------------------------------------------------------------- #
# PyTorch reference (CPU)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.bfloat16, 73.5),
        (torch.float16, 73.3125),
        (torch.float32, torch.tensor(_EMBED_SCALE).item()),
    ],
)
def test_scale_rounds_to_weight_dtype_before_multiplying(dtype, expected):
    assert embedding_scale_for_dtype(_EMBED_SCALE, dtype) == expected


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("hidden", [_HIDDEN, 7])
def test_native_forward_matches_hf_bitwise(dtype, hidden):
    vocab = 301
    weight = _rand_weight(vocab, hidden, dtype=dtype)
    token_ids = _duplicate_heavy_ids(vocab)
    op = NativeScaledTokenEmbeddingOp()

    out = op(token_ids, weight, _EMBED_SCALE)
    assert torch.equal(out, _hf_forward(token_ids, weight))
    assert torch.equal(out, op.forward_fp32(token_ids, weight, _EMBED_SCALE).to(dtype))
    assert torch.equal(op(token_ids.int(), weight, _EMBED_SCALE), out)


def test_unrounded_fp32_scale_breaks_bf16_parity():
    weight = _rand_weight(64, 32, dtype=torch.bfloat16)
    weight[0, 0] = 1.25
    token_ids = torch.arange(64)
    unrounded = (F.embedding(token_ids, weight).float() * _EMBED_SCALE).to(torch.bfloat16)
    assert not torch.equal(unrounded, _hf_forward(token_ids, weight))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_native_backward_matches_contract_oracle(dtype):
    vocab = 37
    weight = _rand_weight(vocab, _HIDDEN, dtype=dtype)
    token_ids = _duplicate_heavy_ids(vocab)
    grad_out = _rand_grad((*token_ids.shape, _HIDDEN), dtype=dtype)

    _, grad_weight = _evaluate(NativeScaledTokenEmbeddingOp(), token_ids, weight, grad_out)
    scale = embedding_scale_for_dtype(_EMBED_SCALE, dtype)
    expected = _contract_grad_weight(token_ids, grad_out, vocab=vocab, scale=scale, dtype=dtype)
    assert torch.equal(grad_weight, expected)


def test_native_backward_keeps_contract_sum_on_cancelling_rows():
    weight = torch.zeros(4, 1, dtype=torch.bfloat16)
    token_ids = torch.tensor([2, 2, 2])
    grad_out = torch.tensor(_CANCELLING_ROWS, dtype=torch.bfloat16)

    _, grad_weight = _evaluate(NativeScaledTokenEmbeddingOp(), token_ids, weight, grad_out)
    assert grad_weight[2, 0].item() == -7.90625
    assert _hf_grad_weight(token_ids, weight, grad_out)[2, 0].item() == -8.1875


def test_native_inference_does_not_require_gradients():
    op = NativeScaledTokenEmbeddingOp()
    weight = _rand_weight(8, 4, dtype=torch.float32)
    token_ids = torch.tensor([[1, 2], [3, 0]])
    assert not op(token_ids, weight, _EMBED_SCALE).requires_grad
    with torch.no_grad():
        assert not op(token_ids, weight.requires_grad_(True), _EMBED_SCALE).requires_grad


@pytest.mark.parametrize("platform", ["cpu", "musa", "npu"])
def test_registry_reference_dispatch(platform, monkeypatch):
    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    op = registry.get_op("scaled_token_embedding")
    assert isinstance(op, NativeScaledTokenEmbeddingOp)


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_registry_falls_back_to_native_when_triton_is_unavailable(platform, monkeypatch):
    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    load_backend = registry._load_backend
    attempted = []

    def without_triton(backend):
        attempted.append(backend)
        if backend == OpBackend.TRITON_SCALED_TOKEN_EMBEDDING:
            return None
        return load_backend(backend)

    monkeypatch.setattr(registry, "_load_backend", without_triton)
    assert isinstance(registry.get_op("scaled_token_embedding"), NativeScaledTokenEmbeddingOp)
    assert attempted == [
        OpBackend.TRITON_SCALED_TOKEN_EMBEDDING,
        OpBackend.PYTORCH_NATIVE_SCALED_TOKEN_EMBEDDING,
    ]


def _harness_args(candidate):
    return argparse.Namespace(
        op="scaled_token_embedding",
        candidate=candidate,
        arch_key=None,
        batch=2,
        seq=16,
        vocab=257,
        normalized_dim=_HIDDEN,
        seed=415,
        input_mode="random",
    )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pytorch_candidate_in_existing_accuracy_harness(dtype):
    args = _harness_args("pytorch")
    case = make_operator_case(args, dtype, torch.device("cpu"))
    assert case.inputs["weight"].shape == (257, _HIDDEN)
    assert case.inputs["embed_scale"] == _EMBED_SCALE
    report = run_operator_suite(
        "scaled_token_embedding", candidates=[make_candidate(args)], cases=[case], check_grad=True
    )
    assert report.passed


# --------------------------------------------------------------------------- #
# Triton (CUDA / ROCm)
# --------------------------------------------------------------------------- #
@pytest.fixture
def triton_op():
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    # On a GPU host a missing Triton backend must fail, not silently test the fallback.
    from rl_engine.backends.shared.triton.embedding.scaled_embedding import (
        TritonScaledTokenEmbeddingOp,
    )

    return TritonScaledTokenEmbeddingOp()


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("hidden", [_HIDDEN, 7])
def test_triton_forward_matches_hf_bitwise(triton_op, dtype, hidden):
    vocab = 301
    weight = _rand_weight(vocab, hidden, dtype=dtype, device="cuda")
    token_ids = _duplicate_heavy_ids(vocab, device="cuda")

    out = triton_op(token_ids, weight, _EMBED_SCALE)
    assert torch.equal(out, _hf_forward(token_ids, weight))
    assert torch.equal(triton_op(token_ids.int(), weight, _EMBED_SCALE), out)
    assert torch.equal(triton_op(token_ids[:0], weight, _EMBED_SCALE), out[:0])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_forward_and_backward_match_native_bitwise(triton_op, dtype):
    vocab = 37
    weight = _rand_weight(vocab, _HIDDEN, dtype=dtype, device="cuda")
    token_ids = _duplicate_heavy_ids(vocab, device="cuda")
    grad_out = _rand_grad((*token_ids.shape, _HIDDEN), dtype=dtype, device="cuda")

    triton_out, triton_grad = _evaluate(triton_op, token_ids, weight, grad_out)
    native_out, native_grad = _evaluate(NativeScaledTokenEmbeddingOp(), token_ids, weight, grad_out)
    assert torch.equal(triton_out, native_out)
    assert torch.equal(triton_grad, native_grad)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("hidden", [_HIDDEN, 7])
def test_triton_backward_matches_contract_oracle(triton_op, dtype, hidden):
    vocab = 37
    weight = _rand_weight(vocab, hidden, dtype=dtype, device="cuda")
    token_ids = _duplicate_heavy_ids(vocab, device="cuda")
    grad_out = _rand_grad((*token_ids.shape, hidden), dtype=dtype, device="cuda")

    _, grad_weight = _evaluate(triton_op, token_ids, weight, grad_out)
    scale = embedding_scale_for_dtype(_EMBED_SCALE, dtype)
    expected = _contract_grad_weight(token_ids, grad_out, vocab=vocab, scale=scale, dtype=dtype)
    assert torch.equal(grad_weight.cpu(), expected)


def test_triton_backward_keeps_contract_sum_on_cancelling_rows(triton_op):
    weight = torch.zeros(4, 1, dtype=torch.bfloat16, device="cuda")
    token_ids = torch.tensor([2, 2, 2], device="cuda")
    grad_out = torch.tensor(_CANCELLING_ROWS, dtype=torch.bfloat16, device="cuda")

    _, grad_weight = _evaluate(triton_op, token_ids, weight, grad_out)
    assert grad_weight[2, 0].item() == -7.90625


@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_noncontiguous_weight_matches_contiguous(triton_op, dtype):
    vocab = 37
    wide = _rand_weight(vocab, 2 * _HIDDEN, dtype=dtype, device="cuda").requires_grad_(True)
    weight = wide[:, :_HIDDEN]
    assert not weight.is_contiguous()
    token_ids = _duplicate_heavy_ids(vocab, device="cuda")
    grad_out = _rand_grad((*token_ids.shape, _HIDDEN), dtype=dtype, device="cuda")

    out = triton_op(token_ids, weight, _EMBED_SCALE)
    (wide_grad,) = torch.autograd.grad(out, wide, grad_outputs=grad_out)
    expected_out, expected_grad = _evaluate(triton_op, token_ids, weight.contiguous(), grad_out)
    assert torch.equal(out, expected_out)
    assert torch.equal(wide_grad[:, :_HIDDEN], expected_grad)
    assert not wide_grad[:, _HIDDEN:].any()


@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_forward_is_batch_invariant(triton_op, dtype):
    vocab = 301
    seq = 8
    weight = _rand_weight(vocab, _HIDDEN, dtype=dtype, device="cuda")
    base = _rand_ids((1, seq), vocab, device="cuda")
    reference = triton_op(base, weight, _EMBED_SCALE)

    assert torch.equal(triton_op(base, weight, _EMBED_SCALE), reference)
    for batch in (2, 16, 300):
        ids = _rand_ids((batch, seq), vocab, device="cuda", seed=batch)
        ids[0] = base[0]
        ids[batch // 2] = base[0]
        out = triton_op(ids, weight, _EMBED_SCALE)
        assert torch.equal(out[0], reference[0])
        assert torch.equal(out[batch // 2], reference[0])
    shifted = triton_op(torch.roll(base, 3, dims=1), weight, _EMBED_SCALE)
    assert torch.equal(torch.roll(shifted, -3, dims=1), reference)
    padding = _rand_ids((1, 5), vocab, device="cuda", seed=99)
    padded = triton_op(torch.cat([base, padding], dim=1), weight, _EMBED_SCALE)
    assert torch.equal(padded[:, :seq], reference)
    for shape in ((seq,), (2, seq // 2), (seq, 1)):
        reshaped = triton_op(base.reshape(shape), weight, _EMBED_SCALE)
        assert torch.equal(reshaped.reshape(1, seq, _HIDDEN), reference)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_backward_is_repeatable(triton_op, dtype):
    vocab = 37
    weight = _rand_weight(vocab, _HIDDEN, dtype=dtype, device="cuda")
    token_ids = _duplicate_heavy_ids(vocab, device="cuda")
    grad_out = _rand_grad((*token_ids.shape, _HIDDEN), dtype=dtype, device="cuda")

    grads = [_evaluate(triton_op, token_ids, weight, grad_out)[1] for _ in range(3)]
    assert torch.equal(grads[0], grads[1])
    assert torch.equal(grads[0], grads[2])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_backward_is_invariant_to_leading_shape(triton_op, dtype):
    vocab = 37
    weight = _rand_weight(vocab, _HIDDEN, dtype=dtype, device="cuda")
    flat_ids = _duplicate_heavy_ids(vocab, device="cuda").reshape(-1)
    flat_grad = _rand_grad((flat_ids.numel(), _HIDDEN), dtype=dtype, device="cuda")

    expected_out, expected_grad = _evaluate(triton_op, flat_ids, weight, flat_grad)
    for shape in ((4, 16), (16, 4), (2, 2, 16)):
        out, grad = _evaluate(
            triton_op, flat_ids.reshape(shape), weight, flat_grad.reshape(*shape, _HIDDEN)
        )
        assert torch.equal(out.reshape(-1, _HIDDEN), expected_out)
        assert torch.equal(grad, expected_grad)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_fused_scale_matches_prescaled_rows_in_reducer(dtype):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.embedding.embedding import _embedding_grad_weight

    vocab = 37
    ids = _duplicate_heavy_ids(vocab, device="cuda").reshape(-1)
    rows = _rand_grad((ids.numel(), _HIDDEN), dtype=dtype, device="cuda")
    scale = embedding_scale_for_dtype(_EMBED_SCALE, dtype)

    fused = _embedding_grad_weight(
        ids, rows, weight_shape=(vocab, _HIDDEN), weight_dtype=dtype, scale=scale
    )
    # fp32 only agrees because the reducer launches with FMA fusion disabled.
    prescaled = _embedding_grad_weight(
        ids, rows.float() * scale, weight_shape=(vocab, _HIDDEN), weight_dtype=dtype
    )
    assert torch.equal(fused, prescaled)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_scale_one_matches_plain_triton_embedding(triton_op, dtype):
    from rl_engine.backends.shared.triton.embedding.embedding import TritonEmbeddingOp

    vocab = 37
    weight = _rand_weight(vocab, _HIDDEN, dtype=dtype, device="cuda")
    token_ids = _duplicate_heavy_ids(vocab, device="cuda")
    grad_out = _rand_grad((*token_ids.shape, _HIDDEN), dtype=dtype, device="cuda")

    out, grad = _evaluate(triton_op, token_ids, weight, grad_out, embed_scale=1.0)
    leaf = weight.detach().clone().requires_grad_(True)
    plain_out = TritonEmbeddingOp()(token_ids, leaf)
    (plain_grad,) = torch.autograd.grad(plain_out, leaf, grad_outputs=grad_out)
    assert torch.equal(out, plain_out)
    assert torch.equal(grad, plain_grad)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_native_fallback_matches_triton_under_pad_flood(triton_op, dtype):
    vocab, hidden, pad_id = 64, 512, 0
    weight = _rand_weight(vocab, hidden, dtype=dtype, device="cuda")
    token_ids = _rand_ids((2, 1000), vocab, device="cuda")
    token_ids[:, 250:] = pad_id
    assert int((token_ids == pad_id).sum()) >= 1500
    grad_out = _rand_grad((*token_ids.shape, hidden), dtype=dtype, device="cuda")

    triton_out, triton_grad = _evaluate(triton_op, token_ids, weight, grad_out)
    native_out, native_grad = _evaluate(NativeScaledTokenEmbeddingOp(), token_ids, weight, grad_out)
    assert torch.equal(triton_out, native_out)
    assert torch.equal(triton_grad, native_grad)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_backward_gap_to_hf_autograd_is_reported(triton_op, dtype):
    """Diagnostic only: HF rounds ``g * s`` to the grad dtype before summing."""
    vocab = 37
    weight = _rand_weight(vocab, _HIDDEN, dtype=dtype, device="cuda")
    token_ids = _duplicate_heavy_ids(vocab, device="cuda")
    grad_out = _rand_grad((*token_ids.shape, _HIDDEN), dtype=dtype, device="cuda")

    _, grad_weight = _evaluate(triton_op, token_ids, weight, grad_out)
    hf_grad = _hf_grad_weight(token_ids, weight, grad_out)
    gap = (grad_weight.float() - hf_grad.float()).abs()
    relative = gap / hf_grad.float().abs().clamp_min(1e-12)
    print(
        f"{dtype}: max |triton - hf| = {gap.max().item():.6e}, "
        f"max relative = {relative.max().item():.3e}, "
        f"max ulp = {_max_ulp_gap(grad_weight, hf_grad)}"
    )
    assert torch.isfinite(grad_weight).all()


@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_candidate_in_existing_accuracy_harness(dtype):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    args = _harness_args("triton")
    case = make_operator_case(args, dtype, torch.device("cuda"))
    report = run_operator_suite(
        "scaled_token_embedding", candidates=[make_candidate(args)], cases=[case], check_grad=True
    )
    assert report.passed


def test_triton_rejects_cpu_tensors(triton_op):
    with pytest.raises(RuntimeError, match="accelerator"):
        triton_op(torch.tensor([0]), torch.ones(2, 4), _EMBED_SCALE)


def _enough_gpu_memory(num_bytes):
    if not torch.cuda.is_available():
        return False
    free, _ = torch.cuda.mem_get_info()
    return free > int(num_bytes * 1.5)


@pytest.mark.skipif(
    not _enough_gpu_memory(_GEMMA_VOCAB * _HIDDEN * 2),
    reason="needs a GPU with room for the ~2.8 GB bf16 Gemma 4 embedding table",
)
def test_triton_real_gemma_shape_forward_matches_hf(triton_op):
    """Forward only: the fp32 golden would copy a second 5.6 GB table."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    weight = torch.randn(
        _GEMMA_VOCAB, _HIDDEN, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    token_ids = torch.randint(0, _GEMMA_VOCAB, (2, 16), generator=generator, device="cuda")
    token_ids[0, 0] = 0
    token_ids[0, 1] = _GEMMA_VOCAB - 1

    out = triton_op(token_ids, weight, _EMBED_SCALE)
    assert out.shape == (2, 16, _HIDDEN)
    assert torch.equal(out, _hf_forward(token_ids, weight))
    assert torch.equal(triton_op(token_ids[:1], weight, _EMBED_SCALE), out[:1])
