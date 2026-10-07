# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Backward of the generic CUDA fused logp: fused kernel and row-chunked fallback (#174)."""

from __future__ import annotations

import pytest
import torch

from rl_engine.kernels.ops.base import _C, _EXT_AVAILABLE
from rl_engine.kernels.ops.cuda.loss.logp import (
    FusedLogpGenericOp,
    _FusedLogpAutograd,
    fused_logp_backward_chunked,
    fused_logp_backward_rows_per_chunk,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
requires_fused_logp = pytest.mark.skipif(
    not (torch.cuda.is_available() and _EXT_AVAILABLE and hasattr(_C, "fused_logp")),
    reason="compiled _C.fused_logp is required",
)
requires_fused_backward = pytest.mark.skipif(
    not (torch.cuda.is_available() and _EXT_AVAILABLE and hasattr(_C, "fused_logp_backward")),
    reason="compiled _C.fused_logp_backward is required",
)

_DEVICES = ["cpu", pytest.param("cuda", marks=requires_cuda)]


def _make_inputs(n_rows, vocab, dtype, device, seed=0):
    gen = torch.Generator().manual_seed(seed)
    logits = (torch.randn(n_rows, vocab, generator=gen) * 4).to(dtype=dtype, device=device)
    labels = torch.randint(0, vocab, (n_rows,), generator=gen).to(device)
    grad = torch.randn(n_rows, generator=gen).to(device)
    return logits, labels, grad


def _unchunked_backward(logits, labels, grad_output, out_dtype):
    # The pre-#174 implementation, kept as the bitwise oracle for valid targets.
    probs = torch.softmax(logits.float(), dim=-1)
    rows = torch.arange(logits.size(0), device=logits.device)
    probs[rows, labels] -= 1.0
    grad = -grad_output.reshape(-1, 1).float() * probs
    return grad.to(out_dtype)


def _reference_grad(logits, labels, grad_output):
    leaf = logits.detach().float().requires_grad_(True)
    logp = torch.log_softmax(leaf, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    (logp * grad_output.float()).sum().backward()
    return leaf.grad


def test_rows_per_chunk_bounds_workspace():
    assert fused_logp_backward_rows_per_chunk(151_936, chunk_elems=1 << 24) == 110
    assert fused_logp_backward_rows_per_chunk(1 << 30, chunk_elems=1 << 24) == 1
    assert fused_logp_backward_rows_per_chunk(17, chunk_elems=170) == 10


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows_per_chunk", [1, 3, 7, 64])
def test_chunked_is_bitwise_equal_to_unchunked(device, dtype, rows_per_chunk):
    vocab = 1031
    logits, labels, grad = _make_inputs(37, vocab, dtype, device)
    expected = _unchunked_backward(logits, labels, grad, dtype)
    actual = fused_logp_backward_chunked(
        logits, labels, grad, dtype, chunk_elems=rows_per_chunk * vocab
    )
    assert actual.dtype == dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", _DEVICES)
def test_chunked_matches_autograd_reference(device):
    logits, labels, grad = _make_inputs(29, 513, torch.float32, device, seed=1)
    actual = fused_logp_backward_chunked(logits, labels, grad, torch.float32, chunk_elems=4 * 513)
    torch.testing.assert_close(actual, _reference_grad(logits, labels, grad), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_out_of_range_targets_get_zero_gradient(device):
    vocab = 257
    logits, labels, grad = _make_inputs(12, vocab, torch.float32, device, seed=2)
    invalid = torch.tensor([1, 4, 9], device=device)
    labels[1], labels[4], labels[9] = -100, vocab, vocab + 5
    actual = fused_logp_backward_chunked(logits, labels, grad, torch.float32, chunk_elems=5 * vocab)

    assert torch.count_nonzero(actual[invalid]) == 0
    valid = torch.ones(12, dtype=torch.bool, device=device)
    valid[invalid] = False
    expected = _unchunked_backward(logits[valid], labels[valid], grad[valid], torch.float32)
    assert torch.equal(actual[valid], expected)


class _ForwardOnlyBackend:
    """A ``_C`` build that predates ``fused_logp_backward``."""

    fused_logp = staticmethod(lambda logits, labels: _C.fused_logp(logits, labels))


@requires_fused_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_generic_op_backward_uses_fused_kernel(dtype):
    logits, labels, grad = _make_inputs(64, 4099, dtype, "cuda", seed=3)
    leaf = logits.clone().reshape(4, 16, 4099).requires_grad_(True)
    out = FusedLogpGenericOp().apply(leaf, labels.reshape(4, 16))
    out.backward(grad.reshape(4, 16).to(out.dtype))

    expected = _C.fused_logp_backward(logits, labels, grad.to(out.dtype).float())
    assert leaf.grad.shape == (4, 16, 4099)
    assert torch.equal(leaf.grad.reshape(64, 4099), expected)


@requires_fused_logp
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backend_without_fused_backward_uses_chunked_vjp(dtype):
    logits, labels, grad = _make_inputs(64, 4099, dtype, "cuda", seed=3)
    leaf = logits.clone().requires_grad_(True)
    out = _FusedLogpAutograd.apply(leaf, labels, _ForwardOnlyBackend())
    out.backward(grad.to(out.dtype))

    expected = fused_logp_backward_chunked(logits, labels, grad.to(out.dtype), dtype)
    assert torch.equal(leaf.grad, expected)


@requires_fused_backward
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [(torch.float32, 1e-5, 1e-4), (torch.bfloat16, 1e-2, 1.6e-2), (torch.float16, 1e-3, 1e-3)],
)
@pytest.mark.parametrize("vocab", [1, 255, 1031, 151_936])
def test_fused_backward_matches_reference(dtype, atol, rtol, vocab):
    logits, labels, grad = _make_inputs(33, vocab, dtype, "cuda", seed=4)
    actual = _C.fused_logp_backward(logits, labels, grad)
    expected = (
        grad.double()[:, None]
        * (torch.nn.functional.one_hot(labels, vocab).double() - torch.softmax(logits.double(), -1))
    ).to(dtype)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@requires_fused_backward
def test_fused_backward_is_deterministic_and_batch_invariant():
    logits, labels, grad = _make_inputs(96, 32_000, torch.bfloat16, "cuda", seed=5)
    full = _C.fused_logp_backward(logits, labels, grad)
    assert torch.equal(full, _C.fused_logp_backward(logits, labels, grad))
    for rows in ([0], [5, 17, 95], list(range(31, 64))):
        idx = torch.tensor(rows, device="cuda")
        part = _C.fused_logp_backward(logits[idx].contiguous(), labels[idx], grad[idx])
        assert torch.equal(part, full[idx])


@requires_fused_backward
def test_fused_backward_out_of_range_targets_and_masked_logits():
    vocab = 1031
    logits, labels, grad = _make_inputs(8, vocab, torch.float32, "cuda", seed=6)
    logits[:, 1000:] = float("-inf")  # padded / masked vocabulary tail
    labels[2], labels[5] = -100, vocab
    actual = _C.fused_logp_backward(logits, labels, grad)

    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[[2, 5]]) == 0
    assert torch.count_nonzero(actual[:, 1000:]) == 0
    keep = torch.tensor([0, 1, 3, 4, 6, 7], device="cuda")
    expected = fused_logp_backward_chunked(logits[keep], labels[keep], grad[keep], torch.float32)
    torch.testing.assert_close(actual[keep], expected, atol=1e-5, rtol=1e-4)


@requires_fused_backward
def test_fused_backward_rejects_bad_inputs():
    logits, labels, grad = _make_inputs(4, 17, torch.float32, "cuda")
    with pytest.raises(RuntimeError, match="float32"):
        _C.fused_logp_backward(logits, labels, grad.double())
    with pytest.raises(RuntimeError, match="int64"):
        _C.fused_logp_backward(logits, labels.int(), grad)
    with pytest.raises(RuntimeError, match="grad_output length"):
        _C.fused_logp_backward(logits, labels, grad[:3])
    assert _C.fused_logp_backward(logits[:0], labels[:0], grad[:0]).shape == (0, 17)


@requires_cuda
@pytest.mark.parametrize("impl", ["chunked", pytest.param("fused", marks=requires_fused_backward)])
@pytest.mark.parametrize("n_rows", [8192, 16384, 32768])
def test_long_sequence_backward_workspace_is_bounded(n_rows, impl):
    vocab = 151_936  # Qwen2/Qwen3 vocabulary
    dtype = torch.bfloat16
    elem = torch.finfo(dtype).bits // 8
    io_bytes = 2 * n_rows * vocab * elem  # logits + returned gradient
    free, _ = torch.cuda.mem_get_info()
    if free < io_bytes + (4 << 30):
        pytest.skip(f"needs {(io_bytes >> 30) + 4} GiB free device memory")

    logits = torch.randn(n_rows, vocab, device="cuda", dtype=dtype)
    labels = torch.randint(0, vocab, (n_rows,), device="cuda")
    grad = torch.randn(n_rows, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    if impl == "fused":
        out = _C.fused_logp_backward(logits, labels, grad)
    else:
        out = fused_logp_backward_chunked(logits, labels, grad, dtype)
    torch.cuda.synchronize()
    transient = torch.cuda.max_memory_allocated() - base - out.numel() * elem

    # The unchunked VJP needed ~10 bytes per logit (19.9 GB at 16k rows).  The
    # chunked fallback keeps two FP32 chunks (upcast + softmax) plus per-row
    # vectors; the fused kernel allocates nothing beyond its output.
    if impl == "fused":
        assert transient == 0
    else:
        rows = fused_logp_backward_rows_per_chunk(vocab)
        assert transient <= 2 * rows * vocab * 4 + 64 * n_rows + (8 << 20)
    assert torch.isfinite(out[:: max(1, n_rows // 64)].float()).all()
