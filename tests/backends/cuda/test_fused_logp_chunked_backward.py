# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Backward of the generic CUDA fused logp: fused kernel and row-chunked fallback (#174)."""

from __future__ import annotations

import pytest
import torch

import rl_engine.backends.cuda.logprob.logp as logp_module
from rl_engine.backends.cuda.logprob.logp import (
    FusedLogpGenericOp,
    _FusedLogpAutograd,
    fused_logp_backward_chunked,
    fused_logp_backward_rows_per_chunk,
)
from rl_engine.backends.extension import _C, _EXT_AVAILABLE

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
requires_fused_logp = pytest.mark.skipif(
    not (torch.cuda.is_available() and _EXT_AVAILABLE and hasattr(_C, "fused_logp")),
    reason="compiled _C.fused_logp is required",
)
requires_fused_backward = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip is None
        and _EXT_AVAILABLE
        and hasattr(_C, "fused_logp_backward")
    ),
    reason="compiled _C.fused_logp_backward is required",
)

_DEVICES = ["cpu", pytest.param("cuda", marks=requires_cuda)]


def _make_inputs(n_rows, vocab, dtype, device, seed=0):
    """Return seeded ``(logits, labels, grad_output)`` with in-range labels and one grad per row."""
    gen = torch.Generator().manual_seed(seed)
    logits = (torch.randn(n_rows, vocab, generator=gen) * 4).to(dtype=dtype, device=device)
    labels = torch.randint(0, vocab, (n_rows,), generator=gen).to(device)
    grad = torch.randn(n_rows, generator=gen).to(device)
    return logits, labels, grad


def _unchunked_backward(logits, labels, grad_output, out_dtype):
    """Return the full-matrix FP32 VJP ``-grad * (softmax - one_hot)`` cast to ``out_dtype``."""
    # The pre-#174 implementation, kept as the bitwise oracle for valid targets.
    probs = torch.softmax(logits.float(), dim=-1)
    rows = torch.arange(logits.size(0), device=logits.device)
    probs[rows, labels] -= 1.0
    grad = -grad_output.reshape(-1, 1).float() * probs
    return grad.to(out_dtype)


def _reference_grad(logits, labels, grad_output):
    """Return the FP32 logits gradient from autograd through ``log_softmax`` and ``gather``."""
    leaf = logits.detach().float().requires_grad_(True)
    logp = torch.log_softmax(leaf, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    (logp * grad_output.float()).sum().backward()
    return leaf.grad


def test_rows_per_chunk_bounds_workspace():
    """The rows per chunk are ``chunk_elems // vocab``, but never fewer than one."""
    assert fused_logp_backward_rows_per_chunk(151_936, chunk_elems=1 << 24) == 110
    assert fused_logp_backward_rows_per_chunk(1 << 30, chunk_elems=1 << 24) == 1
    assert fused_logp_backward_rows_per_chunk(17, chunk_elems=170) == 10


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows_per_chunk", [1, 3, 7, 64])
def test_chunked_is_bitwise_equal_to_unchunked(device, dtype, rows_per_chunk):
    """The chunked backward keeps ``out_dtype`` and equals the unchunked VJP bitwise."""
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
    """The FP32 chunked backward matches the autograd ``log_softmax`` gradient."""
    logits, labels, grad = _make_inputs(29, 513, torch.float32, device, seed=1)
    actual = fused_logp_backward_chunked(logits, labels, grad, torch.float32, chunk_elems=4 * 513)
    torch.testing.assert_close(actual, _reference_grad(logits, labels, grad), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows_per_chunk", [1, 5, 64])
@pytest.mark.parametrize("masked_invalid_rows", [False, True])
def test_out_of_range_targets_get_zero_gradient(device, dtype, rows_per_chunk, masked_invalid_rows):
    """Rows with targets outside ``[0, V)`` get a finite zero gradient, even if all ``-inf``.

    The remaining rows still equal the unchunked VJP bitwise.
    """
    vocab = 257
    logits, labels, grad = _make_inputs(12, vocab, dtype, device, seed=2)
    invalid = torch.tensor([1, 4, 9], device=device)
    labels[1], labels[4], labels[9] = -100, vocab, vocab + 5
    if masked_invalid_rows:
        logits[invalid] = float("-inf")
    actual = fused_logp_backward_chunked(
        logits, labels, grad, dtype, chunk_elems=rows_per_chunk * vocab
    )

    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[invalid]) == 0
    valid = torch.ones(12, dtype=torch.bool, device=device)
    valid[invalid] = False
    expected = _unchunked_backward(logits[valid], labels[valid], grad[valid], dtype)
    assert torch.equal(actual[valid], expected)


class _ForwardOnlyBackend:
    """A ``_C`` build that predates ``fused_logp_backward``."""

    fused_logp = staticmethod(lambda logits, labels: _C.fused_logp(logits, labels))


@pytest.mark.parametrize("hip_version", [None, "6.4.0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_autograd_backward_dispatch_uses_chunked_on_hip(monkeypatch, hip_version, dtype):
    """A stale HIP extension exposing the CUDA symbol must still use the fallback."""
    calls = []

    class Backend:
        @staticmethod
        def fused_logp(logits, labels):
            """Return FP32 selected log-probabilities computed with ``torch.log_softmax``."""
            return torch.log_softmax(logits.float(), -1).gather(1, labels[:, None]).squeeze(1)

        @staticmethod
        def fused_logp_backward(logits, labels, grad):
            """Record a fused dispatch, fail if HIP is mocked, and return the chunked VJP."""
            calls.append("fused")
            assert hip_version is None, "the CUDA backward must not be dispatched on HIP"
            return fused_logp_backward_chunked(logits, labels, grad, logits.dtype)

    def chunked(*args, **kwargs):
        """Record a fallback dispatch and delegate to the real ``fused_logp_backward_chunked``."""
        calls.append("chunked")
        return fused_logp_backward_chunked(*args, **kwargs)

    monkeypatch.setattr(torch.version, "hip", hip_version)
    monkeypatch.setattr(logp_module, "fused_logp_backward_chunked", chunked)
    logits, labels, grad = _make_inputs(6, 17, dtype, "cpu", seed=19)
    leaf = logits.reshape(2, 3, 17).requires_grad_(True)
    out = _FusedLogpAutograd.apply(leaf, labels.reshape(2, 3), Backend())
    out.backward(grad.reshape(2, 3))

    assert calls == (["fused"] if hip_version is None else ["chunked"])
    expected = fused_logp_backward_chunked(logits, labels, grad, dtype)
    assert torch.equal(leaf.grad.reshape_as(logits), expected)


@requires_fused_backward
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_generic_op_backward_uses_fused_kernel(dtype):
    """``FusedLogpGenericOp`` backward on 3-D logits equals ``_C.fused_logp_backward`` bitwise."""
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
    """A backend without ``fused_logp_backward`` gets the chunked VJP, bitwise."""
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
    """The fused kernel keeps the logits dtype and matches an FP64 reference up to 151,936 vocab."""
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
    """The fused kernel is bitwise repeatable and row subsets equal the full-batch rows.

    Both the full and the sliced results also match an FP64 oracle.
    """
    logits, labels, grad = _make_inputs(96, 32_000, torch.bfloat16, "cuda", seed=5)
    full = _C.fused_logp_backward(logits, labels, grad)
    assert torch.equal(full, _C.fused_logp_backward(logits, labels, grad))
    # An independent FP64 oracle, so a row-offset bug that shifts the full and the
    # sliced call identically cannot pass by comparing the kernel with itself.
    one_hot = torch.nn.functional.one_hot(labels, 32_000).double()
    oracle = (grad.double()[:, None] * (one_hot - torch.softmax(logits.double(), -1))).to(
        torch.bfloat16
    )
    torch.testing.assert_close(full, oracle, atol=1e-2, rtol=1.6e-2)
    for rows in ([0], [5, 17, 95], list(range(31, 64))):
        idx = torch.tensor(rows, device="cuda")
        part = _C.fused_logp_backward(logits[idx].contiguous(), labels[idx], grad[idx])
        assert torch.equal(part, full[idx])
        torch.testing.assert_close(part, oracle[idx], atol=1e-2, rtol=1.6e-2)


@requires_fused_backward
def test_fused_backward_out_of_range_targets_and_masked_logits():
    """Out-of-range targets and ``-inf`` columns get zero gradient; other rows match chunked."""
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
    """The binding rejects non-FP32 grads, non-int64 labels and short grads; zero rows work."""
    logits, labels, grad = _make_inputs(4, 17, torch.float32, "cuda")
    with pytest.raises(RuntimeError, match="float32"):
        _C.fused_logp_backward(logits, labels, grad.double())
    with pytest.raises(RuntimeError, match="int64"):
        _C.fused_logp_backward(logits, labels.int(), grad)
    with pytest.raises(RuntimeError, match="grad_output length"):
        _C.fused_logp_backward(logits, labels, grad[:3])
    assert _C.fused_logp_backward(logits[:0], labels[:0], grad[:0]).shape == (0, 17)


@requires_fused_backward
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices are required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_backward_guards_tensor_device_and_uses_its_current_stream(dtype):
    """The binding runs on the inputs' device and that device's current stream.

    The caller's different current device is left unchanged across the call.
    """
    caller_device = torch.cuda.current_device()
    tensor_device = 1 if caller_device == 0 else 0
    previous_stream = torch.cuda.current_stream(tensor_device)
    stream = torch.cuda.Stream(device=tensor_device)
    try:
        # Leave this stream current on the input device, then restore the caller's
        # device before entering C++. The binding must select both independently.
        with torch.cuda.device(tensor_device):
            torch.cuda.set_stream(stream)
            logits, labels, grad = _make_inputs(13, 1031, dtype, f"cuda:{tensor_device}", seed=20)
            true_grad = grad.clone()
            # Queue a long sleep and then the real gradient on the stream. A launch on
            # any other stream would read the zeros below instead of the final values.
            grad.zero_()
            torch.cuda._sleep(200_000_000)
            grad.copy_(true_grad)
        assert torch.cuda.current_device() == caller_device
        assert torch.cuda.current_stream(tensor_device) == stream

        actual = _C.fused_logp_backward(logits, labels, grad)
        assert torch.cuda.current_device() == caller_device
        assert actual.device == logits.device
        stream.synchronize()
        grad = true_grad

        expected = _reference_grad(logits.cpu(), labels.cpu(), grad.cpu()).to(dtype)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1.6e-2)
    finally:
        with torch.cuda.device(tensor_device):
            torch.cuda.set_stream(previous_stream)
        torch.cuda.set_device(caller_device)


@requires_cuda
@pytest.mark.parametrize("impl", ["chunked", pytest.param("fused", marks=requires_fused_backward)])
@pytest.mark.parametrize("n_rows", [8192, 16384, 32768])
def test_long_sequence_backward_workspace_is_bounded(n_rows, impl):
    """BF16 backward with a 151,936 vocab and 8k-32k rows keeps peak transient memory bounded.

    The fused kernel allocates nothing beyond its output; the chunked fallback stays within
    two FP32 row chunks plus 64 bytes per row and 8 MiB of slack. Skips without enough memory.
    """
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
