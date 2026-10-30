# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""First-order native CUDA LogP gradients, including sparse row selection."""

from types import SimpleNamespace

import pytest
import torch

from rl_engine.backends.cuda.logprob.logp import DeterministicLogpCUDAOp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.fixture
def op(record_property):
    """Return a ``DeterministicLogpCUDAOp``, recording its class and backward kernel name."""
    operator = DeterministicLogpCUDAOp()
    record_property("backend", type(operator).__name__)
    record_property("backward", "_C.deterministic_logp_backward")
    return operator


def inputs(vocab, dtype, shape=(2, 3)):
    """Return seeded CUDA logits ``(*shape, vocab)`` requiring grad and ids in ``[0, vocab)``."""
    generator = torch.Generator(device="cuda").manual_seed(2026)
    logits = torch.randn(*shape, vocab, device="cuda", dtype=dtype, generator=generator)
    ids = torch.randint(vocab, shape, device="cuda", generator=generator)
    return logits.requires_grad_(True), ids


def bits(actual, expected):
    """Assert both tensors share dtype and shape and are bitwise identical."""
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def reference(logits, ids):
    """Return FP32 ``log_softmax`` at each target id, zero where the id is out of range."""
    valid = (ids >= 0) & (ids < logits.size(-1))
    result = (
        logits.float().log_softmax(-1).gather(-1, ids.clamp(0, logits.size(-1) - 1).unsqueeze(-1))
    )
    return result.squeeze(-1).masked_fill(~valid, 0)


def check_gradient(logits, ids, actual, active, record_property):
    """Check ``actual``'s logits gradient against the reference restricted to ``active`` rows.

    Uses ``linspace`` upstream weights and a per-dtype ``atol``; asserts the gradient is finite
    and exactly zero on inactive rows, records the max error, and returns the gradient.
    """
    upstream = torch.linspace(-1, 1, ids.numel(), device="cuda").reshape(ids.shape)
    expected = reference(logits, ids).masked_fill(~active, 0)
    grad = torch.autograd.grad(actual, logits, upstream, retain_graph=True)[0]
    ref_grad = torch.autograd.grad(expected, logits, upstream)[0]
    atol = {torch.float32: 3e-6, torch.float16: 5e-4, torch.bfloat16: 4e-3}[logits.dtype]
    error = (grad.float() - ref_grad.float()).abs().max().item() if grad.numel() else 0
    record_property("max_gradient_error", error)
    record_property("gradient_atol", atol)
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(grad, ref_grad, atol=atol, rtol=0)
    assert torch.count_nonzero(grad[~active]) == 0
    return grad


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", [1, 128, 129, 257, 4096, 4097, 128256])
def test_dense_forward_bits_and_gradient(op, dtype, vocab, record_property):
    """The autograd dense forward matches no-grad bits; its gradient matches the FP32 reference."""
    logits, ids = inputs(vocab, dtype)
    with torch.no_grad():
        baseline = op(logits, ids)
    output = op(logits, ids)
    bits(output, baseline)
    check_gradient(logits, ids, output, torch.ones_like(ids, dtype=torch.bool), record_property)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indices", [[], [0, 2, 5], [5, 0, 5, 2, 0], [-1, 2, 999], list(range(6))])
def test_indexed_gradient_and_duplicates(op, dtype, indices, record_property):
    """Indexed gradients ignore out-of-range row indices and count duplicate rows once.

    The grad-mode output bitwise matches ``no_grad``; the gradient matches the reference on the
    valid selected rows, is zero on all other rows, and is bitwise equal to the gradient from
    the deduplicated valid rows.
    """
    logits, ids = inputs(257, dtype)
    rows = torch.tensor(indices, device="cuda", dtype=torch.long)
    valid_rows = rows[(rows >= 0) & (rows < ids.numel())].unique()
    active = torch.zeros(ids.numel(), device="cuda", dtype=torch.bool)
    active[valid_rows] = True
    active = active.reshape(ids.shape)
    output = op.indexed_fp32(logits, ids, rows)
    with torch.no_grad():
        baseline = op.indexed_fp32(logits, ids, rows)
    bits(output, baseline)
    grad = check_gradient(logits, ids, output, active, record_property)
    unique_output = op.indexed_fp32(logits, ids, valid_rows)
    unique_grad = check_gradient(logits, ids, unique_output, active, record_property)
    bits(grad, unique_grad)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indexed", [False, True])
def test_invalid_targets_zero_gradient(op, dtype, indexed, record_property):
    """Out-of-range target ids yield bitwise-zero outputs and zero gradient rows."""
    logits, ids = inputs(257, dtype)
    ids[0, 0], ids[1, 2] = -100, 257
    output = (
        op.indexed_fp32(logits, ids, torch.arange(6, device="cuda")) if indexed else op(logits, ids)
    )
    active = (ids >= 0) & (ids < 257)
    bits(output[~active], torch.zeros_like(output[~active]))
    check_gradient(logits, ids, output, active, record_property)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("indexed", [False, True])
def test_noncontiguous_and_empty(op, dtype, indexed, record_property):
    """Transposed inputs, including an empty batch, match dense no-grad bits and reference grads."""
    for shape in [(2, 3), (0, 3)]:
        logits, ids = inputs(257, dtype, shape)
        logits = logits.transpose(0, 1)
        ids = ids.transpose(0, 1)
        output = (
            op.indexed_fp32(logits, ids, torch.arange(ids.numel(), device="cuda"))
            if indexed
            else op(logits, ids)
        )
        with torch.no_grad():
            bits(output, op(logits, ids))
        check_gradient(logits, ids, output, torch.ones_like(ids, dtype=torch.bool), record_property)


@pytest.mark.parametrize("dtype", DTYPES)
def test_indexed_layout_gradient_bits(op, dtype):
    """Indexed gradients are bitwise stable across reruns, 2-row chunking and row permutation.

    Embedding the rows in a larger batch with zero-weight unselected padding rows leaves the
    original rows' gradient bitwise unchanged and the padding rows' gradient zero.
    """
    logits, ids = inputs(4097, dtype, (7,))
    upstream = torch.linspace(-1, 1, 7, device="cuda")
    selected = torch.tensor([0, 2, 6, 2], device="cuda")

    def gradient(x, labels, rows, weights):
        """Return the ``x`` gradient of ``op.indexed_fp32`` for upstream ``weights``."""
        return torch.autograd.grad(op.indexed_fp32(x, labels, rows), x, weights)[0]

    baseline = gradient(logits, ids, selected, upstream)
    bits(gradient(logits, ids, selected, upstream), baseline)
    chunks = []
    for start in range(0, 7, 2):
        local_rows = selected[(selected >= start) & (selected < start + 2)] - start
        chunks.append(
            gradient(
                logits[start : start + 2],
                ids[start : start + 2],
                local_rows,
                upstream[start : start + 2],
            )
        )
    bits(torch.cat(chunks), baseline)
    order = torch.tensor([6, 0, 4, 2, 5, 1, 3], device="cuda")
    inverse = torch.argsort(order)
    bits(
        gradient(logits[order], ids[order], inverse[selected], upstream[order]),
        baseline[order],
    )
    padded, labels = inputs(4097, dtype, (16,))
    with torch.no_grad():
        padded[3:10].copy_(logits)
        labels[3:10].copy_(ids)
    weights = torch.zeros(16, device="cuda")
    weights[3:10] = upstream
    result = gradient(padded, labels, selected + 3, weights)
    bits(result[3:10], baseline)
    assert torch.count_nonzero(result[:3]) == torch.count_nonzero(result[10:]) == 0


@pytest.mark.parametrize("method", ["out", "indexed_out", "online_out", "online_indexed_out"])
@pytest.mark.parametrize("output_grad", [False, True])
def test_out_buffer_gradient_contract(op, method, output_grad):
    """With grad enabled, out methods raise ``RuntimeError`` for an ``out`` that requires grad.

    A plain buffer is returned in place and backpropagates to ``logits``. Under ``no_grad`` every
    out method returns the given buffer, even one requiring grad, bitwise equal to the dense op.
    """
    logits, ids = inputs(17, torch.float32)
    logits.requires_grad_(not output_grad)
    output = torch.empty_like(ids, dtype=torch.float32, requires_grad=output_grad)
    args = (
        (logits, ids, torch.arange(6, device="cuda"), output)
        if "indexed" in method
        else (logits, ids, output)
    )
    if output_grad:
        with pytest.raises(RuntimeError, match="must not require gradients"):
            getattr(op, method)(*args)
    else:
        result = getattr(op, method)(*args)
        assert result.data_ptr() == output.data_ptr()
        assert torch.autograd.grad(result.sum(), logits)[0].shape == logits.shape
    with torch.no_grad():
        result = getattr(op, method)(*args)
        assert result.data_ptr() == output.data_ptr()
        bits(result, op(logits, ids))


@pytest.mark.parametrize(
    "method",
    ["apply", "apply_fp32", "online_fp32", "indexed_fp32", "online_indexed_fp32"],
)
def test_aliases_and_float64_inference(op, method):
    """FP32 entry points reject float64 logits needing grad with ``TypeError``.

    Under ``no_grad`` they accept float64 logits and return float32; with float32 logits that
    require grad, the output requires grad.
    """
    logits, ids = inputs(17, torch.float64)
    args = (logits, ids, torch.arange(6, device="cuda")) if "indexed" in method else (logits, ids)
    with pytest.raises(TypeError, match="autograd supports"):
        getattr(op, method)(*args)
    with torch.no_grad():
        assert getattr(op, method)(*args).dtype == torch.float32
    args = (logits.float(), *args[1:])
    assert getattr(op, method)(*args).requires_grad


def test_missing_native_symbol_is_actionable(op):
    """With only the FP32 forward symbol, ``no_grad`` works but autograd says to rebuild ``_C``."""
    op._backend = SimpleNamespace(
        deterministic_logp_forward_fp32=op._backend.deterministic_logp_forward_fp32
    )
    logits, ids = inputs(17, torch.float32)
    with torch.no_grad():
        assert op(logits, ids).shape == ids.shape
    with pytest.raises(RuntimeError, match="Rebuild rl_engine._C"):
        op(logits, ids)


def test_second_order_supported(op):
    """Double backward to ``upstream`` is zero within 1e-6: each one_hot - softmax sums to 0."""
    logits, ids = inputs(17, torch.float32)
    upstream = torch.ones_like(ids, dtype=torch.float32, requires_grad=True)
    grad = torch.autograd.grad(op(logits, ids), logits, upstream, create_graph=True)[0]
    actual = torch.autograd.grad(grad.sum(), upstream)[0]
    torch.testing.assert_close(actual, torch.zeros_like(upstream), atol=1e-6, rtol=0)
