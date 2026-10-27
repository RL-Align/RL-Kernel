# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU reference checks and real CUDA/ROCm tests for final-logit softcapping.

The softcap ``c`` is a runtime parameter of the shared operator. Gemma-4-31B
uses 30 and Muse-Glimmer-30B uses 20, so every numerical test runs for both.
"""

import argparse

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.kernels.ops.pytorch.activation import NativeFinalLogitSoftcapOp
from rl_engine.kernels.registry import KernelRegistry, OpBackend
from rl_engine.validation.operators.op_checks import run_operator_suite
from rl_engine.validation.operators.operator_specs import make_candidate, make_operator_case

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SOFTCAPS = (20.0, 30.0)
_CONTRACT = load_contract()
_VALUES = [-3000, -300, -60, -30, -1, -0.0001, 0, 0.0001, 1, 30, 60, 300, 3000]


def _rand(shape, *, dtype=torch.float32, device="cpu", seed=415, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=generator) * scale).to(device=device, dtype=dtype)


def _layout_tensor(layout, *, dtype, device="cpu", seed=415, scale=1.0):
    shapes = {"contiguous": (7, 5), "transpose": (5, 7), "stride": (7, 10), "expand": (1, 5)}
    base = _rand(shapes[layout], dtype=dtype, device=device, seed=seed, scale=scale)
    if layout == "transpose":
        return base.t()
    if layout == "stride":
        return base[:, 1::2]
    if layout == "expand":
        return base.expand(7, 5)
    return base


def _scaled_values(softcap):
    """The same relative range of tanh arguments, rescaled to the current cap."""
    return [softcap * v / 30.0 for v in _VALUES]


def _evaluate(op, x, grad_y):
    """Evaluate an operator and return (fp32 output, input-dtype gradient).

    The PyTorch reference exposes ``forward_fp32`` as the ground-truth path
    while ``forward`` follows the input dtype. The Triton wrapper has a single
    ``forward`` that already returns FP32, so fall back to it when
    ``forward_fp32`` is absent.
    """
    leaf = x.detach().requires_grad_(True)
    fp32_entry = getattr(op, "forward_fp32", None)
    y = fp32_entry(leaf) if fp32_entry is not None else op(leaf)
    (grad_x,) = torch.autograd.grad(y, leaf, grad_outputs=grad_y)
    assert y.shape == x.shape and y.dtype == torch.float32
    assert grad_x.shape == x.shape and grad_x.dtype == x.dtype
    return y.detach(), grad_x.detach()


def _assert_accuracy(actual, expected, judgment):
    tolerance = resolve_tolerance(
        _CONTRACT, judgment=judgment, op_class="elementwise", dtype=actual.dtype
    )
    torch.testing.assert_close(
        actual,
        expected.to(device=actual.device, dtype=actual.dtype),
        atol=tolerance.atol,
        rtol=tolerance.rtol,
    )


def _check_native_against_fp64(x, grad_y, softcap):
    before = x.clone()
    y, grad_x = _evaluate(NativeFinalLogitSoftcapOp(softcap), x, grad_y)
    # FP64 gold uses the already-quantized input; sech^2 avoids repeating the
    # Triton derivative expression and checks multiplication by the upstream
    # gradient. f(x) = c*tanh(x/c) => f'(x) = 1 - tanh(x/c)^2.
    scaled = x.double() / softcap
    _assert_accuracy(y, softcap * torch.tanh(scaled), "forward_accuracy")
    _assert_accuracy(grad_x, grad_y.double() / torch.cosh(scaled).square(), "gradient_accuracy")
    assert torch.equal(x, before)


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_native_values_and_random_upstream_gradient(dtype, softcap):
    x = torch.tensor(_scaled_values(softcap), dtype=dtype)
    _check_native_against_fp64(x, _rand(x.shape, seed=416), softcap)


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", [(), (0, 17), (2, 3, 7)])
def test_native_scalar_empty_and_multidimensional_inputs(dtype, shape, softcap):
    x = _rand(shape, dtype=dtype, scale=softcap)
    _check_native_against_fp64(x, _rand(shape, seed=416), softcap)


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "stride", "expand"])
def test_native_layouts(dtype, layout, softcap):
    x = _layout_tensor(layout, dtype=dtype, scale=softcap)
    grad_y = _layout_tensor(layout, dtype=torch.float32, seed=416)
    if layout != "contiguous":
        assert not x.is_contiguous() and not grad_y.is_contiguous()
    _check_native_against_fp64(x, grad_y, softcap)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.bool, torch.complex64])
def test_native_rejects_unsupported_dtypes(dtype):
    with pytest.raises(TypeError, match="dtype"):
        NativeFinalLogitSoftcapOp()(torch.ones(3, dtype=dtype))


@pytest.mark.parametrize("softcap", [0.0, -1.0, float("inf")])
def test_native_rejects_non_positive_softcap(softcap):
    with pytest.raises(ValueError, match="softcap must be finite and positive"):
        NativeFinalLogitSoftcapOp(softcap)


def test_default_softcap_is_thirty():
    assert NativeFinalLogitSoftcapOp().softcap == 30.0


def test_softcap_thirty_matches_pre_generalization_formula():
    """The generalized kernel must reproduce the original hard-coded 30 exactly."""
    x = torch.tensor(_VALUES, dtype=torch.bfloat16)
    expected = 30.0 * torch.tanh(x.float() / 30.0)
    actual = NativeFinalLogitSoftcapOp(30.0).forward_fp32(x)
    assert torch.equal(actual, expected)


def test_native_inference_does_not_require_gradients():
    op = NativeFinalLogitSoftcapOp()
    x = _rand((2, 7))
    assert not op(x).requires_grad
    with torch.no_grad():
        assert not op(x.requires_grad_(True)).requires_grad


@pytest.mark.parametrize("platform", ["cpu", "musa", "npu"])
def test_registry_reference_dispatch(platform, monkeypatch):
    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    op = registry.get_op("final_logit_softcap")
    assert isinstance(op, NativeFinalLogitSoftcapOp)
    assert op.softcap == 30.0
    x = torch.tensor([0.0, 30.0])
    assert torch.equal(op(x), NativeFinalLogitSoftcapOp()(x))


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_registry_falls_back_to_softcap_when_triton_is_unavailable(platform, monkeypatch):
    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    load_backend = registry._load_backend
    attempted = []

    def without_triton(backend):
        attempted.append(backend)
        if backend == OpBackend.TRITON_FINAL_LOGIT_SOFTCAP:
            return None
        return load_backend(backend)

    monkeypatch.setattr(registry, "_load_backend", without_triton)
    assert isinstance(registry.get_op("final_logit_softcap"), NativeFinalLogitSoftcapOp)
    assert attempted == [
        OpBackend.TRITON_FINAL_LOGIT_SOFTCAP,
        OpBackend.PYTORCH_NATIVE_FINAL_LOGIT_SOFTCAP,
    ]


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pytorch_candidate_in_existing_accuracy_harness(dtype):
    args = argparse.Namespace(
        op="final_logit_softcap",
        candidate="pytorch",
        arch_key=None,
        batch=2,
        seq=3,
        vocab=257,
        seed=415,
        input_mode="random",
    )
    case = make_operator_case(args, dtype, torch.device("cpu"))
    assert case.inputs["x"].shape == (2, 3, 257)
    report = run_operator_suite(
        "final_logit_softcap", candidates=[make_candidate(args)], cases=[case], check_grad=True
    )
    assert report.passed


@pytest.fixture
def triton_op():
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    # On a GPU host, a missing/broken Triton backend must fail rather than
    # silently validating the PyTorch fallback instead.
    from rl_engine.kernels.ops.triton.activation import TritonFinalLogitSoftcapOp

    return TritonFinalLogitSoftcapOp


def _check_triton_against_native(op_cls, softcap, x, grad_y):
    x_before, grad_before = x.clone(), grad_y.clone()
    expected = _evaluate(NativeFinalLogitSoftcapOp(softcap), x.detach().cpu(), grad_y.cpu())
    actual = _evaluate(op_cls(softcap), x, grad_y)
    _assert_accuracy(actual[0], expected[0], "forward_accuracy")
    _assert_accuracy(actual[1], expected[1], "gradient_accuracy")
    assert torch.equal(x, x_before) and torch.equal(grad_y, grad_before)


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "shape", [(), (0, 17), (1,), (1023,), (1024,), (1025,), (2, 3, 257), (2, 262144)]
)
def test_triton_forward_backward_and_block_boundaries(triton_op, dtype, shape, softcap):
    x = _rand(shape, dtype=dtype, device="cuda", scale=softcap)
    grad_y = _rand(shape, device="cuda", seed=416)
    _check_triton_against_native(triton_op, softcap, x, grad_y)


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_forward_backward_at_large_program_ids(triton_op, dtype, softcap):
    """Drive the production kernels with program IDs near 2**31 // BLOCK.

    Both kernels widen the program ID to int64 before multiplying by the block
    size, so element offsets never wrap in 32 bits. What this test covers is the
    behaviour at large program IDs with rebased pointers: that the grid can be
    that long, that only the last blocks touch memory, and that the masked tail
    past ``valid`` is left untouched. A real >2**31-element allocation is not
    attempted.
    """
    import triton
    import triton.language as tl

    from rl_engine.kernels.ops.triton.activation.final_logit_softcap import (
        _BLOCK,
        _final_logit_softcap_bwd_kernel,
        _final_logit_softcap_fwd_kernel,
    )

    @triton.jit
    def run_last_blocks(
        x_ptr,
        grad_y_ptr,
        out_ptr,
        n_elements,
        softcap_arg,
        START_PID: tl.constexpr,
        BACKWARD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # Earlier programs do no memory access. The last three programs call
        # the production kernels with their real, large program IDs. Rebasing
        # maps those logical offsets into the small allocations below.
        if tl.program_id(0) >= START_PID:
            base = tl.full((), START_PID * BLOCK, tl.int64)
            if BACKWARD:
                _final_logit_softcap_bwd_kernel(
                    grad_y_ptr - base,
                    x_ptr - base,
                    out_ptr - base,
                    n_elements,
                    softcap_arg,
                    BLOCK,
                )
            else:
                _final_logit_softcap_fwd_kernel(
                    x_ptr - base, out_ptr - base, n_elements, softcap_arg, BLOCK
                )

    start_pid = (2**31 // _BLOCK) - 1
    valid = 2 * _BLOCK + 17
    n_elements = start_pid * _BLOCK + valid
    x = _rand((3 * _BLOCK,), dtype=dtype, device="cuda", scale=softcap)
    grad_y = _rand(x.shape, device="cuda", seed=416)
    expected = _evaluate(NativeFinalLogitSoftcapOp(softcap), x[:valid].cpu(), grad_y[:valid].cpu())
    grid = (triton.cdiv(n_elements, _BLOCK),)
    for backward, reference in enumerate(expected):
        out = torch.full_like(x, -123.0, dtype=dtype if backward else torch.float32)
        run_last_blocks[grid](
            x,
            grad_y,
            out,
            n_elements,
            softcap,
            START_PID=start_pid,
            BACKWARD=bool(backward),
            BLOCK=_BLOCK,
        )
        _assert_accuracy(
            out[:valid], reference, "gradient_accuracy" if backward else "forward_accuracy"
        )
        assert torch.equal(out[valid:], torch.full_like(out[valid:], -123.0))


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_near_zero_and_saturated_values(triton_op, dtype, softcap):
    x = torch.tensor(_scaled_values(softcap), dtype=dtype, device="cuda")
    _check_triton_against_native(triton_op, softcap, x, _rand(x.shape, device="cuda", seed=416))


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("layout", ["transpose", "stride", "expand"])
def test_triton_noncontiguous_input_and_upstream_gradient(triton_op, dtype, layout, softcap):
    x = _layout_tensor(layout, dtype=dtype, device="cuda", scale=softcap)
    grad_y = _layout_tensor(layout, dtype=torch.float32, device="cuda", seed=416)
    assert not x.is_contiguous() and not grad_y.is_contiguous()
    _check_triton_against_native(triton_op, softcap, x, grad_y)


def _assert_bitwise_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_repeat_slice_padding_reshape_and_inference_invariance(triton_op, dtype, softcap):
    x = _rand((5, 257), dtype=dtype, device="cuda", scale=softcap)
    grad_y = _rand(x.shape, device="cuda", seed=416)
    op = triton_op(softcap)
    expected = _evaluate(op, x, grad_y)
    repeated = _evaluate(op, x, grad_y)
    sliced = _evaluate(op, x[1:4], grad_y[1:4])
    padded = _evaluate(
        op,
        torch.cat((x, _rand((2, 257), dtype=dtype, device="cuda", seed=417, scale=softcap))),
        torch.cat((grad_y, _rand((2, 257), device="cuda", seed=418))),
    )
    flattened = _evaluate(op, x.reshape(-1), grad_y.reshape(-1))
    for i in range(2):
        _assert_bitwise_equal(repeated[i], expected[i])
        _assert_bitwise_equal(sliced[i], expected[i][1:4])
        _assert_bitwise_equal(padded[i][:5], expected[i])
        _assert_bitwise_equal(flattened[i].reshape(x.shape), expected[i])
    with torch.no_grad():
        inference = op(x.detach().requires_grad_(True))
    assert not inference.requires_grad
    _assert_bitwise_equal(inference, expected[0])
    # Every comparison above is against the same wrapper, so a systematic offset
    # or stride error would cancel out. Anchor the full-input result to the
    # independent native reference as well.
    reference = _evaluate(NativeFinalLogitSoftcapOp(softcap), x.detach().cpu(), grad_y.cpu())
    _assert_accuracy(expected[0], reference[0], "forward_accuracy")
    _assert_accuracy(expected[1], reference[1], "gradient_accuracy")


def test_triton_backward_rejects_mismatched_upstream_gradient(triton_op):
    """A direct launcher call with a shorter dy must fail before the kernel runs."""
    from rl_engine.kernels.ops.triton.activation.final_logit_softcap import (
        _launch_final_logit_softcap_bwd,
    )

    x = _rand((8,), device="cuda")
    dy = _rand((4,), device="cuda", seed=416)
    with pytest.raises(ValueError, match="dy must match x"):
        _launch_final_logit_softcap_bwd(dy, x, 30.0)


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_training_and_inference_agree(triton_op, dtype, softcap):
    """Training-mode forward and no_grad forward must be bitwise identical."""
    op = triton_op(softcap)
    x = _rand((4, 257), dtype=dtype, device="cuda", scale=softcap)
    with_grad = op(x.detach().requires_grad_(True))
    with torch.no_grad():
        without_grad = op(x.detach())
    _assert_bitwise_equal(with_grad.detach(), without_grad)
