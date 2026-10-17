# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Additional FP64 mathematical checks, separate from WS1 FP32 acceptance.

Use stable FP64 log_softmax and autograd on the already-quantized inputs.
Retain the earlier tests' numerical targets without borrowing WS1's FP32
reference contract: forward atol=1e-5; gradient atol depends on output dtype.
These are additional test goals, not a project-wide FP64 accuracy guarantee.
The large-cap rounding regressions remain real failures if they exceed the
target; they are neither skipped nor marked as expected failures.
"""

import math

import pytest
import torch

from rl_engine.reference.logprob.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
)

pytestmark = pytest.mark.softcap_math

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SOFTCAPS = (1e-6, 1e-3, 0.125, 0.5, 1.0, 3.7, 10.0, 17.3, 20.0, 30.0, 39.5, 50.0, 64.0, 70.0)
_FORWARD_MATH_ATOL = 1e-5
"""Historical extra FP64 forward target; the WS1 reference itself is FP32."""
_GRADIENT_MATH_ATOL = {
    torch.float16: 5e-3,
    torch.bfloat16: 5e-2,
    torch.float32: 1e-5,
}
"""Keep the prior FP64 gradient targets, including final output quantization."""


@pytest.fixture(
    params=(
        "native_cpu",
        "native_gpu",
        "row",
        "parallel",
        "row_pipelined",
        "auto",
    )
)
def math_impl(request):
    """Qualify Native independently and check each public Triton strategy too."""
    if request.param == "native_cpu":
        return NativeSoftcappedSelectedLogprobOp(), "cpu"
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    if request.param == "native_gpu":
        return NativeSoftcappedSelectedLogprobOp(), "cuda"
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    strategy = None if request.param == "auto" else SoftcappedLogprobStrategy(request.param)
    return TritonSoftcappedSelectedLogprobOp(forward_impl=strategy), "cuda"


def _fp64_reference(logits, token_ids, upstream, *, softcap):
    """Evaluate the original formula in FP64, without calling either implementation.

    Evaluate softcap and log_softmax entirely in FP64, independently of the
    production reference and the kernels' FP32 exp/sum/log reduction.
    Autograd supplies the derivative independently of the Triton formula.
    FP64 starts after input quantization and keeps the oracle outputs in FP64.
    """
    leaf = logits.detach().cpu().double().requires_grad_(True)
    scores = softcap * torch.tanh(leaf / softcap)
    log_probs = torch.log_softmax(scores, dim=-1)
    output = log_probs.gather(-1, token_ids.cpu()[:, None]).squeeze(-1)
    gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream.cpu().double())[0]
    return output.detach(), gradient


def _assert_fp64_math_accuracy(actual, expected, *, atol):
    """Check an extra mathematical target without relabeling it as WS1 acceptance."""
    assert expected.dtype == torch.float64
    torch.testing.assert_close(
        actual.detach().cpu().double(),
        expected,
        atol=atol,
        rtol=0.0,
        msg=lambda message: f"Additional FP64 mathematical target (not WS1 FP32):\n{message}",
    )


def _check_math(op, logits, token_ids, upstream, *, softcap):
    """Check both derivatives and outputs against the independent mathematical oracle."""
    leaf = logits.detach().requires_grad_(True)
    output = op(leaf, token_ids, softcap=softcap)
    gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]
    expected_output, expected_gradient = _fp64_reference(
        logits, token_ids, upstream, softcap=softcap
    )
    _assert_fp64_math_accuracy(gradient, expected_gradient, atol=_GRADIENT_MATH_ATOL[logits.dtype])
    _assert_fp64_math_accuracy(output, expected_output, atol=_FORWARD_MATH_ATOL)


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_ordered_pytorch_reference_against_independent_fp64(dtype, softcap):
    """The same-order reference must not be its own mathematical justification.

    Qualify its forward and explicit analytic backward against independent
    FP64 log_softmax/autograd. Preserve the same extra mathematical targets;
    the ordered reference cannot change the formal FP32 acceptance baseline.
    """
    from tests.helpers.softcapped_ordered_reference import ordered_backward, ordered_forward

    generator = torch.Generator().manual_seed(467)
    logits = (softcap * torch.randn((3, 1025), generator=generator)).to(dtype)
    ids = torch.tensor([0, 512, 1024])
    upstream = torch.tensor([0.5, -2.0, 0.0])
    forward = ordered_forward(logits, ids, softcap=softcap)
    gradient = ordered_backward(forward, ids, upstream, output_dtype=dtype)
    expected_output, expected_gradient = _fp64_reference(logits, ids, upstream, softcap=softcap)
    _assert_fp64_math_accuracy(gradient, expected_gradient, atol=_GRADIENT_MATH_ATOL[dtype])
    _assert_fp64_math_accuracy(forward.output, expected_output, atol=_FORWARD_MATH_ATOL)


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", ((0, 7), (3, 1), (3, 1025)))
def test_fp64_math_configurable_caps(math_impl, softcap, dtype, shape):
    """Retain small caps, fractions, large caps, strided inputs and degenerate shapes."""
    op, device = math_impl
    generator = torch.Generator().manual_seed(436)
    storage = (torch.randn((shape[0], shape[1] * 2), generator=generator) * softcap).to(
        device=device, dtype=dtype
    )
    logits = storage[:, ::2]
    ids = torch.tensor([0, shape[1] // 2, shape[1] - 1][: shape[0]], device=device).long()
    upstream = torch.randn(shape[0] * 2, generator=generator).to(device)[::2]
    _check_math(op, logits, ids, upstream, softcap=softcap)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", ((3, 7), (2, 1023), (2, 1024), (2, 262144)))
def test_fp64_math_default_cap_tile_edges(math_impl, dtype, shape):
    op, device = math_impl
    generator = torch.Generator().manual_seed(415)
    logits = (torch.randn(shape, generator=generator) * 30).to(device=device, dtype=dtype)
    ids = torch.tensor([0, shape[1] - 1, shape[1] // 2][: shape[0]], device=device).long()
    upstream = torch.randn(shape[0], generator=generator).to(device)
    _check_math(op, logits, ids, upstream, softcap=30.0)


@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fp64_math_saturation_and_zero_upstream(math_impl, softcap, dtype):
    op, device = math_impl
    logits = torch.zeros((5, 1025), device=device, dtype=dtype)
    logits[1].fill_(3000.0)
    logits[2].fill_(-3000.0)
    logits[3].fill_(-3000.0)
    logits[3, 1024] = 3000.0
    logits[4].fill_(3000.0)
    logits[4, 1024] = -3000.0
    ids = torch.tensor([0, 512, 1024, 1024, 1024], device=device)
    upstream = torch.tensor([-0.75, 2.0, -3.0, 0.0, 1.5], device=device)
    _check_math(op, logits, ids, upstream, softcap=softcap)


@pytest.mark.parametrize("boundary", ("minimum", "near_maximum"))
@pytest.mark.parametrize("vocab", (1, 1025, 202048, 262144))
@pytest.mark.parametrize("dtype", _DTYPES)
def test_fp64_math_softcap_endpoints(math_impl, boundary, vocab, dtype):
    """Compute valid endpoints independently, without importing an Op validator."""
    op, device = math_impl
    limits = torch.finfo(torch.float32)
    maximum = min(-math.log(2.0 * limits.tiny), math.log(limits.max / 2.0) - math.log(vocab))
    softcap = limits.tiny if boundary == "minimum" else maximum - 1e-4
    logits = torch.stack(
        (
            torch.linspace(-2.0, 2.0, vocab) * softcap,
            torch.full((vocab,), 3000.0),
            torch.full((vocab,), -3000.0),
        )
    ).to(device=device, dtype=dtype)
    ids = torch.tensor([vocab // 2, 0, vocab - 1], device=device)
    upstream = torch.tensor([0.5, -0.5, 0.0], device=device)
    _check_math(op, logits, ids, upstream, softcap=softcap)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("softcap", _SOFTCAPS)
def test_fp64_math_native_wide_reference(device, dtype, softcap):
    """Qualify Native on wide rows separately from candidate/reference agreement."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    generator = torch.Generator().manual_seed(463)
    logits = (torch.randn((3, 202048), generator=generator) * softcap).to(
        device=device, dtype=dtype
    )
    logits[1, 101024] = 39.5
    ids = torch.tensor([0, 101024, 202047], device=device)
    upstream = torch.tensor([0.5, -2.0, 1.5], device=device)
    _check_math(NativeSoftcappedSelectedLogprobOp(), logits, ids, upstream, softcap=softcap)


@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_fp64_math_cap50_division_regression(math_impl, dtype):
    """Retain the independent oracle for the original scalar-division regression."""
    op, device = math_impl
    generator = torch.Generator().manual_seed(463)
    logits = torch.randn((3, 202048), generator=generator)
    logits[0] *= 50.0
    logits[1:] *= 30.0
    logits[1, 101024] = 39.5
    logits = logits.to(device=device, dtype=dtype)
    ids = torch.tensor([0, 101024, 202047], device=device)
    upstream = torch.tensor([0.5, -2.0, 1.5], device=device)
    _check_math(op, logits, ids, upstream, softcap=50.0)


@pytest.mark.parametrize("value", ("output", "gradient"))
@pytest.mark.parametrize(
    ("dtype", "softcap"),
    (
        pytest.param(torch.float32, 70.0, id="fp32-cap70"),
        pytest.param(torch.bfloat16, 70.0, id="bf16-cap70"),
        pytest.param(torch.float32, 64.0, id="fp32-cap64"),
    ),
)
def test_fp64_math_large_cap_rounding_regression(math_impl, value, dtype, softcap):
    """Retain both old and shifted-path rounding cases without xfail or wider limits.

    These are the inputs of the automatic ROW/PARALLEL switch regression.
    FP32 cap 70 exposed the old unshifted Native error. The four-way GPU
    diagnostic found shifted Native/Triton errors just above 1e-5 for BF16
    cap 70 and FP32 cap 64. Keep them as extra mathematical checks; they do
    not redefine WS1's FP32 reference. Separate parameters ensure an output
    failure cannot prevent the gradient assertion from running.
    """
    op, device = math_impl
    generator = torch.Generator().manual_seed(438)
    logits = (torch.randn((17, 32767), generator=generator) * softcap).to(device, dtype)
    ids = torch.arange(17, device=device) * 1025
    upstream = torch.randn(17, generator=generator).to(device)
    leaf = logits.detach().requires_grad_(True)
    output = op(leaf, ids, softcap=softcap)
    gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]
    expected_output, expected_gradient = _fp64_reference(logits, ids, upstream, softcap=softcap)
    if value == "output":
        _assert_fp64_math_accuracy(output, expected_output, atol=_FORWARD_MATH_ATOL)
    else:
        _assert_fp64_math_accuracy(gradient, expected_gradient, atol=_GRADIENT_MATH_ATOL[dtype])


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("softcap", (*_SOFTCAPS, 73.0))
def test_native_division_matches_rounded_fp64_operands(device, softcap, monkeypatch):
    """Catch reciprocal multiplication before tanh can hide the quotient error.

    In particular, CUDA scalar division of 39.5 by 50 rounds differently from
    dividing the two FP32 operands directly. Observe the actual tanh input and
    compare with an independently computed FP64 quotient rounded to FP32.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    logits = torch.tensor([[39.5, -39.5, 0.0, 65.25, -127.0]], device=device)
    ids = torch.tensor([0], device=device)
    tanh = torch.tanh
    quotients = []

    def observe_quotient(value):
        quotients.append(value.detach().cpu())
        return tanh(value)

    monkeypatch.setattr(torch, "tanh", observe_quotient)
    NativeSoftcappedSelectedLogprobOp()(logits, ids, softcap=softcap)
    divisor_fp32 = torch.tensor(softcap, dtype=torch.float32)
    expected = (logits.cpu().double() / divisor_fp32.double()).float()
    assert len(quotients) == 1
    assert torch.equal(quotients[0].view(torch.int32), expected.view(torch.int32))
