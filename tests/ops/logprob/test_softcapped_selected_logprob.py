# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Separately collected FP32 accuracy and bitwise consistency checks.

Independent FP64 mathematical checks live in the companion *_math.py file.
They have their own test targets and do not define the WS1 accuracy reference.
"""

import math
from functools import partial

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.logprob.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
    softcapped_selected_logprob,
)

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CONTRACT = load_contract()
_CHECKS = (
    pytest.param("accuracy", marks=pytest.mark.softcap_accuracy),
    pytest.param("consistency", marks=pytest.mark.softcap_consistency),
)
"""Separate pytest items: an accuracy failure never suppresses a consistency check."""
_VALUE_CHECKS = (
    pytest.param("output", marks=pytest.mark.softcap_accuracy),
    pytest.param("gradient", marks=pytest.mark.softcap_accuracy),
    pytest.param("invariance", marks=pytest.mark.softcap_consistency),
)
_SOFTCAPS = (
    1e-6,
    1e-3,
    0.125,
    0.5,
    1.0,
    3.7,
    10.0,
    17.3,
    20.0,
    30.0,
    39.5,
    50.0,
    64.0,
    70.0,
)
"""Sample small caps, non-binary-exact fractions and larger caps within the tested V bounds."""


def _assert_bitwise(actual, expected):
    """Compare bytes after copying both tensors into an explicit dense layout.

    contiguous() can preserve a non-unit final stride on empty or singleton
    dimensions. A dtype view requires that stride to be one even in those
    cases, so force a contiguous-format clone before viewing the raw bytes.
    """
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    actual_bytes = actual.detach().clone(memory_format=torch.contiguous_format).view(torch.uint8)
    expected_bytes = (
        expected.detach().clone(memory_format=torch.contiguous_format).view(torch.uint8)
    )
    assert torch.equal(actual_bytes, expected_bytes)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", ((0, 7), (1, 1), (3, 1), (1, 7), (3, 7)))
def test_bitwise_comparison_handles_strided_and_degenerate_shapes(dtype, shape):
    """Accept equal bytes even when an empty or single-column view has stride two."""
    storage = torch.arange(shape[0] * shape[1] * 2, dtype=dtype).reshape(shape[0], shape[1] * 2)
    actual = storage[:, ::2]
    expected = actual.clone(memory_format=torch.contiguous_format)
    assert actual.stride(-1) == 2
    _assert_bitwise(actual, expected)
    if actual.numel():
        expected[0, 0] += 1
        with pytest.raises(AssertionError):
            _assert_bitwise(actual, expected)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", _DTYPES)
def test_bitwise_comparison_distinguishes_signed_zero(dtype):
    positive = torch.tensor([0.0], dtype=dtype)
    negative = torch.tensor([-0.0], dtype=dtype)
    assert torch.equal(positive, negative)
    _assert_bitwise(positive, positive.clone())
    with pytest.raises(AssertionError):
        _assert_bitwise(positive, negative)


def _assert_fp32_accuracy(actual, expected, judgment):
    """Apply the unchanged WS1 logprob limits to an unrounded FP32 reference.

    Forward outputs are FP32 and use its 1e-5 absolute limit. Gradients use
    their output dtype's contract row. Promote only for the comparison;
    never round the FP32 reference to the candidate's lower-precision dtype.
    """
    assert expected.dtype == torch.float32, "The WS1 reference must compute in FP32."
    tolerance = resolve_tolerance(
        _CONTRACT, judgment=judgment, op_class="logprob", dtype=actual.dtype
    )
    torch.testing.assert_close(
        actual.to(torch.float64),
        expected.to(device=actual.device, dtype=torch.float64),
        atol=tolerance.atol,
        rtol=tolerance.rtol,
    )


def _native_backward(logits, token_ids, upstream, *, softcap=30.0):
    leaf = logits.detach().requires_grad_(True)
    output = NativeSoftcappedSelectedLogprobOp()(leaf, token_ids, softcap=softcap)
    return torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]


def _triton_forward(logits, token_ids, *, forward_impl="row", softcap=30.0):
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _FORWARD_LAUNCHERS,
        SoftcappedLogprobStrategy,
    )

    launch = _FORWARD_LAUNCHERS[SoftcappedLogprobStrategy(forward_impl)]
    output, _, _ = launch(logits, token_ids, softcap)
    return output


def _triton_backward(logits, token_ids, upstream, *, forward_impl="row", softcap=30.0):
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _FORWARD_LAUNCHERS,
        SoftcappedLogprobStrategy,
        _launch_softcapped_selected_logprob_bwd,
    )

    launch = _FORWARD_LAUNCHERS[SoftcappedLogprobStrategy(forward_impl)]
    _, row_max, log_sum_exp_shifted = launch(logits, token_ids, softcap)
    return _launch_softcapped_selected_logprob_bwd(
        logits, token_ids, upstream, row_max, log_sum_exp_shifted, softcap
    )


@pytest.fixture(params=("pytorch", "triton", "triton_parallel", "triton_row_pipelined"))
def forward_impl(request):
    if request.param == "pytorch":
        return NativeSoftcappedSelectedLogprobOp(), "cpu"
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    impl = request.param.removeprefix("triton_") if request.param != "triton" else "row"
    return partial(_triton_forward, forward_impl=impl), "cuda"


def _fp32_reference(logits, token_ids, softcap=30.0):
    """Evaluate Native in FP32 on the same device and already-quantized inputs."""
    with torch.no_grad(), torch.autocast(device_type=logits.device.type, enabled=False):
        return NativeSoftcappedSelectedLogprobOp()(
            logits.detach().float(), token_ids, softcap=softcap
        )


def _fp32_backward_reference(logits, token_ids, upstream, *, softcap=30.0):
    """Keep the reference gradient in FP32, including for FP16/BF16 candidates."""
    with (
        torch.enable_grad(),
        torch.autocast(device_type=logits.device.type, enabled=False),
    ):
        return _native_backward(logits.float(), token_ids, upstream.float(), softcap=softcap)


def _assert_forward(op, logits, token_ids, *, softcap=30.0, accuracy=True):
    """Check the interface; only accuracy items consult independent Native FP32."""
    before = logits.clone()
    token_ids_before = token_ids.clone()
    actual = op(logits, token_ids)
    assert actual.shape == logits.shape[:1]
    assert actual.dtype == torch.float32
    assert actual.device == logits.device
    if accuracy:
        expected = _fp32_reference(logits, token_ids, softcap)
        _assert_fp32_accuracy(actual, expected, "forward_accuracy")
    assert torch.equal(logits, before)
    assert torch.equal(token_ids, token_ids_before)
    return actual


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "shape", [(0, 7), (3, 1), (3, 7), (2, 1023), (2, 1024), (3, 1025), (2, 262144)]
)
def test_forward_against_native_fp32(forward_impl, dtype, shape):
    op, device = forward_impl
    generator = torch.Generator().manual_seed(415)
    logits = (torch.randn(shape, generator=generator) * 30.0).to(device=device, dtype=dtype)
    # Exercise the first, last, and middle vocabulary positions, including
    # a selected token in the partial tile when V = 1025.
    token_ids = torch.tensor([0, shape[1] - 1, shape[1] // 2][: shape[0]], device=device)
    token_ids = token_ids.to(torch.int64)
    _assert_forward(op, logits, token_ids)


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", _DTYPES)
def test_uniform_and_saturated_rows(forward_impl, dtype):
    op, device = forward_impl
    logits = torch.zeros((5, 1025), device=device, dtype=dtype)
    logits[1].fill_(3000.0)
    logits[2].fill_(-3000.0)
    logits[3].fill_(-3000.0)
    logits[3, 1024] = 3000.0
    logits[4].fill_(3000.0)
    logits[4, 1024] = -3000.0
    token_ids = torch.tensor([0, 512, 1024, 1024, 1024], device=device)
    actual = _assert_forward(op, logits, token_ids)
    # Any uniform row has probability 1 / V, regardless of its score.
    expected_uniform = torch.full((3,), -math.log(1025), device=device)
    _assert_fp32_accuracy(actual[:3], expected_uniform, "forward_accuracy")
    assert torch.isfinite(actual).all()


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", _DTYPES)
def test_noncontiguous_inputs(forward_impl, dtype):
    op, device = forward_impl
    generator = torch.Generator().manual_seed(416)
    logits = (torch.randn((1025, 3), generator=generator) * 30).to(device=device, dtype=dtype).t()
    token_ids = torch.tensor([0, 0, 512, 0, 1024, 0], device=device)[::2]
    assert not logits.is_contiguous() and not token_ids.is_contiguous()
    _assert_forward(op, logits, token_ids)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (1025, 262144))
@pytest.mark.parametrize("forward_impl", ("row", "parallel", "row_pipelined"))
def test_triton_row_invariance(dtype, vocab, forward_impl):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    forward = partial(_triton_forward, forward_impl=forward_impl)

    generator = torch.Generator().manual_seed(417)
    logits = (torch.randn((5, vocab), generator=generator) * 30).to(device="cuda", dtype=dtype)
    token_ids = torch.tensor([0, 1, vocab // 2, vocab - 2, vocab - 1], device="cuda")
    together = forward(logits, token_ids)
    separate = torch.cat([forward(logits[i : i + 1], token_ids[i : i + 1]) for i in range(5)])
    _assert_bitwise(together, separate)

    permutation = torch.tensor([4, 2, 0, 3, 1], device="cuda")
    reordered = forward(logits[permutation], token_ids[permutation])
    _assert_bitwise(reordered, together[permutation])

    logits[0].fill_(3000.0)
    changed = forward(logits, token_ids)
    _assert_bitwise(changed[1:], together[1:])


@pytest.fixture(params=("pytorch", "triton", "triton_parallel", "triton_row_pipelined"))
def backward_impl(request):
    if request.param == "pytorch":
        return _native_backward, "cpu"
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    impl = request.param.removeprefix("triton_") if request.param != "triton" else "row"
    return partial(_triton_backward, forward_impl=impl), "cuda"


def _assert_backward(op, logits, token_ids, grad_selected_logprob, *, softcap=30.0, accuracy=True):
    """Check gradient metadata/purity, optionally comparing with Native autograd."""
    before = logits.clone()
    token_ids_before = token_ids.clone()
    grad_before = grad_selected_logprob.clone()
    actual = op(logits, token_ids, grad_selected_logprob)

    assert actual.shape == logits.shape
    assert actual.dtype == logits.dtype
    assert actual.device == logits.device
    if accuracy:
        expected = _fp32_backward_reference(
            logits, token_ids, grad_selected_logprob, softcap=softcap
        )
        _assert_fp32_accuracy(actual, expected, "gradient_accuracy")
    assert torch.equal(logits, before)
    assert torch.equal(token_ids, token_ids_before)
    assert torch.equal(grad_selected_logprob, grad_before)
    return actual


@pytest.mark.parametrize("check", _CHECKS)
@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", ((0, 7), (3, 1), (3, 1025)))
def test_configurable_softcap_accuracy_and_inference(operator_impl, softcap, dtype, shape, check):
    """Every strategy must use the same cap in forward, saved state and backward."""
    default_op, device = operator_impl
    op = partial(default_op, softcap=softcap)

    generator = torch.Generator().manual_seed(436)
    storage = (torch.randn((shape[0], 2 * shape[1]), generator=generator) * softcap).to(
        device=device, dtype=dtype
    )
    logits = storage[:, ::2].detach().requires_grad_(True)
    ids = torch.tensor([0, shape[1] // 2, shape[1] - 1][: shape[0]], device=device).long()
    upstream = torch.randn(shape[0] * 2, generator=generator).to(device)[::2]
    output = _assert_forward(op, logits, ids, softcap=softcap, accuracy=check == "accuracy")
    if check == "consistency":
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                _assert_bitwise(op(logits, ids), output.detach())
                _assert_bitwise(default_op.forward(logits, ids, softcap=softcap), output.detach())

    def backward_from_saved_graph(x, token_ids, gradient):
        return torch.autograd.grad(output, x, grad_outputs=gradient, retain_graph=True)[0]

    expected_gradient = _assert_backward(
        backward_from_saved_graph,
        logits,
        ids,
        upstream,
        softcap=softcap,
        accuracy=check == "accuracy",
    )
    _assert_backward(
        backward_from_saved_graph,
        logits,
        ids,
        torch.zeros_like(upstream),
        softcap=softcap,
        accuracy=check == "accuracy",
    )
    if check == "consistency":
        repeated_gradient = backward_from_saved_graph(logits, ids, upstream)
        _assert_bitwise(repeated_gradient, expected_gradient)

    # Only Triton promises batch invariance; native reductions may change bits.
    if check == "consistency" and device == "cuda" and shape[0] > 0:
        for row in range(shape[0]):
            single = logits[row : row + 1].detach().requires_grad_(True)
            single_output = op(single, ids[row : row + 1])
            single_gradient = torch.autograd.grad(
                single_output, single, grad_outputs=upstream[row : row + 1]
            )[0]
            _assert_bitwise(single_output.detach(), output[row : row + 1].detach())
            _assert_bitwise(single_gradient, expected_gradient[row : row + 1])


@pytest.mark.parametrize("check", _CHECKS)
@pytest.mark.parametrize("boundary", ("minimum", "near_maximum"))
@pytest.mark.parametrize("vocab", (1, 1025, 202048, 262144))
@pytest.mark.parametrize("dtype", _DTYPES)
def test_softcap_boundary_accuracy_and_inference(operator_impl, boundary, vocab, dtype, check):
    """Cover the supported cap endpoints, not only commonly used model values.

    The upper endpoint depends on V. Leave 1e-4 below it so FP32 conversion
    cannot accidentally push the test outside the documented safety margin.
    Include positive/negative saturation and verify rejection just outside each
    bound. The FP32 reference consumes the same already-quantized inputs.
    """
    default_op, device = operator_impl
    limits = torch.finfo(torch.float32)
    maximum = min(-math.log(2.0 * limits.tiny), math.log(limits.max / 2.0) - math.log(vocab))
    softcap = limits.tiny if boundary == "minimum" else maximum - 1e-4
    invalid = limits.tiny / 2 if boundary == "minimum" else maximum + 1e-4
    logits = torch.stack(
        (
            torch.linspace(-2.0, 2.0, vocab) * softcap,
            torch.full((vocab,), 3000.0),
            torch.full((vocab,), -3000.0),
        )
    ).to(device=device, dtype=dtype)
    logits.requires_grad_(True)
    ids = torch.tensor([vocab // 2, 0, vocab - 1], device=device)
    upstream = torch.tensor([0.5, -0.5, 0.0], device=device)
    op = partial(default_op, softcap=softcap)
    output = _assert_forward(op, logits, ids, softcap=softcap, accuracy=check == "accuracy")
    if check == "consistency":
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                _assert_bitwise(op(logits, ids), output.detach())

    def backward_from_saved_graph(x, token_ids, gradient):
        return torch.autograd.grad(output, x, grad_outputs=gradient, retain_graph=True)[0]

    gradient = _assert_backward(
        backward_from_saved_graph,
        logits,
        ids,
        upstream,
        softcap=softcap,
        accuracy=check == "accuracy",
    )
    if check == "consistency":
        _assert_bitwise(backward_from_saved_graph(logits, ids, upstream), gradient)
    with pytest.raises(ValueError, match="softcap"):
        default_op(logits, ids, softcap=invalid)


@pytest.mark.softcap_accuracy
def test_same_operator_keeps_softcap_per_graph(operator_impl):
    """A later call with another cap must not change an outstanding graph or the default."""
    op, device = operator_impl
    generator = torch.Generator().manual_seed(439)
    logits = (torch.randn((3, 1025), generator=generator) * 30.0).to(device).requires_grad_(True)
    ids = torch.tensor([0, 512, 1024], device=device)
    upstream = torch.tensor([0.25, -2.0, 1.5], device=device)
    caps = (1e-3, 0.125, 17.3, 30.0, 50.0, 70.0)
    outputs = [_assert_forward(partial(op, softcap=cap), logits, ids, softcap=cap) for cap in caps]
    _assert_forward(op, logits, ids)

    def backward_from_saved_graph(output, x, token_ids, gradient):
        return torch.autograd.grad(output, x, grad_outputs=gradient, retain_graph=True)[0]

    # Run backward out of forward order, then repeat it on the first graph.
    for index in (*reversed(range(len(caps))), 0):
        _assert_backward(
            partial(backward_from_saved_graph, outputs[index]),
            logits,
            ids,
            upstream,
            softcap=caps[index],
        )


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_configurable_softcap_saturation(forward_impl, softcap, dtype):
    forward, device = forward_impl
    op = partial(forward, softcap=softcap)
    logits = torch.full((3, 1025), 3000.0, device=device, dtype=dtype)
    logits[1].fill_(-3000.0)
    logits[2, 1024] = -3000.0
    ids = torch.tensor([0, 1024, 1024], device=device)
    output = _assert_forward(op, logits, ids, softcap=softcap)
    assert torch.isfinite(output).all()
    _assert_fp32_accuracy(output[:2], torch.full((2,), -math.log(1025)), "forward_accuracy")


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", _DTYPES)
def test_default_softcap_matches_explicit_30(dtype):
    """Changing the divisor representation must not change the default cap value."""
    generator = torch.Generator().manual_seed(437)
    logits = (torch.randn((3, 1025), generator=generator) * 30).to(dtype).requires_grad_(True)
    ids = torch.tensor([0, 512, 1024])
    upstream = torch.tensor([0.25, -2.0, 1.5])
    expected = softcapped_selected_logprob(logits, ids, softcap=30.0)
    expected_grad = torch.autograd.grad(expected, logits, grad_outputs=upstream)[0]
    op = NativeSoftcappedSelectedLogprobOp()
    for kwargs in ({}, {"softcap": 30}, {"softcap": 30.0}):
        output = op(logits, ids, **kwargs)
        gradient = torch.autograd.grad(output, logits, grad_outputs=upstream)[0]
        _assert_bitwise(output, expected)
        _assert_bitwise(gradient, expected_grad)


@pytest.mark.parametrize("check", _CHECKS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("vocab", (1025, 202048))
def test_native_gpu_reference_accuracy_and_inference(dtype, softcap, vocab, check):
    """Check native GPU autograd and inference; FP64 qualification is separate."""
    generator = torch.Generator().manual_seed(463)
    logits = (torch.randn((3, vocab), generator=generator) * softcap).to(device="cuda", dtype=dtype)
    logits[1, vocab // 2] = 39.5
    ids = torch.tensor([0, vocab // 2, vocab - 1], device="cuda")
    upstream = torch.tensor([0.5, -2.0, 1.5], device="cuda")
    op = partial(NativeSoftcappedSelectedLogprobOp(), softcap=softcap)
    training = _assert_forward(
        op, logits.requires_grad_(True), ids, softcap=softcap, accuracy=check == "accuracy"
    )
    if check == "consistency":
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                _assert_bitwise(op(logits, ids), training.detach())
    _assert_backward(
        partial(_native_backward, softcap=softcap),
        logits,
        ids,
        upstream,
        softcap=softcap,
        accuracy=check == "accuracy",
    )


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", (torch.float16, torch.float32))
def test_softcap50_wide_reference_regression(operator_impl, dtype):
    """Reproduce the wide-row scalar-division mismatch without allocating 4097 rows.

    This deterministic FP32 case failed the old PyTorch/Triton output comparison
    on A6000 at the selected 39.5 logit. Check against the revised FP32
    reference with unchanged output and gradient contract tolerances.
    """
    op, device = operator_impl
    cap = 50.0
    generator = torch.Generator().manual_seed(463)
    logits = torch.randn((3, 202048), generator=generator)
    logits[0] *= cap
    logits[1:] *= 30
    logits[1, 101024] = 39.5
    logits = logits.to(device=device, dtype=dtype).requires_grad_(True)
    ids = torch.tensor([0, 101024, 202047], device=device)
    upstream = torch.tensor([0.5, -2.0, 1.5], device=device)
    output = _assert_forward(partial(op, softcap=cap), logits, ids, softcap=cap)

    def backward_from_saved_graph(x, token_ids, gradient):
        return torch.autograd.grad(output, x, grad_outputs=gradient)[0]

    _assert_backward(backward_from_saved_graph, logits, ids, upstream, softcap=cap)


@pytest.mark.parametrize("check", _CHECKS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value", ("output", "gradient"))
def test_cap64_exp_rounding_regression(operator_impl, dtype, value, check):
    """Keep the large-cap exponential regression across all strategies and AUTO.

    With BF16 inputs on A6000, fast forward exp shifted row 12's output by
    1.52587890625e-5 against Native FP32. Use the original seeded inputs and
    unchanged contract limits, also covering FP16 and FP32 storage. Separate
    cases ensure an output failure cannot prevent the gradient check.
    """
    op, device = operator_impl
    cap = 64.0
    generator = torch.Generator().manual_seed(438)
    logits = (torch.randn((17, 32767), generator=generator) * cap).to(device=device, dtype=dtype)
    logits.requires_grad_(True)
    ids = torch.arange(17, device=device) * 1025
    upstream = torch.randn(17, generator=generator).to(device)
    output = op(logits, ids, softcap=cap)

    if value == "output":
        if check == "accuracy":
            _assert_fp32_accuracy(output, _fp32_reference(logits, ids, cap), "forward_accuracy")
        else:
            for context in (torch.no_grad(), torch.inference_mode()):
                with context:
                    _assert_bitwise(op(logits, ids, softcap=cap), output.detach())
    else:
        gradient = torch.autograd.grad(output, logits, grad_outputs=upstream, retain_graph=True)[0]
        if check == "accuracy":
            _assert_fp32_accuracy(
                gradient,
                _fp32_backward_reference(logits, ids, upstream, softcap=cap),
                "gradient_accuracy",
            )
        else:
            repeated = torch.autograd.grad(output, logits, grad_outputs=upstream)[0]
            _assert_bitwise(repeated, gradient)


@pytest.mark.parametrize(
    ("dtype", "softcap"),
    (
        pytest.param(torch.float16, 64.0, id="fp16-cap64"),
        pytest.param(torch.float16, 70.0, id="fp16-cap70"),
        pytest.param(torch.bfloat16, 70.0, id="bf16-cap70"),
        pytest.param(torch.float32, 64.0, id="fp32-cap64"),
        pytest.param(torch.float32, 70.0, id="fp32-cap70"),
    ),
)
@pytest.mark.parametrize("value", _VALUE_CHECKS)
def test_large_cap_wide_benchmark_regression(dtype, softcap, value):
    """Retain the five H100 benchmark failures with unchanged FP32 contract limits.

    Reproduce the original [1024, 262144] GPU-generated inputs, including RNG
    draw order for token IDs and upstreams. CPU generation or reducing M before
    generation would change the fixture. Other GPUs may generate different
    values; this test checks their accuracy too, without asserting failing rows.

    AUTO is the public implementation that failed the benchmark. Independent
    output, gradient and invariance cases prevent a forward accuracy failure
    from hiding the other checks. No xfail or tolerance relaxation is applied.
    The FP32 reference's large intermediates require several GiB of GPU memory.
    """
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        TritonSoftcappedSelectedLogprobOp,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(415)
    shape = (1024, 262144)
    logits = (
        30.0 * torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    ).to(dtype)
    ids = torch.randint(shape[1], (shape[0],), device=device, generator=generator)
    upstream = torch.randn(shape[0], device=device, dtype=torch.float32, generator=generator)
    op = TritonSoftcappedSelectedLogprobOp()

    if value == "output":
        with torch.no_grad():
            output = op(logits, ids, softcap=softcap)
        _assert_fp32_accuracy(output, _fp32_reference(logits, ids, softcap), "forward_accuracy")
    elif value == "gradient":
        logits.requires_grad_(True)
        output = op(logits, ids, softcap=softcap)
        gradient = torch.autograd.grad(output, logits, grad_outputs=upstream)[0]
        expected = _fp32_backward_reference(logits, ids, upstream, softcap=softcap)
        _assert_fp32_accuracy(gradient, expected, "gradient_accuracy")
    else:
        training = op(logits.requires_grad_(True), ids, softcap=softcap)
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                _assert_bitwise(op(logits, ids, softcap=softcap), training.detach())


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("softcap", (0.0, -1.0, float("nan"), float("inf"), 1e-50, 100.0))
def test_operator_rejects_unsafe_softcap(operator_impl, softcap):
    """Public Ops reject unsafe caps before dispatching to a private launcher."""
    op, device = operator_impl
    logits = torch.zeros((1, 7), device=device)
    ids = torch.tensor([0], device=device)
    with pytest.raises(ValueError, match="softcap"):
        op(logits, ids, softcap=softcap)


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize(
    "logits_shape,logits_dtype,ids_shape,ids_dtype,error,message",
    [
        pytest.param(
            (7,),
            torch.float32,
            (1,),
            torch.int64,
            ValueError,
            "shape",
            id="logits-rank",
        ),
        pytest.param(
            (1, 0),
            torch.float32,
            (1,),
            torch.int64,
            ValueError,
            "V > 0",
            id="empty-vocabulary",
        ),
        pytest.param(
            (1, 7),
            torch.int64,
            (1,),
            torch.int64,
            TypeError,
            "logits.*dtype",
            id="logits-dtype",
        ),
        pytest.param(
            (1, 7),
            torch.float32,
            (2,),
            torch.int64,
            ValueError,
            "token_ids.*shape",
            id="ids-shape",
        ),
        pytest.param(
            (1, 7),
            torch.float32,
            (1,),
            torch.float32,
            TypeError,
            "int64",
            id="ids-dtype",
        ),
    ],
)
def test_operator_rejects_invalid_input_metadata(
    operator_impl, logits_shape, logits_dtype, ids_shape, ids_dtype, error, message
):
    """Input validation belongs to the public interface for every forward strategy."""
    op, device = operator_impl
    logits = torch.zeros(logits_shape, dtype=logits_dtype, device=device)
    ids = torch.zeros(ids_shape, dtype=ids_dtype, device=device)
    with pytest.raises(error, match=message):
        op(logits, ids)


@pytest.mark.softcap_accuracy
def test_softcap_range_depends_on_vocabulary(operator_impl):
    """A valid scalar can still overflow the row sum for a sufficiently large V."""
    op, device = operator_impl
    ids = torch.tensor([0], device=device)
    assert torch.isfinite(op(torch.tensor([[-3000.0]], device=device), ids, softcap=80.0)).all()
    # exp(80) fits in FP32, but summing 32768 such values does not.
    with pytest.raises(ValueError, match="V=32768.*FP32 exp/sum range"):
        op(torch.zeros((1, 32768), device=device), ids, softcap=80.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value", _VALUE_CHECKS)
def test_configurable_softcap_auto_switch(softcap, dtype, value, monkeypatch, request):
    """Check an automatic strategy switch without mixing accuracy and bitwise failures.

    Output accuracy, gradient accuracy and batch-split invariance are separate
    cases. A reference accuracy failure cannot hide the gradient or bitwise
    checks. Invariance compares output, both saved statistics and all gradients.
    """
    from rl_engine.backends.shared.triton.logprob import softcapped_selected_logprob as module
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
        select_softcapped_logprob_strategy,
        softcapped_logprob_device_key,
    )

    select_softcapped_logprob_strategy.cache_clear()
    request.addfinalizer(select_softcapped_logprob_strategy.cache_clear)
    monkeypatch.setattr(
        module,
        "FORWARD_STRATEGY_CONFIGS",
        {
            module.ForwardConfigKey(
                device=module._DEFAULT_DEVICE,
                dtype=dtype,
                min_rows=1,
                max_rows=16,
                min_tiles=32,
                max_tiles=None,
            ): SoftcappedLogprobStrategy.PARALLEL,
        },
    )
    generator = torch.Generator().manual_seed(438)
    logits = (torch.randn((17, 32767), generator=generator) * softcap).to("cuda", dtype)
    ids = torch.arange(17, device="cuda") * 1025
    upstream = torch.randn(17, generator=generator).cuda()
    device = softcapped_logprob_device_key(logits.device)
    assert (
        select_softcapped_logprob_strategy(device, dtype, 17, 32767)
        == SoftcappedLogprobStrategy.ROW
    )
    assert (
        select_softcapped_logprob_strategy(device, dtype, 16, 32767)
        == SoftcappedLogprobStrategy.PARALLEL
    )
    op = partial(TritonSoftcappedSelectedLogprobOp(), softcap=softcap)
    whole = _evaluate_triton_with_statistics(op, logits, ids, upstream)
    output, _, _, gradient = whole
    if value == "output":
        _assert_fp32_accuracy(output, _fp32_reference(logits, ids, softcap), "forward_accuracy")
    elif value == "gradient":
        _assert_fp32_accuracy(
            gradient,
            _fp32_backward_reference(logits, ids, upstream, softcap=softcap),
            "gradient_accuracy",
        )
    else:
        parts = [
            _evaluate_triton_with_statistics(
                op, logits[start:stop], ids[start:stop], upstream[start:stop]
            )
            for start, stop in ((0, 16), (16, 17))
        ]
        # Compare output, row maximum, shifted log-sum and every gradient value.
        for index in range(len(whole)):
            _assert_bitwise(torch.cat([part[index] for part in parts]), whole[index])


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "shape", [(0, 7), (3, 1), (3, 7), (2, 1023), (2, 1024), (3, 1025), (2, 262144)]
)
def test_backward_against_native_fp32_autograd(backward_impl, dtype, shape):
    op, device = backward_impl
    generator = torch.Generator().manual_seed(418)
    logits = (torch.randn(shape, generator=generator) * 15).to(device=device, dtype=dtype)
    token_ids = torch.tensor([0, shape[1] - 1, shape[1] // 2][: shape[0]], device=device)
    token_ids = token_ids.to(torch.int64)
    grad_selected_logprob = torch.randn(shape[0], generator=generator).to(device)
    _assert_backward(op, logits, token_ids, grad_selected_logprob)


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", _DTYPES)
def test_backward_saturation_and_zero_upstream(backward_impl, dtype):
    op, device = backward_impl
    logits = torch.zeros((4, 1025), device=device, dtype=dtype)
    logits[0].fill_(3000.0)
    logits[1].fill_(-3000.0)
    logits[2, 1024] = 15.0
    token_ids = torch.tensor([0, 1024, 1024, 512], device=device)
    grad_selected_logprob = torch.tensor([2.0, -3.0, 0.0, -0.75], device=device)
    actual = _assert_backward(op, logits, token_ids, grad_selected_logprob)
    # Saturated softcap and a zero upstream gradient must both block gradients.
    assert torch.equal(actual[:3], torch.zeros_like(actual[:3]))
    # A negative upstream gradient reverses the signs on the uniform row.
    assert actual[3, 512] < 0 and actual[3, 0] > 0


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", _DTYPES)
def test_backward_noncontiguous_inputs(backward_impl, dtype):
    op, device = backward_impl
    generator = torch.Generator().manual_seed(419)
    logits = (torch.randn((1025, 3), generator=generator) * 15).to(device=device, dtype=dtype).t()
    token_ids = torch.tensor([0, 0, 512, 0, 1024, 0], device=device)[::2]
    grad_selected_logprob = torch.tensor([2.0, 0, -0.5, 0, 1.25, 0], device=device)[::2]
    assert not logits.is_contiguous() and not token_ids.is_contiguous()
    assert not grad_selected_logprob.is_contiguous()
    _assert_backward(op, logits, token_ids, grad_selected_logprob)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (1025, 262144))
@pytest.mark.parametrize("forward_impl", ("row", "parallel", "row_pipelined"))
def test_triton_backward_row_invariance(dtype, vocab, forward_impl):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    backward = partial(_triton_backward, forward_impl=forward_impl)

    generator = torch.Generator().manual_seed(420)
    logits = (torch.randn((3, vocab), generator=generator) * 15).to(device="cuda", dtype=dtype)
    token_ids = torch.tensor([0, vocab // 2, vocab - 1], device="cuda")
    upstream = torch.tensor([0.25, -2.0, 1.5], device="cuda")
    together = backward(logits, token_ids, upstream)
    separate = torch.cat(
        [backward(logits[i : i + 1], token_ids[i : i + 1], upstream[i : i + 1]) for i in range(3)]
    )
    _assert_bitwise(together, separate)

    permutation = torch.tensor([2, 0, 1], device="cuda")
    reordered = backward(logits[permutation], token_ids[permutation], upstream[permutation])
    _assert_bitwise(reordered, together[permutation])

    logits[0].fill_(3000.0)
    changed = backward(logits, token_ids, upstream)
    _assert_bitwise(changed[1:], together[1:])


@pytest.fixture(
    params=(
        "pytorch",
        "triton",
        "triton_parallel",
        "triton_row_pipelined",
        "triton_auto",
    )
)
def operator_impl(request):
    if request.param == "pytorch":
        return NativeSoftcappedSelectedLogprobOp(), "cpu"
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    impl = {
        "triton": SoftcappedLogprobStrategy.ROW,
        "triton_parallel": SoftcappedLogprobStrategy.PARALLEL,
        "triton_row_pipelined": SoftcappedLogprobStrategy.ROW_PIPELINED,
        "triton_auto": None,
    }[request.param]
    return TritonSoftcappedSelectedLogprobOp(forward_impl=impl), "cuda"


@pytest.mark.parametrize("check", _CHECKS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("layout", ("contiguous", "transpose", "empty"))
def test_operator_autograd_and_inference(operator_impl, dtype, layout, check):
    op, device = operator_impl
    shape = {"contiguous": (3, 1025), "transpose": (1025, 3), "empty": (0, 7)}[layout]
    generator = torch.Generator().manual_seed(421)
    base = (torch.randn(shape, generator=generator) * 15).to(device=device, dtype=dtype)
    base.requires_grad_(True)
    # A non-leaf view checks that the Function returns gradients to the
    # surrounding graph, including through a transpose and a prior operation.
    logits = base + 0.5
    if layout == "transpose":
        logits = logits.t()
        assert not logits.is_contiguous()
    n_rows, vocab_size = logits.shape
    token_ids = torch.tensor([0, vocab_size // 2, vocab_size - 1][:n_rows], device=device)
    token_ids = token_ids.to(torch.int64)
    upstream = torch.randn(n_rows * 2, generator=generator).to(device)[::2]

    output = op(logits, token_ids)
    assert output.requires_grad and output.grad_fn is not None
    assert output.shape == (n_rows,) and output.dtype == torch.float32
    if check == "accuracy":
        _assert_fp32_accuracy(output, _fp32_reference(logits, token_ids), "forward_accuracy")
    (grad_base,) = torch.autograd.grad(output, base, grad_outputs=upstream)

    assert grad_base.shape == base.shape and grad_base.dtype == dtype
    if check == "accuracy":
        # Start at the operator boundary, retaining quantized inputs and FP32 gradients.
        expected_grad = _fp32_backward_reference(logits, token_ids, upstream)
        if layout == "transpose":
            expected_grad = expected_grad.t()
        _assert_fp32_accuracy(grad_base, expected_grad, "gradient_accuracy")
    else:
        # The Op must use the same arithmetic with or without graph recording.
        with torch.no_grad():
            no_grad_output = op(logits, token_ids)
        with torch.inference_mode():
            inference_output = op(logits, token_ids)
        detached_output = op(logits.detach(), token_ids)
        for actual in (no_grad_output, inference_output, detached_output):
            assert not actual.requires_grad
            _assert_bitwise(actual, output.detach())


@pytest.mark.parametrize("check", _CHECKS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (1025, 262144))
@pytest.mark.parametrize("forward_impl", (None, "row", "parallel", "row_pipelined"))
@pytest.mark.parametrize("softcap", (30.0, 50.0))
def test_triton_noncurrent_device_forward_backward(dtype, vocab, forward_impl, softcap, check):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA or ROCm GPUs are required for the noncurrent-device regression")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
        _launch_softcapped_selected_logprob_bwd,
    )

    device = torch.device("cuda:1")
    generator = torch.Generator().manual_seed(435)
    # Include copies of strided input tensors while cuda:0 is current.
    storage = (torch.randn((3, 2 * vocab), generator=generator) * 15).to(device=device, dtype=dtype)
    logits = storage[:, ::2].detach().requires_grad_(True)
    token_ids = torch.tensor([0, 0, vocab // 2, 0, vocab - 1, 0], device=device)[::2]
    upstream = torch.tensor([0.25, 0, -2.0, 0, 1.5, 0], device=device)[::2]
    strategy = None if forward_impl is None else SoftcappedLogprobStrategy(forward_impl)
    op = TritonSoftcappedSelectedLogprobOp(forward_impl=strategy)

    with torch.cuda.device(0):
        output = op(logits, token_ids, softcap=softcap)
        assert torch.cuda.current_device() == 0
        assert output.device == device and output.dtype == torch.float32
        saved_logits, saved_ids, row_max, log_sum_exp_shifted = output.grad_fn.saved_tensors
        assert all(
            value.device == device
            for value in (saved_logits, saved_ids, row_max, log_sum_exp_shifted)
        )

        # Call backward directly too: autograd may itself select the forward
        # device, which could otherwise hide a missing guard in the launcher.
        direct_gradient = _launch_softcapped_selected_logprob_bwd(
            logits, token_ids, upstream, row_max, log_sum_exp_shifted, softcap
        )
        assert torch.cuda.current_device() == 0
        (gradient,) = torch.autograd.grad(output, logits, grad_outputs=upstream)
        assert torch.cuda.current_device() == 0
        assert gradient.device == direct_gradient.device == device
        assert gradient.dtype == direct_gradient.dtype == dtype
        if check == "consistency":
            _assert_bitwise(gradient, direct_gradient)
        else:
            _assert_fp32_accuracy(
                output, _fp32_reference(logits, token_ids, softcap), "forward_accuracy"
            )
            _assert_fp32_accuracy(
                gradient,
                _fp32_backward_reference(logits, token_ids, upstream, softcap=softcap),
                "gradient_accuracy",
            )

        if check == "consistency":
            for context in (torch.no_grad(), torch.inference_mode()):
                with context:
                    inference_output = op(logits, token_ids, softcap=softcap)
                assert torch.cuda.current_device() == 0
                assert inference_output.device == device
                _assert_bitwise(inference_output, output.detach())


@pytest.mark.parametrize("check", _CHECKS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", ((0, 7), (3, 1025), (2, 262144)))
@pytest.mark.parametrize("forward_impl", ("row", "parallel", "row_pipelined"))
def test_triton_saved_shifted_statistics_and_repeated_backward(dtype, shape, forward_impl, check):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    generator = torch.Generator().manual_seed(424)
    logits = (torch.randn(shape, generator=generator) * 15).to(device="cuda", dtype=dtype)
    logits.requires_grad_(True)
    ids = torch.tensor(
        [0, shape[1] - 1, shape[1] // 2][: shape[0]], device="cuda", dtype=torch.int64
    )
    op = TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy(forward_impl))
    output = op(logits, ids)
    _, _, row_max, log_sum_exp_shifted = output.grad_fn.saved_tensors
    for statistic in (row_max, log_sum_exp_shifted):
        assert statistic.shape == logits.shape[:1]
        assert statistic.dtype == torch.float32 and statistic.device == logits.device
        assert statistic.numel() * statistic.element_size() == 4 * shape[0]
    if check == "accuracy":
        divisor = torch.tensor(30.0, device=logits.device, dtype=torch.float32)
        scores = 30.0 * torch.tanh(torch.div(logits.detach().float(), divisor))
        expected_max = scores.amax(dim=-1)
        expected_log_sum = (scores - expected_max[:, None]).exp().sum(dim=-1).log()
        _assert_fp32_accuracy(row_max, expected_max, "forward_accuracy")
        _assert_fp32_accuracy(log_sum_exp_shifted, expected_log_sum, "forward_accuracy")
    saved_before = (row_max.clone(), log_sum_exp_shifted.clone())

    def backward_from_saved_graph(x, token_ids, upstream):
        return torch.autograd.grad(output, x, grad_outputs=upstream, retain_graph=True)[0]

    # Different upstream gradients must reuse, rather than overwrite, the same
    # forward statistics. Compare both backward calls with Native FP32 autograd.
    for _ in range(2):
        upstream = torch.randn(shape[0], generator=generator).to(device="cuda")
        _assert_backward(
            backward_from_saved_graph, logits, ids, upstream, accuracy=check == "accuracy"
        )
        if check == "consistency":
            _assert_bitwise(row_max, saved_before[0])
            _assert_bitwise(log_sum_exp_shifted, saved_before[1])

    if check == "consistency":
        with torch.inference_mode():
            _assert_bitwise(op(logits, ids), output.detach())


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("forward_impl", ("row", "parallel", "row_pipelined"))
def test_triton_padding_and_strided_layout_invariance(dtype, forward_impl):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    op = TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy(forward_impl))
    generator = torch.Generator().manual_seed(422)
    row = (torch.randn((1, 1025), generator=generator) * 15).to(device="cuda", dtype=dtype)
    target = torch.tensor([1024], device="cuda")
    upstream = torch.tensor([-1.75], device="cuda")

    def evaluate(logits, ids, grad_output):
        leaf = logits.detach().requires_grad_(True)
        output = op(leaf, ids)
        gradient = torch.autograd.grad(output, leaf, grad_outputs=grad_output)[0]
        return output.detach(), gradient

    expected = evaluate(row, target, upstream)
    storage = torch.zeros((7, 2050), device="cuda", dtype=dtype)
    view = storage[:, ::2]
    view[3:4] = row
    ids = torch.zeros(7, device="cuda", dtype=torch.int64)
    ids[3] = target[0]
    grads = torch.zeros(7, device="cuda")
    grads[3] = upstream[0]
    for _ in range(2):
        actual = evaluate(view, ids, grads)
        _assert_bitwise(actual[0][3:4], expected[0])
        _assert_bitwise(actual[1][3:4], expected[1])
        storage[:, 1::2].fill_(3000.0)
        view[:3].fill_(-3000.0)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "vocab",
    (1, 1023, 1024, 1025, 2048, 2049, 3073, 4096, 4097, 5121, 6145, 7169, 262144),
)
@pytest.mark.parametrize("merge_unroll", (1, 4))
def test_triton_forward_variants_match_bitwise(dtype, vocab, merge_unroll):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _launch_softcapped_selected_logprob_bwd,
        _launch_softcapped_selected_logprob_fwd,
        _launch_softcapped_selected_logprob_fwd_parallel,
    )

    generator = torch.Generator().manual_seed(426)
    logits = (torch.randn((5, vocab), generator=generator) * 30).to(device="cuda", dtype=dtype)
    logits[0].zero_()
    logits[1].fill_(3000.0)
    logits[2].fill_(-3000.0)
    logits[3, vocab - 1] = 3000.0
    ids = torch.tensor([0, vocab - 1, vocab // 2, vocab - 1, 0], device="cuda")
    upstream = torch.tensor([1.0, -2.0, 0.0, -0.75, 0.25], device="cuda")
    expected = _launch_softcapped_selected_logprob_fwd(logits, ids, softcap=30.0)
    actual = _launch_softcapped_selected_logprob_fwd_parallel(
        logits, ids, softcap=30.0, merge_unroll=merge_unroll
    )

    for reference, result in zip(expected, actual, strict=True):
        _assert_bitwise(result, reference)
    grad_expected = _launch_softcapped_selected_logprob_bwd(
        logits, ids, upstream, expected[1], expected[2], softcap=30.0
    )
    grad_actual = _launch_softcapped_selected_logprob_bwd(
        logits, ids, upstream, actual[1], actual[2], softcap=30.0
    )
    _assert_bitwise(grad_actual, grad_expected)


@pytest.mark.parametrize("check", _CHECKS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (1, 1025, 2049, 4097, 262144))
@pytest.mark.parametrize("stages", (2, 3))
def test_triton_pipeline_matches_its_unhinted_family(dtype, vocab, stages, check):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _launch_softcapped_selected_logprob_bwd,
        _launch_softcapped_selected_logprob_fwd,
    )

    generator = torch.Generator().manual_seed(430)
    logits = (torch.randn((5, vocab), generator=generator) * 30).to(device="cuda", dtype=dtype)
    logits[0].zero_()
    logits[1].fill_(3000.0)
    logits[2].fill_(-3000.0)
    ids = torch.tensor([0, vocab - 1, vocab // 2, vocab - 1, 0], device="cuda")
    upstream = torch.tensor([1.0, -2.0, 0.0, -0.75, 0.25], device="cuda")
    expected = _launch_softcapped_selected_logprob_fwd(logits, ids, softcap=30.0)
    actual = _launch_softcapped_selected_logprob_fwd(
        logits, ids, softcap=30.0, pipeline_stages=stages
    )
    if check == "consistency":
        for result, reference in zip(actual, expected, strict=True):
            _assert_bitwise(result, reference)
    grad_expected = _launch_softcapped_selected_logprob_bwd(
        logits, ids, upstream, expected[1], expected[2], softcap=30.0
    )
    grad_actual = _launch_softcapped_selected_logprob_bwd(
        logits, ids, upstream, actual[1], actual[2], softcap=30.0
    )
    if check == "consistency":
        _assert_bitwise(grad_actual, grad_expected)
    else:
        _assert_fp32_accuracy(actual[0], _fp32_reference(logits, ids), "forward_accuracy")
        _assert_fp32_accuracy(
            grad_actual,
            _fp32_backward_reference(logits, ids, upstream),
            "gradient_accuracy",
        )


def _evaluate_triton_with_statistics(op, logits, token_ids, upstream):
    leaf = logits.detach().requires_grad_(True)
    output = op(leaf, token_ids)
    _, _, row_max, log_sum_exp_shifted = output.grad_fn.saved_tensors
    gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]
    return output.detach(), row_max, log_sum_exp_shifted, gradient


@pytest.mark.parametrize("check", _CHECKS)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("softcap", (0.125, 30.0, 50.0, 64.0, 70.0))
def test_shifted_maxima_tails_and_signed_zero_preserve_all_strategy_bits(dtype, softcap, check):
    """Exercise negative maxima, tied maxima, signed zeros and a maximum in a tail.

    Validate outputs and all input gradients against Native FP32, then require
    exact agreement of outputs, both saved statistics and gradients across
    every public strategy and AUTO. Uniform rows also have a known logprob.
    """
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    vocab = 3073
    logits = torch.full((6, vocab), -3000.0, device="cuda", dtype=dtype)
    logits[1].fill_(3000.0)
    logits[2].zero_()
    logits[2, ::2] = -0.0
    logits[3, -1] = 3000.0
    logits[4, 1023:1025] = 3000.0
    logits[5].fill_(-2.0 * softcap)
    logits[5, -1] = -0.5 * softcap
    ids = torch.tensor([0, vocab - 1, 1024, vocab - 1, 1023, 0], device="cuda")
    upstream = torch.tensor([0.5, -1.0, 2.0, -0.75, 0.0, 1.5], device="cuda")
    if check == "accuracy":
        expected_output = _fp32_reference(logits, ids, softcap)
        expected_gradient = _fp32_backward_reference(logits, ids, upstream, softcap=softcap)
        divisor = torch.tensor(softcap, device="cuda", dtype=torch.float32)
        expected_max = (softcap * torch.tanh(torch.div(logits.float(), divisor))).amax(dim=-1)
        expected_uniform = torch.full((3,), -math.log(vocab), device="cuda", dtype=torch.float32)
    reference = None
    for strategy in (*SoftcappedLogprobStrategy, None):
        op = partial(TritonSoftcappedSelectedLogprobOp(forward_impl=strategy), softcap=softcap)
        actual = _evaluate_triton_with_statistics(op, logits, ids, upstream)
        output, row_max, log_sum, gradient = actual
        if check == "accuracy":
            _assert_fp32_accuracy(output, expected_output, "forward_accuracy")
            _assert_fp32_accuracy(gradient, expected_gradient, "gradient_accuracy")
            _assert_fp32_accuracy(row_max, expected_max, "forward_accuracy")
            _assert_fp32_accuracy(output[:3], expected_uniform, "forward_accuracy")
        assert row_max[0] < 0 and row_max[5] < 0
        assert row_max[2] == 0 and not torch.signbit(row_max[2])
        assert torch.all((log_sum >= 0) & (log_sum <= math.log(vocab) + 1e-5))
        if check == "consistency":
            if reference is None:
                reference = actual
            else:
                for value, expected in zip(actual, reference, strict=True):
                    _assert_bitwise(value, expected)
            for context in (torch.no_grad(), torch.inference_mode()):
                with context:
                    _assert_bitwise(op(logits, ids), output)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "shape,softcap",
    [
        ((0, 7), 30.0),
        ((4, 262144), 30.0),
        ((16, 49152), 30.0),
        ((64, 32768), 30.0),
        ((256, 65536), 30.0),
        ((8, 202048), 20.0),
        ((2, 2097153), 30.0),
    ],
)
def test_triton_auto_matches_explicit_row_and_inference(dtype, shape, softcap):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    generator = torch.Generator().manual_seed(428)
    logits = (torch.randn(shape, generator=generator) * 30).to(device="cuda", dtype=dtype)
    ids = torch.arange(shape[0], device="cuda", dtype=torch.int64) * 1025 % shape[1]
    upstream = torch.randn(shape[0], generator=generator).to(device="cuda")
    op = partial(TritonSoftcappedSelectedLogprobOp(forward_impl=None), softcap=softcap)
    reference = _evaluate_triton_with_statistics(
        partial(
            TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy.ROW),
            softcap=softcap,
        ),
        logits,
        ids,
        upstream,
    )
    actual = _evaluate_triton_with_statistics(op, logits, ids, upstream)
    for result, expected in zip(actual, reference, strict=True):
        _assert_bitwise(result, expected)
    with torch.no_grad():
        _assert_bitwise(op(logits, ids), reference[0])
    with torch.inference_mode():
        _assert_bitwise(op(logits, ids), reference[0])


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("softcap", _SOFTCAPS)
@pytest.mark.parametrize("vocab", (1025, 262145))
def test_all_strategies_preserve_output_statistics_and_all_gradients(dtype, softcap, vocab):
    """Every selectable strategy agrees, including wide tails and nondefault caps."""
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    generator = torch.Generator().manual_seed(464)
    logits = (30 * torch.randn((3, vocab), generator=generator)).to(device="cuda", dtype=dtype)
    ids = torch.tensor([0, vocab // 2, vocab - 1], device="cuda")
    upstream = torch.tensor([0.5, -2.0, 0.0], device="cuda")
    expected = None
    for strategy in (*SoftcappedLogprobStrategy, None):
        op = partial(TritonSoftcappedSelectedLogprobOp(forward_impl=strategy), softcap=softcap)
        actual = _evaluate_triton_with_statistics(op, logits, ids, upstream)
        if expected is None:
            expected = actual
        else:
            for result, reference in zip(actual, expected, strict=True):
                _assert_bitwise(result, reference)
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                _assert_bitwise(op(logits, ids), expected[0])


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_auto_batch_split_switches_strategy_without_changing_bits(
    dtype, monkeypatch, request
):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob import softcapped_selected_logprob as module
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
        select_softcapped_logprob_strategy,
        softcapped_logprob_device_key,
    )

    # Exercise an automatic switch with a small batch, independently of the
    # production boundaries covered by the dedicated tests below.
    select_softcapped_logprob_strategy.cache_clear()
    request.addfinalizer(select_softcapped_logprob_strategy.cache_clear)
    monkeypatch.setattr(
        module,
        "FORWARD_STRATEGY_CONFIGS",
        {
            module.ForwardConfigKey(
                device=module._DEFAULT_DEVICE,
                dtype=dtype,
                min_rows=3,
                max_rows=1024,
                min_tiles=256,
                max_tiles=256,
            ): SoftcappedLogprobStrategy.PARALLEL,
        },
    )
    generator = torch.Generator().manual_seed(429)
    logits = (torch.randn((4, 262144), generator=generator) * 30).to(device="cuda", dtype=dtype)
    ids = torch.tensor([0, 1023, 262143, 1024], device="cuda")
    upstream = torch.tensor([0.25, -2.0, 0.0, 1.5], device="cuda")
    key = softcapped_logprob_device_key(logits.device)
    assert (
        select_softcapped_logprob_strategy(key, dtype, 4, 262144)
        is SoftcappedLogprobStrategy.PARALLEL
    )
    assert (
        select_softcapped_logprob_strategy(key, dtype, 2, 262144) is SoftcappedLogprobStrategy.ROW
    )
    op = TritonSoftcappedSelectedLogprobOp()
    together = _evaluate_triton_with_statistics(op, logits, ids, upstream)
    halves = [
        _evaluate_triton_with_statistics(op, logits[i : i + 2], ids[i : i + 2], upstream[i : i + 2])
        for i in (0, 2)
    ]
    for expected, *parts in zip(together, *halves, strict=True):
        _assert_bitwise(torch.cat(parts), expected)
    permutation = torch.tensor([3, 1, 0, 2], device="cuda")
    reordered = _evaluate_triton_with_statistics(
        op, logits[permutation], ids[permutation], upstream[permutation]
    )
    for result, expected in zip(reordered, together, strict=True):
        _assert_bitwise(result, expected[permutation])
    storage = torch.zeros((4, 524288), device="cuda", dtype=dtype)
    view = storage[:, ::2]
    view.copy_(logits)
    for result, expected in zip(
        _evaluate_triton_with_statistics(op, view, ids, upstream), together, strict=True
    ):
        _assert_bitwise(result, expected)


@pytest.mark.softcap_consistency
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (31745, 32768, 32769))
def test_triton_production_range_boundary_preserves_output_statistics_and_gradients(dtype, vocab):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
        select_softcapped_logprob_strategy,
        softcapped_logprob_device_key,
    )

    max_rows = 4096
    # Allocate on the GPU directly: inputs use at most about 513 MiB here.
    # Compare each chunk in place rather than concatenating full-sized gradients.
    generator = torch.Generator(device="cuda").manual_seed(431)
    n_rows = max_rows + 1
    logits = torch.randn((n_rows, vocab), generator=generator, device="cuda", dtype=dtype)
    logits.mul_(15.0)
    ids = torch.arange(n_rows, device="cuda", dtype=torch.int64) * 1025 % vocab
    ids[0], ids[-1] = 0, vocab - 1
    upstream = torch.randn(n_rows, generator=generator, device="cuda")
    key = softcapped_logprob_device_key(logits.device)
    assert (
        select_softcapped_logprob_strategy(key, dtype, max_rows, vocab)
        is SoftcappedLogprobStrategy.PARALLEL
    )
    assert (
        select_softcapped_logprob_strategy(key, dtype, n_rows, vocab)
        is SoftcappedLogprobStrategy.ROW
    )
    reference = _evaluate_triton_with_statistics(
        TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy.ROW),
        logits,
        ids,
        upstream,
    )
    auto = TritonSoftcappedSelectedLogprobOp()
    together = _evaluate_triton_with_statistics(auto, logits, ids, upstream)
    for result, expected in zip(together, reference, strict=True):
        _assert_bitwise(result, expected)
    del together, result, expected
    # The whole batch selects ROW; both chunks select PARALLEL. Output, saved
    # statistics and gradients must keep the same bytes across this real policy switch.
    for start, stop in ((0, max_rows), (max_rows, n_rows)):
        actual = _evaluate_triton_with_statistics(
            auto, logits[start:stop], ids[start:stop], upstream[start:stop]
        )
        for result, expected in zip(actual, reference, strict=True):
            _assert_bitwise(result, expected[start:stop])
        del actual, result, expected


@pytest.mark.softcap_accuracy
def test_triton_explicit_strategy_bypasses_device_lookup_and_auto_validates_first(
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob import softcapped_selected_logprob as module

    def unexpected_lookup(device):
        pytest.fail("Explicit strategies and invalid inputs must not query hardware")

    monkeypatch.setattr(module, "softcapped_logprob_device_key", unexpected_lookup)
    logits = torch.zeros((1, 1025), device="cuda")
    ids = torch.tensor([1024], device="cuda")
    outputs = [
        module.TritonSoftcappedSelectedLogprobOp(forward_impl=choice)(logits, ids)
        for choice in module.SoftcappedLogprobStrategy
    ]
    for output in outputs[1:]:
        _assert_bitwise(output, outputs[0])
    with pytest.raises(ValueError, match="shape"):
        module.TritonSoftcappedSelectedLogprobOp()(logits[0], ids)
    with pytest.raises(TypeError, match="int64"):
        module.TritonSoftcappedSelectedLogprobOp()(logits, ids.float())
    with pytest.raises(ValueError, match="softcap"):
        module.TritonSoftcappedSelectedLogprobOp()(logits, ids, softcap=0.0)


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (2048, 2049, 3073, 4097, 5121, 6145))
@pytest.mark.parametrize("merge_unroll", (1, 4))
def test_triton_parallel_forward_scratch_and_output_bounds(dtype, vocab, merge_unroll):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    import triton

    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _BLOCK_V,
        _launch_softcapped_selected_logprob_fwd,
        _softcapped_selected_logprob_fwd_max_merge_kernel,
        _softcapped_selected_logprob_fwd_max_partial_kernel,
        _softcapped_selected_logprob_fwd_merge_kernel,
        _softcapped_selected_logprob_fwd_partial_kernel,
    )

    n_rows = 4
    n_tiles = triton.cdiv(vocab, _BLOCK_V)
    generator = torch.Generator().manual_seed(427)
    logits = (torch.randn((n_rows, vocab), generator=generator) * 15).to(device="cuda", dtype=dtype)
    # Include invalid IDs to verify the merge masks selected-logit reads.
    ids = torch.tensor([-1, 1024, vocab - 1, vocab], device="cuda")

    def guarded(shape):
        storage = torch.full((math.prod(shape) + 2,), -123.0, device="cuda")
        view = storage[1:-1].view(shape)
        view.fill_(float("nan"))
        return storage, view

    scratch_storage, scratch = guarded((n_rows, n_tiles))
    output_storage, output = guarded((n_rows,))
    max_storage, row_max = guarded((n_rows,))
    lse_storage, log_sum_exp_shifted = guarded((n_rows,))
    _softcapped_selected_logprob_fwd_max_partial_kernel[(n_rows, n_tiles)](
        logits,
        scratch,
        vocab,
        n_tiles,
        _BLOCK_V,
        SOFTCAP=30.0,
        num_warps=4,
        enable_fp_fusion=False,
    )
    max_scratch_before = scratch.clone()
    _softcapped_selected_logprob_fwd_max_merge_kernel[(n_rows,)](
        scratch, row_max, n_tiles, num_warps=4, enable_fp_fusion=False
    )
    _assert_bitwise(scratch, max_scratch_before)
    row_max_before = row_max.clone()
    # The sum stage overwrites every entry; no maximum may remain in the scratch.
    scratch.fill_(float("nan"))
    _softcapped_selected_logprob_fwd_partial_kernel[(n_rows, n_tiles)](
        logits,
        row_max,
        scratch,
        vocab,
        n_tiles,
        _BLOCK_V,
        SOFTCAP=30.0,
        num_warps=4,
        enable_fp_fusion=False,
    )
    scratch_before = scratch.clone()
    _softcapped_selected_logprob_fwd_merge_kernel[(n_rows,)](
        logits,
        ids,
        row_max,
        scratch,
        output,
        log_sum_exp_shifted,
        vocab,
        n_tiles,
        SOFTCAP=30.0,
        MERGE_UNROLL=merge_unroll,
        num_warps=4,
        enable_fp_fusion=False,
    )
    expected, expected_max, expected_log_sum = _launch_softcapped_selected_logprob_fwd(
        logits, ids, softcap=30.0
    )
    assert torch.isfinite(scratch).all()
    _assert_bitwise(scratch, scratch_before)
    _assert_bitwise(row_max, expected_max)
    _assert_bitwise(row_max, row_max_before)
    _assert_bitwise(log_sum_exp_shifted, expected_log_sum)
    _assert_bitwise(output[1:3], expected[1:3])
    assert torch.isnan(output[[0, 3]]).all()
    for storage in (scratch_storage, output_storage, max_storage, lse_storage):
        assert torch.equal(storage[[0, -1]], torch.full((2,), -123.0, device="cuda"))


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (2048, 2049))
def test_triton_backward_tile_coverage_and_output_bounds(dtype, vocab):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    import triton

    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _BLOCK_V,
        _launch_softcapped_selected_logprob_fwd,
        _softcapped_selected_logprob_bwd_kernel,
    )

    n_rows = 4
    generator = torch.Generator().manual_seed(425)
    logits = (torch.randn((n_rows, vocab), generator=generator) * 15).to(device="cuda", dtype=dtype)
    # Selected tokens exercise both sides of a tile boundary and the last lane.
    ids = torch.tensor([0, 1023, 1024, vocab - 1], device="cuda")
    upstream = torch.tensor([0.25, -2.0, 1.5, -0.75], device="cuda")
    storage = torch.full((n_rows * vocab + 2 * _BLOCK_V,), -123.0, device="cuda", dtype=dtype)
    grad_logits = storage[_BLOCK_V:-_BLOCK_V].view(n_rows, vocab)
    grad_logits.fill_(float("nan"))

    def backward_into_guarded_output(x, token_ids, grad_output):
        _, row_max, log_sum_exp_shifted = _launch_softcapped_selected_logprob_fwd(
            x, token_ids, softcap=30.0
        )
        _softcapped_selected_logprob_bwd_kernel[(n_rows, triton.cdiv(vocab, _BLOCK_V))](
            x,
            token_ids,
            grad_output,
            row_max,
            log_sum_exp_shifted,
            grad_logits,
            vocab,
            BLOCK_V=_BLOCK_V,
            SOFTCAP=30.0,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return grad_logits

    _assert_backward(backward_into_guarded_output, logits, ids, upstream)
    # NaN initialization detects missing tiles; guards detect out-of-bounds stores.
    assert torch.isfinite(grad_logits).all()
    for guard in (storage[:_BLOCK_V], storage[-_BLOCK_V:]):
        assert torch.equal(guard, torch.full_like(guard, -123.0))


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_row_offsets_across_int32_boundary(dtype):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    import triton
    import triton.language as tl

    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _BLOCK_V,
        _softcapped_selected_logprob_bwd_kernel,
        _softcapped_selected_logprob_fwd_kernel,
    )

    @triton.jit
    def run_last_rows(
        x,
        ids,
        upstream,
        output,
        row_max,
        log_sum_exp_shifted,
        vocab: tl.constexpr,
        START_ROW: tl.constexpr,
        BACKWARD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        if tl.program_id(0) >= START_ROW:
            base = tl.full((), START_ROW, tl.int64) * vocab
            if BACKWARD:
                _softcapped_selected_logprob_bwd_kernel(
                    x - base,
                    ids - START_ROW,
                    upstream - START_ROW,
                    row_max - START_ROW,
                    log_sum_exp_shifted - START_ROW,
                    output - base,
                    vocab,
                    BLOCK,
                    SOFTCAP=30.0,
                )
            else:
                _softcapped_selected_logprob_fwd_kernel(
                    x - base,
                    ids - START_ROW,
                    output - START_ROW,
                    row_max - START_ROW,
                    log_sum_exp_shifted - START_ROW,
                    vocab,
                    BLOCK,
                    SOFTCAP=30.0,
                )

    vocab = 65537
    start_row = 2**31 // vocab - 1
    generator = torch.Generator().manual_seed(423)
    x = (torch.randn((3, vocab), generator=generator) * 15).to(device="cuda", dtype=dtype)
    ids = torch.tensor([0, 1024, vocab - 1], device="cuda")
    upstream = torch.tensor([0.25, -2.0, 1.5], device="cuda")
    row_max = torch.full((4,), -123.0, device="cuda", dtype=torch.float32)
    log_sum_exp_shifted = torch.full((4,), -123.0, device="cuda", dtype=torch.float32)
    for backward in (False, True):
        shape = (4, vocab) if backward else (4,)
        output = torch.full(
            shape, -123.0, device="cuda", dtype=dtype if backward else torch.float32
        )
        grid = (start_row + 3, triton.cdiv(vocab, _BLOCK_V) if backward else 1)
        run_last_rows[grid](
            x,
            ids,
            upstream,
            output,
            row_max,
            log_sum_exp_shifted,
            vocab,
            START_ROW=start_row,
            BACKWARD=backward,
            BLOCK=_BLOCK_V,
            num_warps=4,
            enable_fp_fusion=False,
        )
        expected = (
            _fp32_backward_reference(x, ids, upstream) if backward else _fp32_reference(x, ids)
        )
        _assert_fp32_accuracy(
            output[:3],
            expected,
            "gradient_accuracy" if backward else "forward_accuracy",
        )
        assert torch.equal(output[3:], torch.full_like(output[3:], -123.0))
        for statistic in (row_max, log_sum_exp_shifted):
            assert torch.equal(statistic[3:], torch.full_like(statistic[3:], -123.0))
