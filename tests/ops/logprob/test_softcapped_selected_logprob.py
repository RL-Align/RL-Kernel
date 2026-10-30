# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Operator interface, autograd and independent FP32 accuracy.

Accuracy and consistency parameters are separately collected pytest items;
one failed layer cannot prevent the other from running."""

import math
from functools import partial

import pytest
import torch

from rl_engine.reference.logprob.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
)
from tests.helpers.softcapped_checks import (
    DTYPES,
    INDEPENDENT_CHECKS,
    SOFTCAPS,
    assert_backward,
    assert_bitwise,
    assert_fp32_accuracy,
    evaluate_triton_with_statistics,
    fp32_backward_reference,
    fp32_reference,
    native_backward,
    triton_backward,
    triton_forward,
)

_VALUE_CHECKS = (
    pytest.param("output", marks=pytest.mark.softcap_accuracy),
    pytest.param("gradient", marks=pytest.mark.softcap_accuracy),
    pytest.param("invariance", marks=pytest.mark.softcap_consistency),
)


@pytest.fixture(params=("pytorch", "triton", "triton_parallel", "triton_row_pipelined"))
def forward_impl(request):
    if request.param == "pytorch":
        return NativeSoftcappedSelectedLogprobOp(), "cpu"
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    impl = request.param.removeprefix("triton_") if request.param != "triton" else "row"
    return partial(triton_forward, forward_impl=impl), "cuda"


def _assert_forward(op, logits, token_ids, *, softcap=30.0, accuracy=True):
    """Check the interface; only accuracy items consult independent Native FP32."""
    before = logits.clone()
    token_ids_before = token_ids.clone()
    actual = op(logits, token_ids, softcap=softcap)
    assert actual.shape == logits.shape[:1]
    assert actual.dtype == torch.float32
    assert actual.device == logits.device
    if accuracy:
        expected = fp32_reference(logits, token_ids, softcap)
        assert_fp32_accuracy(actual, expected, "forward_accuracy")
    assert torch.equal(logits, before)
    assert torch.equal(token_ids, token_ids_before)
    return actual


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", DTYPES)
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
@pytest.mark.parametrize("dtype", DTYPES)
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
    assert_fp32_accuracy(actual[:3], expected_uniform, "forward_accuracy")
    assert torch.isfinite(actual).all()


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", DTYPES)
def test_noncontiguous_inputs(forward_impl, dtype):
    op, device = forward_impl
    generator = torch.Generator().manual_seed(416)
    logits = (torch.randn((1025, 3), generator=generator) * 30).to(device=device, dtype=dtype).t()
    token_ids = torch.tensor([0, 0, 512, 0, 1024, 0], device=device)[::2]
    assert not logits.is_contiguous() and not token_ids.is_contiguous()
    _assert_forward(op, logits, token_ids)


@pytest.fixture(params=("pytorch", "triton", "triton_parallel", "triton_row_pipelined"))
def backward_impl(request):
    if request.param == "pytorch":
        return native_backward, "cpu"
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    impl = request.param.removeprefix("triton_") if request.param != "triton" else "row"
    return partial(triton_backward, forward_impl=impl), "cuda"


@pytest.mark.parametrize("check", INDEPENDENT_CHECKS)
@pytest.mark.parametrize("softcap", SOFTCAPS)
@pytest.mark.parametrize("dtype", DTYPES)
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
                assert_bitwise(op(logits, ids), output.detach())
                assert_bitwise(default_op.forward(logits, ids, softcap=softcap), output.detach())

    def backward_from_saved_graph(x, token_ids, gradient):
        return torch.autograd.grad(output, x, grad_outputs=gradient, retain_graph=True)[0]

    expected_gradient = assert_backward(
        backward_from_saved_graph,
        logits,
        ids,
        upstream,
        softcap=softcap,
        accuracy=check == "accuracy",
    )
    assert_backward(
        backward_from_saved_graph,
        logits,
        ids,
        torch.zeros_like(upstream),
        softcap=softcap,
        accuracy=check == "accuracy",
    )
    if check == "consistency":
        repeated_gradient = backward_from_saved_graph(logits, ids, upstream)
        assert_bitwise(repeated_gradient, expected_gradient)

    # Only Triton promises batch invariance; native reductions may change bits.
    if check == "consistency" and device == "cuda" and shape[0] > 0:
        for row in range(shape[0]):
            single = logits[row : row + 1].detach().requires_grad_(True)
            single_output = op(single, ids[row : row + 1])
            single_gradient = torch.autograd.grad(
                single_output, single, grad_outputs=upstream[row : row + 1]
            )[0]
            assert_bitwise(single_output.detach(), output[row : row + 1].detach())
            assert_bitwise(single_gradient, expected_gradient[row : row + 1])


@pytest.mark.parametrize("check", INDEPENDENT_CHECKS)
@pytest.mark.parametrize("boundary", ("minimum", "near_maximum"))
@pytest.mark.parametrize("vocab", (1, 1025, 202048, 262144))
@pytest.mark.parametrize("dtype", DTYPES)
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
                assert_bitwise(op(logits, ids), output.detach())

    def backward_from_saved_graph(x, token_ids, gradient):
        return torch.autograd.grad(output, x, grad_outputs=gradient, retain_graph=True)[0]

    gradient = assert_backward(
        backward_from_saved_graph,
        logits,
        ids,
        upstream,
        softcap=softcap,
        accuracy=check == "accuracy",
    )
    if check == "consistency":
        assert_bitwise(backward_from_saved_graph(logits, ids, upstream), gradient)
    with pytest.raises(ValueError, match="softcap"):
        default_op(logits, ids, softcap=invalid)


@pytest.mark.softcap_accuracy
def test_same_operator_keeps_softcap_per_graph(operator_impl):
    """A later call with another cap must not change an outstanding graph."""
    op, device = operator_impl
    generator = torch.Generator().manual_seed(439)
    logits = (torch.randn((3, 1025), generator=generator) * 30.0).to(device).requires_grad_(True)
    ids = torch.tensor([0, 512, 1024], device=device)
    upstream = torch.tensor([0.25, -2.0, 1.5], device=device)
    caps = (1e-3, 0.125, 17.3, 30.0, 50.0, 70.0)
    outputs = [_assert_forward(partial(op, softcap=cap), logits, ids, softcap=cap) for cap in caps]
    _assert_forward(op, logits, ids, softcap=30.0)

    def backward_from_saved_graph(output, x, token_ids, gradient):
        return torch.autograd.grad(output, x, grad_outputs=gradient, retain_graph=True)[0]

    # Run backward out of forward order, then repeat it on the first graph.
    for index in (*reversed(range(len(caps))), 0):
        assert_backward(
            partial(backward_from_saved_graph, outputs[index]),
            logits,
            ids,
            upstream,
            softcap=caps[index],
        )


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("softcap", SOFTCAPS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_configurable_softcap_saturation(forward_impl, softcap, dtype):
    forward, device = forward_impl
    op = partial(forward, softcap=softcap)
    logits = torch.full((3, 1025), 3000.0, device=device, dtype=dtype)
    logits[1].fill_(-3000.0)
    logits[2, 1024] = -3000.0
    ids = torch.tensor([0, 1024, 1024], device=device)
    output = _assert_forward(op, logits, ids, softcap=softcap)
    assert torch.isfinite(output).all()
    assert_fp32_accuracy(output[:2], torch.full((2,), -math.log(1025)), "forward_accuracy")


@pytest.mark.parametrize("check", INDEPENDENT_CHECKS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="A CUDA or ROCm GPU is required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("softcap", SOFTCAPS)
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
        op,
        logits.requires_grad_(True),
        ids,
        softcap=softcap,
        accuracy=check == "accuracy",
    )
    if check == "consistency":
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                assert_bitwise(op(logits, ids), training.detach())
    assert_backward(
        partial(native_backward, softcap=softcap),
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

    assert_backward(backward_from_saved_graph, logits, ids, upstream, softcap=cap)


@pytest.mark.parametrize("check", INDEPENDENT_CHECKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab", (2, 1025))
def test_subnormal_probability_survives_large_finite_upstream(operator_impl, dtype, vocab, check):
    """A tiny probability can produce a normal, measurable input gradient.

    At cap 70 the nonselected score is about -19.47 and the maximum is 70.
    Its probability is about 1.39e-39, below FP32's smallest normal value.
    An upstream of 1e36 yields a gradient near 1e-3. Flushing that probability
    to zero loses this contribution before the final FP16/BF16/FP32 cast.
    Neither the input nor the expected gradient comes from a Triton helper.
    """
    op, device = operator_impl
    logits = torch.full((2, vocab), -3000.0, device=device, dtype=dtype)
    logits[:, 0] = 3000.0
    logits[:, -1] = -20.0
    logits.requires_grad_(True)
    ids = torch.zeros(2, device=device, dtype=torch.int64)
    upstream = torch.tensor([1e36, -1e36], device=device, dtype=torch.float32)
    output = op(logits, ids, softcap=70.0)
    gradient = torch.autograd.grad(output, logits, upstream, retain_graph=True)[0]
    if check == "accuracy":
        expected = fp32_backward_reference(logits, ids, upstream, softcap=70.0)
        assert torch.isfinite(gradient).all()
        # This also catches FTZ for lower-precision outputs whose WS1 tolerance
        # would otherwise allow a zero in place of the approximately 1e-3 value.
        assert (expected[:, -1].abs() > 1e-4).all()
        assert (gradient[:, -1].abs() > 1e-4).all()
        assert torch.equal(gradient[:, -1].sign(), -upstream.sign().to(dtype))
        assert_fp32_accuracy(gradient, expected, "gradient_accuracy")
    else:
        repeated = torch.autograd.grad(output, logits, upstream)[0]
        assert_bitwise(gradient, repeated)
        with torch.inference_mode():
            inference_output = op(logits, ids, softcap=70.0)
        assert_bitwise(output.detach(), inference_output)


@pytest.mark.parametrize("check", INDEPENDENT_CHECKS)
@pytest.mark.parametrize("dtype", DTYPES)
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
            assert_fp32_accuracy(output, fp32_reference(logits, ids, cap), "forward_accuracy")
        else:
            for context in (torch.no_grad(), torch.inference_mode()):
                with context:
                    assert_bitwise(op(logits, ids, softcap=cap), output.detach())
    else:
        gradient = torch.autograd.grad(output, logits, grad_outputs=upstream, retain_graph=True)[0]
        if check == "accuracy":
            assert_fp32_accuracy(
                gradient,
                fp32_backward_reference(logits, ids, upstream, softcap=cap),
                "gradient_accuracy",
            )
        else:
            repeated = torch.autograd.grad(output, logits, grad_outputs=upstream)[0]
            assert_bitwise(repeated, gradient)


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
        assert_fp32_accuracy(output, fp32_reference(logits, ids, softcap), "forward_accuracy")
    elif value == "gradient":
        logits.requires_grad_(True)
        output = op(logits, ids, softcap=softcap)
        gradient = torch.autograd.grad(output, logits, grad_outputs=upstream)[0]
        expected = fp32_backward_reference(logits, ids, upstream, softcap=softcap)
        assert_fp32_accuracy(gradient, expected, "gradient_accuracy")
    else:
        training = op(logits.requires_grad_(True), ids, softcap=softcap)
        for context in (torch.no_grad(), torch.inference_mode()):
            with context:
                assert_bitwise(op(logits, ids, softcap=softcap), training.detach())


@pytest.mark.parametrize("entry_point", ("call", "forward"))
def test_operator_requires_explicit_softcap(operator_impl, entry_point):
    """Omitting the model's cap fails at the interface instead of assuming 30."""
    op, device = operator_impl
    logits = torch.zeros((1, 7), device=device)
    ids = torch.zeros(1, device=device, dtype=torch.int64)
    forward = op if entry_point == "call" else op.forward
    with pytest.raises(TypeError, match="softcap"):
        forward(logits, ids)


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
        op(logits, ids, softcap=30.0)


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
@pytest.mark.parametrize("softcap", SOFTCAPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value", _VALUE_CHECKS)
def test_configurable_softcap_auto_switch(softcap, dtype, value, monkeypatch, request):
    """Check an automatic strategy switch without mixing accuracy and bitwise failures.

    Output accuracy, gradient accuracy and batch-split invariance are separate
    cases. A reference accuracy failure cannot hide the gradient or bitwise
    checks. Invariance compares output, both saved statistics and all gradients.
    """
    from rl_engine.backends.shared.triton.logprob import softcapped_selected_logprob as config
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
        select_softcapped_logprob_strategy,
        softcapped_logprob_device_key,
    )

    select_softcapped_logprob_strategy.cache_clear()
    request.addfinalizer(select_softcapped_logprob_strategy.cache_clear)
    monkeypatch.setattr(
        config,
        "FORWARD_STRATEGY_CONFIGS",
        {
            config.ForwardConfigKey(
                device=config._DEFAULT_DEVICE,
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
    whole = evaluate_triton_with_statistics(op, logits, ids, upstream)
    output, _, _, gradient = whole
    if value == "output":
        assert_fp32_accuracy(output, fp32_reference(logits, ids, softcap), "forward_accuracy")
    elif value == "gradient":
        assert_fp32_accuracy(
            gradient,
            fp32_backward_reference(logits, ids, upstream, softcap=softcap),
            "gradient_accuracy",
        )
    else:
        parts = [
            evaluate_triton_with_statistics(
                op, logits[start:stop], ids[start:stop], upstream[start:stop]
            )
            for start, stop in ((0, 16), (16, 17))
        ]
        # Compare output, row maximum, shifted log-sum and every gradient value.
        for index in range(len(whole)):
            assert_bitwise(torch.cat([part[index] for part in parts]), whole[index])


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", DTYPES)
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
    assert_backward(op, logits, token_ids, grad_selected_logprob)


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_saturation_and_zero_upstream(backward_impl, dtype):
    op, device = backward_impl
    logits = torch.zeros((4, 1025), device=device, dtype=dtype)
    logits[0].fill_(3000.0)
    logits[1].fill_(-3000.0)
    logits[2, 1024] = 15.0
    token_ids = torch.tensor([0, 1024, 1024, 512], device=device)
    grad_selected_logprob = torch.tensor([2.0, -3.0, 0.0, -0.75], device=device)
    actual = assert_backward(op, logits, token_ids, grad_selected_logprob)
    # Saturated softcap and a zero upstream gradient must both block gradients.
    assert torch.equal(actual[:3], torch.zeros_like(actual[:3]))
    # A negative upstream gradient reverses the signs on the uniform row.
    assert actual[3, 512] < 0 and actual[3, 0] > 0


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("dtype", DTYPES)
def test_backward_noncontiguous_inputs(backward_impl, dtype):
    op, device = backward_impl
    generator = torch.Generator().manual_seed(419)
    logits = (torch.randn((1025, 3), generator=generator) * 15).to(device=device, dtype=dtype).t()
    token_ids = torch.tensor([0, 0, 512, 0, 1024, 0], device=device)[::2]
    grad_selected_logprob = torch.tensor([2.0, 0, -0.5, 0, 1.25, 0], device=device)[::2]
    assert not logits.is_contiguous() and not token_ids.is_contiguous()
    assert not grad_selected_logprob.is_contiguous()
    assert_backward(op, logits, token_ids, grad_selected_logprob)


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


@pytest.mark.parametrize("check", INDEPENDENT_CHECKS)
@pytest.mark.parametrize("dtype", DTYPES)
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

    output = op(logits, token_ids, softcap=30.0)
    assert output.requires_grad and output.grad_fn is not None
    assert output.shape == (n_rows,) and output.dtype == torch.float32
    if check == "accuracy":
        assert_fp32_accuracy(output, fp32_reference(logits, token_ids), "forward_accuracy")
    (grad_base,) = torch.autograd.grad(output, base, grad_outputs=upstream)

    assert grad_base.shape == base.shape and grad_base.dtype == dtype
    if check == "accuracy":
        # Start at the operator boundary, retaining quantized inputs and FP32 gradients.
        expected_grad = fp32_backward_reference(logits, token_ids, upstream)
        if layout == "transpose":
            expected_grad = expected_grad.t()
        assert_fp32_accuracy(grad_base, expected_grad, "gradient_accuracy")
    else:
        # The Op must use the same arithmetic with or without graph recording.
        with torch.no_grad():
            no_grad_output = op(logits, token_ids, softcap=30.0)
        with torch.inference_mode():
            inference_output = op(logits, token_ids, softcap=30.0)
        detached_output = op(logits.detach(), token_ids, softcap=30.0)
        for actual in (no_grad_output, inference_output, detached_output):
            assert not actual.requires_grad
            assert_bitwise(actual, output.detach())


@pytest.mark.parametrize("check", INDEPENDENT_CHECKS)
@pytest.mark.parametrize("dtype", DTYPES)
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
    output = op(logits, ids, softcap=30.0)
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
        assert_fp32_accuracy(row_max, expected_max, "forward_accuracy")
        assert_fp32_accuracy(log_sum_exp_shifted, expected_log_sum, "forward_accuracy")
    saved_before = (row_max.clone(), log_sum_exp_shifted.clone())

    def backward_from_saved_graph(x, token_ids, upstream):
        return torch.autograd.grad(output, x, grad_outputs=upstream, retain_graph=True)[0]

    # Different upstream gradients must reuse, rather than overwrite, the same
    # forward statistics. Compare both backward calls with Native FP32 autograd.
    for _ in range(2):
        upstream = torch.randn(shape[0], generator=generator).to(device="cuda")
        assert_backward(
            backward_from_saved_graph,
            logits,
            ids,
            upstream,
            accuracy=check == "accuracy",
        )
        if check == "consistency":
            assert_bitwise(row_max, saved_before[0])
            assert_bitwise(log_sum_exp_shifted, saved_before[1])

    if check == "consistency":
        with torch.inference_mode():
            assert_bitwise(op(logits, ids, softcap=30.0), output.detach())


@pytest.mark.parametrize("check", INDEPENDENT_CHECKS)
@pytest.mark.parametrize("dtype", DTYPES)
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
        assert_bitwise(actual.selected_logprob, expected.selected_logprob)
        assert_bitwise(actual.row_max, expected.row_max)
        assert_bitwise(actual.log_sum_exp_shifted, expected.log_sum_exp_shifted)
    grad_expected = _launch_softcapped_selected_logprob_bwd(
        logits, ids, upstream, expected.row_max, expected.log_sum_exp_shifted, softcap=30.0
    )
    grad_actual = _launch_softcapped_selected_logprob_bwd(
        logits, ids, upstream, actual.row_max, actual.log_sum_exp_shifted, softcap=30.0
    )
    if check == "consistency":
        assert_bitwise(grad_actual, grad_expected)
    else:
        assert_fp32_accuracy(
            actual.selected_logprob, fp32_reference(logits, ids), "forward_accuracy"
        )
        assert_fp32_accuracy(
            grad_actual,
            fp32_backward_reference(logits, ids, upstream),
            "gradient_accuracy",
        )


@pytest.mark.parametrize("check", INDEPENDENT_CHECKS)
@pytest.mark.parametrize("dtype", DTYPES)
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
        expected_output = fp32_reference(logits, ids, softcap)
        expected_gradient = fp32_backward_reference(logits, ids, upstream, softcap=softcap)
        divisor = torch.tensor(softcap, device="cuda", dtype=torch.float32)
        expected_max = (softcap * torch.tanh(torch.div(logits.float(), divisor))).amax(dim=-1)
        expected_uniform = torch.full((3,), -math.log(vocab), device="cuda", dtype=torch.float32)
    reference = None
    for strategy in (*SoftcappedLogprobStrategy, None):
        op = partial(TritonSoftcappedSelectedLogprobOp(forward_impl=strategy), softcap=softcap)
        actual = evaluate_triton_with_statistics(op, logits, ids, upstream)
        output, row_max, log_sum, gradient = actual
        if check == "accuracy":
            assert_fp32_accuracy(output, expected_output, "forward_accuracy")
            assert_fp32_accuracy(gradient, expected_gradient, "gradient_accuracy")
            assert_fp32_accuracy(row_max, expected_max, "forward_accuracy")
            assert_fp32_accuracy(output[:3], expected_uniform, "forward_accuracy")
        assert row_max[0] < 0 and row_max[5] < 0
        assert row_max[2] == 0 and not torch.signbit(row_max[2])
        assert torch.all((log_sum >= 0) & (log_sum <= math.log(vocab) + 1e-5))
        if check == "consistency":
            if reference is None:
                reference = actual
            else:
                for value, expected in zip(actual, reference, strict=True):
                    assert_bitwise(value, expected)
            for context in (torch.no_grad(), torch.inference_mode()):
                with context:
                    assert_bitwise(op(logits, ids), output)


@pytest.mark.softcap_accuracy
def test_triton_explicit_row_and_parallel_bypass_device_lookup_and_auto_validates_first(
    monkeypatch,
):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.backends.shared.triton.logprob import softcapped_selected_logprob as module
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
    )

    def unexpected_lookup(device):
        pytest.fail("Explicit ROW/PARALLEL and invalid inputs must not query hardware")

    monkeypatch.setattr(module, "softcapped_logprob_device_key", unexpected_lookup)
    logits = torch.zeros((1, 1025), device="cuda")
    ids = torch.tensor([1024], device="cuda")
    outputs = [
        module.TritonSoftcappedSelectedLogprobOp(forward_impl=choice)(logits, ids, softcap=30.0)
        for choice in (SoftcappedLogprobStrategy.ROW, SoftcappedLogprobStrategy.PARALLEL)
    ]
    for output in outputs[1:]:
        assert_bitwise(output, outputs[0])
    with pytest.raises(ValueError, match="shape"):
        module.TritonSoftcappedSelectedLogprobOp()(logits[0], ids, softcap=30.0)
    with pytest.raises(TypeError, match="int64"):
        module.TritonSoftcappedSelectedLogprobOp()(logits, ids.float(), softcap=30.0)
    with pytest.raises(ValueError, match="softcap"):
        module.TritonSoftcappedSelectedLogprobOp()(logits, ids, softcap=0.0)


@pytest.mark.softcap_accuracy
@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("softcap", (*SOFTCAPS, 73.0))
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
