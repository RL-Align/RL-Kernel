# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared test inputs, independent FP32 checks and raw-bit comparisons.

These helpers belong to this operator's tests. They do not reuse another
operator's tests or validators. The ordered arithmetic reference is separate
from Native FP32, which supplies the numerical-accuracy baseline.
"""

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.reference.logprob.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
)
from tests.helpers.tensor_checks import assert_bitwise as assert_bitwise  # noqa: F401

DTYPES = (torch.float16, torch.bfloat16, torch.float32)

SOFTCAPS = (
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

_CONTRACT = load_contract()

INDEPENDENT_CHECKS = (
    pytest.param("accuracy", marks=pytest.mark.softcap_accuracy),
    pytest.param("consistency", marks=pytest.mark.softcap_consistency),
)

"""Independent collected cases: an accuracy failure cannot suppress consistency."""


def assert_fp32_accuracy(actual, expected, judgment):
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


def native_backward(logits, token_ids, upstream, *, softcap=30.0):
    """Differentiate Native on a detached leaf, preserving the input dtype."""
    leaf = logits.detach().requires_grad_(True)
    output = NativeSoftcappedSelectedLogprobOp()(leaf, token_ids, softcap=softcap)
    return torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]


def triton_forward(logits, token_ids, *, forward_impl="row", softcap=30.0):
    """Exercise a forward launcher directly, without the public autograd wrapper."""
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _FORWARD_LAUNCHERS,
        SoftcappedLogprobStrategy,
    )

    launch = _FORWARD_LAUNCHERS[SoftcappedLogprobStrategy(forward_impl)]
    return launch(logits, token_ids, softcap).selected_logprob


def triton_backward(logits, token_ids, upstream, *, forward_impl="row", softcap=30.0):
    """Pass matching forward statistics to the direct backward launcher."""
    from rl_engine.backends.shared.triton.logprob.softcapped_selected_logprob import (
        _FORWARD_LAUNCHERS,
        SoftcappedLogprobStrategy,
        _launch_softcapped_selected_logprob_bwd,
    )

    launch = _FORWARD_LAUNCHERS[SoftcappedLogprobStrategy(forward_impl)]
    result = launch(logits, token_ids, softcap)
    return _launch_softcapped_selected_logprob_bwd(
        logits, token_ids, upstream, result.row_max, result.log_sum_exp_shifted, softcap
    )


def fp32_reference(logits, token_ids, softcap=30.0):
    """Evaluate Native in FP32 on the same device and already-quantized inputs."""
    with torch.no_grad(), torch.autocast(device_type=logits.device.type, enabled=False):
        return NativeSoftcappedSelectedLogprobOp()(
            logits.detach().float(), token_ids, softcap=softcap
        )


def fp32_backward_reference(logits, token_ids, upstream, *, softcap=30.0):
    """Keep the reference gradient in FP32, including for FP16/BF16 candidates."""
    with (
        torch.enable_grad(),
        torch.autocast(device_type=logits.device.type, enabled=False),
    ):
        return native_backward(logits.float(), token_ids, upstream.float(), softcap=softcap)


def evaluate_triton_with_statistics(op, logits, token_ids, upstream):
    """Return public output, saved maximum/log-sum and autograd gradient for comparison."""
    leaf = logits.detach().requires_grad_(True)
    output = op(leaf, token_ids)
    _, _, row_max, log_sum_exp_shifted = output.grad_fn.saved_tensors
    gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]
    return output.detach(), row_max, log_sum_exp_shifted, gradient


def assert_backward(op, logits, token_ids, grad_selected_logprob, *, softcap=30.0, accuracy=True):
    """Check gradient metadata/purity, optionally comparing with Native autograd."""
    before = logits.clone()
    token_ids_before = token_ids.clone()
    grad_before = grad_selected_logprob.clone()
    actual = op(logits, token_ids, grad_selected_logprob)

    assert actual.shape == logits.shape
    assert actual.dtype == logits.dtype
    assert actual.device == logits.device
    if accuracy:
        expected = fp32_backward_reference(
            logits, token_ids, grad_selected_logprob, softcap=softcap
        )
        assert_fp32_accuracy(actual, expected, "gradient_accuracy")
    assert torch.equal(logits, before)
    assert torch.equal(token_ids, token_ids_before)
    assert torch.equal(grad_selected_logprob, grad_before)
    return actual
