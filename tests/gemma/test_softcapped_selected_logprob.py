# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Forward/backward reference accuracy and direct CUDA/ROCm kernel checks."""

import argparse
import math
from functools import partial

import pytest
import torch

from rl_engine.kernels.gtest.op_checks import run_operator_suite
from rl_engine.kernels.gtest.operator_specs import make_candidate, make_operator_case
from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance
from rl_engine.kernels.ops.pytorch.loss.softcapped_selected_logprob import (
    NativeSoftcappedSelectedLogprobOp,
    softcapped_selected_logprob,
)
from rl_engine.kernels.registry import KernelRegistry, OpBackend

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_CONTRACT = load_contract()


def _assert_bitwise(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_bitwise_comparison_distinguishes_signed_zero(dtype):
    positive = torch.tensor([0.0], dtype=dtype)
    negative = torch.tensor([-0.0], dtype=dtype)
    assert torch.equal(positive, negative)
    _assert_bitwise(positive, positive.clone())
    with pytest.raises(AssertionError):
        _assert_bitwise(positive, negative)


def _assert_contract(actual, expected, judgment):
    tolerance = resolve_tolerance(
        _CONTRACT, judgment=judgment, op_class="logprob", dtype=actual.dtype
    )
    torch.testing.assert_close(
        actual.to(torch.float64),
        expected.to(device=actual.device, dtype=torch.float64),
        atol=tolerance.atol,
        rtol=tolerance.rtol,
    )


def _native_backward(logits, token_ids, upstream):
    leaf = logits.detach().requires_grad_(True)
    output = NativeSoftcappedSelectedLogprobOp()(leaf, token_ids)
    return torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]


def _triton_forward(logits, token_ids, *, forward_impl="row"):
    from rl_engine.kernels.ops.triton.loss.softcapped_selected_logprob import (
        _launch_softcapped_selected_logprob_fwd,
        _launch_softcapped_selected_logprob_fwd_parallel,
    )

    launch = (
        _launch_softcapped_selected_logprob_fwd_parallel
        if forward_impl == "parallel"
        else _launch_softcapped_selected_logprob_fwd
    )
    output, _ = launch(logits, token_ids)
    return output


def _triton_backward(logits, token_ids, upstream, *, forward_impl="row"):
    from rl_engine.kernels.ops.triton.loss.softcapped_selected_logprob import (
        _launch_softcapped_selected_logprob_bwd,
        _launch_softcapped_selected_logprob_fwd,
        _launch_softcapped_selected_logprob_fwd_parallel,
    )

    launch = (
        _launch_softcapped_selected_logprob_fwd_parallel
        if forward_impl == "parallel"
        else _launch_softcapped_selected_logprob_fwd
    )
    _, log_sum_exp = launch(logits, token_ids)
    return _launch_softcapped_selected_logprob_bwd(logits, token_ids, upstream, log_sum_exp)


@pytest.fixture(params=("pytorch", "triton", "triton_parallel"))
def forward_impl(request):
    if request.param == "pytorch":
        return softcapped_selected_logprob, "cpu"
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    impl = "parallel" if request.param == "triton_parallel" else "row"
    return partial(_triton_forward, forward_impl=impl), "cuda"


def _fp64_reference(logits, token_ids):
    # Use stable FP64 log_softmax on the already-quantized input, rather than
    # repeating the candidate's explicit exp -> sum -> log implementation.
    softcapped = 30.0 * torch.tanh(logits.detach().cpu().to(torch.float64) / 30.0)
    log_probs = torch.log_softmax(softcapped, dim=-1)
    return log_probs.gather(-1, token_ids.cpu()[:, None]).squeeze(-1)


def _assert_forward(op, logits, token_ids):
    before = logits.clone()
    token_ids_before = token_ids.clone()
    actual = op(logits, token_ids)
    expected = _fp64_reference(logits, token_ids)
    assert actual.shape == logits.shape[:1]
    assert actual.dtype == torch.float32
    assert actual.device == logits.device
    _assert_contract(actual, expected, "forward_accuracy")
    assert torch.equal(logits, before)
    assert torch.equal(token_ids, token_ids_before)
    return actual


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "shape", [(0, 7), (3, 1), (3, 7), (2, 1023), (2, 1024), (3, 1025), (2, 262144)]
)
def test_forward_against_fp64(forward_impl, dtype, shape):
    op, device = forward_impl
    generator = torch.Generator().manual_seed(415)
    logits = (torch.randn(shape, generator=generator) * 30.0).to(device=device, dtype=dtype)
    # Exercise the first, last, and middle vocabulary positions, including
    # a selected token in the partial tile when V = 1025.
    token_ids = torch.tensor([0, shape[1] - 1, shape[1] // 2][: shape[0]], device=device)
    token_ids = token_ids.to(torch.int64)
    _assert_forward(op, logits, token_ids)


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
    _assert_contract(actual[:3], expected_uniform, "forward_accuracy")
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("dtype", _DTYPES)
def test_noncontiguous_inputs(forward_impl, dtype):
    op, device = forward_impl
    generator = torch.Generator().manual_seed(416)
    logits = (torch.randn((1025, 3), generator=generator) * 30).to(device=device, dtype=dtype).t()
    token_ids = torch.tensor([0, 0, 512, 0, 1024, 0], device=device)[::2]
    assert not logits.is_contiguous() and not token_ids.is_contiguous()
    _assert_forward(op, logits, token_ids)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (1025, 262144))
@pytest.mark.parametrize("forward_impl", ("row", "parallel"))
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


@pytest.fixture(params=("pytorch", "triton", "triton_parallel"))
def backward_impl(request):
    if request.param == "pytorch":
        return _native_backward, "cpu"
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    impl = "parallel" if request.param == "triton_parallel" else "row"
    return partial(_triton_backward, forward_impl=impl), "cuda"


def _assert_backward(op, logits, token_ids, grad_selected_logprob):
    before = logits.clone()
    token_ids_before = token_ids.clone()
    grad_before = grad_selected_logprob.clone()
    actual = op(logits, token_ids, grad_selected_logprob)

    # Independent FP64 autograd reference through stable log_softmax; do not
    # repeat the manually implemented (selected - p) derivative formula.
    leaf = logits.detach().cpu().to(torch.float64).requires_grad_(True)
    softcapped = 30.0 * torch.tanh(leaf / 30.0)
    log_probs = torch.log_softmax(softcapped, dim=-1)
    selected_logprob = log_probs.gather(-1, token_ids.cpu()[:, None]).squeeze(-1)
    (expected,) = torch.autograd.grad(
        selected_logprob, leaf, grad_outputs=grad_selected_logprob.cpu().to(torch.float64)
    )
    assert actual.shape == logits.shape
    assert actual.dtype == logits.dtype
    assert actual.device == logits.device
    _assert_contract(actual, expected, "gradient_accuracy")
    assert torch.equal(logits, before)
    assert torch.equal(token_ids, token_ids_before)
    assert torch.equal(grad_selected_logprob, grad_before)
    return actual


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "shape", [(0, 7), (3, 1), (3, 7), (2, 1023), (2, 1024), (3, 1025), (2, 262144)]
)
def test_backward_against_fp64_autograd(backward_impl, dtype, shape):
    op, device = backward_impl
    generator = torch.Generator().manual_seed(418)
    logits = (torch.randn(shape, generator=generator) * 15).to(device=device, dtype=dtype)
    token_ids = torch.tensor([0, shape[1] - 1, shape[1] // 2][: shape[0]], device=device)
    token_ids = token_ids.to(torch.int64)
    grad_selected_logprob = torch.randn(shape[0], generator=generator).to(device)
    _assert_backward(op, logits, token_ids, grad_selected_logprob)


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


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (1025, 262144))
@pytest.mark.parametrize("forward_impl", ("row", "parallel"))
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


@pytest.fixture(params=("pytorch", "triton", "triton_parallel", "triton_auto"))
def operator_impl(request):
    if request.param == "pytorch":
        return NativeSoftcappedSelectedLogprobOp(), "cpu"
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.kernels.ops.triton.loss.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    impl = {
        "triton": SoftcappedLogprobStrategy.ROW,
        "triton_parallel": SoftcappedLogprobStrategy.PARALLEL,
        "triton_auto": None,
    }[request.param]
    return TritonSoftcappedSelectedLogprobOp(forward_impl=impl), "cuda"


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("layout", ("contiguous", "transpose", "empty"))
def test_operator_autograd_and_inference(operator_impl, dtype, layout):
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
    _assert_contract(output, _fp64_reference(logits, token_ids), "forward_accuracy")
    (grad_base,) = torch.autograd.grad(output, base, grad_outputs=upstream)

    # Start FP64 at the operator boundary, using the already-rounded logits.
    reference_logits = logits.detach().cpu().to(torch.float64).requires_grad_(True)
    log_probs = torch.log_softmax(30.0 * torch.tanh(reference_logits / 30.0), dim=-1)
    selected = log_probs.gather(-1, token_ids.cpu()[:, None]).squeeze(-1)
    (expected_grad,) = torch.autograd.grad(
        selected, reference_logits, grad_outputs=upstream.cpu().to(torch.float64)
    )
    if layout == "transpose":
        expected_grad = expected_grad.t()
    assert grad_base.shape == base.shape and grad_base.dtype == dtype
    _assert_contract(grad_base, expected_grad, "gradient_accuracy")

    # The Op must use the same forward arithmetic with or without graph recording.
    with torch.no_grad():
        no_grad_output = op(logits, token_ids)
    with torch.inference_mode():
        inference_output = op(logits, token_ids)
    detached_output = op(logits.detach(), token_ids)
    for actual in (no_grad_output, inference_output, detached_output):
        assert not actual.requires_grad
        _assert_bitwise(actual, output.detach())


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", ((0, 7), (3, 1025), (2, 262144)))
@pytest.mark.parametrize("forward_impl", ("row", "parallel"))
def test_triton_saved_log_sum_exp_and_repeated_backward(dtype, shape, forward_impl):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.kernels.ops.triton.loss import (
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
    _, _, log_sum_exp = output.grad_fn.saved_tensors
    assert log_sum_exp.shape == logits.shape[:1]
    assert log_sum_exp.dtype == torch.float32 and log_sum_exp.device == logits.device
    assert log_sum_exp.numel() * log_sum_exp.element_size() == 4 * shape[0]
    expected_lse = torch.logsumexp(
        30.0 * torch.tanh(logits.detach().cpu().to(torch.float64) / 30.0), dim=-1
    )
    _assert_contract(log_sum_exp, expected_lse, "forward_accuracy")
    saved_before = log_sum_exp.clone()

    def backward_from_saved_graph(x, token_ids, upstream):
        return torch.autograd.grad(output, x, grad_outputs=upstream, retain_graph=True)[0]

    # Different upstream gradients must reuse, rather than overwrite, the same
    # forward statistics. Compare both backward calls with independent FP64 autograd.
    for _ in range(2):
        upstream = torch.randn(shape[0], generator=generator).to(device="cuda")
        _assert_backward(backward_from_saved_graph, logits, ids, upstream)
        _assert_bitwise(log_sum_exp, saved_before)

    with torch.inference_mode():
        _assert_bitwise(op(logits, ids), output.detach())


@pytest.mark.parametrize("platform", ("cpu", "musa", "npu"))
def test_registry_native_dispatch(platform, monkeypatch):
    from rl_engine.kernels.ops.pytorch.loss import NativeSoftcappedSelectedLogprobOp

    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    op = registry.get_op("softcapped_selected_logprob")
    assert isinstance(op, NativeSoftcappedSelectedLogprobOp)
    _assert_forward(op, torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([1]))


@pytest.mark.parametrize("platform", ("cuda", "rocm"))
def test_registry_fallback_when_triton_cannot_load(platform, monkeypatch):
    registry = KernelRegistry()
    monkeypatch.setattr(registry, "_platform", lambda: platform)
    load_backend = registry._load_backend
    attempted = []

    def without_triton(backend):
        attempted.append(backend)
        if backend == OpBackend.TRITON_SOFTCAPPED_SELECTED_LOGPROB:
            return None
        return load_backend(backend)

    monkeypatch.setattr(registry, "_load_backend", without_triton)
    assert isinstance(
        registry.get_op("softcapped_selected_logprob"), NativeSoftcappedSelectedLogprobOp
    )
    assert attempted == [
        OpBackend.TRITON_SOFTCAPPED_SELECTED_LOGPROB,
        OpBackend.PYTORCH_NATIVE_SOFTCAPPED_SELECTED_LOGPROB,
    ]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("backend", ("pytorch", "triton"))
def test_shared_accuracy_harness(dtype, backend):
    if backend == "triton" and not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    device = torch.device("cuda" if backend == "triton" else "cpu")
    args = argparse.Namespace(
        op="softcapped_selected_logprob",
        candidate=backend,
        arch_key=None,
        batch=2,
        seq=3,
        vocab=1025,
        seed=415,
        input_mode="random",
    )
    case = make_operator_case(args, dtype, device)
    assert case.inputs["logits"].shape == (6, 1025)
    assert case.inputs["token_ids"].shape == (6,)
    report = run_operator_suite(
        args.op,
        candidates=[make_candidate(args)],
        cases=[case],
        check_grad=True,
    )
    assert report.passed


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("forward_impl", ("row", "parallel"))
def test_triton_padding_and_strided_layout_invariance(dtype, forward_impl):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.kernels.ops.triton.loss import (
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


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (1, 1023, 1024, 1025, 2048, 2049, 262144))
def test_triton_forward_variants_match_bitwise(dtype, vocab):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.kernels.ops.triton.loss.softcapped_selected_logprob import (
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
    expected = _launch_softcapped_selected_logprob_fwd(logits, ids)
    actual = _launch_softcapped_selected_logprob_fwd_parallel(logits, ids)

    for reference, result in zip(expected, actual, strict=True):
        _assert_bitwise(result, reference)
    grad_expected = _launch_softcapped_selected_logprob_bwd(logits, ids, upstream, expected[1])
    grad_actual = _launch_softcapped_selected_logprob_bwd(logits, ids, upstream, actual[1])
    _assert_bitwise(grad_actual, grad_expected)


def _evaluate_triton_with_statistics(op, logits, token_ids, upstream):
    leaf = logits.detach().requires_grad_(True)
    output = op(leaf, token_ids)
    _, _, log_sum_exp = output.grad_fn.saved_tensors
    gradient = torch.autograd.grad(output, leaf, grad_outputs=upstream)[0]
    return output.detach(), log_sum_exp, gradient


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape", [(0, 7), (4, 36864), (16, 49152), (64, 32768), (256, 65536)])
def test_triton_auto_matches_explicit_row_and_inference(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.kernels.ops.triton.loss import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
    )

    generator = torch.Generator().manual_seed(428)
    logits = (torch.randn(shape, generator=generator) * 30).to(device="cuda", dtype=dtype)
    ids = torch.arange(shape[0], device="cuda", dtype=torch.int64) * 1025 % shape[1]
    upstream = torch.randn(shape[0], generator=generator).to(device="cuda")
    op = TritonSoftcappedSelectedLogprobOp(forward_impl=None)
    reference = _evaluate_triton_with_statistics(
        TritonSoftcappedSelectedLogprobOp(forward_impl=SoftcappedLogprobStrategy.ROW),
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


@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_auto_batch_split_switches_strategy_without_changing_bits(dtype):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.kernels.ops.triton.loss.softcapped_selected_logprob import (
        SoftcappedLogprobStrategy,
        TritonSoftcappedSelectedLogprobOp,
        select_softcapped_logprob_strategy,
        softcapped_logprob_device_key,
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


def test_triton_explicit_strategy_bypasses_device_lookup_and_auto_validates_first(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    from rl_engine.kernels.ops.triton.loss import softcapped_selected_logprob as module

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


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (2048, 2049))
def test_triton_parallel_forward_scratch_and_output_bounds(dtype, vocab):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    import triton

    from rl_engine.kernels.ops.triton.loss.softcapped_selected_logprob import (
        _BLOCK_V,
        _launch_softcapped_selected_logprob_fwd,
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
    lse_storage, log_sum_exp = guarded((n_rows,))
    _softcapped_selected_logprob_fwd_partial_kernel[(n_rows, n_tiles)](
        logits, scratch, vocab, n_tiles, _BLOCK_V, num_warps=4, enable_fp_fusion=False
    )
    _softcapped_selected_logprob_fwd_merge_kernel[(n_rows,)](
        logits,
        ids,
        scratch,
        output,
        log_sum_exp,
        vocab,
        n_tiles,
        num_warps=4,
        enable_fp_fusion=False,
    )
    expected, expected_lse = _launch_softcapped_selected_logprob_fwd(logits, ids)
    assert torch.isfinite(scratch).all()
    _assert_bitwise(log_sum_exp, expected_lse)
    _assert_bitwise(output[1:3], expected[1:3])
    assert torch.isnan(output[[0, 3]]).all()
    for storage in (scratch_storage, output_storage, lse_storage):
        assert torch.equal(storage[[0, -1]], torch.full((2,), -123.0, device="cuda"))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("vocab", (2048, 2049))
def test_triton_backward_tile_coverage_and_output_bounds(dtype, vocab):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    import triton

    from rl_engine.kernels.ops.triton.loss.softcapped_selected_logprob import (
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
        _, log_sum_exp = _launch_softcapped_selected_logprob_fwd(x, token_ids)
        _softcapped_selected_logprob_bwd_kernel[(n_rows, triton.cdiv(vocab, _BLOCK_V))](
            x,
            token_ids,
            grad_output,
            log_sum_exp,
            grad_logits,
            vocab,
            BLOCK_V=_BLOCK_V,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return grad_logits

    _assert_backward(backward_into_guarded_output, logits, ids, upstream)
    # NaN initialization detects missing tiles; guards detect out-of-bounds stores.
    assert torch.isfinite(grad_logits).all()
    for guard in (storage[:_BLOCK_V], storage[-_BLOCK_V:]):
        assert torch.equal(guard, torch.full_like(guard, -123.0))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_triton_row_offsets_across_int32_boundary(dtype):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA or ROCm GPU is required")
    import triton
    import triton.language as tl

    from rl_engine.kernels.ops.triton.loss.softcapped_selected_logprob import (
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
        log_sum_exp,
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
                    log_sum_exp - START_ROW,
                    output - base,
                    vocab,
                    BLOCK,
                )
            else:
                _softcapped_selected_logprob_fwd_kernel(
                    x - base,
                    ids - START_ROW,
                    output - START_ROW,
                    log_sum_exp - START_ROW,
                    vocab,
                    BLOCK,
                )

    vocab = 65537
    start_row = 2**31 // vocab - 1
    generator = torch.Generator().manual_seed(423)
    x = (torch.randn((3, vocab), generator=generator) * 15).to(device="cuda", dtype=dtype)
    ids = torch.tensor([0, 1024, vocab - 1], device="cuda")
    upstream = torch.tensor([0.25, -2.0, 1.5], device="cuda")
    log_sum_exp = torch.full((4,), -123.0, device="cuda", dtype=torch.float32)
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
            log_sum_exp,
            vocab,
            START_ROW=start_row,
            BACKWARD=backward,
            BLOCK=_BLOCK_V,
            num_warps=4,
            enable_fp_fusion=False,
        )
        expected = (
            _native_backward(x.cpu(), ids.cpu(), upstream.cpu())
            if backward
            else _fp64_reference(x, ids)
        )
        _assert_contract(
            output[:3], expected, "gradient_accuracy" if backward else "forward_accuracy"
        )
        assert torch.equal(output[3:], torch.full_like(output[3:], -123.0))
        assert torch.equal(log_sum_exp[3:], torch.full_like(log_sum_exp[3:], -123.0))
