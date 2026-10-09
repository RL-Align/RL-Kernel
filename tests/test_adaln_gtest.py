# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU checks of AdaLN gtest wiring, independent gold and gradient sensitivity."""

import os
from argparse import Namespace
from dataclasses import replace

import pytest
import torch

from rl_engine.kernels.gtest import run_operator_suite
from rl_engine.kernels.gtest.operator_inputs import operator_shape_name
from rl_engine.kernels.gtest.operator_specs import (
    GtestAdaLNReference,
    make_candidate,
    make_operator_case,
)
from rl_engine.kernels.gtest.tolerance import load_contract, resolve_tolerance
from rl_engine.kernels.ops.pytorch.norm.adaln_modulation import NativeAdaLNModulationOp
from rl_engine.kernels.registry import OpBackend, kernel_registry


def _args(hidden=7):
    return Namespace(
        op="adaln_modulation",
        candidate="pytorch",
        arch_key=None,
        batch=2,
        seq=3,
        normalized_dim=hidden,
        seed=386,
    )


@pytest.fixture(
    params=["pytorch", "triton"]
    + [
        pytest.param(("cuda", threads), id=f"cuda-threads{threads}")
        for threads in (128, 256, 512)
    ]
)
def shared_op(request):
    backend, threads = (
        request.param if isinstance(request.param, tuple) else (request.param, None)
    )
    if backend != "pytorch" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    args = _args()
    args.candidate = backend
    try:
        op = make_candidate(args).fn
        if threads is not None:
            op = type(op)(threads=threads)
    except RuntimeError:
        if backend != "cuda":
            raise
        if os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
            raise
        pytest.skip("compiled CUDA AdaLN extension required")
    return op, "cpu" if backend == "pytorch" else "cuda"


def _bytes_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().cpu().contiguous().view(torch.uint8),
        expected.detach().cpu().contiguous().view(torch.uint8),
    )


@pytest.mark.parametrize("nan_upstream", [False, True])
def test_adaln_gate_preserves_bf16_nan_payload_and_identity_vjp(
    shared_op, nan_upstream
):
    op, device = shared_op
    x = torch.tensor([[[-1.0, 1.0]]], device=device, dtype=torch.bfloat16)
    m = torch.zeros(1, 6, device=device, dtype=torch.bfloat16)
    # Gate is a passthrough: converting these payloads through FP32 is lossy.
    payload = torch.tensor([0x7F81, -63], dtype=torch.int16).view(torch.bfloat16)
    m[:, 4:] = payload.to(device)
    m.requires_grad_()
    _, gate = op(x, m)
    _bytes_equal(gate, payload.reshape(1, 1, 2))
    dg = (
        torch.tensor([0x7F82, 0x7FFF], dtype=torch.int16).view(m.dtype).reshape(1, 1, 2)
        if nan_upstream
        else torch.tensor([[[2.0, -3.0]]], dtype=m.dtype)
    )
    (dm,) = torch.autograd.grad(gate, (m,), dg.to(device))
    expected = torch.zeros(1, 6, dtype=m.dtype)
    expected[:, 4:] = dg[:, 0]
    _bytes_equal(dm, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaln_signed_zero_follows_hidden_tree(shared_op, dtype):
    op, device = shared_op
    x = torch.full((1, 1, 2), -0.0, device=device, dtype=dtype)
    m = torch.zeros(1, 6, device=device, dtype=dtype)
    m[:, :2] = -0.0
    y, _ = op(x, m)
    # (-0 + -0)/2 = -0, so centered = -0 - -0 = +0. An extra +0 in
    # the final reduction erases mean's sign and incorrectly makes y=-0.
    _bytes_equal(y, torch.zeros(1, 1, 2, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "eps,rstd",
    [
        (1e-40, 1.000002658868784e20),
        (2**-149, 2.671373844909537e22),
        (3.4028234663852886e38, 5.421011508662376e-20),
    ],
)
def test_adaln_subnormal_eps_is_not_flushed_to_zero(shared_op, dtype, eps, rstd):
    op, device = shared_op
    x = torch.zeros(1, 1, 2, device=device, dtype=dtype, requires_grad=True)
    m = torch.zeros(1, 6, device=device, dtype=dtype, requires_grad=True)
    y, gate = op(x, m, eps=eps)
    _bytes_equal(y, torch.zeros(1, 1, 2, dtype=dtype))
    dy = torch.tensor([[[1.0, -1.0]]], device=device, dtype=dtype)
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, torch.zeros_like(gate)))
    # Independent scalar rounding literals: FP32(1 / FP32(sqrt(FP32(eps)))).
    _bytes_equal(dx, torch.tensor([[[rstd, -rstd]]], dtype=dtype))
    _bytes_equal(dm, torch.tensor([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]], dtype=dtype))


@pytest.mark.parametrize("scale, positive", [(0.0, -(2**-21)), (1 / 256, 1 / 256)])
def test_adaln_bf16_forward_casts_only_at_output(shared_op, scale, positive):
    op, device = shared_op
    x = torch.tensor([[[-1.0, 1.0]]], device=device, dtype=torch.bfloat16)
    m = torch.tensor(
        [[-1.0, -1.0, scale, scale, -0.0, 2.0]], device=device, dtype=torch.bfloat16
    )
    y, gate = op(x, m)
    # FP32 sqrt(1 + 1e-6) = 1 + 2^-21, reciprocal = 1 - 2^-21.
    # Rounding norm or (1 + 1/256) to BF16 early erases the positive residual.
    _bytes_equal(y, torch.tensor([[[-2.0, positive]]], dtype=torch.bfloat16))
    _bytes_equal(gate, torch.tensor([[[-0.0, 2.0]]], dtype=torch.bfloat16))


def test_adaln_bf16_backward_keeps_partials_and_accumulators_fp32(shared_op):
    op, device = shared_op
    x = torch.tensor(
        [[[-3.0, 3.0]] * 3], device=device, dtype=torch.bfloat16, requires_grad=True
    )
    m = torch.tensor(
        [[0.0, 0.0, 1 / 256, 1 / 256, 2.0, 3.0]],
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    dy = torch.tensor(
        [[[256.0, 0.0], [1.0, 0.0], [-256.0, 0.0]]], device=device, dtype=torch.bfloat16
    )
    dg = torch.tensor([[[-2.0, 0.5]]], device=device, dtype=torch.bfloat16)
    y, gate = op(x, m, eps=7.0)
    torch.autograd.backward((y, gate), (dy, dg))
    # variance+eps=16, rstd=1/4, norm=[-3/4,+3/4]. The exact VJP coefficient
    # is (1+1/256)*(1-9/16)/8 = 1799/32768, independent of kernel helpers.
    coefficient = 1799 / 32768
    expected_dx = torch.tensor(
        [
            [
                [256 * coefficient, -256 * coefficient],
                [coefficient, -coefficient],
                [-256 * coefficient, 256 * coefficient],
            ]
        ],
        dtype=torch.bfloat16,
    )
    _bytes_equal(x.grad, expected_dx)
    # FP32 left folds: 256+1-256=1; -192-3/4+192=-3/4.
    _bytes_equal(
        m.grad, torch.tensor([[1.0, 0.0, -0.75, 0.0, -2.0, 0.5]], dtype=torch.bfloat16)
    )


@pytest.mark.parametrize("hidden", [7, 3072])
def test_adaln_bf16_vjp_matches_fp32_with_one_final_cast(shared_op, hidden):
    op, device = shared_op
    torch.manual_seed(389)
    x = torch.randn(2, 3, hidden, device=device, dtype=torch.bfloat16)
    m = torch.randn(2, 3 * hidden, device=device, dtype=torch.bfloat16)
    dy, dg = torch.randn_like(x), torch.randn(
        2, 1, hidden, device=device, dtype=x.dtype
    )
    observations = []
    for dtype in (torch.bfloat16, torch.float32):
        xx, mm = (
            x.to(dtype).detach().requires_grad_(),
            m.to(dtype).detach().requires_grad_(),
        )
        y, gate = op(xx, mm)
        dx, dm = torch.autograd.grad((y, gate), (xx, mm), (dy.to(dtype), dg.to(dtype)))
        observations.append((y, gate, dx, dm))
    # Identical quantized inputs also expose BF16 saved norms/partials used only
    # in backward. Independent LayerNorm math is checked in a separate test.
    for actual, fp32 in zip(*observations, strict=True):
        _bytes_equal(actual, fp32.to(torch.bfloat16))


@pytest.mark.parametrize("fraction, rounded", [(1 / 256, 1.0), (3 / 256, 1 + 4 / 256)])
def test_adaln_bf16_boundary_rounds_halfway_values_to_even(
    shared_op, fraction, rounded
):
    op, device = shared_op
    x = torch.tensor([[[-1.0, 1.0]] * 2], device=device, dtype=torch.bfloat16)
    m = torch.zeros(1, 6, device=device, dtype=x.dtype)
    m[:, :2] = fraction
    m.requires_grad_()
    y, _ = op(x, m, eps=0.0)
    _bytes_equal(y, torch.tensor([[[-1 + fraction, rounded]] * 2], dtype=x.dtype))
    dy = torch.tensor([[[0.0, 1.0], [0.0, fraction]]], device=device, dtype=x.dtype)
    (dm,) = torch.autograd.grad(y, (m,), dy)
    _bytes_equal(
        dm, torch.tensor([[0.0, rounded, 0.0, rounded, 0.0, 0.0]], dtype=x.dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaln_h3072_sample_bytes_survive_batch_positions_and_padding(shared_op, dtype):
    _assert_h3072_relocation(shared_op, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "backend,options",
    [
        ("triton", {"num_warps": w, "reduction_tile": t})
        for w in (4, 8)
        for t in (64, 128, 256)
    ]
    + [("cuda", {"threads": n}) for n in (128, 256, 512)],
)
def test_adaln_h3072_relocation_with_all_launch_configurations(dtype, backend, options):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    args = _args()
    args.candidate = backend
    try:
        default_op = make_candidate(args).fn
    except RuntimeError:
        if backend != "cuda" or os.environ.get("RL_KERNEL_REQUIRE_EXT") == "1":
            raise
        pytest.skip("compiled CUDA AdaLN extension required")
    op_class = type(default_op)
    op = op_class(**options)
    _assert_h3072_relocation((op, "cuda"), dtype, reference_op=default_op)


def _assert_h3072_relocation(shared_op, dtype, *, reference_op=None):
    op, device = shared_op
    torch.manual_seed(386)
    hidden, valid = 3072, 3
    x0 = torch.randn(1, valid, hidden, device=device, dtype=dtype)
    full0 = torch.randn(1, 6 * hidden, device=device, dtype=dtype)
    dy0 = torch.randn_like(x0)
    dg0 = torch.randn(1, 1, hidden, device=device, dtype=dtype)

    def observe(x, full, dy, dg, *, candidate=op):
        x = x.detach().requires_grad_()
        full = full.detach().requires_grad_()
        y, gate = candidate(x, full[:, : 3 * hidden])
        dx, dfull = torch.autograd.grad((y, gate), (x, full), (dy, dg))
        _bytes_equal(dfull[:, 3 * hidden :], torch.zeros_like(dfull[:, 3 * hidden :]))
        return y, gate, dx, dfull[:, : 3 * hidden]

    baseline = observe(x0, full0, dy0, dg0, candidate=reference_op or op)
    for batch, seq in ((1, 5), (3, 3), (3, 5)):
        for position in range(batch):
            # Strides and six-chunk views match model callers, alongside topology changes.
            x = torch.randn(batch, 2 * seq, hidden, device=device, dtype=dtype)[:, ::2]
            full = torch.randn(batch, 6 * hidden, device=device, dtype=dtype)
            dy = torch.randn_like(x)
            dg = torch.randn(batch, 1, hidden, device=device, dtype=dtype)
            x[position, :valid], full[position] = x0[0], full0[0]
            dy[position, :valid], dg[position] = dy0[0], dg0[0]
            dy[position, valid:] = 0
            actual = observe(x, full, dy, dg)
            for index, (result, expected) in enumerate(
                zip(actual, baseline, strict=True)
            ):
                result = (
                    result[position, :valid] if index in (0, 2) else result[position]
                )
                _bytes_equal(result, expected[0])
            # Padding has zero VJP; IEEE arithmetic may produce either zero sign.
            assert torch.equal(
                actual[2][position, valid:],
                torch.zeros_like(actual[2][position, valid:]),
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaln_backward_accepts_transposed_and_broadcast_upstreams(shared_op, dtype):
    op, device = shared_op
    torch.manual_seed(391)
    x = torch.randn(2, 3, 7, device=device, dtype=dtype, requires_grad=True)
    m = torch.randn(2, 21, device=device, dtype=dtype, requires_grad=True)
    dy = torch.randn(2, 7, 3, device=device, dtype=dtype).transpose(1, 2)
    dg_storage = torch.randn(2, 1, 7, device=device, dtype=dtype)
    dg = dg_storage[:1].expand(2, 1, 7)
    assert not dy.is_contiguous() and dg.stride(0) == 0
    actual = torch.autograd.grad(op(x, m), (x, m), (dy, dg))
    contiguous = torch.autograd.grad(
        op(x, m), (x, m), (dy.contiguous(), dg.contiguous())
    )
    for result, expected in zip(actual, contiguous, strict=True):
        _bytes_equal(result, expected)
    xx = x.detach().cpu().float().requires_grad_()
    mm = m.detach().cpu().float().requires_grad_()
    gold = torch.autograd.grad(
        GtestAdaLNReference().forward_fp32(xx, mm),
        (xx, mm),
        (dy.cpu().float(), dg.cpu().float()),
    )
    tol = resolve_tolerance(
        load_contract(), judgment="gradient_accuracy", op_class="reduction", dtype=dtype
    )
    for result, expected in zip(actual, gold, strict=True):
        torch.testing.assert_close(
            result.cpu().float(), expected, atol=tol.atol, rtol=tol.rtol
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("trainable", ["x", "modulation", "gate_only"])
def test_adaln_single_trainable_input_and_unused_output(shared_op, dtype, trainable):
    op, device = shared_op
    x_trainable = trainable != "modulation"
    x = torch.tensor(
        [[[-3.0, 3.0] if trainable == "x" else [-1.0, 1.0]]],
        device=device,
        dtype=dtype,
        requires_grad=x_trainable,
    )
    m = torch.zeros(1, 6, device=device, dtype=dtype, requires_grad=trainable != "x")
    y, gate = op(x, m, eps=7.0 if trainable == "x" else 0.0)
    if trainable == "x":
        y.backward(torch.tensor([[[1.0, 0.0]]], device=device, dtype=dtype))
        _bytes_equal(x.grad, torch.tensor([[[7 / 128, -7 / 128]]], dtype=dtype))
        assert m.grad is None
    elif trainable == "modulation":
        y.sum().backward()  # Gate is genuinely unused, rather than given a zero VJP.
        assert x.grad is None
        _bytes_equal(
            m.grad, torch.tensor([[1.0, 1.0, -1.0, 1.0, 0.0, 0.0]], dtype=dtype)
        )
    else:
        gate.backward(torch.tensor([[[2.0, -3.0]]], device=device, dtype=dtype))
        assert torch.equal(x.grad, torch.zeros_like(x))
        _bytes_equal(
            m.grad, torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, -3.0]], dtype=dtype)
        )


def test_adaln_hidden_reduction_uses_declared_pairwise_order(shared_op):
    op, device = shared_op
    x = torch.tensor([[[2**24, 1.0, -(2**24), 1.0]]], device=device)
    m = torch.zeros(1, 12, device=device)
    y, _ = op(x, m, eps=0.0)
    # Lower+upper sum is 2 (left fold is 1), mean=1/2, FP32 variance=2^47.
    _bytes_equal(
        y,
        torch.tensor(
            [
                [
                    [
                        1.4142135381698608,
                        4.214684778958144e-8,
                        -1.4142135381698608,
                        4.214684778958144e-8,
                    ]
                ]
            ]
        ),
    )


def test_adaln_variance_reduction_uses_declared_pairwise_order(shared_op):
    op, device = shared_op
    x = torch.tensor([[[4096.0, 1.25, -4096.0, -1.25]]], device=device)
    y, _ = op(x, torch.zeros(1, 12, device=device), eps=0.0)
    # Mean is zero under either tree. Pairwise variance=8388609, left fold=8388608.
    _bytes_equal(
        y,
        torch.tensor(
            [
                [
                    [
                        1.4142134189605713,
                        0.0004315836704336107,
                        -1.4142134189605713,
                        -0.0004315836704336107,
                    ]
                ]
            ]
        ),
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_adaln_backward_hidden_sums_use_declared_pairwise_order(shared_op, sign):
    op, device = shared_op
    x = torch.tensor([[[-1.0, 1.0, -1.0, 1.0]]], device=device, requires_grad=True)
    m = torch.zeros(1, 12, device=device)
    y, _ = op(x, m, eps=0.0)
    dy = torch.tensor([[[sign * 2**24, 1.0, -sign * 2**24, 1.0]]], device=device)
    (dx,) = torch.autograd.grad(y, (x,), dy)
    # Both hidden means are 1/2. A left fold in either reduction leaves 1/4
    # in channels 1 and 3; the two signs discriminate the reductions separately.
    _bytes_equal(dx, torch.tensor([[[sign * 2**24, 0.0, -sign * 2**24, 0.0]]]))


def test_adaln_modulation_gradient_uses_ascending_token_order(shared_op):
    op, device = shared_op
    x = torch.tensor([[[-1.0, 1.0]] * 4], device=device, requires_grad=True)
    m = torch.zeros(1, 6, device=device, requires_grad=True)
    y, gate = op(x, m, eps=0.0)
    dy = torch.tensor(
        [[[2**24, 0.0], [1.0, 0.0], [-(2**24), 0.0], [1.0, 0.0]]], device=device
    )
    (dm,) = torch.autograd.grad((y, gate), (m,), (dy, torch.zeros_like(gate)))
    # Ascending FP32 fold is 1; reordering to lower+upper pairs would yield 2.
    _bytes_equal(dm, torch.tensor([[1.0, 0.0, -1.0, 0.0, 0.0, 0.0]]))


def test_adaln_independent_gold_uses_contract_token_order():
    x = torch.tensor([[[-1.0, 1.0]] * 8])
    m = torch.zeros(1, 6, requires_grad=True)
    dy = torch.tensor([[[2**24, 0.0], [1.0, 0.0], [-(2**24), 0.0], [1.0, 0.0]]])
    y, gate = GtestAdaLNReference().forward_fp32(x, m, eps=0.0)
    (dm,) = torch.autograd.grad(
        (y, gate), (m,), (dy.repeat(1, 2, 1), torch.zeros_like(gate))
    )
    # The carry from the first block is lost at 1+2^24; each ascending fold ends at 1.
    # A reassociated CPU broadcast backward returns +/-2 for two blocks.
    _bytes_equal(dm, torch.tensor([[1.0, 0.0, -1.0, 0.0, 0.0, 0.0]]))


@pytest.mark.parametrize(
    "invalid, error, message",
    [
        ("rank", ValueError, "nonempty shape"),
        ("empty_batch", ValueError, "nonempty shape"),
        ("empty_sequence", ValueError, "nonempty shape"),
        ("empty_hidden", ValueError, "nonempty shape"),
        ("modulation_shape", ValueError, "modulation must have shape"),
        ("mixed_dtype", ValueError, "share device and dtype"),
        ("fp16", TypeError, "FP32 or BF16"),
        ("negative_eps", ValueError, "finite and nonnegative"),
        ("nan_eps", ValueError, "finite and nonnegative"),
        ("infinite_eps", ValueError, "finite and nonnegative"),
    ],
)
def test_adaln_shared_rejects_unsupported_inputs(shared_op, invalid, error, message):
    op, device = shared_op
    shape = {
        "rank": (1, 2),
        "empty_batch": (0, 2, 2),
        "empty_sequence": (1, 0, 2),
        "empty_hidden": (1, 2, 0),
    }.get(invalid, (1, 2, 2))
    x = torch.zeros(shape, device=device)
    m = torch.zeros(shape[0], 3 * shape[-1], device=device)
    if invalid == "modulation_shape":
        m = m[:, :-1]
    if invalid == "mixed_dtype":
        m = m.bfloat16()
    if invalid == "fp16":
        x, m = x.half(), m.half()
    eps = {
        "negative_eps": -1.0,
        "nan_eps": float("nan"),
        "infinite_eps": float("inf"),
    }.get(invalid, 1e-6)
    with pytest.raises(error, match=message):
        op(x, m, eps=eps)


def test_adaln_shared_rejects_mixed_devices(shared_op):
    op, device = shared_op
    with pytest.raises(ValueError, match="share device and dtype"):
        op(torch.zeros(1, 2, 2, device=device), torch.empty(1, 6, device="meta"))


@pytest.mark.parametrize("eps", [1e39, 1e-46])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaln_shared_rejects_eps_overflow_and_positive_underflow(
    shared_op, dtype, eps
):
    op, device = shared_op
    x = torch.zeros(1, 1, 2, device=device, dtype=dtype)
    m = torch.zeros(1, 6, device=device, dtype=dtype)
    with pytest.raises(ValueError, match="finite FP32.*underflow"):
        op(x, m, eps=eps)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaln_nonfinite_activation_propagates_nan_and_preserves_gate_vjp(
    shared_op, dtype, value
):
    op, device = shared_op
    x = torch.tensor([[[value, 1.0]]], device=device, dtype=dtype, requires_grad=True)
    m = torch.zeros(1, 6, device=device, dtype=dtype)
    m[:, 4:] = torch.tensor([2.0, -0.0], device=device, dtype=dtype)
    m.requires_grad_()
    y, gate = op(x, m)
    assert torch.isnan(y).all()
    _bytes_equal(gate, torch.tensor([[[2.0, -0.0]]], dtype=dtype))
    dy = torch.tensor([[[1.0, -1.0]]], device=device, dtype=dtype)
    dg = torch.tensor([[[2.0, -3.0]]], device=device, dtype=dtype)
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy, dg))
    assert torch.isnan(dx).all() and torch.isnan(dm[:, 2:4]).all()
    _bytes_equal(dm[:, :2], torch.tensor([[1.0, -1.0]], dtype=dtype))
    _bytes_equal(dm[:, 4:], dg[:, 0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden", [7, 3072])
def test_adaln_gtest_outputs_and_gradients_use_shared_contract(dtype, hidden):
    args = _args(hidden)
    case = make_operator_case(args, dtype, torch.device("cpu"))
    assert operator_shape_name(args.op, args) == f"2x3x{hidden}"
    assert case.inputs["modulation"].shape == (2, 3 * hidden)
    gold = case.gold_fn(**case.inputs)
    assert all(t.device.type == "cpu" and t.dtype == torch.float32 for t in gold)
    assert gold[0].shape == (2, 3, hidden) and gold[1].shape == (2, 1, hidden)
    report = run_operator_suite(
        args.op, candidates=[make_candidate(args)], cases=[case], check_grad=True
    )
    assert report.passed, report.to_dict()
    checks = report.candidates[0].cases[0].outputs
    assert len(checks) == 4
    assert all(c.gold_dtype == "torch.float32" for c in checks)
    assert [c.message for c in checks[2:]] == ["gradient:x", "gradient:modulation"]
    for check in checks:
        threshold = resolve_tolerance(
            load_contract(), judgment=check.judgment, op_class="reduction", dtype=dtype
        )
        assert (check.atol, check.rtol) == (threshold.atol, threshold.rtol)


@pytest.mark.parametrize("broken", ["gate", "x", "modulation"])
def test_adaln_gtest_detects_wrong_output_and_backward(broken):
    args = _args()
    case = make_operator_case(args, torch.float32, torch.device("cpu"))
    good = make_candidate(args)

    def corrupt(x, modulation, eps):
        # Zero-valued terms preserve connectivity while deliberately erasing a VJP.
        xx = x.detach() + x * 0 if broken == "x" else x
        mm = (
            modulation.detach() + modulation * 0
            if broken == "modulation"
            else modulation
        )
        y, gate = good.fn(xx, mm, eps=eps)
        return y, gate + 1 if broken == "gate" else gate

    report = run_operator_suite(
        args.op, candidates=[replace(good, fn=corrupt)], cases=[case], check_grad=True
    )
    assert not report.passed
    checks = report.candidates[0].cases[0].outputs
    check = (
        checks[1]
        if broken == "gate"
        else next(c for c in checks if c.message == f"gradient:{broken}")
    )
    assert not check.passed


@pytest.mark.parametrize("check_grad", [False, True])
def test_adaln_gtest_rejects_fp32_outputs_for_bf16_inputs(check_grad):
    args = _args()
    case = make_operator_case(args, torch.bfloat16, torch.device("cpu"))
    good = make_candidate(args)

    def wrong_dtype(**inputs):
        return tuple(output.float() for output in good.fn(**inputs))

    report = run_operator_suite(
        args.op,
        candidates=[replace(good, fn=wrong_dtype)],
        cases=[case],
        check_grad=check_grad,
    )
    assert not report.passed, report.to_dict()
    checks = report.candidates[0].cases[0].outputs[:2]
    assert all(
        not check.passed and "dtype mismatch" in check.message for check in checks
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("triple", [0, 1])
def test_adaln_gate_only_and_separate_stream_chunks(shared_op, dtype, triple):
    op, device = shared_op
    torch.manual_seed(386)
    hidden = 7
    # Separate image/text projections, each exposing a strided triple of six chunks.
    image = torch.randn(2, 5, hidden, device=device, dtype=dtype, requires_grad=True)
    text = torch.randn(2, 3, hidden, device=device, dtype=dtype, requires_grad=True)
    image_full = torch.randn(
        2, 6 * hidden, device=device, dtype=dtype, requires_grad=True
    )
    text_full = torch.randn(
        2, 6 * hidden, device=device, dtype=dtype, requires_grad=True
    )
    start, stop = triple * 3 * hidden, (triple + 1) * 3 * hidden
    unused = slice(3 * hidden, None) if triple == 0 else slice(None, 3 * hidden)
    for x, full, other in (
        (image, image_full, text_full),
        (text, text_full, image_full),
    ):
        m = full[:, start:stop]
        other_before = None if other.grad is None else other.grad.clone()
        assert not m.is_contiguous()
        y, gate = op(x, m)
        dg = torch.randn_like(gate)
        torch.autograd.backward((y, gate), (torch.zeros_like(y), dg))
        assert torch.equal(x.grad, torch.zeros_like(x))
        expected = torch.zeros_like(full)
        expected[:, start + 2 * hidden : stop] = dg[:, 0]
        assert torch.equal(full.grad, expected)
        if other_before is None:
            assert other.grad is None
        else:
            _bytes_equal(other.grad, other_before)
        # Independent CPU gold with nonuniform y/gate upstreams on each stream.
        ref_x = x.detach().cpu().float().requires_grad_()
        ref_m = m.detach().cpu().float().requires_grad_()
        ref_y, ref_gate = GtestAdaLNReference().forward_fp32(ref_x, ref_m)
        forward_tol = resolve_tolerance(
            load_contract(),
            judgment="forward_accuracy",
            op_class="reduction",
            dtype=dtype,
        )
        torch.testing.assert_close(
            y.detach().cpu().float(),
            ref_y.detach(),
            atol=forward_tol.atol,
            rtol=forward_tol.rtol,
        )
        _bytes_equal(gate, m[:, None, 2 * hidden :])
        dy = torch.randn_like(y)
        actual = torch.autograd.grad(op(x, m), (x, full), (dy, dg))
        gold = torch.autograd.grad(
            (ref_y, ref_gate), (ref_x, ref_m), (dy.cpu().float(), dg.cpu().float())
        )
        tol = resolve_tolerance(
            load_contract(),
            judgment="gradient_accuracy",
            op_class="reduction",
            dtype=dtype,
        )
        for result, expected_grad in (
            (actual[0], gold[0]),
            (actual[1][:, start:stop], gold[1]),
        ):
            torch.testing.assert_close(
                result.cpu().float(), expected_grad, atol=tol.atol, rtol=tol.rtol
            )
        assert torch.equal(actual[1][:, unused], torch.zeros_like(actual[1][:, unused]))


@pytest.mark.parametrize("hidden", [4095, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaln_hidden_boundary_matches_independent_math_and_fixed_bytes(
    shared_op, dtype, hidden
):
    op, device = shared_op
    torch.manual_seed(395)
    x_cpu = torch.randn(2, 3, hidden, dtype=dtype)
    m_cpu = torch.randn(2, 3 * hidden, dtype=dtype)
    dy, dg = torch.randn_like(x_cpu), torch.randn(2, 1, hidden, dtype=dtype)
    x = x_cpu.to(device).detach().requires_grad_()
    m = m_cpu.to(device).detach().requires_grad_()
    y, gate = op(x, m)
    dx, dm = torch.autograd.grad((y, gate), (x, m), (dy.to(device), dg.to(device)))
    xx, mm = x_cpu.float().requires_grad_(), m_cpu.float().requires_grad_()
    gold_y, gold_gate = GtestAdaLNReference().forward_fp32(xx, mm)
    gold_dx, gold_dm = torch.autograd.grad(
        (gold_y, gold_gate), (xx, mm), (dy.float(), dg.float())
    )
    for actual, expected, judgment in (
        (y, gold_y, "forward_accuracy"),
        (gate, gold_gate, "forward_accuracy"),
        (dx, gold_dx, "gradient_accuracy"),
        (dm, gold_dm, "gradient_accuracy"),
    ):
        tol = resolve_tolerance(
            load_contract(), judgment=judgment, op_class="reduction", dtype=dtype
        )
        torch.testing.assert_close(
            actual.cpu().float(), expected, atol=tol.atol, rtol=tol.rtol
        )
    fx, fm = x_cpu.detach().requires_grad_(), m_cpu.detach().requires_grad_()
    fixed_y, fixed_gate = NativeAdaLNModulationOp()(fx, fm)
    fixed_dx, fixed_dm = torch.autograd.grad((fixed_y, fixed_gate), (fx, fm), (dy, dg))
    for actual, expected in zip(
        (y, gate, dx, dm), (fixed_y, fixed_gate, fixed_dx, fixed_dm), strict=True
    ):
        _bytes_equal(actual, expected)


def test_adaln_declared_unavailable_fallbacks_and_exhaustion(monkeypatch):
    original = kernel_registry._get_or_create_backend
    missing = set()

    def resolve(backend):
        if backend in missing:
            return None
        if backend is OpBackend.TRITON_ADALN_MODULATION:
            return object()  # Resolver-only sentinel: CPU CI does not install Triton.
        return original(backend)

    monkeypatch.setattr(kernel_registry, "_get_or_create_backend", resolve)
    missing.add(OpBackend.CUDA_ADALN_MODULATION)
    # No GPU execution: test resolution and trace, then exercise the CPU fallback.
    _, trace = kernel_registry.get_adaln_modulation_op("cuda", hidden=7)
    assert trace["selected_backend"] == OpBackend.TRITON_ADALN_MODULATION.name
    assert trace["fallback"] is True
    missing.add(OpBackend.TRITON_ADALN_MODULATION)
    op, trace = kernel_registry.get_adaln_modulation_op("cuda", hidden=7)
    assert isinstance(op, NativeAdaLNModulationOp)
    assert trace["rejected_backends"] == [
        OpBackend.CUDA_ADALN_MODULATION.name,
        OpBackend.TRITON_ADALN_MODULATION.name,
    ]
    x, m = torch.randn(1, 2, 7), torch.randn(1, 21)
    y, _ = op(x, m)
    gold, _ = GtestAdaLNReference().forward_fp32(x, m)
    tol = resolve_tolerance(
        load_contract(),
        judgment="forward_accuracy",
        op_class="reduction",
        dtype=torch.float32,
    )
    torch.testing.assert_close(y, gold, atol=tol.atol, rtol=tol.rtol)
    missing.add(OpBackend.PYTORCH_ADALN_MODULATION)
    with pytest.raises(RuntimeError, match="No functional adaln_modulation"):
        kernel_registry.get_adaln_modulation_op("cuda", hidden=7)
