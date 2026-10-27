# SPDX-License-Identifier: Apache-2.0
"""Task-29 mathematical accuracy, raw-bit BI, and fail-closed behavior."""

import pytest
import torch

from rl_engine.contracts.operators.router_sqrt_softplus import NEGATIVE_TAIL_CUTOFF
from rl_engine.ops.moe.router_sqrt_softplus import router_sqrt_softplus
from rl_engine.reference.moe.router_sqrt_softplus import (
    MATHEMATICAL_ORACLE_TAIL_CUTOFF,
    mlite_sqrt_softplus,
    sqrt_softplus_fp64,
    sqrt_softplus_vjp_fp64,
)

gpu = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(), reason="requires ROCm GPU"
)
QUANTUM_TOLERANCE = 2.0**-146  # Eight minimum FP32 subnormal quanta.


def test_mathematical_oracle_keeps_the_more_conservative_tail_cutoff():
    # This interval distinguishes mathematical validation from repeating the
    # production approximation. -20 exposes a difference well above FP64 epsilon.
    x = torch.tensor([-36.0, -30.0, -20.0, -16.00001], dtype=torch.float64)
    mathematical = torch.sqrt(torch.logaddexp(x, torch.zeros_like(x)))
    mathematical_slope = torch.sigmoid(x) / (2 * mathematical)
    oracle = sqrt_softplus_fp64(x)
    oracle_slope = sqrt_softplus_vjp_fp64(x, torch.ones_like(x))
    assert MATHEMATICAL_ORACLE_TAIL_CUTOFF == -36.0 < NEGATIVE_TAIL_CUTOFF
    torch.testing.assert_close(oracle, mathematical, rtol=0, atol=0)
    torch.testing.assert_close(oracle_slope, mathematical_slope, rtol=0, atol=0)
    production_tail = torch.exp(x / 2)
    assert production_tail[2] > oracle[2]
    assert 0.5 * production_tail[2] > oracle_slope[2]
    assert ((production_tail - oracle).abs() / oracle).max() < torch.exp(
        torch.tensor(NEGATIVE_TAIL_CUTOFF, dtype=torch.float64)
    ) / 4


@gpu
def test_production_tail_formula_and_vjp_at_fp32_branch_neighbors():
    cutoff = torch.tensor(NEGATIVE_TAIL_CUTOFF, dtype=torch.float32)
    values = torch.tensor([-40.0, -36.0, -30.0, -20.0, -17.0, -16.0, -15.0, 0.0])
    neighbors = torch.stack(
        [
            torch.nextafter(cutoff, torch.tensor(-torch.inf)),
            torch.nextafter(cutoff, torch.tensor(torch.inf)),
        ]
    )
    x_cpu = torch.cat((values, neighbors)).reshape(1, -1)
    # An explicit FP32 contract emulator is separate from the FP64 math oracle.
    e = torch.exp(-x_cpu.abs())
    expected = torch.where(
        x_cpu < NEGATIVE_TAIL_CUTOFF,
        torch.exp(x_cpu / 2),
        torch.sqrt(torch.clamp_min(x_cpu, 0) + torch.log1p(e)),
    )
    sigmoid = torch.where(x_cpu >= 0, 1 / (1 + e), e / (1 + e))
    slope = torch.where(x_cpu < NEGATIVE_TAIL_CUTOFF, 0.5 * expected, sigmoid / (2 * expected))
    x = x_cpu.cuda().requires_grad_()
    scores = transform(x).scores
    dx = torch.autograd.grad(scores, x, torch.ones_like(scores))[0]
    # CPU libm and AMD OCML may differ by a small number of ULPs. This is an
    # arithmetic-contract check, independent of broad mathematical tolerances.
    assert (bits(scores).cpu().long() - bits(expected).long()).abs().max() <= 2
    assert (bits(dx).cpu().long() - bits(slope).long()).abs().max() <= 3


def transform(x):
    return router_sqrt_softplus(x, model_version="task29-synthetic-v1")


def bits(x):
    return x.detach().contiguous().view(torch.int32)


def layout(x, kind):
    if kind == "transpose":
        return x.T.contiguous().T
    if kind == "padded":
        output = torch.empty((x.shape[0], 2 * x.shape[1] + 3), device=x.device)
        output = output[:, 1 : 2 * x.shape[1] + 1 : 2]
        output.copy_(x)
        return output
    return x


@gpu
@pytest.mark.parametrize("columns", [1, 6, 255, 256, 257])
@pytest.mark.parametrize("kind", ["contiguous", "padded", "transpose"])
def test_forward_and_vjp_bytes_across_batches_and_layouts(columns, kind):
    generator = torch.Generator().manual_seed(29)
    row = (torch.randn(1, columns, generator=generator) * 32).cuda()
    grad_row = torch.randn(1, columns, generator=generator).cuda()
    baseline_x = row.clone().requires_grad_()
    baseline = transform(baseline_x)
    baseline_grad = torch.autograd.grad(baseline.scores, baseline_x, grad_row)[0]
    for m in (1, 16, 17, 32, 65, 257):
        x = (torch.randn(m, columns, generator=generator) * 32).cuda()
        positions = sorted({0, m // 2, m - 1})
        x[positions] = row
        grad = torch.randn(m, columns, generator=generator).cuda()
        grad[positions] = grad_row
        x = layout(x, kind).requires_grad_()
        result = transform(x)
        dx = torch.autograd.grad(result.scores, x, layout(grad, kind))[0]
        with torch.inference_mode():
            replay = transform(x)
        assert result.scores.dtype == dx.dtype == torch.float32
        assert torch.equal(bits(result.scores), bits(replay.scores))
        for i in positions:
            assert torch.equal(bits(result.scores[i]), bits(baseline.scores[0]))
            assert torch.equal(bits(dx[i]), bits(baseline_grad[0]))
        reference = sqrt_softplus_fp64(x.detach().cpu())
        reference_dx = sqrt_softplus_vjp_fp64(x.detach().cpu(), grad.cpu())
        torch.testing.assert_close(
            result.scores.detach().cpu().double(), reference, rtol=5e-7, atol=QUANTUM_TOLERANCE
        )
        torch.testing.assert_close(
            dx.cpu().double(), reference_dx, rtol=2e-5, atol=QUANTUM_TOLERANCE
        )


@gpu
def test_extreme_finite_values_branch_neighbors_and_wide_gradients():
    maximum = torch.finfo(torch.float32).max
    values = [
        -maximum,
        -1000.0,
        -400.0,
        -300.0,
        -220.0,
        -200.0,
        -180.0,
        -160.0,
        -120.0,
        -100.0,
        -80.0,
        -20.0,
        -16.0,
        -1.0,
        -0.0,
        0.0,
        1.0,
        20.0,
        80.0,
        1000.0,
        maximum,
    ]
    boundary = torch.tensor([-160.0, -16.0, 0.0, 20.0], dtype=torch.float32)
    values.extend(torch.nextafter(boundary, torch.full_like(boundary, -torch.inf)).tolist())
    values.extend(torch.nextafter(boundary, torch.full_like(boundary, torch.inf)).tolist())
    for upstream in (1.0, -1.0, 0.0, -0.0, maximum, -maximum, torch.finfo(torch.float32).tiny):
        x = torch.tensor([values], device="cuda", requires_grad=True)
        grad = torch.full_like(x, upstream)
        result = transform(x)
        actual_dx = torch.autograd.grad(result.scores, x, grad)[0]
        assert torch.isfinite(result.scores).all() and torch.isfinite(actual_dx).all()
        reference = sqrt_softplus_fp64(x.detach().cpu())
        reference_dx = sqrt_softplus_vjp_fp64(x.detach().cpu(), grad.cpu())
        torch.testing.assert_close(
            result.scores.detach().cpu().double(), reference, rtol=5e-7, atol=QUANTUM_TOLERANCE
        )
        torch.testing.assert_close(
            actual_dx.cpu().double(), reference_dx, rtol=2e-5, atol=QUANTUM_TOLERANCE
        )
    # Forward score is below FP32, but this finite large-upstream VJP is representable.
    x = torch.tensor([[-300.0]], device="cuda", requires_grad=True)
    score = transform(x).scores
    dx = torch.autograd.grad(score, x, torch.full_like(x, maximum))[0]
    assert score.item() == 0 and dx.item() > 0


@gpu
@pytest.mark.parametrize("seed", range(10))
def test_accuracy_and_repeated_vjp_against_independent_oracle(seed):
    generator = torch.Generator().manual_seed(seed)
    x_cpu = torch.rand(17, 256, generator=generator) * 320 - 160
    grad_cpu = torch.randn(17, 256, generator=generator)
    x = x_cpu.cuda().requires_grad_()
    grad = grad_cpu.cuda()
    first = transform(x)
    dx1 = torch.autograd.grad(first.scores, x, grad)[0]
    second = transform(x)
    dx2 = torch.autograd.grad(second.scores, x, grad)[0]
    assert torch.equal(bits(first.scores), bits(second.scores))
    assert torch.equal(bits(dx1), bits(dx2))
    torch.testing.assert_close(
        first.scores.detach().cpu().double(),
        sqrt_softplus_fp64(x_cpu),
        rtol=5e-7,
        atol=QUANTUM_TOLERANCE,
    )
    torch.testing.assert_close(
        dx1.cpu().double(),
        sqrt_softplus_vjp_fp64(x_cpu, grad_cpu),
        rtol=2e-5,
        atol=QUANTUM_TOLERANCE,
    )


@gpu
def test_mlite_comparison_and_negative_tail_failure_control():
    x = torch.linspace(-80, 80, 1025, device="cuda").reshape(1, -1).requires_grad_()
    ours = transform(x).scores
    upstream = mlite_sqrt_softplus(x)
    ours_dx = torch.autograd.grad(ours, x, torch.ones_like(ours))[0]
    upstream_dx = torch.autograd.grad(upstream, x, torch.ones_like(upstream))[0]
    torch.testing.assert_close(ours, upstream, rtol=1e-6, atol=QUANTUM_TOLERANCE)
    torch.testing.assert_close(ours_dx, upstream_dx, rtol=1e-6, atol=QUANTUM_TOLERANCE)
    tail = torch.tensor([[-120.0]], device="cuda", requires_grad=True)
    corrected = transform(tail).scores
    native = mlite_sqrt_softplus(tail)
    corrected_grad = torch.autograd.grad(corrected, tail, torch.ones_like(tail))[0]
    native_grad = torch.autograd.grad(native, tail, torch.ones_like(tail))[0]
    assert corrected.item() > 0 and corrected_grad.item() > 0
    assert native.item() == 0 and torch.isnan(native_grad).all()


@gpu
@pytest.mark.parametrize("m", [0, 1, 17])
def test_sum_backward_zero_stride_gradient_and_empty_rows(m):
    x = torch.zeros(m, 256, device="cuda", requires_grad=True)
    result = transform(x)
    result.scores.sum().backward()
    assert x.grad.shape == x.shape and torch.isfinite(x.grad).all()
    assert result.execution["kernel_launched"] == (m != 0)


@gpu
@pytest.mark.parametrize("invalid", [torch.nan, torch.inf, -torch.inf])
def test_nonfinite_inputs_and_upstream_gradients_rejected(invalid):
    x = torch.tensor([[invalid]], device="cuda")
    with pytest.raises(ValueError, match="finite FP32 logits"):
        transform(x)
    x = torch.zeros(1, 6, device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="upstream gradients"):
        torch.autograd.grad(transform(x).scores, x, torch.full_like(x, invalid))


@gpu
def test_graph_and_expanded_inputs_rejected():
    x = torch.ones(1, 256, device="cuda")
    with pytest.raises(ValueError, match="positive input strides"):
        transform(x.expand(17, -1))
    with pytest.raises(RuntimeError, match="Graph capture"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            transform(x)


@gpu
@pytest.mark.parametrize("target", ["input", "scores"])
def test_saved_tensor_mutation_detected(target):
    x = torch.ones(17, 256, device="cuda", requires_grad=True)
    result = transform(x)
    with torch.no_grad():
        (x if target == "input" else result.scores).add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        result.scores.sum().backward()


@gpu
def test_higher_order_derivative_is_rejected():
    x = torch.ones(1, 6, device="cuda", requires_grad=True)
    scores = transform(x).scores
    gradient = torch.autograd.grad(scores, x, torch.ones_like(scores), create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), x)


@gpu
@pytest.mark.parametrize("profile", ["topk", "hash", "grouped"])
def test_existing_aiter_paths_reuse_the_same_score_values(profile):
    from aiter.ops.triton.moe.moe_routing.topk import grouped_topk, hash_routing, topk

    x = torch.linspace(-80, -20, 17 * 256, device="cuda").reshape(17, 256)
    if profile == "topk":
        weights, indices, _ = topk(x, 6, apply_softmax=False, score_mode="sqrtsoftplus")
    elif profile == "hash":
        ids = torch.arange(17, dtype=torch.int32, device="cuda")
        table = torch.arange(17 * 6, device="cuda", dtype=torch.int32).reshape(17, 6) % 256
        weights, indices, _ = hash_routing(x, table, ids, 6, renorm=False)
    else:
        weights, indices, _ = grouped_topk(
            x, 6, num_expert_group=8, topk_group=4, score_mode="sqrtsoftplus"
        )
    expected = transform(x).scores.gather(1, indices.long())
    assert torch.equal(bits(weights), bits(expected))


@pytest.mark.parametrize(
    "x,error",
    [
        (torch.ones(4), ValueError),
        (torch.ones(3, 0), ValueError),
        (torch.ones(3, 6, dtype=torch.bfloat16), TypeError),
        (torch.ones(3, 6), RuntimeError),
        ([1.0, 2.0], TypeError),
    ],
)
def test_invalid_requests_rejected_without_a_gpu(x, error):
    with pytest.raises(error):
        transform(x)


def test_model_version_required():
    with pytest.raises(ValueError, match="model_version"):
        router_sqrt_softplus(torch.zeros(1, 6), model_version="")


def test_reference_derivative_by_fp64_finite_difference():
    x = torch.linspace(-80, 80, 2001, dtype=torch.float64)
    epsilon = 1e-4
    difference = (sqrt_softplus_fp64(x + epsilon) - sqrt_softplus_fp64(x - epsilon)) / (2 * epsilon)
    analytic = sqrt_softplus_vjp_fp64(x, torch.ones_like(x))
    torch.testing.assert_close(difference, analytic, rtol=2e-8, atol=1e-20)
