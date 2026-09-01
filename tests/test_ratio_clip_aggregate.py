# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.pytorch.loss.ratio_clip_aggregate import NativeRatioClipAggregateOp

try:
    from rl_engine.kernels.ops.triton.loss.ratio_clip_aggregate import TritonRatioClipAggregateOp

    _HAS_TRITON = True
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name not in {"triton", "triton.language"}:
        raise
    TritonRatioClipAggregateOp = None
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton ratio-clip-aggregate requires a CUDA device and Triton.",
)


def test_native_matches_asymmetric_clip_reference():
    ratio = torch.tensor([[0.60, 0.85, 1.10], [1.25, 1.50, 0.95]])
    advantages = torch.tensor([[1.0, -2.0, 0.5], [-1.5, 2.0, -0.25]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    penalty = torch.tensor([[0.1, 0.2, 9.0], [0.3, 0.4, 0.5]])
    clip_low, clip_high, penalty_coef = 0.15, 0.25, 0.07

    total, policy, mean_penalty, clip_fraction = NativeRatioClipAggregateOp()(
        ratio,
        advantages,
        mask,
        clip_low=clip_low,
        clip_high=clip_high,
        penalty_terms=penalty,
        penalty_coef=penalty_coef,
    )

    clipped_ratio = ratio.clamp(1.0 - clip_low, 1.0 + clip_high)
    terms = -torch.minimum(ratio * advantages, clipped_ratio * advantages)
    expected_policy = terms[mask].mean()
    expected_penalty = penalty[mask].mean()
    expected_clip_fraction = (
        ((ratio < 1.0 - clip_low) | (ratio > 1.0 + clip_high))[mask].float().mean()
    )

    torch.testing.assert_close(policy, expected_policy)
    torch.testing.assert_close(mean_penalty, expected_penalty)
    torch.testing.assert_close(total, expected_policy + penalty_coef * expected_penalty)
    torch.testing.assert_close(clip_fraction, expected_clip_fraction)


def test_native_accepts_per_sequence_advantages_without_expansion():
    ratio = torch.tensor([[0.70, 1.10, 1.30], [0.80, 1.00, 1.40]])
    sequence_advantages = torch.tensor([2.0, -3.0])
    mask = torch.tensor([[True, False, True], [True, True, True]])

    _, policy, _, _ = NativeRatioClipAggregateOp()(
        ratio,
        sequence_advantages,
        mask,
        clip_low=0.1,
        clip_high=0.2,
    )

    token_advantages = sequence_advantages[:, None].expand_as(ratio)
    expected_terms = -torch.minimum(
        ratio * token_advantages,
        ratio.clamp(0.9, 1.2) * token_advantages,
    )
    torch.testing.assert_close(policy, expected_terms[mask].mean())


def test_native_total_loss_gradients_match_reference():
    ratio = torch.tensor([[0.70, 0.95, 1.30], [0.75, 1.05, 1.45]], requires_grad=True)
    ratio_ref = ratio.detach().clone().requires_grad_(True)
    advantages = torch.tensor([[2.0, -1.0, 0.5], [-3.0, 1.5, -0.25]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    penalty = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], requires_grad=True)
    penalty_ref = penalty.detach().clone().requires_grad_(True)

    total, _, _, _ = NativeRatioClipAggregateOp()(
        ratio,
        advantages,
        mask,
        clip_low=0.1,
        clip_high=0.2,
        penalty_terms=penalty,
        penalty_coef=0.05,
    )
    total.backward()

    reference_terms = -torch.minimum(
        ratio_ref * advantages,
        ratio_ref.clamp(0.9, 1.2) * advantages,
    )
    reference_total = reference_terms[mask].mean() + 0.05 * penalty_ref[mask].mean()
    reference_total.backward()

    torch.testing.assert_close(ratio.grad, ratio_ref.grad)
    torch.testing.assert_close(penalty.grad, penalty_ref.grad)


def test_native_empty_mask_returns_finite_zeros_and_zero_gradients():
    ratio = torch.tensor([[0.5, 1.5]], requires_grad=True)
    penalty = torch.tensor([[2.0, 3.0]], requires_grad=True)

    outputs = NativeRatioClipAggregateOp()(
        ratio,
        torch.tensor([1.0]),
        torch.zeros_like(ratio, dtype=torch.bool),
        penalty_terms=penalty,
        penalty_coef=0.2,
    )

    for output in outputs:
        torch.testing.assert_close(output, torch.zeros_like(output))
    outputs[0].backward()
    torch.testing.assert_close(ratio.grad, torch.zeros_like(ratio))
    torch.testing.assert_close(penalty.grad, torch.zeros_like(penalty))


def test_native_rejects_invalid_public_contract():
    op = NativeRatioClipAggregateOp()
    ratio = torch.ones(2, 3)
    advantages = torch.ones(2)
    mask = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="mask shape"):
        op(ratio, advantages, mask[:, :2])
    with pytest.raises(ValueError, match="penalty_terms shape"):
        op(ratio, advantages, mask, penalty_terms=torch.ones(2, 2))
    with pytest.raises(ValueError, match="clip_low"):
        op(ratio, advantages, mask, clip_low=1.0)
    with pytest.raises(TypeError, match="mask"):
        op(ratio, advantages, mask.float())
    with pytest.raises(ValueError, match="detached"):
        op(ratio, advantages.requires_grad_(True), mask)
    with pytest.raises(TypeError, match="floating-point"):
        op(ratio.to(torch.int64), advantages, mask)
    with pytest.raises(ValueError, match="at least one"):
        op(torch.empty(0), torch.empty(0), torch.empty(0, dtype=torch.bool))


@requires_triton_cuda
def test_triton_forward_matches_native_with_sequence_advantages_and_penalty():
    generator = torch.Generator(device="cuda").manual_seed(17)
    ratio = torch.exp(torch.randn(37, 129, generator=generator, device="cuda") * 0.3)
    advantages = torch.randn(37, generator=generator, device="cuda")
    mask = torch.rand(37, 129, generator=generator, device="cuda") > 0.2
    penalty = torch.rand(37, 129, generator=generator, device="cuda")
    kwargs = dict(
        clip_low=0.1,
        clip_high=0.25,
        penalty_terms=penalty,
        penalty_coef=0.07,
    )

    expected = NativeRatioClipAggregateOp()(ratio, advantages, mask, **kwargs)
    actual = TritonRatioClipAggregateOp()(ratio, advantages, mask, **kwargs)

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


@requires_triton_cuda
def test_triton_backward_matches_native_for_all_differentiable_outputs():
    generator = torch.Generator(device="cuda").manual_seed(23)
    base_ratio = torch.exp(torch.randn(19, 257, generator=generator, device="cuda") * 0.4)
    advantages = torch.randn(19, 257, generator=generator, device="cuda")
    mask = torch.rand(19, 257, generator=generator, device="cuda") > 0.25
    base_penalty = torch.rand(19, 257, generator=generator, device="cuda")
    kwargs = dict(clip_low=0.12, clip_high=0.28, penalty_coef=0.03)

    native_ratio = base_ratio.clone().requires_grad_(True)
    native_penalty = base_penalty.clone().requires_grad_(True)
    native_outputs = NativeRatioClipAggregateOp()(
        native_ratio,
        advantages,
        mask,
        penalty_terms=native_penalty,
        **kwargs,
    )
    (1.7 * native_outputs[0] + 0.3 * native_outputs[1] - 0.2 * native_outputs[2]).backward()

    triton_ratio = base_ratio.clone().requires_grad_(True)
    triton_penalty = base_penalty.clone().requires_grad_(True)
    triton_outputs = TritonRatioClipAggregateOp()(
        triton_ratio,
        advantages,
        mask,
        penalty_terms=triton_penalty,
        **kwargs,
    )
    (1.7 * triton_outputs[0] + 0.3 * triton_outputs[1] - 0.2 * triton_outputs[2]).backward()

    torch.testing.assert_close(triton_ratio.grad, native_ratio.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(triton_penalty.grad, native_penalty.grad, atol=1e-7, rtol=1e-6)


@requires_triton_cuda
def test_triton_backward_supports_policy_output_without_materialized_grads():
    ratio = torch.tensor([[0.7, 0.95, 1.3]], device="cuda", requires_grad=True)
    ratio_ref = ratio.detach().clone().requires_grad_(True)
    advantages = torch.tensor([[2.0, -1.0, 0.5]], device="cuda")
    mask = torch.tensor([[True, True, True]], device="cuda")

    TritonRatioClipAggregateOp()(ratio, advantages, mask)[1].backward()
    NativeRatioClipAggregateOp()(ratio_ref, advantages, mask)[1].backward()

    torch.testing.assert_close(ratio.grad, ratio_ref.grad)


@requires_triton_cuda
def test_triton_output_mutation_does_not_invalidate_backward_state():
    ratio = torch.tensor([[0.7, 0.95, 1.3]], device="cuda", requires_grad=True)
    advantages = torch.tensor([[2.0, -1.0, 0.5]], device="cuda")
    mask = torch.tensor([[True, True, True]], device="cuda")

    total, policy, _, _ = TritonRatioClipAggregateOp()(ratio, advantages, mask)
    with torch.no_grad():
        total.add_(1.0)
    policy.backward()

    assert ratio.grad is not None


@requires_triton_cuda
def test_triton_backward_preserves_mixed_input_gradient_dtypes():
    ratio = torch.tensor([[0.7, 0.95, 1.3]], device="cuda", requires_grad=True)
    penalty = torch.tensor(
        [[0.2, 0.3, 0.4]], device="cuda", dtype=torch.float16, requires_grad=True
    )
    advantages = torch.tensor([[2.0, -1.0, 0.5]], device="cuda")
    mask = torch.tensor([[True, True, True]], device="cuda")

    TritonRatioClipAggregateOp()(
        ratio,
        advantages,
        mask,
        penalty_terms=penalty,
        penalty_coef=0.04,
    )[0].backward()

    assert ratio.grad.dtype == ratio.dtype
    assert penalty.grad.dtype == penalty.dtype
    torch.testing.assert_close(
        penalty.grad,
        torch.full_like(penalty, 0.04 / penalty.numel()),
    )


@requires_triton_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_triton_matches_native_without_penalty_for_supported_dtypes(dtype):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")
    generator = torch.Generator(device="cuda").manual_seed(29)
    ratio_storage = torch.exp(torch.randn(11, 258, generator=generator, device="cuda") * 0.25).to(
        dtype
    )
    advantage_storage = torch.randn(11, 258, generator=generator, device="cuda").to(dtype)
    mask_storage = torch.rand(11, 258, generator=generator, device="cuda") > 0.15
    ratio = ratio_storage[:, ::2]
    advantages = advantage_storage[:, ::2]
    mask = mask_storage[:, ::2]

    expected = NativeRatioClipAggregateOp()(
        ratio,
        advantages,
        mask,
        clip_low=0.2,
        clip_high=0.3,
    )
    actual = TritonRatioClipAggregateOp()(
        ratio,
        advantages,
        mask,
        clip_low=0.2,
        clip_high=0.3,
    )

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)


@requires_triton_cuda
def test_triton_empty_mask_returns_finite_zeros_and_zero_gradients():
    ratio = torch.tensor([[0.5, 1.5]], device="cuda", requires_grad=True)
    penalty = torch.tensor([[2.0, 3.0]], device="cuda", requires_grad=True)
    outputs = TritonRatioClipAggregateOp()(
        ratio,
        torch.tensor([1.0], device="cuda"),
        torch.zeros_like(ratio, dtype=torch.bool),
        penalty_terms=penalty,
        penalty_coef=0.2,
    )

    for output in outputs:
        torch.testing.assert_close(output, torch.zeros_like(output))
    outputs[0].backward()
    torch.testing.assert_close(ratio.grad, torch.zeros_like(ratio))
    torch.testing.assert_close(penalty.grad, torch.zeros_like(penalty))


@requires_triton_cuda
@pytest.mark.parametrize("elements", [65536, 65537])
def test_triton_matches_native_at_single_and_staged_reduction_boundary(elements):
    generator = torch.Generator(device="cuda").manual_seed(elements)
    ratio = torch.exp(torch.randn(elements, generator=generator, device="cuda") * 0.3)
    ratio.requires_grad_(True)
    advantages = torch.randn(elements, generator=generator, device="cuda")
    mask = torch.rand(elements, generator=generator, device="cuda") > 0.1
    penalty = torch.rand(elements, generator=generator, device="cuda")
    kwargs = dict(
        clip_low=0.15,
        clip_high=0.25,
        penalty_terms=penalty,
        penalty_coef=0.04,
    )

    expected = NativeRatioClipAggregateOp()(ratio, advantages, mask, **kwargs)
    actual = TritonRatioClipAggregateOp()(ratio, advantages, mask, **kwargs)

    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)
    assert not expected[3].requires_grad
    assert not actual[3].requires_grad


@requires_triton_cuda
def test_triton_staged_reduction_is_bitwise_deterministic():
    generator = torch.Generator(device="cuda").manual_seed(41)
    elements = 65537
    ratio = torch.exp(torch.randn(elements, generator=generator, device="cuda") * 0.3)
    advantages = torch.randn(elements, generator=generator, device="cuda")
    mask = torch.rand(elements, generator=generator, device="cuda") > 0.1
    penalty = torch.rand(elements, generator=generator, device="cuda")
    op = TritonRatioClipAggregateOp()

    expected = op(ratio, advantages, mask, penalty_terms=penalty, penalty_coef=0.04)
    for _ in range(5):
        actual = op(ratio, advantages, mask, penalty_terms=penalty, penalty_coef=0.04)
        assert all(torch.equal(got, want) for got, want in zip(actual, expected, strict=True))


def test_registry_dispatches_ratio_clip_aggregate():
    from rl_engine.kernels.registry import kernel_registry

    op = kernel_registry.get_op("ratio_clip_aggregate")
    if _HAS_TRITON and torch.cuda.is_available():
        assert TritonRatioClipAggregateOp is not None
        assert isinstance(op, TritonRatioClipAggregateOp)
    else:
        assert isinstance(op, NativeRatioClipAggregateOp)
