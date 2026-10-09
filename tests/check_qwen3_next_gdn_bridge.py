# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Explicit subprocess provider tests; never import vLLM into ordinary pytest."""

import pytest
import torch

from rl_engine.integrations.qwen3_next_gdn import _provider, packed_decode_training_step
from rl_engine.kernels.ops.pytorch.linear_attn import GatedDeltaRuleRecurrentStepOp
from rl_engine.testing.tensor_identity import assert_tensor_bitwise_equal, tensor_bitwise_equal

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _inputs(seed=1):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=gen)

    return [
        rand(2, 512),
        rand(2, 2),
        rand(2, 2),
        rand(2, dtype=torch.float32),
        rand(2, dtype=torch.float32),
        rand(3, 2, 128, 128, dtype=torch.float32) * 0.01,
    ]


def test_bridge_forward_is_provider_exact_and_does_not_mutate_inputs():
    inputs = _inputs()
    state_before = inputs[-1].clone()
    indices = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    out, state = packed_decode_training_step(*inputs, indices, num_k_heads=1)
    expected = torch.empty_like(out)
    expected_state = state_before.clone()
    _provider()(
        *inputs[:-1], 128**-0.5, expected_state, expected, indices, use_qk_l2norm_in_kernel=True
    )
    assert_tensor_bitwise_equal(out, expected)
    assert_tensor_bitwise_equal(state, expected_state)
    assert tensor_bitwise_equal(inputs[-1], state_before)


def test_bridge_keeps_prompt_state_gradient_through_response():
    inputs = [x.requires_grad_() for x in _inputs()]
    indices = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    _, prompt_state = packed_decode_training_step(*inputs, indices, num_k_heads=1)
    prompt_state.retain_grad()
    response = _inputs(seed=2)[0].requires_grad_()
    out, final_state = packed_decode_training_step(
        response, *inputs[1:5], prompt_state, indices, num_k_heads=1
    )
    (out.float().square().sum() + final_state.square().sum() * 0.01).backward()
    assert prompt_state.grad is not None and prompt_state.grad.abs().max() > 0
    for tensor in [*inputs, response]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().max() > 0


def test_bridge_backward_matches_recomputed_recurrence_vjp():
    inputs = [x.requires_grad_() for x in _inputs()]
    indices = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    out, state = packed_decode_training_step(*inputs, indices, num_k_heads=1)
    grad_out = torch.ones_like(out) * 0.125
    grad_state = torch.ones_like(state) * 0.001
    torch.autograd.backward((out, state), (grad_out, grad_state))
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    ref_out, ref_state = GatedDeltaRuleRecurrentStepOp()(
        *refs, indices, scale=128**-0.5, num_k_heads=1
    )
    torch.autograd.backward((ref_out, ref_state), (grad_out, grad_state))
    for actual, ref in zip(inputs, refs):
        assert tensor_bitwise_equal(actual.grad, ref.grad)


def test_initial_state_vjp_matches_provider_directional_difference():
    inputs = _inputs()
    inputs[-1].requires_grad_()
    indices = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    _, output = packed_decode_training_step(*inputs, indices, num_k_heads=1)
    direction = torch.ones_like(inputs[-1]) * 0.03125
    probe = torch.linspace(-0.01, 0.01, output.numel(), device="cuda").reshape_as(output)
    (output.double() * probe.double()).sum().backward()
    analytic = (inputs[-1].grad.double() * direction.double()).sum()
    eps = 0.01
    with torch.no_grad():
        _, plus = packed_decode_training_step(
            *inputs[:-1], inputs[-1] + eps * direction, indices, num_k_heads=1
        )
        _, minus = packed_decode_training_step(
            *inputs[:-1], inputs[-1] - eps * direction, indices, num_k_heads=1
        )
        numerical = ((plus.double() - minus.double()) * probe.double()).sum() / (2 * eps)
    torch.testing.assert_close(analytic, numerical, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("empty", [False, True])
def test_inactive_state_is_identity_in_backward(empty):
    inputs = _inputs()
    if empty:
        inputs[:3] = [x[:0] for x in inputs[:3]]
    inputs = [x.requires_grad_() for x in inputs]
    indices = torch.zeros(inputs[0].shape[0], device="cuda", dtype=torch.int32)
    out, state = packed_decode_training_step(*inputs, indices, num_k_heads=1)
    (out.float().sum() + state.sum()).backward()
    assert tensor_bitwise_equal(inputs[-1].grad, torch.ones_like(state))
    for tensor in inputs[:-1]:
        assert torch.count_nonzero(tensor.grad) == 0
