# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.pytorch.loss.grpo_loss import NativeGRPOLossOp
from rl_engine.kernels.ops.triton.loss.grpo_loss import TritonGRPOLossOp
from rl_engine.testing import (
    compute_policy_ratio,
    compute_reference_kl,
    make_synthetic_rl_kernel_batch,
    masked_mean,
    selected_logprobs_reference,
)

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton GRPO loss requires a CUDA device and Triton.",
)

_NUM_PROMPTS = 3
_SPP = 4
_COMP_LEN = 6
_VOCAB = 64


# Shared helpers
def _batch(seed=0, *, device="cpu", valid_density=0.9):
    return make_synthetic_rl_kernel_batch(
        num_prompts=_NUM_PROMPTS,
        samples_per_prompt=_SPP,
        prompt_len=0,
        completion_len=_COMP_LEN,
        vocab_size=_VOCAB,
        valid_density=valid_density,
        device=device,
        seed=seed,
    )


def _logits(batch, seed, *, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(batch.batch_size, batch.completion_len, _VOCAB, generator=gen, device=device)


def _logit_pair(batch, seed, *, device="cpu"):
    """(policy_logits, ref_logits) for a batch."""
    return _logits(batch, seed, device=device), _logits(batch, seed + 1, device=device)


def _reference_group_advantages(rewards, samples_per_prompt, eps=1e-6):
    """Mirror of examples.grpo_single_gpu.make_group_advantages normalization."""
    grouped = rewards.view(-1, samples_per_prompt)
    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std = grouped.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return ((grouped - group_mean) / group_std).reshape(-1)


def _reference_loss(batch, policy_logits, ref_logits, advantages, clip_eps, beta):
    """Independent reference: logits -> selected logp -> clipped surrogate + KL."""
    current = selected_logprobs_reference(policy_logits, batch.token_ids).float()
    ref = selected_logprobs_reference(ref_logits, batch.token_ids).float()
    mask = batch.completion_mask
    ratio = compute_policy_ratio(current, batch.old_logps, mask)
    unclipped = ratio * advantages.float()
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.float()
    policy_loss_terms = -torch.minimum(unclipped, clipped)
    kl_terms = compute_reference_kl(current, ref, mask)
    policy_loss = masked_mean(policy_loss_terms, mask)
    kl = masked_mean(kl_terms, mask)
    return policy_loss + beta * kl, policy_loss, kl


def _adv_tokens(batch):
    sample_adv = _reference_group_advantages(batch.rewards, _SPP)
    return (
        sample_adv[:, None]
        .expand_as(batch.completion_mask)
        .clone()
        .masked_fill(~batch.completion_mask, 0.0)
    )


# pure-PyTorch reference op (group advantages)
def test_group_advantages_matches_reference_uniform():
    op = NativeGRPOLossOp()
    rewards = _batch(seed=1).rewards
    got = op.group_advantages(rewards, samples_per_prompt=_SPP)
    expected = _reference_group_advantages(rewards, samples_per_prompt=_SPP)
    assert torch.allclose(got, expected, atol=1e-6)


def test_boundaries_agree_with_uniform():
    op = NativeGRPOLossOp()
    rewards = _batch(seed=2).rewards
    base = op.group_advantages(rewards, samples_per_prompt=_SPP)
    bounds = list(range(0, _NUM_PROMPTS * _SPP + 1, _SPP))
    by_bounds = op.group_advantages(rewards, group_boundaries=bounds)
    assert torch.allclose(base, by_bounds, atol=1e-6)


def test_variable_group_boundaries():
    op = NativeGRPOLossOp()
    rewards = torch.tensor([1.0, 3.0, 10.0, 20.0, 30.0])
    got = op.group_advantages(rewards, group_boundaries=[0, 2, 5])
    g0 = torch.tensor([-1.0, 1.0])
    g1 = (torch.tensor([10.0, 20.0, 30.0]) - 20.0) / (200.0 / 3.0) ** 0.5
    expected = torch.cat([g0, g1])
    assert torch.allclose(got, expected, atol=1e-5)


def test_requires_exactly_one_group_spec():
    op = NativeGRPOLossOp()
    rewards = _batch(seed=6).rewards
    with pytest.raises(ValueError):
        op.group_advantages(rewards)
    with pytest.raises(ValueError):
        op.group_advantages(rewards, samples_per_prompt=_SPP, group_boundaries=[0, 4, 8, 12])


# pure-PyTorch reference op (loss from logits)
def test_forward_loss_matches_reference():
    op = NativeGRPOLossOp()
    batch = _batch(seed=0)
    policy_logits, ref_logits = _logit_pair(batch, seed=100)
    clip_eps, beta = 0.2, 0.01

    loss, policy_loss, kl = op.forward(
        policy_logits,
        ref_logits,
        batch.token_ids,
        batch.old_logps,
        batch.rewards,
        batch.completion_mask,
        clip_eps=clip_eps,
        beta=beta,
        samples_per_prompt=_SPP,
    )

    exp_loss, exp_policy, exp_kl = _reference_loss(
        batch, policy_logits, ref_logits, _adv_tokens(batch), clip_eps, beta
    )
    assert torch.allclose(loss, exp_loss, atol=1e-5)
    assert torch.allclose(policy_loss, exp_policy, atol=1e-5)
    assert torch.allclose(kl, exp_kl, atol=1e-5)


def test_gradient_flows_to_policy_logits():
    op = NativeGRPOLossOp()
    batch = _batch(seed=4)
    policy_logits, ref_logits = _logit_pair(batch, seed=104)
    policy_logits = policy_logits.clone().requires_grad_(True)
    ref_logits = ref_logits.clone().requires_grad_(True)

    loss, _, _ = op.forward(
        policy_logits,
        ref_logits,
        batch.token_ids,
        batch.old_logps,
        batch.rewards,
        batch.completion_mask,
        clip_eps=0.2,
        beta=0.01,
        samples_per_prompt=_SPP,
    )
    loss.backward()

    assert policy_logits.grad is not None
    assert torch.isfinite(policy_logits.grad).all()
    # Reference is frozen: no gradient should reach ref_logits.
    assert ref_logits.grad is None
    # Masked-out tokens (whole logits rows) must receive zero gradient.
    assert torch.all(policy_logits.grad[~batch.completion_mask.bool()] == 0.0)


# Triton fused op (validated against the native reference)
@requires_triton_cuda
def test_triton_forward_matches_native():
    native = NativeGRPOLossOp()
    fused = TritonGRPOLossOp()
    batch = _batch(seed=0, device="cuda")
    policy_logits, ref_logits = _logit_pair(batch, seed=100, device="cuda")
    args = (
        policy_logits,
        ref_logits,
        batch.token_ids,
        batch.old_logps,
        batch.rewards,
        batch.completion_mask,
    )
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)

    n_loss, n_policy, n_kl = native.forward(*args, **kwargs)
    t_loss, t_policy, t_kl = fused.forward(*args, **kwargs)

    assert torch.allclose(t_loss, n_loss, atol=1e-4, rtol=1e-4)
    assert torch.allclose(t_policy, n_policy, atol=1e-4, rtol=1e-4)
    assert torch.allclose(t_kl, n_kl, atol=1e-4, rtol=1e-4)


@requires_triton_cuda
def test_triton_backward_matches_native():
    native = NativeGRPOLossOp()
    fused = TritonGRPOLossOp()
    batch = _batch(seed=7, device="cuda")
    policy_logits, ref_logits = _logit_pair(batch, seed=107, device="cuda")
    rest = (ref_logits, batch.token_ids, batch.old_logps, batch.rewards, batch.completion_mask)
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)

    pol_n = policy_logits.clone().requires_grad_(True)
    native.forward(pol_n, *rest, **kwargs)[0].backward()

    pol_t = policy_logits.clone().requires_grad_(True)
    fused.forward(pol_t, *rest, **kwargs)[0].backward()

    assert pol_t.grad is not None
    assert torch.allclose(pol_t.grad, pol_n.grad, atol=1e-4, rtol=1e-4)
    assert torch.all(pol_t.grad[~batch.completion_mask.bool()] == 0.0)


@requires_triton_cuda
def test_triton_backward_with_grad_scaling():
    """A non-unit upstream gradient must scale the policy-logits gradient linearly."""
    fused = TritonGRPOLossOp()
    batch = _batch(seed=3, device="cuda")
    policy_logits, ref_logits = _logit_pair(batch, seed=103, device="cuda")
    rest = (ref_logits, batch.token_ids, batch.old_logps, batch.rewards, batch.completion_mask)
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)

    pol1 = policy_logits.clone().requires_grad_(True)
    fused.forward(pol1, *rest, **kwargs)[0].backward()

    pol2 = policy_logits.clone().requires_grad_(True)
    (3.0 * fused.forward(pol2, *rest, **kwargs)[0]).backward()

    assert torch.allclose(pol2.grad, 3.0 * pol1.grad, atol=1e-4, rtol=1e-4)


@requires_triton_cuda
def test_triton_group_advantages_matches_native():
    native = NativeGRPOLossOp()
    fused = TritonGRPOLossOp()
    rewards = _batch(seed=5, device="cuda").rewards
    got = fused.group_advantages(rewards, samples_per_prompt=_SPP)
    expected = native.group_advantages(rewards, samples_per_prompt=_SPP)
    assert torch.allclose(got, expected, atol=1e-5)
    got_b = fused.group_advantages(rewards, group_boundaries=[0, 5, 12])
    exp_b = native.group_advantages(rewards, group_boundaries=[0, 5, 12])
    assert torch.allclose(got_b, exp_b, atol=1e-5)


@requires_triton_cuda
def test_triton_apply_with_per_sequence_advantages_matches_native():
    native = NativeGRPOLossOp()
    fused = TritonGRPOLossOp()
    batch = _batch(seed=11, device="cuda")
    policy_logits, ref_logits = _logit_pair(batch, seed=111, device="cuda")
    sample_adv = native.group_advantages(batch.rewards, samples_per_prompt=_SPP)
    args = (
        policy_logits,
        ref_logits,
        batch.token_ids,
        batch.old_logps,
        sample_adv,
        batch.completion_mask,
    )
    kwargs = dict(clip_eps=0.2, beta=0.05)

    n_loss, _, _ = native.apply(*args, **kwargs)
    t_loss, _, _ = fused.apply(*args, **kwargs)
    assert torch.allclose(t_loss, n_loss, atol=1e-4, rtol=1e-4)


# Loss step: masked-token invariance and a gradient step
def _perturb_inactive_logits(batch, policy_logits):
    """Set garbage at masked positions' logits; the loss must ignore them."""
    pol = policy_logits.clone()
    pol[~batch.completion_mask.bool()] = 1000.0
    return pol


def test_masked_tokens_do_not_affect_native_loss():
    op = NativeGRPOLossOp()
    batch = _batch(seed=8, valid_density=0.75)
    policy_logits, ref_logits = _logit_pair(batch, seed=108)
    args = (ref_logits, batch.token_ids, batch.old_logps, batch.rewards, batch.completion_mask)
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)

    base, _, _ = op.forward(policy_logits, *args, **kwargs)
    pert, _, _ = op.forward(_perturb_inactive_logits(batch, policy_logits), *args, **kwargs)
    assert torch.allclose(base, pert)


@requires_triton_cuda
def test_masked_tokens_do_not_affect_triton_loss():
    fused = TritonGRPOLossOp()
    batch = _batch(seed=8, device="cuda", valid_density=0.75)
    policy_logits, ref_logits = _logit_pair(batch, seed=108, device="cuda")
    args = (ref_logits, batch.token_ids, batch.old_logps, batch.rewards, batch.completion_mask)
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)

    base, _, _ = fused.forward(policy_logits, *args, **kwargs)
    pert, _, _ = fused.forward(_perturb_inactive_logits(batch, policy_logits), *args, **kwargs)
    assert torch.allclose(base, pert, atol=1e-5)


def _descend(op, batch, policy_logits, ref_logits, *, steps=5, lr=0.05):
    rest = (ref_logits, batch.token_ids, batch.old_logps, batch.rewards, batch.completion_mask)
    kwargs = dict(clip_eps=0.2, beta=0.05, samples_per_prompt=_SPP)
    initial = op.forward(policy_logits, *rest, **kwargs)[0]
    params = policy_logits.clone().requires_grad_(True)
    for _ in range(steps):
        loss, _, _ = op.forward(params, *rest, **kwargs)
        (grad,) = torch.autograd.grad(loss, params)
        params = (params - lr * grad).detach().requires_grad_(True)
    final = op.forward(params, *rest, **kwargs)[0]
    return initial, final


def test_grpo_gradient_step_reduces_loss():
    """Full loss step: forward -> backward -> SGD on the policy logits lowers the loss."""
    op = NativeGRPOLossOp()
    batch = _batch(seed=9)
    policy_logits, ref_logits = _logit_pair(batch, seed=109)
    initial, final = _descend(op, batch, policy_logits, ref_logits)
    assert final.item() < initial.item()


@requires_triton_cuda
def test_triton_grpo_gradient_step_reduces_loss():
    fused = TritonGRPOLossOp()
    batch = _batch(seed=9, device="cuda")
    policy_logits, ref_logits = _logit_pair(batch, seed=109, device="cuda")
    initial, final = _descend(fused, batch, policy_logits, ref_logits)
    assert final.item() < initial.item()


# Registry dispatch (device-dependent backend selection)
def test_registry_dispatches_grpo_loss():
    from rl_engine.kernels.registry import kernel_registry

    op = kernel_registry.get_op("grpo_loss")
    assert hasattr(op, "forward") and hasattr(op, "group_advantages")
    if _HAS_TRITON and torch.cuda.is_available():
        assert isinstance(op, TritonGRPOLossOp)
    else:
        assert isinstance(op, NativeGRPOLossOp)
