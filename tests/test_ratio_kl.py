# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.pytorch.loss.ratio_kl import NativeRatioKLOp
from rl_engine.kernels.ops.triton.loss.ratio_kl import TritonRatioKLOp
from rl_engine.testing import make_synthetic_rl_kernel_batch, selected_logprobs_reference

try:
    import triton  # noqa: F401

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

requires_triton_cuda = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="Triton ratio/KL op requires a CUDA device and Triton.",
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


def _logits(batch, seed, *, vocab=_VOCAB, device="cpu"):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(batch.batch_size, batch.completion_len, vocab, generator=gen, device=device)


def _inputs(seed, *, device="cpu", valid_density=0.9, vocab=_VOCAB):
    """A full ratio/KL input set: (policy_logits, ref_logits, action_ids, mask, old_logps)."""
    batch = _batch(seed=seed, device=device, valid_density=valid_density)
    policy_logits = _logits(batch, seed=seed + 100, vocab=vocab, device=device)
    ref_logits = _logits(batch, seed=seed + 200, vocab=vocab, device=device)
    return (
        policy_logits,
        ref_logits,
        batch.token_ids,
        batch.completion_mask,
        batch.old_logps,
    )


def _reference_ratio_kl(policy_logits, ref_logits, action_ids, mask, old_logps):
    """Independent reference using the testing log-prob helper + mask-before-exp."""
    logp_policy = selected_logprobs_reference(policy_logits, action_ids).float()
    logp_ref = selected_logprobs_reference(ref_logits, action_ids).float()
    bool_mask = mask.to(torch.bool)
    delta = (logp_policy - old_logps.float()).masked_fill(~bool_mask, 0.0)
    diff = (logp_ref - logp_policy).masked_fill(~bool_mask, 0.0)
    return torch.exp(delta), torch.exp(diff) - diff - 1.0


# pure-PyTorch reference op
def test_native_matches_reference():
    op = NativeRatioKLOp()
    inputs = _inputs(seed=0)
    ratio, kl = op(*inputs)
    exp_ratio, exp_kl = _reference_ratio_kl(*inputs)
    assert torch.allclose(ratio, exp_ratio, atol=1e-6)
    assert torch.allclose(kl, exp_kl, atol=1e-6)


def test_native_masked_tokens_are_neutral():
    op = NativeRatioKLOp()
    *_, mask, _ = inputs = _inputs(seed=1, valid_density=0.6)
    ratio, kl = op(*inputs)
    inactive = ~mask.to(torch.bool)
    # mask-before-exp convention: ratio = exp(0) = 1, kl = 0 on inactive tokens.
    assert torch.allclose(ratio[inactive], torch.ones_like(ratio[inactive]))
    assert torch.all(kl[inactive] == 0.0)


def test_native_ratio_is_one_when_old_equals_policy():
    op = NativeRatioKLOp()
    policy_logits, ref_logits, action_ids, mask, _ = _inputs(seed=2, valid_density=1.0)
    old = selected_logprobs_reference(policy_logits, action_ids).float()
    ratio, _ = op(policy_logits, ref_logits, action_ids, mask, old)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5)


def test_native_gradient_flows_to_policy_logits():
    op = NativeRatioKLOp()
    policy_logits, ref_logits, action_ids, mask, old = _inputs(seed=3)
    policy_logits = policy_logits.clone().requires_grad_(True)
    ref_logits = ref_logits.clone().requires_grad_(True)

    ratio, kl = op(policy_logits, ref_logits, action_ids, mask, old)
    (ratio.sum() + kl.sum()).backward()

    assert policy_logits.grad is not None
    assert torch.isfinite(policy_logits.grad).all()
    # Reference is frozen: no gradient should reach ref_logits.
    assert ref_logits.grad is None


# Triton fused op (validated against the native reference)
@requires_triton_cuda
@pytest.mark.parametrize("vocab", [_VOCAB, 50257])
def test_triton_forward_matches_native(vocab):
    native = NativeRatioKLOp()
    fused = TritonRatioKLOp()
    inputs = _inputs(seed=4, device="cuda", vocab=vocab)
    r_t, k_t = fused(*inputs)
    r_n, k_n = native(*inputs)
    assert torch.allclose(r_t, r_n, atol=1e-4, rtol=1e-4)
    assert torch.allclose(k_t, k_n, atol=1e-4, rtol=1e-4)


@requires_triton_cuda
def test_triton_backward_matches_native():
    native = NativeRatioKLOp()
    fused = TritonRatioKLOp()
    policy_logits, ref_logits, action_ids, mask, old = _inputs(seed=5, device="cuda")
    gr = torch.randn(policy_logits.shape[:-1], device="cuda")
    gk = torch.randn(policy_logits.shape[:-1], device="cuda")

    pol_t = policy_logits.clone().requires_grad_(True)
    r_t, k_t = fused(pol_t, ref_logits, action_ids, mask, old)
    (r_t * gr + k_t * gk).sum().backward()

    pol_n = policy_logits.clone().requires_grad_(True)
    r_n, k_n = native(pol_n, ref_logits, action_ids, mask, old)
    (r_n * gr + k_n * gk).sum().backward()

    assert pol_t.grad is not None
    assert torch.isfinite(pol_t.grad).all()
    assert torch.allclose(pol_t.grad, pol_n.grad, atol=1e-4, rtol=1e-4)


@requires_triton_cuda
def test_triton_no_grad_to_ref():
    """The reference is frozen: the fused backward must not reach ref_logits."""
    fused = TritonRatioKLOp()
    pol, ref, act, mask, old = _inputs(seed=8, device="cuda")
    pol = pol.clone().requires_grad_(True)
    ref = ref.clone().requires_grad_(True)
    r, k = fused(pol, ref, act, mask, old)
    (r.sum() + k.sum()).backward()
    assert pol.grad is not None
    assert ref.grad is None


@requires_triton_cuda
def test_triton_backward_with_grad_scaling():
    """A non-unit upstream gradient must scale the policy-logits gradient linearly."""
    fused = TritonRatioKLOp()
    policy_logits, ref_logits, action_ids, mask, old = _inputs(seed=6, device="cuda")

    pol1 = policy_logits.clone().requires_grad_(True)
    r1, k1 = fused(pol1, ref_logits, action_ids, mask, old)
    (r1.sum() + k1.sum()).backward()

    pol2 = policy_logits.clone().requires_grad_(True)
    r2, k2 = fused(pol2, ref_logits, action_ids, mask, old)
    (3.0 * (r2.sum() + k2.sum())).backward()

    assert torch.allclose(pol2.grad, 3.0 * pol1.grad, atol=1e-4, rtol=1e-4)


@requires_triton_cuda
def test_triton_masked_tokens_do_not_affect_active():
    """Garbage logits at masked positions must not change active outputs."""
    fused = TritonRatioKLOp()
    policy_logits, ref_logits, action_ids, mask, old = _inputs(
        seed=7, device="cuda", valid_density=0.7
    )
    base_r, base_k = fused(policy_logits, ref_logits, action_ids, mask, old)

    inactive = ~mask.to(torch.bool)
    pert = policy_logits.clone()
    pert[inactive] = 1000.0
    pert_r, pert_k = fused(pert, ref_logits, action_ids, mask, old)

    active = mask.to(torch.bool)
    assert torch.allclose(base_r[active], pert_r[active], atol=1e-5)
    assert torch.allclose(base_k[active], pert_k[active], atol=1e-5)
    assert torch.allclose(pert_r[inactive], torch.ones_like(pert_r[inactive]))


@requires_triton_cuda
def test_triton_handles_oob_action_ids():
    """Out-of-range ids at masked positions must not fault; the fused op clamps
    them (like the native op) and stays finite, agreeing on active outputs."""
    native = NativeRatioKLOp()
    fused = TritonRatioKLOp()
    pol, ref, act, mask, old = _inputs(seed=9, device="cuda", valid_density=0.7)
    act = act.clone()
    inactive = ~mask.to(torch.bool)
    act[inactive] = _VOCAB + 999  # garbage id at masked positions

    r_t, k_t = fused(pol, ref, act, mask, old)
    r_n, k_n = native(pol, ref, act, mask, old)

    active = mask.to(torch.bool)
    assert torch.isfinite(r_t).all() and torch.isfinite(k_t).all()
    assert torch.allclose(r_t[active], r_n[active], atol=1e-4, rtol=1e-4)
    assert torch.allclose(k_t[active], k_n[active], atol=1e-4, rtol=1e-4)


# Registry dispatch (device-dependent backend selection)
def test_registry_dispatches_ratio_kl():
    from rl_engine.kernels.registry import kernel_registry

    op = kernel_registry.get_op("ratio_kl")
    if _HAS_TRITON and torch.cuda.is_available():
        assert isinstance(op, TritonRatioKLOp)
    else:
        assert isinstance(op, NativeRatioKLOp)
