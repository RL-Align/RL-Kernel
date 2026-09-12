# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.kernels.ops.pytorch.loss.ratio_kl import NativeRatioKLOp
from rl_engine.kernels.ops.triton.loss import ratio_kl as ratio_kl_module
from rl_engine.kernels.ops.triton.loss.ratio_kl import (
    TritonRatioKLOp,
    _ratio_kl_bwd_kernel,
    _ratio_kl_fwd_kernel,
)
from rl_engine.platforms.device import device_ctx
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
requires_nvidia_triton = pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available() and torch.version.hip is None),
    reason="Direct-output ratio/KL backward requires NVIDIA CUDA and Triton.",
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


def _logits(batch, seed, *, vocab=_VOCAB, device="cpu", dtype=torch.float32):
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        batch.batch_size,
        batch.completion_len,
        vocab,
        generator=gen,
        device=device,
        dtype=dtype,
    )


def _inputs(seed, *, device="cpu", valid_density=0.9, vocab=_VOCAB, dtype=torch.float32):
    """A full ratio/KL input set: (policy_logits, ref_logits, action_ids, mask, old_logps)."""
    batch = _batch(seed=seed, device=device, valid_density=valid_density)
    policy_logits = _logits(batch, seed=seed + 100, vocab=vocab, device=device, dtype=dtype)
    ref_logits = _logits(batch, seed=seed + 200, vocab=vocab, device=device, dtype=dtype)
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


def _kernel_case(
    dtype,
    *,
    n_rows=8,
    vocab=17,
    density=0.5,
    upstream="combined",
    masked_oob=False,
):
    torch.manual_seed(n_rows + vocab)
    policy = torch.randn(n_rows, vocab, device="cuda", dtype=dtype)
    ref = torch.randn_like(policy)
    action = torch.randint(vocab, (n_rows,), device="cuda", dtype=torch.int64)
    mask = torch.arange(n_rows, device="cuda") < round(n_rows * density)
    if masked_oob:
        action = action.clone()
        action[~mask] = vocab + 999
    mask = mask.to(torch.int32)
    old = torch.randn(n_rows, device="cuda", dtype=torch.float32)
    ratio, kl, diff, logz = (
        torch.empty(n_rows, device="cuda", dtype=torch.float32) for _ in range(4)
    )
    block_v = min(2048, triton.next_power_of_2(vocab))
    _ratio_kl_fwd_kernel[(n_rows,)](
        policy,
        ref,
        action,
        mask,
        old,
        ratio,
        kl,
        diff,
        logz,
        vocab,
        BLOCK_V=block_v,
    )
    grad_ratio = torch.randn(n_rows, 2, device="cuda", dtype=torch.float32)[:, 0]
    grad_kl = torch.randn(n_rows, 2, device="cuda", dtype=torch.float32)[:, 0]
    if upstream == "ratio":
        grad_kl.zero_()
    elif upstream == "kl":
        grad_ratio.zero_()
    elif upstream == "zero":
        grad_ratio.zero_()
        grad_kl.zero_()
    elif upstream == "large":
        grad_ratio.mul_(65536)
        grad_kl.mul_(65536)
    elif upstream == "small":
        grad_ratio.mul_(2**-14)
        grad_kl.mul_(2**-14)
    return (
        policy,
        ref,
        action,
        mask,
        old,
        ratio,
        diff,
        logz,
        grad_ratio,
        grad_kl,
        block_v,
    )


def _backward_kernel_output(case, *, write_inactive_zero):
    policy, _, action, mask, _, ratio, diff, logz, grad_ratio, grad_kl, block_v = case
    n_rows, vocab = policy.shape
    grad_ratio = grad_ratio.contiguous()
    grad_kl = grad_kl.contiguous()
    output = (
        torch.empty_like(policy)
        if write_inactive_zero
        else torch.zeros_like(policy, dtype=torch.float32)
    )
    if write_inactive_zero:
        output.fill_(torch.nan)
    _ratio_kl_bwd_kernel[(n_rows,)](
        policy,
        action,
        mask,
        ratio,
        diff,
        logz,
        grad_ratio,
        grad_kl,
        output,
        vocab,
        BLOCK_V=block_v,
        WRITE_INACTIVE_ZERO=write_inactive_zero,
    )
    return output if write_inactive_zero else output.to(policy.dtype)


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
@requires_nvidia_triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_direct_backward_matches_staging_bitwise(dtype):
    case = _kernel_case(dtype)
    staged = _backward_kernel_output(case, write_inactive_zero=False)
    direct = _backward_kernel_output(case, write_inactive_zero=True)
    _, _, _, mask, *_ = case
    inactive = ~mask.bool()

    assert torch.equal(direct, staged)
    assert torch.count_nonzero(direct[inactive]) == 0
    assert not torch.signbit(direct[inactive]).any()


@requires_nvidia_triton
def test_triton_fp32_backward_matches_staging_bitwise():
    case = _kernel_case(torch.float32, vocab=64)
    staged = _backward_kernel_output(case, write_inactive_zero=False)
    policy, ref, action, mask, old, *_, grad_ratio, grad_kl, _ = case
    production_policy = policy.clone().requires_grad_(True)

    ratio, kl = TritonRatioKLOp()(production_policy, ref, action, mask, old)
    torch.autograd.backward((ratio, kl), (grad_ratio, grad_kl))

    assert torch.equal(production_policy.grad, staged)


@requires_nvidia_triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("n_rows", "vocab", "density", "masked_oob"),
    [
        (0, 17, 0.0, False),
        (4, 17, 0.0, True),
        (8, 64, 0.1, True),
        (8, 2048, 0.5, False),
        (8, 2049, 0.9, False),
        (2, 50257, 1.0, False),
    ],
)
def test_triton_direct_backward_edge_matrix(dtype, n_rows, vocab, density, masked_oob):
    case = _kernel_case(
        dtype,
        n_rows=n_rows,
        vocab=vocab,
        density=density,
        masked_oob=masked_oob,
    )
    staged = _backward_kernel_output(case, write_inactive_zero=False)
    direct = _backward_kernel_output(case, write_inactive_zero=True)
    _, _, _, mask, *_ = case
    inactive = ~mask.bool()

    assert torch.equal(direct, staged)
    assert torch.count_nonzero(direct[inactive]) == 0
    assert not torch.signbit(direct[inactive]).any()


@requires_nvidia_triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("upstream", ["ratio", "kl", "combined", "zero", "large", "small"])
def test_triton_direct_backward_upstream_matrix(dtype, upstream):
    case = _kernel_case(dtype, vocab=64, upstream=upstream)
    staged = _backward_kernel_output(case, write_inactive_zero=False)
    direct = _backward_kernel_output(case, write_inactive_zero=True)
    *_, grad_ratio, grad_kl, _ = case

    assert not grad_ratio.is_contiguous()
    assert not grad_kl.is_contiguous()
    if upstream == "combined":
        assert grad_ratio.min() < 0 < grad_ratio.max()
        assert grad_kl.min() < 0 < grad_kl.max()
    assert torch.equal(direct, staged)

    policy, ref, action, mask, old, *_ = case
    production_policy = policy.clone().requires_grad_(True)
    ratio, kl = TritonRatioKLOp()(production_policy, ref, action, mask, old)
    torch.autograd.backward((ratio, kl), (grad_ratio, grad_kl))
    assert torch.equal(production_policy.grad, staged)


@requires_nvidia_triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_direct_backward_is_bitwise_deterministic(dtype):
    case = _kernel_case(dtype, vocab=2049, density=0.9)
    outputs = [_backward_kernel_output(case, write_inactive_zero=True) for _ in range(5)]

    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


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


@requires_nvidia_triton
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float16, 2e-3), (torch.bfloat16, 2e-2), (torch.float32, 1e-4)],
)
def test_triton_backward_dtype_paths_match_native(dtype, tolerance):
    native = NativeRatioKLOp()
    fused = TritonRatioKLOp()
    policy, ref, action, mask, old = _inputs(seed=11, device="cuda", dtype=dtype)
    grad_ratio = torch.randn(*mask.shape, 2, device="cuda")[:, :, 0]
    grad_kl = torch.randn(*mask.shape, 2, device="cuda")[:, :, 0]

    policy_t = policy.clone().requires_grad_(True)
    ref_t = ref.clone().requires_grad_(True)
    old_t = old.clone().requires_grad_(True)
    ratio_t, kl_t = fused(policy_t, ref_t, action, mask, old_t)
    torch.autograd.backward((ratio_t, kl_t), (grad_ratio, grad_kl))

    policy_n = policy.clone().requires_grad_(True)
    ratio_n, kl_n = native(policy_n, ref, action, mask, old)
    torch.autograd.backward((ratio_n, kl_n), (grad_ratio, grad_kl))

    assert policy_t.grad.dtype == dtype
    assert torch.allclose(
        policy_t.grad.float(), policy_n.grad.float(), atol=tolerance, rtol=tolerance
    )
    assert ref_t.grad is None
    assert old_t.grad is None
    assert not action.requires_grad
    assert not mask.requires_grad


@requires_nvidia_triton
@pytest.mark.parametrize(
    ("dtype", "uses_staging"),
    [(torch.float16, False), (torch.bfloat16, False), (torch.float32, True)],
)
def test_triton_backward_selects_direct_or_staging_path(dtype, uses_staging, monkeypatch):
    policy, ref, action, mask, old = _inputs(seed=12, device="cuda", dtype=dtype)
    policy = policy.requires_grad_(True)
    real_zeros_like = torch.zeros_like
    staging_shape = (policy.numel() // policy.shape[-1], policy.shape[-1])
    staging_allocations = []

    def track_zeros_like(tensor, *args, **kwargs):
        if tensor.shape == staging_shape and kwargs.get("dtype") == torch.float32:
            staging_allocations.append(tensor.shape)
        return real_zeros_like(tensor, *args, **kwargs)

    monkeypatch.setattr(ratio_kl_module.torch, "zeros_like", track_zeros_like)
    ratio, kl = TritonRatioKLOp()(policy, ref, action, mask, old)
    (ratio.sum() + kl.sum()).backward()

    assert len(staging_allocations) == int(uses_staging)


@requires_nvidia_triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_triton_empty_input_preserves_backward_contract(dtype):
    policy = torch.empty(0, 17, device="cuda", dtype=dtype, requires_grad=True)
    ref = torch.empty_like(policy, requires_grad=True)
    action = torch.empty(0, device="cuda", dtype=torch.int64)
    mask = torch.empty(0, device="cuda", dtype=torch.bool)
    old = torch.empty(0, device="cuda", dtype=torch.float32, requires_grad=True)

    ratio, kl = TritonRatioKLOp()(policy, ref, action, mask, old)
    (ratio.sum() + kl.sum()).backward()

    assert ratio.shape == kl.shape == torch.Size([0])
    assert policy.grad.shape == policy.shape
    assert policy.grad.dtype == dtype
    assert ref.grad is None
    assert old.grad is None


@requires_nvidia_triton
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_triton_all_inactive_is_neutral_with_positive_zero_gradient(dtype):
    policy = torch.randn(2, 3, 17, device="cuda", dtype=dtype, requires_grad=True)
    ref = torch.randn_like(policy)
    action = torch.full((2, 3), 999, device="cuda", dtype=torch.int64)
    mask = torch.zeros(2, 3, device="cuda", dtype=torch.bool)
    old = torch.randn(2, 3, device="cuda")

    ratio, kl = TritonRatioKLOp()(policy, ref, action, mask, old)
    (ratio.sum() + kl.sum()).backward()

    assert torch.equal(ratio, torch.ones_like(ratio))
    assert torch.equal(kl, torch.zeros_like(kl))
    assert torch.count_nonzero(policy.grad) == 0
    assert not torch.signbit(policy.grad).any()


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
    if _HAS_TRITON and (torch.cuda.is_available() or device_ctx.is_musa):
        assert isinstance(op, TritonRatioKLOp)
    else:
        assert isinstance(op, NativeRatioKLOp)
