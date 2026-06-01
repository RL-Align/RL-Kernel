# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.testing import (
    active_token_count,
    compute_policy_ratio,
    compute_reference_kl,
    make_synthetic_rl_kernel_batch,
    masked_mean,
    selected_logprobs_reference,
    summarize_kernel_drift,
)


def _minimal_rl_loss(
    current_logps,
    old_logps,
    ref_logps,
    advantages,
    completion_mask,
    eps=0.2,
    beta=0.01,
):
    ratio = compute_policy_ratio(current_logps, old_logps, completion_mask)
    unclipped = ratio * advantages.float()
    clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages.float()
    policy_loss = -torch.minimum(unclipped, clipped)
    kl = compute_reference_kl(current_logps, ref_logps, completion_mask)
    loss_terms = policy_loss + beta * kl
    loss = masked_mean(loss_terms, completion_mask)
    return {
        "loss": loss,
        "ratio": ratio,
        "kl": kl,
        "active_tokens": active_token_count(completion_mask),
    }


def _make_loss_inputs(device="cpu", dtype=torch.float32):
    batch = make_synthetic_rl_kernel_batch(
        num_prompts=2,
        samples_per_prompt=2,
        prompt_len=4,
        completion_len=6,
        vocab_size=64,
        valid_density=0.75,
        dtype=dtype,
        device=device,
        seed=42,
    )
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(99)
    logits = torch.randn(
        batch.batch_size,
        batch.completion_len,
        batch.metadata["vocab_size"],
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return batch, logits


def test_minimal_rl_loss_step_reference_path():
    batch, logits = _make_loss_inputs()

    current_logps = selected_logprobs_reference(
        logits,
        batch.token_ids,
        mask=batch.completion_mask,
        output_dtype=torch.float32,
    )
    result = _minimal_rl_loss(
        current_logps=current_logps,
        old_logps=batch.old_logps,
        ref_logps=batch.ref_logps,
        advantages=batch.advantages,
        completion_mask=batch.completion_mask,
    )

    assert torch.isfinite(result["loss"])
    assert result["active_tokens"].item() == batch.completion_mask.sum().item()
    assert torch.equal(
        result["ratio"][~batch.completion_mask],
        torch.zeros_like(result["ratio"][~batch.completion_mask]),
    )
    assert torch.equal(
        result["kl"][~batch.completion_mask],
        torch.zeros_like(result["kl"][~batch.completion_mask]),
    )


def test_masked_tokens_do_not_affect_minimal_loss():
    batch, logits = _make_loss_inputs()
    current_logps = selected_logprobs_reference(
        logits,
        batch.token_ids,
        mask=batch.completion_mask,
        output_dtype=torch.float32,
    )

    baseline = _minimal_rl_loss(
        current_logps,
        batch.old_logps,
        batch.ref_logps,
        batch.advantages,
        batch.completion_mask,
    )["loss"]

    perturbed_current = current_logps.clone()
    perturbed_old = batch.old_logps.clone()
    perturbed_ref = batch.ref_logps.clone()
    perturbed_advantages = batch.advantages.clone()
    inactive = ~batch.completion_mask
    perturbed_current[inactive] = 1000.0
    perturbed_old[inactive] = -1000.0
    perturbed_ref[inactive] = 500.0
    perturbed_advantages[inactive] = -500.0

    perturbed = _minimal_rl_loss(
        perturbed_current,
        perturbed_old,
        perturbed_ref,
        perturbed_advantages,
        batch.completion_mask,
    )["loss"]

    assert torch.allclose(baseline, perturbed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_minimal_rl_loss_step_fused_logp_candidate_cuda():
    from rl_engine.kernels.registry import kernel_registry

    batch, logits = _make_loss_inputs(device="cuda", dtype=torch.float16)

    reference_logps = selected_logprobs_reference(
        logits,
        batch.token_ids,
        mask=batch.completion_mask,
        output_dtype=torch.float32,
    )
    old_logps = reference_logps - 0.01
    ref_logps = reference_logps - 0.02
    candidate_op = kernel_registry.get_op("logp")
    if candidate_op.__class__.__name__ != "FusedLogpGenericOp":
        pytest.skip("fused logp CUDA backend is unavailable")

    candidate_logps = candidate_op(logits, batch.token_ids).float()
    candidate_logps = candidate_logps.masked_fill(~batch.completion_mask, 0.0)

    reference = _minimal_rl_loss(
        reference_logps,
        old_logps,
        ref_logps,
        batch.advantages,
        batch.completion_mask,
    )
    candidate = _minimal_rl_loss(
        candidate_logps,
        old_logps,
        ref_logps,
        batch.advantages,
        batch.completion_mask,
    )

    logp_drift = summarize_kernel_drift(
        candidate_logps,
        reference_logps,
        batch.completion_mask,
    )
    ratio_drift = summarize_kernel_drift(
        candidate["ratio"], reference["ratio"], batch.completion_mask
    )
    kl_drift = summarize_kernel_drift(candidate["kl"], reference["kl"], batch.completion_mask)

    assert logp_drift["max_abs_error"] < 1e-2
    assert ratio_drift["max_abs_error"] < 2e-2
    assert kl_drift["max_abs_error"] < 2e-2
    assert torch.allclose(
        candidate["loss"],
        reference["loss"],
        atol=2e-2,
        rtol=2e-2,
    )
