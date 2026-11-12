# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rl_engine.reference.embedding.embedding import NativeEmbeddingOp
from rl_engine.reference.logprob.linear_logp import NativeLinearLogpOp
from rl_engine.validation.common import (
    SyntheticRLKernelBatch,
    make_synthetic_rl_kernel_batch,
    selected_logprobs_reference,
)

VOCAB_SIZE = 4096
HIDDEN_DIM = 128
PROMPT_PROBE_POS = 1
COMPLETION_PROBE_POS = 5
PROMPT_PROBE_TOKEN = 1234
COMPLETION_PROBE_TOKEN = 2345

BATCH_LAYOUTS = (
    dict(
        num_prompts=1,
        samples_per_prompt=2,
        prompt_len=4,
        completion_len=6,
        vocab_size=VOCAB_SIZE,
        valid_density=1.0,
        seed=151,
    ),
    dict(
        num_prompts=2,
        samples_per_prompt=3,
        prompt_len=4,
        completion_len=8,
        vocab_size=VOCAB_SIZE,
        valid_density=0.5,
        seed=152,
    ),
    dict(
        num_prompts=3,
        samples_per_prompt=4,
        prompt_len=4,
        completion_len=10,
        vocab_size=VOCAB_SIZE,
        valid_density=0.75,
        seed=153,
    ),
)

CUDA_CASE = pytest.param(
    "cuda",
    torch.bfloat16,
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
)


def _make_embedding_weight(*, device: str, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    return torch.randn(
        VOCAB_SIZE,
        HIDDEN_DIM,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(dtype=dtype)


def _stamp_probe_tokens(batch: SyntheticRLKernelBatch) -> SyntheticRLKernelBatch:
    completion_offset = COMPLETION_PROBE_POS - batch.prompt_len
    if completion_offset < 0 or completion_offset >= batch.completion_len:
        raise ValueError("completion probe position must fall inside completion tokens")

    input_ids = batch.input_ids.clone()
    token_ids = batch.token_ids.clone()
    completion_mask = batch.completion_mask.clone()
    attention_mask = batch.attention_mask.clone()

    input_ids[:, PROMPT_PROBE_POS] = PROMPT_PROBE_TOKEN
    input_ids[:, COMPLETION_PROBE_POS] = COMPLETION_PROBE_TOKEN
    token_ids[:, completion_offset] = COMPLETION_PROBE_TOKEN
    completion_mask[:, completion_offset] = True
    attention_mask[:, COMPLETION_PROBE_POS] = True
    valid_indices = completion_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)

    return replace(
        batch,
        input_ids=input_ids,
        token_ids=token_ids,
        completion_mask=completion_mask,
        attention_mask=attention_mask,
        valid_indices=valid_indices,
        metadata={
            **batch.metadata,
            "valid_tokens": int(completion_mask.sum().item()),
            "valid_density": float(completion_mask.float().mean().item()),
        },
    )


def _make_layout(
    layout: dict[str, int | float], *, device: str, dtype: torch.dtype
) -> SyntheticRLKernelBatch:
    batch = make_synthetic_rl_kernel_batch(device=device, dtype=dtype, **layout)
    return _stamp_probe_tokens(batch)


def _permute_rows(batch: SyntheticRLKernelBatch, perm: torch.Tensor) -> SyntheticRLKernelBatch:
    completion_mask = batch.completion_mask.index_select(0, perm)
    return replace(
        batch,
        input_ids=batch.input_ids.index_select(0, perm),
        attention_mask=batch.attention_mask.index_select(0, perm),
        prompt_mask=batch.prompt_mask.index_select(0, perm),
        completion_mask=completion_mask,
        token_ids=batch.token_ids.index_select(0, perm),
        rewards=batch.rewards.index_select(0, perm),
        advantages=batch.advantages.index_select(0, perm),
        old_logps=batch.old_logps.index_select(0, perm),
        ref_logps=batch.ref_logps.index_select(0, perm),
        valid_indices=completion_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1),
    )


def _assert_probe_vectors(
    output: torch.Tensor,
    *,
    batch_size: int,
    prompt_reference: torch.Tensor,
    completion_reference: torch.Tensor,
) -> None:
    assert torch.equal(
        output[:, PROMPT_PROBE_POS, :],
        prompt_reference.expand(batch_size, -1),
    )
    assert torch.equal(
        output[:, COMPLETION_PROBE_POS, :],
        completion_reference.expand(batch_size, -1),
    )


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32), CUDA_CASE])
def test_embedding_lookup_is_bitwise_identical_across_rl_batch_layouts(
    device: str, dtype: torch.dtype
) -> None:
    op = NativeEmbeddingOp()
    weight = _make_embedding_weight(device=device, dtype=dtype, seed=2026)
    prompt_reference = weight[PROMPT_PROBE_TOKEN].detach()
    completion_reference = weight[COMPLETION_PROBE_TOKEN].detach()

    for layout in BATCH_LAYOUTS:
        batch = _make_layout(layout, device=device, dtype=dtype)
        output = op.forward(batch.input_ids, weight)
        _assert_probe_vectors(
            output,
            batch_size=batch.batch_size,
            prompt_reference=prompt_reference,
            completion_reference=completion_reference,
        )


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32), CUDA_CASE])
def test_embedding_lookup_is_row_order_invariant_under_rl_permutation(
    device: str, dtype: torch.dtype
) -> None:
    op = NativeEmbeddingOp()
    weight = _make_embedding_weight(device=device, dtype=dtype, seed=2027)
    batch = _make_layout(BATCH_LAYOUTS[2], device=device, dtype=dtype)

    perm = torch.arange(batch.batch_size - 1, -1, -1, device=torch.device(device))
    original = op.forward(batch.input_ids, weight)
    permuted = op.forward(_permute_rows(batch, perm).input_ids, weight)

    assert torch.equal(permuted, original.index_select(0, perm))


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32), CUDA_CASE])
def test_embedding_lookup_active_tokens_ignore_padding_tail_mutations(
    device: str, dtype: torch.dtype
) -> None:
    op = NativeEmbeddingOp()
    weight = _make_embedding_weight(device=device, dtype=dtype, seed=2028)
    batch = _make_layout(BATCH_LAYOUTS[1], device=device, dtype=dtype)
    inactive = ~batch.attention_mask
    assert bool(inactive.any().item())

    mutated_input_ids = batch.input_ids.clone()
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(404)
    random_tokens = torch.randint(
        0,
        VOCAB_SIZE,
        mutated_input_ids.shape,
        device=device,
        generator=generator,
        dtype=torch.long,
    )
    mutated_input_ids[inactive] = random_tokens[inactive]

    baseline = op.forward(batch.input_ids, weight)
    candidate = op.forward(mutated_input_ids, weight)

    assert torch.equal(candidate[batch.attention_mask], baseline[batch.attention_mask])


def test_embedding_backward_weight_gradient_is_layout_invariant() -> None:
    torch.manual_seed(2030)
    op = NativeEmbeddingOp()
    base_token_ids = torch.tensor([2, 5, 1, 5, 2, 3], dtype=torch.long)
    base_upstream = torch.randn(6, HIDDEN_DIM)
    layouts = [
        ((2, 3), [0, 1, 2, 3, 4, 5]),
        ((3, 2), [5, 2, 1, 0, 4, 3]),
        ((1, 6), [3, 0, 5, 2, 1, 4]),
    ]

    canonical_grad = None
    for lead_shape, order in layouts:
        order_t = torch.tensor(order, dtype=torch.long)
        token_ids = base_token_ids.index_select(0, order_t).reshape(lead_shape)
        upstream = base_upstream.index_select(0, order_t).reshape(*lead_shape, HIDDEN_DIM)
        weight = _make_embedding_weight(device="cpu", dtype=torch.float32, seed=2030)
        weight.requires_grad_(True)

        (op.forward(token_ids, weight) * upstream).sum().backward()
        grad = weight.grad.detach().clone()

        if canonical_grad is None:
            canonical_grad = grad
        else:
            assert torch.allclose(grad, canonical_grad, atol=1e-6)


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32), CUDA_CASE])
def test_lm_head_linear_logp_handoff_is_layout_invariant_for_rl_batches(
    device: str, dtype: torch.dtype
) -> None:
    target_device = torch.device(device)
    generator = torch.Generator(device=target_device)
    generator.manual_seed(2031)
    op = NativeLinearLogpOp()
    weight = torch.randn(29, 7, device=target_device, dtype=dtype, generator=generator)
    bias = torch.randn(29, device=target_device, dtype=dtype, generator=generator)
    base_hidden = torch.randn(6, 7, device=target_device, dtype=dtype, generator=generator)
    base_target = torch.tensor([3, 7, 1, 9, 4, 6], device=target_device, dtype=torch.long)
    base_mask = torch.tensor(
        [True, False, True, True, False, True], device=target_device, dtype=torch.bool
    )
    layouts = [
        ((2, 3), [0, 1, 2, 3, 4, 5]),
        ((3, 2), [5, 1, 3, 0, 4, 2]),
        ((1, 6), [2, 4, 1, 5, 0, 3]),
    ]

    canonical = None
    for lead_shape, order in layouts:
        order_t = torch.tensor(order, device=target_device, dtype=torch.long)
        hidden = base_hidden.index_select(0, order_t).reshape(*lead_shape, -1)
        target = base_target.index_select(0, order_t).reshape(lead_shape)
        mask = base_mask.index_select(0, order_t).reshape(lead_shape)
        safe_target = target.masked_fill(~mask, 0)
        padded_hidden = hidden.clone()
        padding_values = torch.randn(
            padded_hidden.shape,
            device=target_device,
            dtype=dtype,
            generator=generator,
        )
        padded_hidden[~mask] = padding_values[~mask]

        logps = op(padded_hidden, weight, safe_target, bias).masked_fill(~mask, 0.0)
        logits = torch.nn.functional.linear(padded_hidden.float(), weight.float(), bias.float())
        expected = selected_logprobs_reference(logits, target, mask=mask)

        assert torch.allclose(logps, expected, atol=5e-2 if dtype is torch.bfloat16 else 1e-5)
        recovered = logps.reshape(-1)[torch.argsort(order_t)].float()
        if canonical is None:
            canonical = recovered
        else:
            assert torch.allclose(
                recovered, canonical, atol=5e-2 if dtype is torch.bfloat16 else 1e-6
            )
