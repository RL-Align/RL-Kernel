# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import pytest
import torch

from rl_engine.testing import SyntheticRLKernelBatch, make_synthetic_rl_kernel_batch

VOCAB_SIZE = 32_768
HIDDEN_DIM = 256
PROMPT_PROBE_POS = 1
COMPLETION_PROBE_POS = 5
PROMPT_PROBE_TOKEN = 12_345
COMPLETION_PROBE_TOKEN = 23_456

BATCH_LAYOUTS = (
    dict(
        num_prompts=1,
        samples_per_prompt=2,
        prompt_len=4,
        completion_len=6,
        vocab_size=VOCAB_SIZE,
        valid_density=1.0,
        seed=11,
    ),
    dict(
        num_prompts=2,
        samples_per_prompt=3,
        prompt_len=4,
        completion_len=8,
        vocab_size=VOCAB_SIZE,
        valid_density=0.5,
        seed=12,
    ),
    dict(
        num_prompts=3,
        samples_per_prompt=4,
        prompt_len=4,
        completion_len=10,
        vocab_size=VOCAB_SIZE,
        valid_density=0.75,
        seed=13,
    ),
)

CUDA_CASE = pytest.param(
    "cuda",
    torch.bfloat16,
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
)


def _make_embedding(*, device: str, dtype: torch.dtype, seed: int) -> torch.nn.Embedding:
    embedding = torch.nn.Embedding(VOCAB_SIZE, HIDDEN_DIM).to(device=device, dtype=dtype)
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    weights = torch.randn(
        VOCAB_SIZE,
        HIDDEN_DIM,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ).to(dtype=dtype)
    with torch.no_grad():
        embedding.weight.copy_(weights)
    return embedding


def _stamp_probe_tokens(batch: SyntheticRLKernelBatch) -> SyntheticRLKernelBatch:
    completion_offset = COMPLETION_PROBE_POS - batch.prompt_len
    if completion_offset < 0 or completion_offset >= batch.completion_len:
        raise ValueError("completion probe position must fall inside completion tokens")

    batch.input_ids[:, PROMPT_PROBE_POS] = PROMPT_PROBE_TOKEN
    batch.input_ids[:, COMPLETION_PROBE_POS] = COMPLETION_PROBE_TOKEN
    batch.token_ids[:, completion_offset] = COMPLETION_PROBE_TOKEN
    batch.completion_mask[:, completion_offset] = True
    batch.attention_mask[:, COMPLETION_PROBE_POS] = True
    return batch


def _make_layout(
    layout: dict[str, int | float], *, device: str, dtype: torch.dtype
) -> SyntheticRLKernelBatch:
    batch = make_synthetic_rl_kernel_batch(device=device, dtype=dtype, **layout)
    return _stamp_probe_tokens(batch)


def _permute_rows(batch: SyntheticRLKernelBatch, perm: torch.Tensor) -> SyntheticRLKernelBatch:
    completion_mask = batch.completion_mask.index_select(0, perm)
    return SyntheticRLKernelBatch(
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
        metadata=dict(batch.metadata),
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
def test_embedding_lookup_is_bitwise_identical_across_batch_layouts(
    device: str, dtype: torch.dtype
) -> None:
    embedding = _make_embedding(device=device, dtype=dtype, seed=2026)
    prompt_reference = embedding.weight[PROMPT_PROBE_TOKEN].detach()
    completion_reference = embedding.weight[COMPLETION_PROBE_TOKEN].detach()

    for layout in BATCH_LAYOUTS:
        batch = _make_layout(layout, device=device, dtype=dtype)
        output = embedding(batch.input_ids)
        _assert_probe_vectors(
            output,
            batch_size=batch.batch_size,
            prompt_reference=prompt_reference,
            completion_reference=completion_reference,
        )


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32), CUDA_CASE])
def test_embedding_lookup_is_row_order_invariant_under_permutation(
    device: str, dtype: torch.dtype
) -> None:
    embedding = _make_embedding(device=device, dtype=dtype, seed=2026)
    batch = _make_layout(BATCH_LAYOUTS[2], device=device, dtype=dtype)

    perm = torch.arange(batch.batch_size - 1, -1, -1, device=torch.device(device))
    original = embedding(batch.input_ids)
    permuted_batch = _permute_rows(batch, perm)
    permuted = embedding(permuted_batch.input_ids)

    assert torch.equal(permuted, original.index_select(0, perm))


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32), CUDA_CASE])
def test_embedding_lookup_is_unaffected_by_padding_tail_mutations(
    device: str, dtype: torch.dtype
) -> None:
    embedding = _make_embedding(device=device, dtype=dtype, seed=2026)
    batch = _make_layout(BATCH_LAYOUTS[1], device=device, dtype=dtype)
    inactive = ~batch.attention_mask
    assert bool(inactive.any())

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

    baseline = embedding(batch.input_ids)
    candidate = embedding(mutated_input_ids)

    assert torch.equal(
        batch.input_ids[batch.attention_mask], mutated_input_ids[batch.attention_mask]
    )
    assert torch.equal(candidate[batch.attention_mask], baseline[batch.attention_mask])
