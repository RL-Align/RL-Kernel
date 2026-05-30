# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

import pytest
import torch

from rl_engine.testing import make_synthetic_rl_kernel_batch


def test_synthetic_rl_batch_shapes_and_masks():
    batch = make_synthetic_rl_kernel_batch(
        num_prompts=2,
        samples_per_prompt=3,
        prompt_len=4,
        completion_len=5,
        vocab_size=128,
        valid_density=0.6,
        dtype=torch.float32,
        seed=7,
    )

    assert batch.batch_size == 6
    assert batch.total_seq_len == 9
    assert batch.input_ids.shape == (6, 9)
    assert batch.attention_mask.shape == (6, 9)
    assert batch.prompt_mask.shape == (6, 9)
    assert batch.completion_mask.shape == (6, 5)
    assert batch.token_ids.shape == (6, 5)
    assert batch.advantages.shape == (6, 5)
    assert batch.old_logps.shape == (6, 5)
    assert batch.ref_logps.shape == (6, 5)

    assert batch.prompt_mask[:, :4].all()
    assert not batch.prompt_mask[:, 4:].any()
    assert torch.equal(batch.attention_mask[:, 4:], batch.completion_mask)
    assert batch.valid_indices is not None
    assert batch.valid_indices.numel() == int(round(6 * 5 * 0.6))


def test_synthetic_rl_batch_is_deterministic():
    kwargs = dict(
        num_prompts=2,
        samples_per_prompt=2,
        prompt_len=3,
        completion_len=4,
        vocab_size=64,
        valid_density=0.5,
        dtype=torch.float32,
        seed=123,
    )
    first = make_synthetic_rl_kernel_batch(**kwargs)
    second = make_synthetic_rl_kernel_batch(**kwargs)

    for field in (
        "input_ids",
        "attention_mask",
        "prompt_mask",
        "completion_mask",
        "token_ids",
        "rewards",
        "advantages",
        "old_logps",
        "ref_logps",
        "valid_indices",
    ):
        assert torch.equal(getattr(first, field), getattr(second, field))


def test_grouped_samples_share_prompt_tokens():
    batch = make_synthetic_rl_kernel_batch(
        num_prompts=2,
        samples_per_prompt=3,
        prompt_len=4,
        completion_len=5,
        vocab_size=64,
        seed=21,
    )

    prompt_tokens = batch.input_ids[:, : batch.prompt_len]
    assert torch.equal(prompt_tokens[0], prompt_tokens[1])
    assert torch.equal(prompt_tokens[1], prompt_tokens[2])
    assert torch.equal(prompt_tokens[3], prompt_tokens[4])
    assert torch.equal(prompt_tokens[4], prompt_tokens[5])
    assert not torch.equal(prompt_tokens[0], prompt_tokens[3])


def test_compact_completion_values_follow_valid_indices():
    batch = make_synthetic_rl_kernel_batch(
        num_prompts=1,
        samples_per_prompt=2,
        prompt_len=2,
        completion_len=6,
        vocab_size=32,
        valid_density=0.5,
        seed=9,
    )
    values = torch.arange(batch.batch_size * batch.completion_len).reshape(
        batch.batch_size, batch.completion_len
    )

    compact = batch.compact_completion_values(values)
    expected = values.reshape(-1)[batch.flat_completion_mask]

    assert torch.equal(compact, expected)
    assert torch.equal(batch.compact_token_ids(), batch.flat_token_ids[batch.flat_completion_mask])
    assert torch.equal(
        batch.valid_indices, batch.flat_completion_mask.nonzero(as_tuple=False).squeeze(-1)
    )


def test_benchmark_metadata_is_reproducible():
    batch = make_synthetic_rl_kernel_batch(
        num_prompts=3,
        samples_per_prompt=2,
        prompt_len=1,
        completion_len=5,
        vocab_size=100,
        valid_density=0.4,
        dtype=torch.float16,
        seed=11,
    )
    metadata = batch.benchmark_metadata()

    assert metadata["num_prompts"] == 3
    assert metadata["samples_per_prompt"] == 2
    assert metadata["batch_size"] == 6
    assert metadata["prompt_len"] == 1
    assert metadata["completion_len"] == 5
    assert metadata["vocab_size"] == 100
    assert metadata["valid_tokens"] == int(batch.flat_completion_mask.sum().item())
    assert metadata["seed"] == 11


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_synthetic_rl_batch_cuda_smoke():
    batch = make_synthetic_rl_kernel_batch(
        num_prompts=1,
        samples_per_prompt=1,
        prompt_len=2,
        completion_len=3,
        vocab_size=16,
        dtype=torch.float16,
        device="cuda",
        seed=1,
    )

    assert batch.input_ids.is_cuda
    assert batch.completion_mask.is_cuda
    assert batch.advantages.dtype == torch.float16
