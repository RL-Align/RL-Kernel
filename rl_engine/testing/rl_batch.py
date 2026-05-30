# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kernel-Align Contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SyntheticRLKernelBatch:
    """Synthetic RL-shaped tensors shared by kernel tests and benchmarks."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_mask: torch.Tensor
    completion_mask: torch.Tensor
    token_ids: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    old_logps: torch.Tensor
    ref_logps: torch.Tensor
    valid_indices: torch.Tensor | None
    metadata: dict[str, Any]

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.size(0))

    @property
    def total_seq_len(self) -> int:
        return int(self.input_ids.size(1))

    @property
    def prompt_len(self) -> int:
        return int(self.metadata["prompt_len"])

    @property
    def completion_len(self) -> int:
        return int(self.metadata["completion_len"])

    @property
    def flat_completion_mask(self) -> torch.Tensor:
        return self.completion_mask.reshape(-1)

    @property
    def flat_token_ids(self) -> torch.Tensor:
        return self.token_ids.reshape(-1)

    def dense_completion_token_ids(self) -> torch.Tensor:
        return self.token_ids

    def dense_completion_values(self, values: torch.Tensor) -> torch.Tensor:
        expected_shape = (self.batch_size, self.completion_len)
        if tuple(values.shape[:2]) != expected_shape:
            raise ValueError(
                f"expected leading shape {expected_shape}, got {tuple(values.shape[:2])}"
            )
        return values

    def compact_completion_values(self, values: torch.Tensor) -> torch.Tensor:
        dense = self.dense_completion_values(values)
        return dense.reshape(-1, *dense.shape[2:])[self.flat_completion_mask]

    def compact_token_ids(self) -> torch.Tensor:
        return self.flat_token_ids[self.flat_completion_mask]

    def benchmark_metadata(self) -> dict[str, Any]:
        return {
            "num_prompts": self.metadata["num_prompts"],
            "samples_per_prompt": self.metadata["samples_per_prompt"],
            "batch_size": self.batch_size,
            "prompt_len": self.prompt_len,
            "completion_len": self.completion_len,
            "total_seq_len": self.total_seq_len,
            "vocab_size": self.metadata["vocab_size"],
            "valid_density": self.metadata["valid_density"],
            "valid_tokens": int(self.flat_completion_mask.sum().item()),
            "dtype": str(self.metadata["dtype"]),
            "device": str(self.input_ids.device),
            "seed": self.metadata["seed"],
        }


def make_synthetic_rl_kernel_batch(
    *,
    num_prompts: int,
    samples_per_prompt: int,
    prompt_len: int,
    completion_len: int,
    vocab_size: int,
    valid_density: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> SyntheticRLKernelBatch:
    """Create deterministic RL-shaped tensors for kernel tests."""

    if num_prompts <= 0:
        raise ValueError("num_prompts must be greater than zero")
    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be greater than zero")
    if prompt_len < 0:
        raise ValueError("prompt_len must be non-negative")
    if completion_len <= 0:
        raise ValueError("completion_len must be greater than zero")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than one")
    if not 0.0 <= valid_density <= 1.0:
        raise ValueError("valid_density must be in [0.0, 1.0]")

    target_device = torch.device(device)
    batch_size = num_prompts * samples_per_prompt
    total_seq_len = prompt_len + completion_len

    generator = torch.Generator(device=target_device)
    generator.manual_seed(seed)

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, total_seq_len),
        device=target_device,
        generator=generator,
        dtype=torch.long,
    )
    if prompt_len:
        prompt_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_prompts, prompt_len),
            device=target_device,
            generator=generator,
            dtype=torch.long,
        )
        shared_prompt_ids = prompt_ids.repeat_interleave(samples_per_prompt, dim=0)
        input_ids[:, :prompt_len] = shared_prompt_ids

    attention_mask = torch.ones((batch_size, total_seq_len), device=target_device, dtype=torch.bool)
    prompt_mask = torch.zeros_like(attention_mask)
    if prompt_len:
        prompt_mask[:, :prompt_len] = True

    completion_mask = torch.zeros(
        (batch_size, completion_len), device=target_device, dtype=torch.bool
    )
    total_completion_tokens = batch_size * completion_len
    valid_count = int(round(total_completion_tokens * valid_density))
    if valid_count:
        order = torch.randperm(total_completion_tokens, device=target_device, generator=generator)
        completion_mask.reshape(-1)[order[:valid_count]] = True

    attention_completion = attention_mask[:, prompt_len:]
    attention_completion.copy_(completion_mask)

    token_ids = input_ids[:, prompt_len:].clone()
    rewards = torch.randn((batch_size,), device=target_device, generator=generator, dtype=dtype)
    advantages = torch.randn(
        (batch_size, completion_len), device=target_device, generator=generator, dtype=dtype
    )
    old_logps = torch.randn(
        (batch_size, completion_len), device=target_device, generator=generator, dtype=dtype
    )
    ref_logps = torch.randn(
        (batch_size, completion_len), device=target_device, generator=generator, dtype=dtype
    )

    valid_indices = completion_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)

    metadata: dict[str, Any] = {
        "num_prompts": num_prompts,
        "samples_per_prompt": samples_per_prompt,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "completion_len": completion_len,
        "total_seq_len": total_seq_len,
        "vocab_size": vocab_size,
        "valid_density": valid_density,
        "valid_tokens": int(valid_count),
        "dtype": dtype,
        "device": str(target_device),
        "seed": seed,
    }

    return SyntheticRLKernelBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_mask=prompt_mask,
        completion_mask=completion_mask,
        token_ids=token_ids,
        rewards=rewards,
        advantages=advantages,
        old_logps=old_logps,
        ref_logps=ref_logps,
        valid_indices=valid_indices,
        metadata=metadata,
    )
