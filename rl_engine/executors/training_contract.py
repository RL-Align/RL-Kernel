# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

import torch

from rl_engine.kernels.registry import resolve_logp_op_type
from rl_engine.testing import SyntheticRLKernelBatch, make_synthetic_rl_kernel_batch


@dataclass(frozen=True)
class RolloutStageResult:
    """Result consumed by training workers."""

    iteration: int
    weight_version: int
    payload: Any
    started_at: float
    finished_at: float
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at


@dataclass(frozen=True)
class TrainingStageResult:
    """Result produced by training workers."""

    iteration: int
    consumed_weight_version: int
    published_weight_version: Optional[int]
    metrics: Mapping[str, Any]
    started_at: float
    finished_at: float

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at


class TrainingWorker(Protocol):
    def train(self, rollout: RolloutStageResult) -> TrainingStageResult: ...


@dataclass(frozen=True)
class TorchRLTrainingConfig:
    """Config shared by local and DeepSpeed training workers."""

    num_prompts: int = 1
    samples_per_prompt: int = 2
    prompt_len: int = 4
    completion_len: int = 8
    vocab_size: int = 64
    hidden_dim: int = 32
    valid_density: float = 0.75
    lr: float = 1e-3
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 0
    min_completion_len: int = 1
    logp_backend: str = "auto"
    require_batch_invariant_logp: bool = False

    def __post_init__(self) -> None:
        resolve_logp_op_type(
            self.logp_backend,
            require_batch_invariant=self.require_batch_invariant_logp,
        )


class RolloutBatchMixin:
    config: TorchRLTrainingConfig
    device: torch.device

    def _batch_from_rollout_or_synthetic(
        self,
        rollout: RolloutStageResult,
    ) -> tuple[SyntheticRLKernelBatch, dict[str, Any]]:
        token_groups = extract_rollout_token_groups(rollout.payload)
        if token_groups:
            return self._batch_from_token_groups(token_groups, rollout), {
                "training_data_source": "rollout_payload",
                "rollout_sequences": len(token_groups),
                "rollout_tokens": sum(len(group) for group in token_groups),
            }

        seed = self.config.seed + int(rollout.iteration)
        batch = make_synthetic_rl_kernel_batch(
            num_prompts=self.config.num_prompts,
            samples_per_prompt=self.config.samples_per_prompt,
            prompt_len=self.config.prompt_len,
            completion_len=self.config.completion_len,
            vocab_size=self.config.vocab_size,
            valid_density=self.config.valid_density,
            dtype=self.config.dtype,
            device=self.device,
            seed=seed,
        )
        return batch, {
            "training_data_source": "synthetic_fallback",
            "rollout_sequences": 0,
            "rollout_tokens": 0,
        }

    def _batch_from_token_groups(
        self,
        token_groups: Sequence[Sequence[int]],
        rollout: RolloutStageResult,
    ) -> SyntheticRLKernelBatch:
        completion_len = max(
            self.config.min_completion_len,
            min(self.config.completion_len, max(len(group) for group in token_groups)),
        )
        batch_size = len(token_groups)
        token_ids = torch.zeros(
            (batch_size, completion_len),
            device=self.device,
            dtype=torch.long,
        )
        completion_mask = torch.zeros(
            (batch_size, completion_len),
            device=self.device,
            dtype=torch.bool,
        )
        for row, group in enumerate(token_groups):
            clipped = [int(token) % self.config.vocab_size for token in group[:completion_len]]
            if not clipped:
                continue
            values = torch.tensor(clipped, device=self.device, dtype=torch.long)
            token_ids[row, : values.numel()] = values
            completion_mask[row, : values.numel()] = True

        prompt_tokens = torch.zeros(
            (batch_size, self.config.prompt_len),
            device=self.device,
            dtype=torch.long,
        )
        input_ids = torch.cat([prompt_tokens, token_ids], dim=1)
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.config.prompt_len:
            prompt_mask[:, : self.config.prompt_len] = True
        attention_mask = torch.cat(
            [
                prompt_mask[:, : self.config.prompt_len],
                completion_mask,
            ],
            dim=1,
        )

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.seed + int(rollout.iteration))
        advantages = torch.randn(
            (batch_size, completion_len),
            device=self.device,
            generator=generator,
            dtype=self.config.dtype,
        )
        old_logps = torch.zeros(
            (batch_size, completion_len),
            device=self.device,
            dtype=self.config.dtype,
        )
        ref_logps = torch.zeros(
            (batch_size, completion_len),
            device=self.device,
            dtype=self.config.dtype,
        )
        rewards = torch.randn(
            (batch_size,),
            device=self.device,
            generator=generator,
            dtype=self.config.dtype,
        )
        valid_indices = completion_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        metadata: dict[str, Any] = {
            "num_prompts": batch_size,
            "samples_per_prompt": 1,
            "batch_size": batch_size,
            "prompt_len": self.config.prompt_len,
            "completion_len": completion_len,
            "total_seq_len": self.config.prompt_len + completion_len,
            "vocab_size": self.config.vocab_size,
            "valid_density": float(completion_mask.float().mean().item()),
            "valid_tokens": int(completion_mask.sum().item()),
            "dtype": self.config.dtype,
            "device": str(self.device),
            "seed": self.config.seed + int(rollout.iteration),
            "source": "rollout_payload",
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


def make_rollout_result(
    *,
    iteration: int,
    weight_version: int,
    payload: Any,
    metrics: Optional[Mapping[str, Any]] = None,
) -> RolloutStageResult:
    now = time.perf_counter()
    return RolloutStageResult(
        iteration=iteration,
        weight_version=weight_version,
        payload=payload,
        started_at=now,
        finished_at=time.perf_counter(),
        metrics=dict(metrics or {}),
    )


def extract_rollout_token_groups(payload: Any) -> list[list[int]]:
    """Extract generated token ids from RL-Kernel/vLLM-style rollout payloads."""

    groups: list[list[int]] = []
    if not isinstance(payload, Mapping):
        return groups

    normalized_outputs = payload.get("normalized_outputs")
    if isinstance(normalized_outputs, Sequence) and not isinstance(
        normalized_outputs, (str, bytes)
    ):
        for group in normalized_outputs:
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                continue
            for candidate in group:
                token_ids = _candidate_token_ids(candidate)
                if token_ids:
                    groups.append(token_ids)
        if groups:
            return groups

    outputs = payload.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        for group in outputs:
            if isinstance(group, Sequence) and not isinstance(group, (str, bytes)):
                candidates = group
            else:
                candidates = [group]
            for candidate in candidates:
                token_ids = _candidate_token_ids(candidate)
                if token_ids:
                    groups.append(token_ids)
    return groups


def _candidate_token_ids(candidate: Any) -> list[int]:
    if candidate is None:
        return []
    if isinstance(candidate, Mapping):
        nested_outputs = candidate.get("outputs")
        if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
            for nested in nested_outputs:
                token_ids = _candidate_token_ids(nested)
                if token_ids:
                    return token_ids
        value = candidate.get("token_ids")
        return _copy_int_list(value)

    value = getattr(candidate, "token_ids", None)
    if value is not None:
        return _copy_int_list(value)
    nested_outputs = getattr(candidate, "outputs", None)
    if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
        for nested in nested_outputs:
            token_ids = _candidate_token_ids(nested)
            if token_ids:
                return token_ids
    return []


def _copy_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    return []
