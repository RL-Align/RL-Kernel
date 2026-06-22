# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

import torch

from rl_engine.executors.stateless_executor import (
    StatelessForwardExecutor,
    StatelessForwardInputs,
    StatelessForwardResult,
)
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
    advantage_eps: float = 1e-8


class RolloutBatchMixin:
    config: TorchRLTrainingConfig
    device: torch.device

    def _batch_from_rollout_or_synthetic(
        self,
        rollout: RolloutStageResult,
    ) -> tuple[SyntheticRLKernelBatch, dict[str, Any]]:
        candidate_groups = extract_rollout_candidate_groups(rollout.payload)
        if candidate_groups:
            reward_groups = extract_rollout_reward_groups(rollout.payload)
            reference_logp_groups = extract_rollout_reference_logp_groups(rollout.payload)
            has_payload_rewards = _reward_groups_match(candidate_groups, reward_groups)
            has_payload_reference_logps = _reference_logp_groups_match(
                candidate_groups,
                reference_logp_groups,
                max_completion_len=self.config.completion_len,
            )
            batch = self._batch_from_candidate_groups(
                candidate_groups,
                rollout,
                reward_groups=reward_groups if has_payload_rewards else None,
                reference_logp_groups=(
                    reference_logp_groups if has_payload_reference_logps else None
                ),
            )
            token_groups = [tokens for group in candidate_groups for tokens in group]
            return batch, {
                "training_data_source": "rollout_payload",
                "reward_source": "payload_rewards" if has_payload_rewards else "token_id_proxy",
                "reference_logp_source": (
                    "payload_reference_logps"
                    if has_payload_reference_logps
                    else "synthetic_current_offset"
                ),
                "rollout_prompt_groups": len(candidate_groups),
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
            "reward_source": "synthetic_rewards",
            "reference_logp_source": "synthetic_current_offset",
        }

    def _batch_from_token_groups(
        self,
        token_groups: Sequence[Sequence[int]],
        rollout: RolloutStageResult,
    ) -> SyntheticRLKernelBatch:
        return self._batch_from_candidate_groups(
            [[group] for group in token_groups],
            rollout,
            reward_groups=None,
            reference_logp_groups=None,
        )

    def _batch_from_candidate_groups(
        self,
        candidate_groups: Sequence[Sequence[Sequence[int]]],
        rollout: RolloutStageResult,
        *,
        reward_groups: Optional[Sequence[Sequence[float]]] = None,
        reference_logp_groups: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    ) -> SyntheticRLKernelBatch:
        flat_token_groups = [tokens for group in candidate_groups for tokens in group]
        completion_len = max(
            self.config.min_completion_len,
            min(self.config.completion_len, max(len(group) for group in flat_token_groups)),
        )
        batch_size = len(flat_token_groups)
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
        flat_rewards: list[float] = []
        flat_group_ids: list[int] = []
        flat_reference_logps: list[list[float]] = []
        row = 0
        for group_index, group in enumerate(candidate_groups):
            for candidate_index, candidate_tokens in enumerate(group):
                clipped = [
                    int(token) % self.config.vocab_size
                    for token in candidate_tokens[:completion_len]
                ]
                if clipped:
                    values = torch.tensor(clipped, device=self.device, dtype=torch.long)
                    token_ids[row, : values.numel()] = values
                    completion_mask[row, : values.numel()] = True
                flat_rewards.append(
                    _candidate_reward_value(
                        candidate_tokens,
                        reward_groups,
                        group_index,
                        candidate_index,
                        vocab_size=self.config.vocab_size,
                    )
                )
                if reference_logp_groups is not None:
                    flat_reference_logps.append(
                        _candidate_reference_logps(
                            reference_logp_groups,
                            group_index,
                            candidate_index,
                            completion_len=completion_len,
                        )
                    )
                flat_group_ids.append(group_index)
                row += 1

        if not bool(completion_mask.any().item()):
            completion_mask[:, :1] = True

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

        rewards = torch.tensor(
            flat_rewards,
            device=self.device,
            dtype=self.config.dtype,
        )
        group_ids = torch.tensor(flat_group_ids, device=self.device, dtype=torch.long)
        sequence_advantages = _compute_group_relative_advantages(
            rewards,
            group_ids=group_ids,
            num_groups=len(candidate_groups),
            eps=self.config.advantage_eps,
        )
        advantages = _broadcast_sequence_advantages(sequence_advantages, completion_mask).to(
            dtype=self.config.dtype
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
        if reference_logp_groups is not None:
            for row_index, reference_values in enumerate(flat_reference_logps):
                if reference_values:
                    values = torch.tensor(
                        reference_values,
                        device=self.device,
                        dtype=self.config.dtype,
                    )
                    ref_logps[row_index, : values.numel()] = values
        valid_indices = completion_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        metadata: dict[str, Any] = {
            "num_prompts": len(candidate_groups),
            "samples_per_prompt": max(len(group) for group in candidate_groups),
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
            "group_ids": flat_group_ids,
            "reward_source": "payload_rewards" if reward_groups is not None else "token_id_proxy",
            "reference_logp_source": (
                "payload_reference_logps"
                if reference_logp_groups is not None
                else "synthetic_current_offset"
            ),
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


class StatelessScoringWorker:
    """Attach no-cache reference/reward scores to a completed rollout payload."""

    def __init__(
        self,
        executor: StatelessForwardExecutor,
        collate_inputs: Callable[[RolloutStageResult], StatelessForwardInputs],
    ):
        self.executor = executor
        self.collate_inputs = collate_inputs

    def score(self, rollout: RolloutStageResult) -> RolloutStageResult:
        started_at = time.perf_counter()
        inputs = self.collate_inputs(rollout)
        result = self.executor.score(inputs)
        finished_at = time.perf_counter()
        return RolloutStageResult(
            iteration=rollout.iteration,
            weight_version=rollout.weight_version,
            payload=attach_stateless_scores_to_payload(rollout.payload, result, inputs=inputs),
            started_at=started_at,
            finished_at=finished_at,
            metrics={
                **dict(rollout.metrics),
                "scoring_backend": "stateless",
                "scoring_mode": result.metrics["mode"],
                "scoring_elapsed_ms": result.metrics["elapsed_ms"],
                "scoring_active_completion_tokens": result.metrics["active_completion_tokens"],
                "scoring_zero_kv_cache": result.metrics["zero_kv_cache"],
                "scoring_attention_backend": result.metrics["attention_backend"],
                "scoring_kv_cache_output_mb": result.metrics["kv_cache_output_mb"],
            },
        )


def attach_stateless_scores_to_payload(
    payload: Any,
    result: StatelessForwardResult,
    *,
    inputs: Optional[StatelessForwardInputs] = None,
) -> dict[str, Any]:
    """Return a rollout payload augmented with stateless scoring outputs."""

    scored_payload = dict(payload) if isinstance(payload, Mapping) else {"raw_payload": payload}
    scored_payload["stateless_scores"] = {
        "reference_logps": result.reference_logps,
        "rewards": result.rewards,
        "token_scores": result.token_scores,
        "metrics": dict(result.metrics),
    }
    if result.rewards is not None:
        _attach_reward_tensor_to_grouped_candidates(scored_payload, result.rewards)
    if result.reference_logps is not None and inputs is not None:
        _attach_reference_logps_to_grouped_candidates(
            scored_payload,
            result.reference_logps,
            inputs.completion_mask,
        )
    return scored_payload


def build_stateless_inputs_from_rollout_payload(
    payload: Any,
    *,
    prompt_len: Optional[int] = 1,
    prompt_token_id: int = 0,
    max_completion_len: Optional[int] = None,
    device: torch.device | str = "cpu",
) -> StatelessForwardInputs:
    """Build dense no-cache scoring inputs from grouped rollout token payloads."""

    if prompt_len is not None and prompt_len < 0:
        raise ValueError("prompt_len must be non-negative")
    if max_completion_len is not None and max_completion_len <= 0:
        raise ValueError("max_completion_len must be greater than zero")

    candidate_records = _candidate_records_from_payload(payload)
    if not candidate_records:
        raise ValueError("rollout payload does not contain candidate token ids")

    flat_token_groups = [tokens for _prompt_tokens, tokens in candidate_records]
    completion_len = max(len(tokens) for tokens in flat_token_groups)
    if max_completion_len is not None:
        completion_len = min(completion_len, max_completion_len)
    completion_len = max(1, completion_len)
    resolved_prompt_len = (
        max((len(prompt_tokens) for prompt_tokens, _tokens in candidate_records), default=0)
        if prompt_len is None
        else prompt_len
    )
    seq_len = resolved_prompt_len + completion_len
    resolved_device = torch.device(device)

    input_ids = torch.full(
        (len(flat_token_groups), seq_len),
        int(prompt_token_id),
        device=resolved_device,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if resolved_prompt_len and prompt_len is not None:
        attention_mask[:, :resolved_prompt_len] = True

    for row, (prompt_tokens, token_ids) in enumerate(candidate_records):
        if resolved_prompt_len and prompt_tokens:
            clipped_prompt = [int(token) for token in prompt_tokens[:resolved_prompt_len]]
            values = torch.tensor(clipped_prompt, device=resolved_device, dtype=torch.long)
            input_ids[row, : values.numel()] = values
            if prompt_len is None:
                attention_mask[row, : values.numel()] = True
        clipped = [int(token) for token in token_ids[:completion_len]]
        if clipped:
            values = torch.tensor(clipped, device=resolved_device, dtype=torch.long)
            start = resolved_prompt_len
            end = resolved_prompt_len + values.numel()
            input_ids[row, start:end] = values
            attention_mask[row, start:end] = True
            completion_mask[row, start:end] = True

    if not bool(completion_mask.any().item()):
        raise ValueError("rollout payload does not contain active completion tokens")
    return StatelessForwardInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
    )


def extract_rollout_token_groups(payload: Any) -> list[list[int]]:
    """Extract generated token ids from RL-Kernel/vLLM-style rollout payloads."""

    return [tokens for group in extract_rollout_candidate_groups(payload) for tokens in group]


def extract_rollout_candidate_groups(payload: Any) -> list[list[list[int]]]:
    """Extract generated token ids while preserving prompt-level candidate groups."""

    if not isinstance(payload, Mapping):
        return []

    normalized_outputs = payload.get("normalized_outputs")
    if isinstance(normalized_outputs, Sequence) and not isinstance(
        normalized_outputs, (str, bytes)
    ):
        grouped = _candidate_groups_from_grouped_outputs(normalized_outputs)
        if grouped:
            return grouped

    outputs = payload.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return _candidate_groups_from_grouped_outputs(outputs)
    return []


def extract_rollout_reward_groups(payload: Any) -> list[list[float]]:
    """Extract scalar reward groups from rollout payloads when they are present."""

    if not isinstance(payload, Mapping):
        return []

    normalized_outputs = payload.get("normalized_outputs")
    if isinstance(normalized_outputs, Sequence) and not isinstance(
        normalized_outputs, (str, bytes)
    ):
        groups = _reward_groups_from_grouped_outputs(normalized_outputs)
        if groups:
            return groups

    outputs = payload.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return _reward_groups_from_grouped_outputs(outputs)
    return []


def extract_rollout_reference_logp_groups(payload: Any) -> list[list[list[float]]]:
    """Extract per-candidate reference logprobs from rollout payloads when present."""

    if not isinstance(payload, Mapping):
        return []

    normalized_outputs = payload.get("normalized_outputs")
    if isinstance(normalized_outputs, Sequence) and not isinstance(
        normalized_outputs, (str, bytes)
    ):
        groups = _reference_logp_groups_from_grouped_outputs(normalized_outputs)
        if groups:
            return groups

    outputs = payload.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return _reference_logp_groups_from_grouped_outputs(outputs)
    return []


def _candidate_records_from_payload(payload: Any) -> list[tuple[list[int], list[int]]]:
    if not isinstance(payload, Mapping):
        return []

    normalized_outputs = payload.get("normalized_outputs")
    if isinstance(normalized_outputs, Sequence) and not isinstance(
        normalized_outputs,
        (str, bytes),
    ):
        records = _candidate_records_from_grouped_outputs(normalized_outputs)
        if records:
            return records

    outputs = payload.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return _candidate_records_from_grouped_outputs(outputs)
    return []


def _candidate_records_from_grouped_outputs(
    grouped_outputs: Sequence[Any],
) -> list[tuple[list[int], list[int]]]:
    records: list[tuple[list[int], list[int]]] = []
    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            candidates = group
        else:
            candidates = [group]
        for candidate in candidates:
            token_ids = _candidate_token_ids(candidate)
            if token_ids:
                records.append((_candidate_prompt_token_ids(candidate), token_ids))
    return records


def _attach_reward_tensor_to_grouped_candidates(
    payload: dict[str, Any],
    rewards: torch.Tensor,
) -> None:
    reward_values = [float(value) for value in rewards.detach().cpu().reshape(-1).tolist()]
    if not reward_values:
        return

    for key in ("normalized_outputs", "outputs"):
        grouped_outputs = payload.get(key)
        if not isinstance(grouped_outputs, Sequence) or isinstance(grouped_outputs, (str, bytes)):
            continue
        updated_outputs, used = _attach_rewards_to_grouped_outputs(
            grouped_outputs,
            reward_values,
        )
        if used == 0:
            continue
        if used != len(reward_values):
            raise ValueError(
                "stateless reward count must match rollout candidate count, got "
                f"{len(reward_values)} rewards for {used} candidates"
            )
        payload[key] = updated_outputs
        return


def _attach_reference_logps_to_grouped_candidates(
    payload: dict[str, Any],
    reference_logps: torch.Tensor,
    completion_mask: torch.Tensor,
) -> None:
    if reference_logps.shape != completion_mask.shape:
        raise ValueError("reference_logps shape must match completion_mask shape")

    mask = completion_mask.detach().to(dtype=torch.bool, device=reference_logps.device)
    logp_rows = [
        [float(value) for value in reference_logps[row][mask[row]].detach().cpu().tolist()]
        for row in range(reference_logps.shape[0])
    ]
    if not logp_rows:
        return

    for key in ("normalized_outputs", "outputs"):
        grouped_outputs = payload.get(key)
        if not isinstance(grouped_outputs, Sequence) or isinstance(grouped_outputs, (str, bytes)):
            continue
        updated_outputs, used = _attach_reference_logps_to_grouped_outputs(
            grouped_outputs,
            logp_rows,
        )
        if used == 0:
            continue
        if used != len(logp_rows):
            raise ValueError(
                "stateless reference_logps row count must match rollout candidate count, got "
                f"{len(logp_rows)} rows for {used} candidates"
            )
        payload[key] = updated_outputs
        return


def _candidate_groups_from_grouped_outputs(grouped_outputs: Sequence[Any]) -> list[list[list[int]]]:
    groups: list[list[list[int]]] = []
    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            candidates = group
        else:
            candidates = [group]
        candidate_group = []
        for candidate in candidates:
            token_ids = _candidate_token_ids(candidate)
            if token_ids:
                candidate_group.append(token_ids)
        if candidate_group:
            groups.append(candidate_group)
    return groups


def _reward_groups_from_grouped_outputs(grouped_outputs: Sequence[Any]) -> list[list[float]]:
    groups: list[list[float]] = []
    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            candidates = group
        else:
            candidates = [group]
        reward_group = []
        for candidate in candidates:
            reward = _candidate_reward(candidate)
            if reward is not None:
                reward_group.append(reward)
        if reward_group:
            groups.append(reward_group)
    return groups


def _reference_logp_groups_from_grouped_outputs(
    grouped_outputs: Sequence[Any],
) -> list[list[list[float]]]:
    groups: list[list[list[float]]] = []
    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            candidates = group
        else:
            candidates = [group]
        reference_group = []
        for candidate in candidates:
            reference_logps = _candidate_reference_logp_values(candidate)
            if reference_logps:
                reference_group.append(reference_logps)
        if reference_group:
            groups.append(reference_group)
    return groups


def _attach_rewards_to_grouped_outputs(
    grouped_outputs: Sequence[Any],
    rewards: Sequence[float],
) -> tuple[list[Any], int]:
    updated_groups: list[Any] = []
    reward_index = 0

    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            updated_candidates = []
            for candidate in group:
                updated_candidate, consumed = _attach_reward_to_candidate(
                    candidate,
                    rewards,
                    reward_index,
                )
                reward_index += consumed
                updated_candidates.append(updated_candidate)
            updated_groups.append(updated_candidates)
        else:
            updated_candidate, consumed = _attach_reward_to_candidate(
                group,
                rewards,
                reward_index,
            )
            reward_index += consumed
            updated_groups.append(updated_candidate)

    return updated_groups, reward_index


def _attach_reward_to_candidate(
    candidate: Any,
    rewards: Sequence[float],
    reward_index: int,
) -> tuple[Any, int]:
    token_ids = _candidate_token_ids(candidate)
    if not token_ids:
        return candidate, 0
    if reward_index >= len(rewards):
        raise ValueError(
            "stateless reward count must match rollout candidate count, got fewer rewards "
            "than candidates"
        )

    reward = float(rewards[reward_index])
    if isinstance(candidate, Mapping):
        updated = dict(candidate)
        updated["reward"] = reward
        updated["reward_source"] = "stateless_executor"
        return updated, 1

    try:
        updated = dict(vars(candidate))
    except TypeError:
        updated = {"candidate": candidate}
    updated["reward"] = reward
    updated["reward_source"] = "stateless_executor"
    return updated, 1


def _attach_reference_logps_to_grouped_outputs(
    grouped_outputs: Sequence[Any],
    logp_rows: Sequence[Sequence[float]],
) -> tuple[list[Any], int]:
    updated_groups: list[Any] = []
    row_index = 0

    for group in grouped_outputs:
        if isinstance(group, Sequence) and not isinstance(group, (str, bytes, Mapping)):
            updated_candidates = []
            for candidate in group:
                updated_candidate, consumed = _attach_reference_logps_to_candidate(
                    candidate,
                    logp_rows,
                    row_index,
                )
                row_index += consumed
                updated_candidates.append(updated_candidate)
            updated_groups.append(updated_candidates)
        else:
            updated_candidate, consumed = _attach_reference_logps_to_candidate(
                group,
                logp_rows,
                row_index,
            )
            row_index += consumed
            updated_groups.append(updated_candidate)

    return updated_groups, row_index


def _attach_reference_logps_to_candidate(
    candidate: Any,
    logp_rows: Sequence[Sequence[float]],
    row_index: int,
) -> tuple[Any, int]:
    token_ids = _candidate_token_ids(candidate)
    if not token_ids:
        return candidate, 0
    if row_index >= len(logp_rows):
        raise ValueError(
            "stateless reference_logps row count must match rollout candidate count, got "
            "fewer rows than candidates"
        )

    values = [float(value) for value in logp_rows[row_index]]
    if isinstance(candidate, Mapping):
        updated = dict(candidate)
        updated["reference_logps"] = values
        updated["reference_logp_source"] = "stateless_executor"
        return updated, 1

    try:
        updated = dict(vars(candidate))
    except TypeError:
        updated = {"candidate": candidate}
    updated["reference_logps"] = values
    updated["reference_logp_source"] = "stateless_executor"
    return updated, 1


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


def _candidate_prompt_token_ids(candidate: Any) -> list[int]:
    if candidate is None:
        return []
    if isinstance(candidate, Mapping):
        for key in ("prompt_token_ids", "prompt_ids", "input_token_ids", "input_ids"):
            if key in candidate:
                return _copy_int_list(candidate[key])
        return []

    for attr in ("prompt_token_ids", "prompt_ids", "input_token_ids", "input_ids"):
        value = getattr(candidate, attr, None)
        if value is not None:
            return _copy_int_list(value)
    return []


def _candidate_reward(candidate: Any) -> Optional[float]:
    if candidate is None:
        return None
    if isinstance(candidate, Mapping):
        nested_outputs = candidate.get("outputs")
        if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
            for nested in nested_outputs:
                reward = _candidate_reward(nested)
                if reward is not None:
                    return reward
        for key in ("reward", "score", "scalar_reward", "reward_score"):
            if key in candidate:
                return _safe_float(candidate[key])
        return None

    nested_outputs = getattr(candidate, "outputs", None)
    if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
        for nested in nested_outputs:
            reward = _candidate_reward(nested)
            if reward is not None:
                return reward
    for attr in ("reward", "score", "scalar_reward", "reward_score"):
        if hasattr(candidate, attr):
            return _safe_float(getattr(candidate, attr))
    return None


def _candidate_reference_logp_values(candidate: Any) -> list[float]:
    if candidate is None:
        return []
    if isinstance(candidate, Mapping):
        nested_outputs = candidate.get("outputs")
        if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
            for nested in nested_outputs:
                reference_logps = _candidate_reference_logp_values(nested)
                if reference_logps:
                    return reference_logps
        for key in ("reference_logps", "ref_logps", "reference_logprobs", "ref_logprobs"):
            if key in candidate:
                return _copy_float_list(candidate[key])
        return []

    nested_outputs = getattr(candidate, "outputs", None)
    if isinstance(nested_outputs, Sequence) and not isinstance(nested_outputs, (str, bytes)):
        for nested in nested_outputs:
            reference_logps = _candidate_reference_logp_values(nested)
            if reference_logps:
                return reference_logps
    for attr in ("reference_logps", "ref_logps", "reference_logprobs", "ref_logprobs"):
        if hasattr(candidate, attr):
            return _copy_float_list(getattr(candidate, attr))
    return []


def _safe_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() != 1:
            raise ValueError("rollout reward tensors must contain exactly one value")
        return float(flat[0].item())
    return float(value)


def _reward_groups_match(
    candidate_groups: Sequence[Sequence[Sequence[int]]],
    reward_groups: Sequence[Sequence[float]],
) -> bool:
    if len(candidate_groups) != len(reward_groups):
        return False
    return all(
        len(candidates) == len(rewards)
        for candidates, rewards in zip(candidate_groups, reward_groups, strict=False)
    )


def _reference_logp_groups_match(
    candidate_groups: Sequence[Sequence[Sequence[int]]],
    reference_logp_groups: Sequence[Sequence[Sequence[float]]],
    *,
    max_completion_len: int,
) -> bool:
    if len(candidate_groups) != len(reference_logp_groups):
        return False
    for candidates, reference_group in zip(
        candidate_groups,
        reference_logp_groups,
        strict=False,
    ):
        if len(candidates) != len(reference_group):
            return False
        for token_ids, reference_logps in zip(candidates, reference_group, strict=False):
            required_len = min(len(token_ids), max_completion_len)
            if len(reference_logps) < required_len:
                return False
    return True


def _candidate_reward_value(
    token_ids: Sequence[int],
    reward_groups: Optional[Sequence[Sequence[float]]],
    group_index: int,
    candidate_index: int,
    *,
    vocab_size: int,
) -> float:
    if reward_groups is not None:
        return float(reward_groups[group_index][candidate_index])
    if not token_ids:
        return 0.0
    clipped = [int(token) % vocab_size for token in token_ids]
    return float(sum(clipped)) / float(max(len(clipped), 1) * max(vocab_size - 1, 1))


def _candidate_reference_logps(
    reference_logp_groups: Sequence[Sequence[Sequence[float]]],
    group_index: int,
    candidate_index: int,
    *,
    completion_len: int,
) -> list[float]:
    return [
        float(value)
        for value in reference_logp_groups[group_index][candidate_index][:completion_len]
    ]


def objective_reference_logps(
    current_logps: torch.Tensor,
    batch: SyntheticRLKernelBatch,
) -> torch.Tensor:
    """Return payload reference logps when available, otherwise a synthetic offset."""

    if batch.metadata.get("reference_logp_source") != "payload_reference_logps":
        return current_logps.detach() - 0.02
    if batch.ref_logps.shape != current_logps.shape:
        raise ValueError("payload reference logps shape must match current logps shape")
    return batch.ref_logps.detach().to(device=current_logps.device, dtype=torch.float32)


def _compute_group_relative_advantages(
    rewards: torch.Tensor,
    *,
    group_ids: torch.Tensor,
    num_groups: int,
    eps: float,
) -> torch.Tensor:
    flat_rewards = rewards.reshape(-1).float()
    flat_group_ids = group_ids.reshape(-1).to(device=flat_rewards.device, dtype=torch.long)
    if flat_rewards.numel() != flat_group_ids.numel():
        raise ValueError("rewards and group_ids must contain the same number of elements")

    counts = flat_rewards.new_zeros(num_groups).index_add_(
        0,
        flat_group_ids,
        torch.ones_like(flat_rewards),
    )
    sums = flat_rewards.new_zeros(num_groups).index_add_(0, flat_group_ids, flat_rewards)
    sq_sums = flat_rewards.new_zeros(num_groups).index_add_(
        0,
        flat_group_ids,
        flat_rewards * flat_rewards,
    )
    safe_counts = counts.clamp_min(1.0)
    means = sums / safe_counts
    variances = (sq_sums / safe_counts) - means * means
    stds = torch.sqrt(torch.clamp(variances, min=0.0) + eps)
    advantages = (flat_rewards - means[flat_group_ids]) / stds[flat_group_ids]
    return advantages.to(dtype=rewards.dtype)


def _broadcast_sequence_advantages(
    sequence_advantages: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    return sequence_advantages.reshape(-1, 1) * completion_mask.to(
        device=sequence_advantages.device,
        dtype=sequence_advantages.dtype,
    )


def _copy_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    return []


def _copy_float_list(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(item) for item in value]
    return []
