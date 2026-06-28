# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

import time

import torch

from rl_engine.executors.stateless_executor import StatelessForwardConfig, StatelessForwardExecutor
from rl_engine.executors.training_contract import (
    RolloutBatchMixin,
    RolloutStageResult,
    StatelessScoringWorker,
    TorchRLTrainingConfig,
    build_stateless_inputs_from_rollout_payload,
    extract_rollout_reference_logp_groups,
    extract_rollout_reward_groups,
)


class BatchOnlyWorker(RolloutBatchMixin):
    def __init__(self, config: TorchRLTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)


def _rollout(payload) -> RolloutStageResult:
    return RolloutStageResult(
        iteration=0,
        weight_version=4,
        payload=payload,
        started_at=time.perf_counter(),
        finished_at=time.perf_counter(),
    )


def test_build_stateless_inputs_from_rollout_payload_preserves_candidate_order_and_masks():
    payload = {
        "normalized_outputs": [
            [{"token_ids": [1, 2, 3]}, {"token_ids": [4]}],
            [{"token_ids": [5, 6]}],
        ]
    }

    inputs = build_stateless_inputs_from_rollout_payload(
        payload,
        prompt_len=2,
        prompt_token_id=9,
        max_completion_len=2,
    )

    assert inputs.input_ids.tolist() == [
        [9, 9, 1, 2],
        [9, 9, 4, 9],
        [9, 9, 5, 6],
    ]
    assert inputs.attention_mask.tolist() == [
        [True, True, True, True],
        [True, True, True, False],
        [True, True, True, True],
    ]
    assert inputs.completion_mask.tolist() == [
        [False, False, True, True],
        [False, False, True, False],
        [False, False, True, True],
    ]


def test_build_stateless_inputs_preserves_payload_prompt_token_ids_when_available():
    payload = {
        "normalized_outputs": [
            [{"prompt_token_ids": [7, 8, 9], "token_ids": [1, 2]}],
            [{"prompt_token_ids": [5], "token_ids": [3]}],
        ]
    }

    inputs = build_stateless_inputs_from_rollout_payload(
        payload,
        prompt_len=None,
        max_completion_len=2,
    )

    assert inputs.input_ids.tolist() == [
        [7, 8, 9, 1, 2],
        [5, 0, 0, 3, 0],
    ]
    assert inputs.attention_mask.tolist() == [
        [True, True, True, True, True],
        [True, False, False, True, False],
    ]
    assert inputs.completion_mask.tolist() == [
        [False, False, False, True, True],
        [False, False, False, True, False],
    ]


def test_stateless_scoring_worker_attaches_rewards_for_training_payload():
    class FakeRewardScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_cache_calls = []

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            del attention_mask
            self.use_cache_calls.append(use_cache)
            rewards = torch.tensor([1.0, 3.0], device=input_ids.device)
            return {"rewards": rewards}

    payload = {
        "normalized_outputs": [
            [
                {"token_ids": [1, 2]},
                {"token_ids": [3, 4]},
            ]
        ]
    }
    scorer_model = FakeRewardScorer()
    scorer = StatelessScoringWorker(
        StatelessForwardExecutor(
            scorer_model,
            StatelessForwardConfig(mode="reward"),
        ),
        lambda rollout: build_stateless_inputs_from_rollout_payload(
            rollout.payload,
            prompt_len=1,
            device="cpu",
        ),
    )

    scored = scorer.score(_rollout(payload))

    assert scorer_model.use_cache_calls == [False]
    assert scored.payload["stateless_scores"]["rewards"].tolist() == [1.0, 3.0]
    assert extract_rollout_reward_groups(scored.payload) == [[1.0, 3.0]]
    assert scored.metrics["scoring_backend"] == "stateless"
    assert scored.metrics["scoring_mode"] == "reward"
    assert scored.metrics["scoring_zero_kv_cache"] is True
    assert scored.metrics["scoring_attention_backend"] == "flash_attention_2"

    worker = BatchOnlyWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            valid_density=1.0,
            seed=17,
        )
    )
    batch, payload_metrics = worker._batch_from_rollout_or_synthetic(scored)

    assert payload_metrics["reward_source"] == "payload_rewards"
    assert payload_metrics["rollout_prompt_groups"] == 1
    assert batch.rewards.tolist() == [1.0, 3.0]
    assert torch.allclose(
        batch.advantages,
        torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=batch.advantages.dtype),
        atol=1e-5,
    )


def test_stateless_scoring_worker_attaches_reference_logps_for_training_payload():
    class FakeReferenceRewardScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_cache_calls = []

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            del attention_mask
            self.use_cache_calls.append(use_cache)
            logits = torch.full((*input_ids.shape, 16), -5.0, device=input_ids.device)
            for row in range(input_ids.shape[0]):
                for pos in range(input_ids.shape[1] - 1):
                    token = int(input_ids[row, pos + 1].item())
                    logits[row, pos, token] = 5.0 + row + pos
            return {
                "logits": logits,
                "rewards": torch.tensor([1.0, 3.0], device=input_ids.device),
            }

    payload = {
        "normalized_outputs": [
            [
                {"token_ids": [1, 2]},
                {"token_ids": [3, 4]},
            ]
        ]
    }
    scorer_model = FakeReferenceRewardScorer()
    scorer = StatelessScoringWorker(
        StatelessForwardExecutor(
            scorer_model,
            StatelessForwardConfig(mode="both"),
        ),
        lambda rollout: build_stateless_inputs_from_rollout_payload(
            rollout.payload,
            prompt_len=1,
            device="cpu",
        ),
    )

    scored = scorer.score(_rollout(payload))

    assert scorer_model.use_cache_calls == [False]
    assert extract_rollout_reward_groups(scored.payload) == [[1.0, 3.0]]
    assert scored.metrics["scoring_zero_kv_cache"] is True
    reference_groups = extract_rollout_reference_logp_groups(scored.payload)
    assert len(reference_groups) == 1
    assert [len(row) for row in reference_groups[0]] == [2, 2]
    assert scored.payload["normalized_outputs"][0][0]["reference_logp_source"] == (
        "stateless_executor"
    )

    worker = BatchOnlyWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=2,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            valid_density=1.0,
            seed=19,
        )
    )
    batch, payload_metrics = worker._batch_from_rollout_or_synthetic(scored)

    assert payload_metrics["reward_source"] == "payload_rewards"
    assert payload_metrics["reference_logp_source"] == "payload_reference_logps"
    assert torch.allclose(
        batch.ref_logps,
        torch.tensor(reference_groups[0], dtype=batch.ref_logps.dtype),
    )


def test_training_contract_accepts_reference_logps_for_truncated_completions():
    payload = {
        "normalized_outputs": [
            [
                {
                    "token_ids": [1, 2, 3],
                    "reward": 1.0,
                    "reference_logps": [-0.1, -0.2],
                }
            ]
        ]
    }
    worker = BatchOnlyWorker(
        TorchRLTrainingConfig(
            num_prompts=1,
            samples_per_prompt=1,
            prompt_len=1,
            completion_len=2,
            vocab_size=16,
            valid_density=1.0,
        )
    )

    batch, payload_metrics = worker._batch_from_rollout_or_synthetic(_rollout(payload))

    assert payload_metrics["reference_logp_source"] == "payload_reference_logps"
    assert batch.token_ids.tolist() == [[1, 2]]
    assert torch.allclose(
        batch.ref_logps,
        torch.tensor([[-0.1, -0.2]], dtype=batch.ref_logps.dtype),
    )
