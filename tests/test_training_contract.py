# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch

from rl_engine.executors.training_contract import (
    RolloutBatchMixin,
    TorchRLTrainingConfig,
    extract_rollout_token_groups,
    make_rollout_result,
)
from rl_engine.testing import teacher_forced_logprobs_reference


class _BatchBuilder(RolloutBatchMixin):
    def __init__(self):
        self.config = TorchRLTrainingConfig(
            prompt_len=2,
            completion_len=3,
            vocab_size=8,
            dtype=torch.float32,
            seed=17,
        )
        self.device = torch.device("cpu")


def _rollout_payload(*, token_groups: list[list[int]], sampling_params: dict) -> dict:
    return {
        "sampling_params": dict(sampling_params),
        "normalized_outputs": [
            [
                {
                    "token_ids": list(token_ids),
                    "sampling_params": dict(sampling_params),
                    "text": f"sample-{index}",
                }
            ]
            for index, token_ids in enumerate(token_groups)
        ],
    }


def test_rollout_token_extraction_ignores_sampling_params():
    token_groups = [[2, 1, 3], [4, 0, 2]]
    payloads = [
        _rollout_payload(token_groups=token_groups, sampling_params={"temperature": 0.1}),
        _rollout_payload(
            token_groups=token_groups,
            sampling_params={"temperature": 1.8, "top_p": 0.4, "top_k": 8},
        ),
    ]

    assert [extract_rollout_token_groups(payload) for payload in payloads] == [
        token_groups,
        token_groups,
    ]


def test_same_sampled_tokens_score_the_same_under_different_sampling_params():
    token_groups = [[2, 1, 3], [4, 0, 2]]
    logits = torch.tensor(
        [
            [[0.0, 0.5, 2.0, -1.0, 1.0, 0.2, -0.4, 0.1]] * 3,
            [[1.5, -0.5, 0.25, 0.75, 2.0, -1.0, 0.0, 0.3]] * 3,
        ]
    )
    builder = _BatchBuilder()

    scored_batches = []
    for sampling_params in (
        {"temperature": 0.1, "top_p": 0.95},
        {"temperature": 1.6, "top_p": 0.4, "top_k": 5},
    ):
        batch, metrics = builder._batch_from_rollout_or_synthetic(
            make_rollout_result(
                iteration=3,
                weight_version=1,
                payload=_rollout_payload(
                    token_groups=token_groups,
                    sampling_params=sampling_params,
                ),
            )
        )

        assert metrics["training_data_source"] == "rollout_payload"
        assert metrics["logprob_scoring_mode"] == "teacher_forcing"
        assert batch.metadata["logprob_scoring_mode"] == "teacher_forcing"
        scored_batches.append(
            teacher_forced_logprobs_reference(
                logits,
                batch.token_ids,
                mask=batch.completion_mask,
            )
        )

    assert torch.equal(scored_batches[0], scored_batches[1])


def test_teacher_forced_score_changes_when_fixed_token_sequence_changes():
    logits = torch.tensor(
        [
            [[0.0, 0.5, 2.0, -1.0, 1.0, 0.2, -0.4, 0.1]] * 3,
            [[1.5, -0.5, 0.25, 0.75, 2.0, -1.0, 0.0, 0.3]] * 3,
        ]
    )
    builder = _BatchBuilder()

    first, _ = builder._batch_from_rollout_or_synthetic(
        make_rollout_result(
            iteration=3,
            weight_version=1,
            payload=_rollout_payload(
                token_groups=[[2, 1, 3], [4, 0, 2]],
                sampling_params={"temperature": 0.7},
            ),
        )
    )
    second, _ = builder._batch_from_rollout_or_synthetic(
        make_rollout_result(
            iteration=3,
            weight_version=1,
            payload=_rollout_payload(
                token_groups=[[2, 1, 3], [0, 4, 2]],
                sampling_params={"temperature": 0.7},
            ),
        )
    )

    first_score = teacher_forced_logprobs_reference(
        logits,
        first.token_ids,
        mask=first.completion_mask,
    )
    second_score = teacher_forced_logprobs_reference(
        logits,
        second.token_ids,
        mask=second.completion_mask,
    )

    assert not torch.equal(first_score, second_score)
