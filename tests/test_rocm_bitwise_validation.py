# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from examples.vime_rocm_attention_ablation.validate_artifacts import (
    SIDECAR_SCHEMA_VERSION,
    compare_train_rollout_logps,
)


@pytest.mark.parametrize("rollout_zero,passed", [(0.0, True), (-0.0, False)])
def test_strict_validator_compares_bits_including_signed_zero(tmp_path, rollout_zero, passed):
    torch.save(
        {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "rank": 0,
            "call_index": 0,
            "train_log_probs": [torch.tensor([0.0, -1.0])],
            "rollout_log_probs": [torch.tensor([rollout_zero, -1.0])],
            "loss_masks": [torch.ones(2)],
            "total_lengths": [3],
            "response_lengths": [2],
        },
        tmp_path / "sidecar.pt",
    )
    result = compare_train_rollout_logps(tmp_path, require_exact=True)
    assert result["passed"] is passed
    assert result["bitwise_equal"] is passed
    assert result["bitwise_mismatch_count"] == (0 if passed else 1)
    assert result["torch_equal"] is True
