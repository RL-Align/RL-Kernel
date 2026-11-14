# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from examples.operators.dsv4_4l import cycle


def test_single_strided_fp32_and_signed_zero_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(cycle, "ROOT", tmp_path)
    case = dict(id="one", prompt_token_ids=[1], response_token_ids=[2])
    strided = torch.tensor([0.0, 9.0])[::2]
    row = cycle.compare_values("equal", [case], [strided], [torch.tensor([0.0])], 0)[0]
    assert row["compared_tokens"] == 1 and row["mismatch_count"] == 0
    row = cycle.compare_values("bits", [case], [strided], [torch.tensor([-0.0])], 0)[0]
    assert row["mismatch_count"] == 1
    assert row["first_mismatch"]["rollout_bits"] == "00000000"
    assert row["first_mismatch"]["actor_bits"] == "80000000"


@pytest.mark.parametrize(
    "score", [torch.tensor([]), torch.tensor([float("nan")]), torch.tensor([float("inf")])]
)
def test_invalid_or_empty_scores_rejected(tmp_path, monkeypatch, score):
    monkeypatch.setattr(cycle, "ROOT", tmp_path)
    case = dict(id="invalid", prompt_token_ids=[1], response_token_ids=[2])
    with pytest.raises(AssertionError):
        cycle.compare_values("invalid", [case], [score], [score], 0)
