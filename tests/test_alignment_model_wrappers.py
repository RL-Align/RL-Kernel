# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from rl_engine.alignment import PolicyModelWrapper, ReferenceModelWrapper, extract_logits
from rl_engine.testing import selected_logprobs_reference


@dataclass
class ObjectOutput:
    logits: torch.Tensor


class FixedLogitsModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor, *, output_kind: str = "tensor"):
        super().__init__()
        self.output_kind = output_kind
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("base_logits", logits.clone())

    def forward(self, input_ids: torch.Tensor, *, logit_bias: float = 0.0):
        del input_ids
        logits = self.base_logits * self.weight + float(logit_bias)
        if self.output_kind == "tensor":
            return logits
        if self.output_kind == "mapping":
            return {"logits": logits}
        if self.output_kind == "object":
            return ObjectOutput(logits)
        if self.output_kind == "tuple":
            return (logits, {"hidden_states": None})
        raise AssertionError(f"unknown output kind: {self.output_kind}")


@pytest.mark.parametrize("output_kind", ("tensor", "mapping", "object", "tuple"))
def test_extract_logits_accepts_common_model_outputs(output_kind: str):
    logits = torch.randn(2, 3, 5)
    model = FixedLogitsModel(logits, output_kind=output_kind)

    actual = extract_logits(model(torch.zeros(2, 3, dtype=torch.long)))

    assert torch.allclose(actual, logits)


def test_extract_logits_rejects_missing_or_invalid_logits():
    with pytest.raises(TypeError, match="logits"):
        extract_logits({"hidden_states": torch.randn(1, 2)})

    with pytest.raises(TypeError, match="torch.Tensor"):
        extract_logits(ObjectOutput(logits="not-a-tensor"))


def test_policy_wrapper_keeps_model_trainable():
    model = FixedLogitsModel(torch.randn(2, 3, 7))

    PolicyModelWrapper(model)

    assert any(parameter.requires_grad for parameter in model.parameters())
    assert model.training is True


def test_reference_wrapper_freezes_and_evals_model():
    model = FixedLogitsModel(torch.randn(2, 3, 7))
    model.train()

    wrapper = ReferenceModelWrapper(model)

    assert wrapper.training is False
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_policy_wrapper_selected_logprobs_matches_reference():
    logits = torch.randn(2, 3, 7)
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    mask = torch.tensor([[True, False, True], [True, True, False]])
    input_ids = torch.zeros_like(token_ids)
    model = FixedLogitsModel(logits, output_kind="mapping")
    wrapper = PolicyModelWrapper(model)

    actual = wrapper.selected_logprobs(input_ids, token_ids, mask=mask, logit_bias=0.5)
    expected_logits = model(input_ids, logit_bias=0.5)["logits"]
    expected = selected_logprobs_reference(expected_logits, token_ids, mask=mask)

    assert torch.allclose(actual, expected)
    assert torch.equal(actual[~mask], torch.zeros_like(actual[~mask]))


def test_policy_wrapper_selected_logprobs_preserves_gradient():
    logits = torch.randn(2, 3, 7)
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    input_ids = torch.zeros_like(token_ids)
    model = FixedLogitsModel(logits)
    wrapper = PolicyModelWrapper(model)

    logps = wrapper.selected_logprobs(input_ids, token_ids)
    loss = logps.sum()
    loss.backward()

    assert logps.requires_grad is True
    assert model.weight.grad is not None


def test_reference_wrapper_selected_logprobs_does_not_track_gradient():
    logits = torch.randn(2, 3, 7)
    token_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    input_ids = torch.zeros_like(token_ids)
    model = FixedLogitsModel(logits)
    wrapper = ReferenceModelWrapper(model)

    logps = wrapper.selected_logprobs(input_ids, token_ids)

    assert logps.requires_grad is False
    assert all(parameter.grad is None for parameter in model.parameters())
