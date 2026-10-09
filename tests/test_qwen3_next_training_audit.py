# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CPU regression of observations; these are not native model training acceptance."""

import random

import numpy as np
import pytest
import torch

from rl_engine.integrations.qwen3_next_training import (
    capture_rng_state,
    gradient_evidence,
    observe_native_optimizer_step,
    restore_rng_state,
    state_digest,
    tensor_digest,
)


def test_state_identity_preserves_bits_dtype_structure_and_key_order():
    assert tensor_digest(torch.tensor(0.0)) != tensor_digest(torch.tensor(-0.0))
    assert tensor_digest(torch.ones(2)) != tensor_digest(torch.ones(2, dtype=torch.bfloat16))
    assert state_digest({"a": [1], "b": 2}) == state_digest({"b": 2, "a": [1]})
    assert state_digest([1, 2]) != state_digest((1, 2))
    assert state_digest({1: "a"}) != state_digest({"1": "a"})
    value = torch.arange(12).reshape(3, 4).T
    assert tensor_digest(value, chunk_bytes=3) == tensor_digest(value.contiguous())


@pytest.mark.parametrize("value", [torch.tensor(float("nan")), float("inf"), object()])
def test_state_identity_rejects_unknown_or_nonfinite_state(value):
    with pytest.raises((ValueError, TypeError)):
        state_digest(value)


def test_rng_restores_python_numpy_torch_exactly():
    state = capture_rng_state()
    first = (random.random(), np.random.random(), torch.rand(4))
    restore_rng_state(state)
    second = (random.random(), np.random.random(), torch.rand(4))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_gradients_observe_main_grad_and_allow_inactive_expert():
    active, inactive = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
    active.main_grad = torch.ones(2)
    inactive.grad = torch.zeros(2)
    evidence = gradient_evidence(
        {"layer.router": active, "layer.expert42": inactive},
        required_groups={"routing": ["router"]},
    )
    assert evidence["active_groups"] == {"routing": ["layer.router"]}
    assert not evidence["parameters"]["layer.expert42"]["nonzero"]
    with pytest.raises(ValueError, match="Active paths"):
        gradient_evidence({"expert": inactive}, required_groups={"experts": ["expert"]})
    inactive.grad[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        gradient_evidence({"expert": inactive}, required_groups={"experts": ["expert"]})


class _Optimizer:
    """Test double only: exercise observer timing and cleanup, never acceptance."""

    def __init__(self, parameter, change=True):
        self.parameter, self.change = parameter, change

    def step(self):
        if self.change:
            with torch.no_grad():
                self.parameter.add_(1)
        self.parameter.grad = None
        return True, 1.0, 0


def test_observer_captures_before_native_clear_and_restores_method():
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer, evidence = _Optimizer(model.weight), {}
    with observe_native_optimizer_step(
        optimizer, [model], required_groups={"output": ["weight"]}, evidence=evidence
    ):
        optimizer.step()
    assert model.weight.grad is None
    assert "step" not in vars(optimizer)
    assert evidence["changed_parameters"] == ["chunk0.weight"]
    assert evidence["gradients"]["parameters"]["chunk0.weight"]["nonzero"]


@pytest.mark.parametrize("mode", ["no_call", "no_change", "double_call"])
def test_observer_rejects_incomplete_or_fake_update(mode):
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer = _Optimizer(model.weight, change=mode != "no_change")
    with pytest.raises(ValueError):
        with observe_native_optimizer_step(
            optimizer, [model], required_groups={"output": ["weight"]}, evidence={}
        ):
            if mode != "no_call":
                optimizer.step()
            if mode == "double_call":
                optimizer.step()
    assert "step" not in vars(optimizer)


def test_gradient_nonzero_is_judged_across_tensor_parallel_ranks():
    """A vocab-parallel shard with no tokens is all-zero locally yet trains as a whole."""
    embedding, router = torch.nn.Parameter(torch.ones(4, 2)), torch.nn.Parameter(torch.ones(2))
    embedding.main_grad = torch.zeros(4, 2)
    router.main_grad = torch.ones(2)
    groups = {"embedding": ["embedding"], "router": ["router"]}

    def other_rank_saw_tokens(flags):
        return [True for _ in flags]

    evidence = gradient_evidence(
        {"chunk0.embedding.word_embeddings.weight": embedding, "layer.router": router},
        required_groups=groups,
        reduce_flags=other_rank_saw_tokens,
    )
    record = evidence["parameters"]["chunk0.embedding.word_embeddings.weight"]
    assert record["nonzero"] is True and record["nonzero_local"] is False
    assert evidence["active_groups"]["embedding"] == ["chunk0.embedding.word_embeddings.weight"]
    with pytest.raises(ValueError, match="Active paths.*embedding"):
        gradient_evidence(
            {"chunk0.embedding.word_embeddings.weight": embedding, "layer.router": router},
            required_groups=groups,
            reduce_flags=lambda flags: list(flags),
        )
    with pytest.raises(ValueError, match="one per parameter"):
        gradient_evidence(
            {"layer.router": router},
            required_groups={"router": ["router"]},
            reduce_flags=lambda flags: [],
        )
