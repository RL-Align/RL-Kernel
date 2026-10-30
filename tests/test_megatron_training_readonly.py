# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""T05 P0: the observer must not change outputs, inputs, autograd, RNG or gradients."""

import pytest
import torch
from p7_doubles.fixtures import build_fixtures, vime_request
from test_megatron_training_readback import FakeMegatronLogpOperator

from rl_engine.integrations.megatron_training_readback import TrainingLogpObserver

pytestmark = pytest.mark.unit

FIXTURES = build_fixtures()


def _bytes(tensor):
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).clone()


def _run(observe: bool):
    request = vime_request(FIXTURES["normal"])
    request.logits = request.logits.float().requires_grad_(True)
    hidden_before = _bytes(request.context.hidden)
    logits_before = _bytes(request.logits)
    observer = TrainingLogpObserver()
    if observe:
        observer.install(FakeMegatronLogpOperator)
    try:
        rng_before = torch.get_rng_state()
        result = FakeMegatronLogpOperator()(request)
        rng_after = torch.get_rng_state()
        result.logp.sum().backward()
    finally:
        observer.uninstall()
    assert torch.equal(rng_before, rng_after), "observer consumed RNG state"
    assert torch.equal(_bytes(request.context.hidden), hidden_before), "input hidden mutated"
    assert torch.equal(_bytes(request.logits), logits_before), "input logits mutated"
    return result, request.logits.grad


def test_outputs_and_gradients_are_bitwise_identical_with_and_without_observer():
    plain_result, plain_grad = _run(observe=False)
    observed_result, observed_grad = _run(observe=True)
    assert torch.equal(_bytes(plain_result.logp), _bytes(observed_result.logp))
    assert torch.equal(_bytes(plain_grad), _bytes(observed_grad))
    assert observed_result.logp.requires_grad, "observer detached the autograd result"


def test_install_is_idempotent_and_uninstall_restores_the_exact_function():
    original = FakeMegatronLogpOperator.__dict__["__call__"]
    observer = TrainingLogpObserver()
    observer.install(FakeMegatronLogpOperator)
    observer.install(FakeMegatronLogpOperator)
    observer.uninstall()
    observer.uninstall()
    assert FakeMegatronLogpOperator.__dict__["__call__"] is original
    assert "_rlk_t05_training_observer" not in FakeMegatronLogpOperator.__dict__


def test_second_observer_is_refused():
    first, second = TrainingLogpObserver(), TrainingLogpObserver()
    first.install(FakeMegatronLogpOperator)
    try:
        with pytest.raises(RuntimeError):
            second.install(FakeMegatronLogpOperator)
    finally:
        first.uninstall()


def test_uninstall_refuses_when_wrapper_was_replaced():
    observer = TrainingLogpObserver()
    observer.install(FakeMegatronLogpOperator)
    marker = FakeMegatronLogpOperator.__dict__["_rlk_t05_training_observer"]
    other = TrainingLogpObserver()
    FakeMegatronLogpOperator._rlk_t05_training_observer = other, marker[1]
    try:
        with pytest.raises(RuntimeError):
            observer.uninstall()
    finally:
        FakeMegatronLogpOperator.__call__ = marker[1]
        delattr(FakeMegatronLogpOperator, "_rlk_t05_training_observer")


def test_instance_identity_is_preserved_for_integration_readback():
    operator = FakeMegatronLogpOperator()
    registry = {"logp": operator}
    observer = TrainingLogpObserver()
    observer.install(FakeMegatronLogpOperator)
    try:
        assert registry["logp"] is operator
        operator(vime_request(FIXTURES["normal"]))
    finally:
        observer.uninstall()
    assert len(observer.snapshots) == 1


def test_real_megatron_operator_class_round_trips():
    from rl_engine.integrations.framework_operators import MegatronLogpOperator

    original = MegatronLogpOperator.__dict__["__call__"]
    observer = TrainingLogpObserver()
    observer.install()
    try:
        assert MegatronLogpOperator.__call__ is not original
        assert MegatronLogpOperator.__call__.__wrapped__ is original
    finally:
        observer.uninstall()
    assert MegatronLogpOperator.__dict__["__call__"] is original
